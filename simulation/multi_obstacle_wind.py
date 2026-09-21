
"""
Multi-obstacle MuJoCo wind-response demonstration
with pole-placement-derived attitude PID gains.

The gains are calculated from the actual MuJoCo inertias.

Simplified model:
    I * angular_acceleration = control_torque

This is not a calibrated four-motor aerodynamic model.
"""

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# 1. MODEL
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_DIR
    / "simulation"
    / "drone_multi_obstacle.xml"
)


# ============================================================
# 2. SETTINGS
# ============================================================

DT = 0.002
GRAVITY = 9.81

# Pole-placement design specifications
SETTLING_TIME = 1.0
ZETA = 1.0
THIRD_POLE_FACTOR = 5.0

# Position controller (existing simplified PD controller)
KP_POS = np.array([2.0, 2.0, 5.0])
KD_POS = np.array([2.5, 2.5, 3.0])

# Attitude gains will be calculated from MuJoCo inertia.
KP_ATT = np.zeros(3)
KI_ATT = np.zeros(3)
KD_ATT = np.zeros(3)

MAX_TILT_DEG = 20.0
MAX_TILT = np.deg2rad(MAX_TILT_DEG)

MAX_HORIZONTAL_ACCEL = 3.0
MAX_VERTICAL_ACCEL = 4.0

MAX_TORQUE = 5.0
TARGET_TOLERANCE = 0.30
MAX_SIMULATION_TIME = 100.0

WIND_START_TIME = 5.0
WIND_END_TIME = 10.0
WIND_FORCE = np.array([2.0, 0.0, 0.0])

RECOVERY_HOLD_TIME = 1.5

# Limit integral accumulation to reduce windup.
MAX_INTEGRAL_ERROR = 0.5


# ============================================================
# 3. WAYPOINTS
# ============================================================

WAYPOINTS = [
    np.array([0.0, 0.0, 1.0]),

    # Obstacle 1
    np.array([1.0, 1.0, 1.0]),
    np.array([3.0, 1.0, 1.0]),
    np.array([3.0, 0.0, 1.0]),

    # Obstacle 2
    np.array([3.0, 2.5, 1.0]),
    np.array([5.0, 2.5, 1.0]),
    np.array([5.0, 0.0, 1.0]),

    # Obstacle 3
    np.array([5.0, -2.5, 1.0]),
    np.array([7.0, -2.5, 1.0]),
    np.array([7.0, 0.0, 1.0]),

    # Obstacle 4
    np.array([7.0, 1.0, 1.0]),
    np.array([9.0, 1.0, 1.0]),

    # Final target
    np.array([10.0, 0.0, 1.0]),
]


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def quaternion_to_euler(q):
    """MuJoCo quaternion order: w, x, y, z."""

    w, x, y, z = q

    roll = np.arctan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y)
    )

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))

    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z)
    )

    return np.array([roll, pitch, yaw])


def clamp_vector(vector, maximum):
    magnitude = np.linalg.norm(vector)

    if magnitude > maximum and magnitude > 0.0:
        return vector * maximum / magnitude

    return vector


def get_drone_body_id(model):
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "drone"
    )

    if body_id < 0:
        raise RuntimeError(
            "Could not find a MuJoCo body named 'drone'."
        )

    return body_id


# ============================================================
# 5. POLE-PLACEMENT GAIN CALCULATION
# ============================================================

def calculate_pid_gains(inertia):
    """
    Calculate PID gains by matching coefficients.

    Plant:
        I * theta_ddot = torque

    Desired polynomial:
        (s^2 + 2*zeta*wn*s + wn^2) * (s + p3)

    Settling-time approximation:
        wn = 4 / (zeta * settling_time)
    """

    wn = 4.0 / (ZETA * SETTLING_TIME)
    p3 = THIRD_POLE_FACTOR * wn

    kp = inertia * (
        wn**2
        + 2.0 * ZETA * wn * p3
    )

    ki = inertia * (
        wn**2 * p3
    )

    kd = inertia * (
        2.0 * ZETA * wn + p3
    )

    return kp, ki, kd, wn, p3


# ============================================================
# 6. PID CONTROLLER
# ============================================================

class AttitudePID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral_error = 0.0

    def reset_integral(self):
        self.integral_error = 0.0

    def update(
        self,
        desired_angle,
        actual_angle,
        actual_rate,
        dt
    ):
        error = wrap_angle(
            desired_angle - actual_angle
        )

        # Integral term
        self.integral_error += error * dt

        self.integral_error = np.clip(
            self.integral_error,
            -MAX_INTEGRAL_ERROR,
            MAX_INTEGRAL_ERROR
        )

        # Assume desired angular rate is zero.
        # Therefore derivative of error = -actual_rate.
        derivative_error = -actual_rate

        torque = (
            self.kp * error
            + self.ki * self.integral_error
            + self.kd * derivative_error
        )

        return torque


# ============================================================
# 7. POSITION CONTROLLER
# ============================================================

def position_controller(target, position, velocity):
    error = target - position

    acceleration = (
        KP_POS * error
        - KD_POS * velocity
    )

    acceleration[:2] = clamp_vector(
        acceleration[:2],
        MAX_HORIZONTAL_ACCEL
    )

    acceleration[2] = np.clip(
        acceleration[2],
        -MAX_VERTICAL_ACCEL,
        MAX_VERTICAL_ACCEL
    )

    return acceleration


# ============================================================
# 8. ATTITUDE COMMAND
# ============================================================

def calculate_tilt_command(wind_active):
    """
    Explicit demonstration tilt during the wind gust.

    +X wind uses a negative pitch command.
    After the gust, desired attitude returns to zero.
    """

    if not wind_active:
        return np.zeros(3)

    pitch_command = np.deg2rad(-12.0)

    return np.array([
        0.0,
        np.clip(
            pitch_command,
            -MAX_TILT,
            MAX_TILT
        ),
        0.0
    ])


# ============================================================
# 9. APPLY CONTROL
# ============================================================

def apply_control(
    model,
    data,
    body_id,
    target,
    wind_active,
    roll_pid,
    pitch_pid,
    yaw_pid
):
    position = data.qpos[0:3].copy()
    quaternion = data.qpos[3:7].copy()

    velocity = data.qvel[0:3].copy()

    actual_euler = quaternion_to_euler(
        quaternion
    )

    # --------------------------------------------------------
    # Position control
    # --------------------------------------------------------

    desired_acceleration = position_controller(
        target,
        position,
        velocity
    )

    control_force = (
        desired_acceleration
        + np.array([0.0, 0.0, GRAVITY])
    )

    if wind_active:
        total_force = (
            control_force + WIND_FORCE
        )
    else:
        total_force = control_force

    # --------------------------------------------------------
    # Attitude PID control
    # --------------------------------------------------------

    desired_euler = calculate_tilt_command(
        wind_active
    )

    # Free-joint angular velocity.
    # This simplified implementation uses qvel[3:6].
    angular_velocity = data.qvel[3:6].copy()

    roll_torque = roll_pid.update(
        desired_euler[0],
        actual_euler[0],
        angular_velocity[0],
        DT
    )

    pitch_torque = pitch_pid.update(
        desired_euler[1],
        actual_euler[1],
        angular_velocity[1],
        DT
    )

    yaw_torque = yaw_pid.update(
        desired_euler[2],
        actual_euler[2],
        angular_velocity[2],
        DT
    )

    torque = np.array([
        roll_torque,
        pitch_torque,
        yaw_torque
    ])

    torque = np.clip(
        torque,
        -MAX_TORQUE,
        MAX_TORQUE
    )

    # --------------------------------------------------------
    # Apply world-frame force and torque
    # --------------------------------------------------------

    data.xfrc_applied[body_id, :] = 0.0

    data.xfrc_applied[
        body_id, 0:3
    ] = total_force

    data.xfrc_applied[
        body_id, 3:6
    ] = torque

    return (
        position,
        velocity,
        actual_euler,
        desired_euler
    )


# ============================================================
# 10. MAIN
# ============================================================

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    print("Loading model:", MODEL_PATH)

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(model)

    model.opt.timestep = DT

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    body_id = get_drone_body_id(model)

    # --------------------------------------------------------
    # Read actual MuJoCo inertias
    # --------------------------------------------------------

    inertia = model.body(body_id).inertia.copy()

    IXX = float(inertia[0])
    IYY = float(inertia[1])
    IZZ = float(inertia[2])

    # Calculate roll, pitch, and yaw gains.
    roll_gains = calculate_pid_gains(IXX)
    pitch_gains = calculate_pid_gains(IYY)
    yaw_gains = calculate_pid_gains(IZZ)

    global KP_ATT, KI_ATT, KD_ATT

    KP_ATT = np.array([
        roll_gains[0],
        pitch_gains[0],
        yaw_gains[0]
    ])

    KI_ATT = np.array([
        roll_gains[1],
        pitch_gains[1],
        yaw_gains[1]
    ])

    KD_ATT = np.array([
        roll_gains[2],
        pitch_gains[2],
        yaw_gains[2]
    ])

    # Create independent PID controllers.
    roll_pid = AttitudePID(
        KP_ATT[0],
        KI_ATT[0],
        KD_ATT[0]
    )

    pitch_pid = AttitudePID(
        KP_ATT[1],
        KI_ATT[1],
        KD_ATT[1]
    )

    yaw_pid = AttitudePID(
        KP_ATT[2],
        KI_ATT[2],
        KD_ATT[2]
    )

    print("\nPOLE-PLACEMENT PID GAINS")
    print("------------------------")
    print(f"Mass: {model.body(body_id).mass[0]:.4f} kg")
    print(f"Ixx:  {IXX:.8f} kg m^2")
    print(f"Iyy:  {IYY:.8f} kg m^2")
    print(f"Izz:  {IZZ:.8f} kg m^2")
    print(f"Settling-time target: {SETTLING_TIME:.2f} s")
    print(f"Damping ratio: {ZETA:.2f}")
    print(f"Natural frequency: {roll_gains[3]:.2f} rad/s")
    print(f"Third pole: {roll_gains[4]:.2f} rad/s")

    print("\nRoll PID:")
    print(
        f"Kp={KP_ATT[0]:.8f}, "
        f"Ki={KI_ATT[0]:.8f}, "
        f"Kd={KD_ATT[0]:.8f}"
    )

    print("\nPitch PID:")
    print(
        f"Kp={KP_ATT[1]:.8f}, "
        f"Ki={KI_ATT[1]:.8f}, "
        f"Kd={KD_ATT[1]:.8f}"
    )

    print("\nYaw PID:")
    print(
        f"Kp={KP_ATT[2]:.8f}, "
        f"Ki={KI_ATT[2]:.8f}, "
        f"Kd={KD_ATT[2]:.8f}"
    )

    # --------------------------------------------------------
    # Simulation state
    # --------------------------------------------------------

    simulation_time = 0.0
    waypoint_index = 0
    last_print_time = -1.0

    wind_active = False
    wind_finished = False

    recovery_start_time = None
    wind_hold_position = None

    reached_final_target = False

    print("\nMULTI-OBSTACLE WIND SIMULATION")
    print("--------------------------------")
    print("Wind force:", WIND_FORCE, "N")
    print(f"Wind starts: {WIND_START_TIME:.1f} s")
    print(f"Wind ends:   {WIND_END_TIME:.1f} s")
    print("Watch the drone and roll/pitch logs.\n")

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:

        while viewer.is_running():

            if simulation_time >= MAX_SIMULATION_TIME:
                print("\nMaximum simulation time reached.")
                break

            position = data.qpos[0:3].copy()

            # -----------------------------------------------
            # Start wind gust
            # -----------------------------------------------

            if (
                not wind_active
                and not wind_finished
                and simulation_time >= WIND_START_TIME
            ):
                wind_active = True
                wind_hold_position = position.copy()

                print("\nWIND DISTURBANCE STARTED")
                print(
                    "Hold position:",
                    wind_hold_position.round(2)
                )

            # -----------------------------------------------
            # Select target
            # -----------------------------------------------

            recovery_hold = (
                wind_finished
                and recovery_start_time is not None
                and simulation_time
                < recovery_start_time + RECOVERY_HOLD_TIME
            )

            if wind_active or recovery_hold:
                target = wind_hold_position
            else:
                target = WAYPOINTS[waypoint_index]

            # -----------------------------------------------
            # Apply control and step
            # -----------------------------------------------

            (
                old_position,
                velocity,
                actual_euler,
                desired_euler
            ) = apply_control(
                model,
                data,
                body_id,
                target,
                wind_active,
                roll_pid,
                pitch_pid,
                yaw_pid
            )

            mujoco.mj_step(model, data)
            simulation_time += model.opt.timestep

            new_position = data.qpos[0:3].copy()

            # -----------------------------------------------
            # End wind gust
            # -----------------------------------------------

            if (
                wind_active
                and simulation_time >= WIND_END_TIME
            ):
                wind_active = False
                wind_finished = True
                recovery_start_time = simulation_time

                # Clear integral memory at the mode transition.
                roll_pid.reset_integral()
                pitch_pid.reset_integral()
                yaw_pid.reset_integral()

                print("\nWIND DISTURBANCE ENDED")
                print("Recovery hold started.\n")

            # -----------------------------------------------
            # Waypoint progression
            # -----------------------------------------------

            recovery_hold = (
                wind_finished
                and recovery_start_time is not None
                and simulation_time
                < recovery_start_time + RECOVERY_HOLD_TIME
            )

            if not wind_active and not recovery_hold:

                target = WAYPOINTS[waypoint_index]

                distance = np.linalg.norm(
                    target - new_position
                )

                if distance < TARGET_TOLERANCE:

                    print(
                        f"Reached waypoint "
                        f"{waypoint_index + 1}/"
                        f"{len(WAYPOINTS)}: "
                        f"{target.round(2)}"
                    )

                    waypoint_index += 1

                    if waypoint_index >= len(WAYPOINTS):
                        reached_final_target = True
                        print("\nFINAL TARGET REACHED!")
                        break

            # -----------------------------------------------
            # Progress log
            # -----------------------------------------------

            if simulation_time - last_print_time >= 0.25:
                last_print_time = simulation_time

                if wind_active:
                    status = "WIND ACTIVE"
                elif recovery_hold:
                    status = "RECOVERING"
                elif wind_finished:
                    status = "WIND ENDED / NAVIGATING"
                else:
                    status = "NORMAL FLIGHT"

                roll_deg = np.rad2deg(
                    actual_euler[0]
                )

                pitch_deg = np.rad2deg(
                    actual_euler[1]
                )

                cmd_roll = np.rad2deg(
                    desired_euler[0]
                )

                cmd_pitch = np.rad2deg(
                    desired_euler[1]
                )

                print(
                    f"t={simulation_time:6.2f}s | "
                    f"pos={new_position.round(2)} | "
                    f"roll={roll_deg:6.1f}° "
                    f"(cmd {cmd_roll:5.1f}°) | "
                    f"pitch={pitch_deg:6.1f}° "
                    f"(cmd {cmd_pitch:5.1f}°) | "
                    f"{status}"
                )

            viewer.sync()
            time.sleep(DT)

    if reached_final_target:
        print("\nRESULT: Final target reached.")
    else:
        print("\nRESULT: Simulation stopped before final target.")
        print("Check the last position and waypoint in the log.")


if __name__ == "__main__":
    main()