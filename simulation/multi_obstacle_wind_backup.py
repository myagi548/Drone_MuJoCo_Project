
"""
Multi-obstacle MuJoCo simulation with a visible wind response.

Wind scenario:
1. Drone follows waypoints.
2. A wind gust pushes the drone sideways.
3. Position control counters the disturbance.
4. The drone visibly tilts during the gust.
5. After the gust, the drone returns upright and resumes navigation.

IMPORTANT:
This is a simplified simulation controller.
Horizontal position force is controlled in world coordinates;
the tilt is commanded separately for a visible attitude response.
It is not a calibrated four-motor aerodynamic model.
"""

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# 1. MODEL PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "simulation" / "drone_multi_obstacle.xml"


# ============================================================
# 2. SETTINGS
# ============================================================

DT = 0.002
MASS = 1.0
GRAVITY = 9.81

# Position PD controller
KP_POS = np.array([2.0, 2.0, 5.0])
KD_POS = np.array([2.5, 2.5, 3.0])

# Attitude PD controller
KP_ATT = np.array([8.0, 8.0, 5.0])
KD_ATT = np.array([2.5, 2.5, 2.0])

# Maximum commanded tilt
MAX_TILT_DEG = 20.0
MAX_TILT = np.deg2rad(MAX_TILT_DEG)

# Limit acceleration to prevent runaway
MAX_HORIZONTAL_ACCEL = 3.0
MAX_VERTICAL_ACCEL = 4.0

TARGET_TOLERANCE = 0.30
MAX_SIMULATION_TIME = 100.0

# Wind window
WIND_START_TIME = 5.0
WIND_END_TIME = 10.0

# External horizontal force in Newtons
WIND_FORCE = np.array([2.0, 0.0, 0.0])

# Time to hold position after the gust
RECOVERY_HOLD_TIME = 1.5


# ============================================================
# 3. WAYPOINTS
# ============================================================

WAYPOINTS = [
    np.array([0.0, 0.0, 1.0]),

    # Avoid obstacle 1
    np.array([1.0, 1.0, 1.0]),
    np.array([3.0, 1.0, 1.0]),
    np.array([3.0, 0.0, 1.0]),

    # Avoid obstacle 2
    np.array([3.0, 2.5, 1.0]),
    np.array([5.0, 2.5, 1.0]),
    np.array([5.0, 0.0, 1.0]),

    # Avoid obstacle 3
    np.array([5.0, -2.5, 1.0]),
    np.array([7.0, -2.5, 1.0]),
    np.array([7.0, 0.0, 1.0]),

    # Avoid obstacle 4
    np.array([7.0, 1.0, 1.0]),
    np.array([9.0, 1.0, 1.0]),

    # Final target
    np.array([10.0, 0.0, 1.0]),
]


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def wrap_angle(angle):
    """Keep an angle between -pi and +pi."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def quaternion_to_euler(q):
    """
    Convert MuJoCo quaternion [w, x, y, z]
    into roll, pitch, yaw.
    """

    w, x, y, z = q

    roll = np.arctan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y)
    )

    sin_pitch = 2.0 * (w * y - z * x)
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)

    pitch = np.arcsin(sin_pitch)

    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z)
    )

    return np.array([roll, pitch, yaw])


def clamp_vector(vector, maximum):
    """Limit a vector's magnitude."""
    magnitude = np.linalg.norm(vector)

    if magnitude > maximum and magnitude > 0.0:
        return vector * (maximum / magnitude)

    return vector


# ============================================================
# 5. POSITION CONTROLLER
# ============================================================

def position_controller(target, position, velocity):
    """
    Position PD controller.

    Calculates desired acceleration from position error
    and velocity feedback.
    """

    position_error = target - position

    acceleration = (
        KP_POS * position_error
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
# 6. WIND-INDUCED TILT COMMAND
# ============================================================

def calculate_tilt_command(wind_force, wind_active):
    """
    Produce a visible tilt command during wind.

    This is a deliberately limited attitude response.
    It is not a full aerodynamic mapping from wind to tilt.
    """

    if not wind_active:
        return np.array([0.0, 0.0, 0.0])

    # Wind along +X produces a visible pitch command.
    # Wind along +Y produces a visible roll command.
    pitch_command = np.clip(
        -0.10 * wind_force[0],
        -MAX_TILT,
        MAX_TILT
    )

    roll_command = np.clip(
        0.10 * wind_force[1],
        -MAX_TILT,
        MAX_TILT
    )

    return np.array([
        roll_command,
        pitch_command,
        0.0
    ])


# ============================================================
# 7. APPLY CONTROL
# ============================================================

def apply_control(
    model,
    data,
    target,
    wind_force,
    wind_active
):
    """
    Apply world-frame position force and attitude torque.

    Wind is applied as an external force.
    The position controller responds using feedback.
    Attitude commands produce visible tilt during wind.
    """

    drone_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "drone"
    )

    if drone_id < 0:
        raise RuntimeError("Could not find body named 'drone'.")

    position = data.qpos[0:3].copy()
    quaternion = data.qpos[3:7].copy()

    velocity = data.qvel[0:3].copy()
    angular_velocity = data.qvel[3:6].copy()

    # --------------------------------------------------------
    # A. Position control
    # --------------------------------------------------------

    desired_acceleration = position_controller(
        target,
        position,
        velocity
    )

    # Force needed for desired acceleration plus gravity.
    control_force = MASS * (
        desired_acceleration
        + np.array([0.0, 0.0, GRAVITY])
    )

    # External wind force is added separately.
    total_force = control_force + wind_force

    # --------------------------------------------------------
    # B. Attitude control
    # --------------------------------------------------------

    actual_euler = quaternion_to_euler(quaternion)

    desired_euler = calculate_tilt_command(
        wind_force,
        wind_active
    )

    attitude_error = np.array([
        wrap_angle(desired_euler[0] - actual_euler[0]),
        wrap_angle(desired_euler[1] - actual_euler[1]),
        wrap_angle(desired_euler[2] - actual_euler[2])
    ])

    # PD torque command.
    torque_world = (
        KP_ATT * attitude_error
        - KD_ATT * angular_velocity
    )

    # Limit torque to avoid extreme angular acceleration.
    torque_world = np.clip(
        torque_world,
        -5.0,
        5.0
    )

    # --------------------------------------------------------
    # C. Apply wrench to drone body
    # --------------------------------------------------------

    data.xfrc_applied[drone_id, :] = 0.0

    # xfrc_applied:
    # [force_x, force_y, force_z, torque_x, torque_y, torque_z]
    data.xfrc_applied[drone_id, 0:3] = total_force
    data.xfrc_applied[drone_id, 3:6] = torque_world

    return (
        position,
        velocity,
        actual_euler,
        desired_euler,
        desired_acceleration
    )


# ============================================================
# 8. MAIN SIMULATION
# ============================================================

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    print("Loading model:", MODEL_PATH)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    model.opt.timestep = DT

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    simulation_time = 0.0
    waypoint_index = 0
    last_print_time = -1.0

    wind_active = False
    wind_finished = False
    recovery_start_time = None
    wind_hold_position = None

    print("\nMULTI-OBSTACLE WIND SIMULATION")
    print("--------------------------------")
    print("Wind force:", WIND_FORCE, "N")
    print(f"Wind starts at {WIND_START_TIME:.1f} s")
    print(f"Wind ends at   {WIND_END_TIME:.1f} s")
    print("Watch the drone tilt and drift during the gust.\n")

    reached_final_target = False

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            if simulation_time >= MAX_SIMULATION_TIME:
                print("\nMaximum simulation time reached.")
                break

            position = data.qpos[0:3].copy()

            # ------------------------------------------------
            # Wind starts
            # ------------------------------------------------

            if (
                not wind_active
                and not wind_finished
                and simulation_time >= WIND_START_TIME
            ):
                wind_active = True
                wind_hold_position = position.copy()

                print("\nWIND DISTURBANCE STARTED")
                print(
                    "Wind hold point:",
                    wind_hold_position.round(2)
                )

            # ------------------------------------------------
            # Choose target and wind
            # ------------------------------------------------

            if wind_active:
                target = wind_hold_position
                current_wind = WIND_FORCE.copy()

            elif (
                wind_finished
                and recovery_start_time is not None
                and simulation_time
                < recovery_start_time + RECOVERY_HOLD_TIME
            ):
                target = wind_hold_position
                current_wind = np.zeros(3)

            else:
                target = WAYPOINTS[waypoint_index]
                current_wind = np.zeros(3)

            # ------------------------------------------------
            # Apply controller and step physics
            # ------------------------------------------------

            (
                old_position,
                velocity,
                actual_euler,
                desired_euler,
                desired_acceleration
            ) = apply_control(
                model,
                data,
                target,
                current_wind,
                wind_active
            )

            mujoco.mj_step(model, data)
            simulation_time += model.opt.timestep

            new_position = data.qpos[0:3].copy()

            # ------------------------------------------------
            # Wind ends
            # ------------------------------------------------

            if (
                wind_active
                and simulation_time >= WIND_END_TIME
            ):
                wind_active = False
                wind_finished = True
                recovery_start_time = simulation_time

                print("\nWIND DISTURBANCE ENDED")
                print("Recovery hold started.\n")

            # ------------------------------------------------
            # Waypoint progression
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Progress display
            # ------------------------------------------------

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

                actual_roll = np.rad2deg(actual_euler[0])
                actual_pitch = np.rad2deg(actual_euler[1])

                cmd_roll = np.rad2deg(desired_euler[0])
                cmd_pitch = np.rad2deg(desired_euler[1])

                print(
                    f"t={simulation_time:6.2f}s | "
                    f"pos={new_position.round(2)} | "
                    f"roll={actual_roll:6.1f}° "
                    f"(cmd {cmd_roll:5.1f}°) | "
                    f"pitch={actual_pitch:6.1f}° "
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