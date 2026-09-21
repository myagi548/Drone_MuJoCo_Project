
"""
Multi-obstacle MuJoCo simulation — NO WIND.

Uses:
    simulation/drone_multi_obstacle.xml

This is a separate scenario. It does not modify the original
single-obstacle simulation or its controller files.
"""

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "simulation" / "drone_multi_obstacle.xml"


# ------------------------------------------------------------
# 2. SIMULATION SETTINGS
# ------------------------------------------------------------

DT = 0.002
MASS = 1.0
GRAVITY = 9.81

# Controller gains
KP_POS = np.array([2.0, 2.0, 5.0])
KD_POS = np.array([2.5, 2.5, 3.0])

KP_ATTITUDE = 8.0
KD_ATTITUDE = 2.5

MAX_HORIZONTAL_ACCEL = 2.0
MAX_VERTICAL_ACCEL = 5.0

TARGET_TOLERANCE = 0.25
MAX_SIMULATION_TIME = 100.0

# No wind is applied in this scenario.
WIND_FORCE = np.array([0.0, 0.0, 0.0])


# ------------------------------------------------------------
# 3. TARGETS AND SAFE WAYPOINTS
# ------------------------------------------------------------

# Each waypoint is [x, y, z].
#
# The route passes around the obstacles instead of flying
# directly through their centers.

WAYPOINTS = [
    np.array([0.0, 0.0, 1.0]),

    # Go around obstacle 1 at x=2, y=0.
    np.array([1.0, 1.0, 1.0]),
    np.array([3.0, 1.0, 1.0]),

    # Reach target 1.
    np.array([3.0, 0.0, 1.0]),

    # Go around obstacle 2 at x=4, y=1.5.
    np.array([3.0, 2.5, 1.0]),
    np.array([5.0, 2.5, 1.0]),

    # Reach target 2.
    np.array([5.0, 0.0, 1.0]),

    # Go around obstacle 3 at x=6, y=-1.5.
    np.array([5.0, -2.5, 1.0]),
    np.array([7.0, -2.5, 1.0]),

    # Reach target 3.
    np.array([7.0, 0.0, 1.0]),

    # Go around obstacle 4 at x=8, y=0.
    np.array([7.0, 1.0, 1.0]),
    np.array([9.0, 1.0, 1.0]),

    # Final target.
    np.array([10.0, 0.0, 1.0]),
]


# ------------------------------------------------------------
# 4. QUATERNION HELPER
# ------------------------------------------------------------

def quaternion_to_rotation_matrix(q):
    """
    MuJoCo freejoint quaternion order:
        [w, x, y, z]

    Returns the body-to-world rotation matrix.
    """

    w, x, y, z = q

    return np.array([
        [
            1 - 2 * (y*y + z*z),
            2 * (x*y - z*w),
            2 * (x*z + y*w),
        ],
        [
            2 * (x*y + z*w),
            1 - 2 * (x*x + z*z),
            2 * (y*z - x*w),
        ],
        [
            2 * (x*z - y*w),
            2 * (y*z + x*w),
            1 - 2 * (x*x + y*y),
        ],
    ])


# ------------------------------------------------------------
# 5. ATTITUDE ERROR
# ------------------------------------------------------------

def upright_attitude_error(q):
    """
    Calculates a small-angle attitude error that drives the
    drone back toward the upright identity orientation.

    The desired quaternion is [1, 0, 0, 0].
    """

    w, x, y, z = q

    # q and -q represent the same physical orientation.
    if w < 0:
        x, y, z = -x, -y, -z

    return np.array([x, y, z])


# ------------------------------------------------------------
# 6. POSITION CONTROLLER
# ------------------------------------------------------------

def position_controller(target, position, velocity):
    """
    PD position controller.

    Returns a desired world-frame acceleration.
    """

    position_error = target - position

    acceleration = (
        KP_POS * position_error
        - KD_POS * velocity
    )

    # Limit horizontal acceleration.
    horizontal = acceleration[:2]
    horizontal_norm = np.linalg.norm(horizontal)

    if horizontal_norm > MAX_HORIZONTAL_ACCEL:
        acceleration[:2] *= (
            MAX_HORIZONTAL_ACCEL / horizontal_norm
        )

    # Limit vertical acceleration.
    acceleration[2] = np.clip(
        acceleration[2],
        -MAX_VERTICAL_ACCEL,
        MAX_VERTICAL_ACCEL,
    )

    return acceleration


# ------------------------------------------------------------
# 7. APPLY FORCE AND TORQUE
# ------------------------------------------------------------

def apply_drone_control(model, data, target):
    """
    Applies a simplified force/torque controller to the
    drone's free body.

    This is a simplified simulation controller, not a
    physically calibrated four-rotor motor mixer.
    """

    drone_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "drone",
    )

    if drone_id < 0:
        raise RuntimeError("Could not find body named 'drone'.")

    # Freejoint position and velocity.
    position = data.qpos[0:3].copy()
    quaternion = data.qpos[3:7].copy()

    # For a freejoint, qvel[0:3] is translational velocity
    # and qvel[3:6] is angular velocity.
    velocity = data.qvel[0:3].copy()
    angular_velocity = data.qvel[3:6].copy()

    # Position controller.
    desired_acceleration = position_controller(
        target,
        position,
        velocity,
    )

    # Convert desired acceleration into world force.
    force_world = MASS * (
        desired_acceleration
        + np.array([0.0, 0.0, GRAVITY])
    )

    # Add wind disturbance (zero in this scenario).
    force_world += WIND_FORCE

    # Attitude controller: keep the drone upright.
    attitude_error = upright_attitude_error(quaternion)

    torque_body = (
        -KP_ATTITUDE * attitude_error
        -KD_ATTITUDE * angular_velocity
    )

    # Convert body-frame torque to world frame.
    rotation = quaternion_to_rotation_matrix(quaternion)
    torque_world = rotation @ torque_body

    # Clear previous applied wrench.
    data.xfrc_applied[drone_id, :] = 0.0

    # MuJoCo xfrc_applied layout:
    # [world force x,y,z, world torque x,y,z]
    data.xfrc_applied[drone_id, 0:3] = force_world
    data.xfrc_applied[drone_id, 3:6] = torque_world

    return position, velocity


# ------------------------------------------------------------
# 8. MAIN SIMULATION
# ------------------------------------------------------------

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MuJoCo model not found:\n{MODEL_PATH}"
        )

    print("Loading model:", MODEL_PATH)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    model.opt.timestep = DT

    # Reset to initial model state.
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    waypoint_index = 0
    simulation_time = 0.0
    last_print_time = -1.0

    print("\nNO-WIND MULTI-OBSTACLE SIMULATION")
    print("---------------------------------")
    print("Wind force:", WIND_FORCE)
    print("Waypoints:", len(WAYPOINTS))
    print("Starting simulation...\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            if simulation_time >= MAX_SIMULATION_TIME:
                print("Maximum simulation time reached.")
                break

            target = WAYPOINTS[waypoint_index]

            position, velocity = apply_drone_control(
                model,
                data,
                target,
            )

            # Advance MuJoCo physics.
            mujoco.mj_step(model, data)
            simulation_time += model.opt.timestep

            # Check whether current waypoint is reached.
            new_position = data.qpos[0:3].copy()
            distance = np.linalg.norm(target - new_position)

            if distance < TARGET_TOLERANCE:
                print(
                    f"Reached waypoint {waypoint_index + 1}/"
                    f"{len(WAYPOINTS)}: "
                    f"{target.round(2)}"
                )

                waypoint_index += 1

                if waypoint_index >= len(WAYPOINTS):
                    print("\nFINAL TARGET REACHED!")
                    break

            # Print progress twice per simulation second.
            if simulation_time - last_print_time >= 0.5:
                last_print_time = simulation_time

                print(
                    f"t={simulation_time:6.2f}s | "
                    f"pos={new_position.round(2)} | "
                    f"target={target.round(2)} | "
                    f"error={distance:.2f} m"
                )

            viewer.sync()
            time.sleep(DT)

    print("\nSimulation ended.")


if __name__ == "__main__":
    main()