
"""
Multi-obstacle MuJoCo simulation — NO WIND + COLLISION DETECTION.

Separate diagnostic scenario.
Does not modify the original single-obstacle baseline.

Note:
This uses a simplified direct force/torque controller,
not the four-rotor mixer.
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

KP_POS = np.array([2.0, 2.0, 5.0])
KD_POS = np.array([2.5, 2.5, 3.0])

KP_ATTITUDE = 8.0
KD_ATTITUDE = 2.5

MAX_HORIZONTAL_ACCEL = 2.0
MAX_VERTICAL_ACCEL = 5.0

TARGET_TOLERANCE = 0.25
MAX_SIMULATION_TIME = 100.0

WIND_FORCE = np.array([0.0, 0.0, 0.0])


# ------------------------------------------------------------
# 3. WAYPOINTS
# ------------------------------------------------------------

WAYPOINTS = [
    np.array([0.0, 0.0, 1.0]),
    np.array([1.0, 1.0, 1.0]),
    np.array([3.0, 1.0, 1.0]),
    np.array([3.0, 0.0, 1.0]),
    np.array([3.0, 2.5, 1.0]),
    np.array([5.0, 2.5, 1.0]),
    np.array([5.0, 0.0, 1.0]),
    np.array([5.0, -2.5, 1.0]),
    np.array([7.0, -2.5, 1.0]),
    np.array([7.0, 0.0, 1.0]),
    np.array([7.0, 1.0, 1.0]),
    np.array([9.0, 1.0, 1.0]),
    np.array([10.0, 0.0, 1.0]),
]


# ------------------------------------------------------------
# 4. QUATERNION HELPER
# ------------------------------------------------------------

def quaternion_to_rotation_matrix(q):
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
    w, x, y, z = q

    if w < 0:
        x, y, z = -x, -y, -z

    return np.array([x, y, z])


# ------------------------------------------------------------
# 6. POSITION CONTROLLER
# ------------------------------------------------------------

def position_controller(target, position, velocity):
    position_error = target - position

    acceleration = (
        KP_POS * position_error
        - KD_POS * velocity
    )

    horizontal_norm = np.linalg.norm(acceleration[:2])

    if horizontal_norm > MAX_HORIZONTAL_ACCEL:
        acceleration[:2] *= (
            MAX_HORIZONTAL_ACCEL / horizontal_norm
        )

    acceleration[2] = np.clip(
        acceleration[2],
        -MAX_VERTICAL_ACCEL,
        MAX_VERTICAL_ACCEL,
    )

    return acceleration


# ------------------------------------------------------------
# 7. APPLY SIMPLIFIED FORCE AND TORQUE
# ------------------------------------------------------------

def apply_drone_control(model, data, target, drone_id):
    position = data.qpos[0:3].copy()
    quaternion = data.qpos[3:7].copy()

    velocity = data.qvel[0:3].copy()
    angular_velocity = data.qvel[3:6].copy()

    desired_acceleration = position_controller(
        target,
        position,
        velocity,
    )

    force_world = MASS * (
        desired_acceleration
        + np.array([0.0, 0.0, GRAVITY])
    )

    force_world += WIND_FORCE

    attitude_error = upright_attitude_error(quaternion)

    torque_body = (
        -KP_ATTITUDE * attitude_error
        -KD_ATTITUDE * angular_velocity
    )

    rotation = quaternion_to_rotation_matrix(quaternion)
    torque_world = rotation @ torque_body

    data.xfrc_applied[drone_id, :] = 0.0
    data.xfrc_applied[drone_id, 0:3] = force_world
    data.xfrc_applied[drone_id, 3:6] = torque_world

    return position, velocity


# ------------------------------------------------------------
# 8. COLLISION DETECTION
# ------------------------------------------------------------

def detect_drone_collision(model, data, drone_id):
    """
    Return contact details for contacts involving a geom
    attached directly to the drone body.

    This checks MuJoCo's current contact list. It does not
    detect near misses or guarantee clearance between steps.
    """

    collisions = []

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]

        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])

        drone_contact = (
            (body1 == drone_id and body2 != 0)
            or
            (body2 == drone_id and body1 != 0)
        )

        if not drone_contact:
            continue

        name1 = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom1,
        ) or f"geom_{geom1}"

        name2 = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom2,
        ) or f"geom_{geom2}"

        collisions.append((name1, name2))

    return collisions


# ------------------------------------------------------------
# 9. MAIN SIMULATION
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

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    drone_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "drone",
    )

    if drone_id < 0:
        raise RuntimeError("Could not find body named 'drone'.")

    waypoint_index = 0
    simulation_time = 0.0
    last_print_time = -1.0

    collision_count = 0
    collided_pairs = set()
    collision_stop = False
    target_reached = False

    print("\nMULTI-OBSTACLE NO-WIND COLLISION TEST")
    print("-------------------------------------")
    print("Wind force:", WIND_FORCE)
    print("Waypoints:", len(WAYPOINTS))
    print("Starting simulation...\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            if simulation_time >= MAX_SIMULATION_TIME:
                print("Maximum simulation time reached.")
                break

            target = WAYPOINTS[waypoint_index]

            apply_drone_control(
                model,
                data,
                target,
                drone_id,
            )

            mujoco.mj_step(model, data)
            simulation_time += model.opt.timestep

            # Detect contacts after this physics step.
            collisions = detect_drone_collision(
                model,
                data,
                drone_id,
            )

            if collisions:
                collision_count += len(collisions)

                for name1, name2 in collisions:
                    pair = tuple(sorted((name1, name2)))

                    if pair not in collided_pairs:
                        collided_pairs.add(pair)

                        print(
                            "\nCOLLISION DETECTED!"
                            f"\n  Time: {simulation_time:.3f} s"
                            f"\n  Geoms: {name1} <-> {name2}"
                        )

                collision_stop = True
                break

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
                    target_reached = True
                    break

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

    print("\nSIMULATION SUMMARY")
    print("------------------")
    print(f"Simulation time: {simulation_time:.3f} s")
    print(f"Collision event count: {collision_count}")
    print(f"Unique contact pairs: {len(collided_pairs)}")
    print(f"Final position: {data.qpos[0:3].round(3)}")

    if collision_stop:
        print("RESULT: STOPPED AFTER DRONE CONTACT.")
    elif target_reached:
        print("RESULT: FINAL TARGET REACHED; NO CONTACT DETECTED.")
    else:
        print("RESULT: SIMULATION ENDED BEFORE FINAL TARGET.")


if __name__ == "__main__":
    main()