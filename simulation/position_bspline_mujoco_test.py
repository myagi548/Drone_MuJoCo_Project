
import time
import numpy as np
import mujoco
import mujoco.viewer

from navigation.obstacle_planner import (
    BoxObstacle,
    path_is_collision_free,
)

from navigation.multi_obstacle_bspline import (
    grid_route,
    simplify_route,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_PATH = "simulation/drone_multi_obstacle.xml"

START = np.array([0.0, 0.0, 1.0])

TARGETS = [
    np.array([3.0, 0.0, 1.0]),
    np.array([5.0, 0.0, 1.0]),
    np.array([7.0, 0.0, 1.0]),
    np.array([10.0, 0.0, 1.0]),
]

GOAL = TARGETS[-1]

OBSTACLES = [
    BoxObstacle([2.0, 0.0, 1.0], [0.8, 0.8, 2.0]),
    BoxObstacle([4.0, 1.5, 1.0], [0.8, 0.8, 2.0]),
    BoxObstacle([6.0, -1.5, 1.0], [0.8, 0.8, 2.0]),
    BoxObstacle([8.0, 0.0, 1.0], [0.8, 0.8, 2.0]),
]

# IMPORTANT: use the clearance that worked in your planner.
SAFETY_DISTANCE = 0.25

WAYPOINT_TOLERANCE = 0.25
GOAL_TOLERANCE = 0.30
MAX_SIMULATION_TIME = 120.0

MAX_SPEED = 0.60
MAX_ACCELERATION = 1.5

KP_POSITION = np.array([2.0, 2.0, 3.0])
KD_POSITION = np.array([2.5, 2.5, 3.0])

KP_ATTITUDE = np.array([8.0, 8.0, 4.0])
KD_ATTITUDE = np.array([2.5, 2.5, 2.0])

MAX_FORCE = 30.0
MAX_TORQUE = 8.0


# ==========================================================
# LOAD MUJOCO MODEL
# ==========================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

drone_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "drone"
)

if drone_id < 0:
    raise RuntimeError(
        "Could not find body named 'drone' in the XML."
    )

mass = float(model.body(drone_id).mass[0])
gravity_vector = np.asarray(model.opt.gravity, dtype=float)
gravity = abs(float(gravity_vector[2]))
dt = float(model.opt.timestep)

print("=" * 60)
print("MULTI-OBSTACLE WAYPOINT FLIGHT")
print("=" * 60)
print(f"Mass: {mass:.3f} kg")
print(f"Gravity: {gravity:.3f} m/s^2")
print(f"Timestep: {dt:.4f} s")
print(f"Safety distance: {SAFETY_DISTANCE:.2f} m")


# ==========================================================
# PLAN ROUTE
# ==========================================================

waypoint_list = []
current_start = START.copy()

print()
print("Planning routes...")

for target_number, target in enumerate(TARGETS, start=1):

    print(f"Planning target {target_number}: {target}")

    route = grid_route(
        current_start,
        target,
        OBSTACLES,
        SAFETY_DISTANCE
    )

    if route is None or len(route) == 0:
        raise RuntimeError(
            f"A* failed to find a route to target {target_number}."
        )

    route = simplify_route(
        route,
        OBSTACLES,
        SAFETY_DISTANCE
    )

    route = np.asarray(route, dtype=float)

    if not path_is_collision_free(
        route,
        OBSTACLES,
        SAFETY_DISTANCE
    ):
        raise RuntimeError(
            f"Route to target {target_number} failed validation."
        )

    if waypoint_list and np.linalg.norm(
        waypoint_list[-1] - route[0]
    ) < 1e-8:
        route = route[1:]

    waypoint_list.extend(route)

    current_start = target.copy()

waypoints = np.asarray(waypoint_list, dtype=float)

if len(waypoints) == 0:
    raise RuntimeError("No waypoints were generated.")

print()
print("=" * 60)
print("ROUTE VALIDATION")
print("=" * 60)
print(f"Targets: {len(TARGETS)}")
print(f"Waypoints: {len(waypoints)}")
print(
    "Collision check:",
    path_is_collision_free(
        waypoints,
        OBSTACLES,
        SAFETY_DISTANCE
    )
)

for i, point in enumerate(waypoints):
    print(f"Waypoint {i}: {point}")


# ==========================================================
# RESET DRONE STATE
# ==========================================================

data.qpos[0:3] = START
data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
data.qvel[:] = 0.0

mujoco.mj_forward(model, data)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def clamp_norm(vector, maximum):
    norm = np.linalg.norm(vector)

    if norm > maximum and norm > 1e-9:
        return vector * (maximum / norm)

    return vector


def get_euler_angles(quaternion):
    rotation_flat = np.zeros(9)

    mujoco.mju_quat2Mat(
        rotation_flat,
        quaternion
    )

    rotation = rotation_flat.reshape(3, 3)

    roll = np.arctan2(
        rotation[2, 1],
        rotation[2, 2]
    )

    pitch = np.arcsin(
        np.clip(
            -rotation[2, 0],
            -1.0,
            1.0
        )
    )

    yaw = np.arctan2(
        rotation[1, 0],
        rotation[0, 0]
    )

    return np.array([roll, pitch, yaw])


def get_desired_velocity(position, target):
    error = target - position
    distance = np.linalg.norm(error)

    if distance < 1e-8:
        return np.zeros(3)

    # Reduce speed near the waypoint so the drone can brake.
    speed = min(
        MAX_SPEED,
        np.sqrt(2.0 * 0.8 * distance)
    )

    return (error / distance) * speed


# ==========================================================
# FLIGHT LOOP
# ==========================================================

waypoint_index = 0
simulation_time = 0.0
last_print_time = 0.0

goal_reached = False
final_position = START.copy()

print()
print("=" * 60)
print("STARTING MUJOCO FLIGHT")
print("=" * 60)
print("Close the viewer to stop.")
print()

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        loop_start = time.perf_counter()

        # --------------------------------------------------
        # READ POSITION AND VELOCITY
        # --------------------------------------------------

        position = data.qpos[0:3].copy()
        velocity = data.qvel[0:3].copy()

        quaternion = data.qpos[3:7].copy()
        angular_velocity = data.qvel[3:6].copy()

        angles = get_euler_angles(quaternion)

        # --------------------------------------------------
        # SELECT WAYPOINT
        # --------------------------------------------------

        target = waypoints[waypoint_index]

        distance_to_waypoint = np.linalg.norm(
            target - position
        )

        speed = np.linalg.norm(velocity)

        # Advance only when close and moving slowly.
        if (
            distance_to_waypoint < WAYPOINT_TOLERANCE
            and speed < 0.25
        ):
            if waypoint_index < len(waypoints) - 1:
                waypoint_index += 1

            target = waypoints[waypoint_index]

        # --------------------------------------------------
        # POSITION CONTROLLER
        # --------------------------------------------------

        position_error = target - position

        desired_velocity = get_desired_velocity(
            position,
            target
        )

        desired_acceleration = (
            KP_POSITION * position_error
            + KD_POSITION * (desired_velocity - velocity)
        )

        desired_acceleration = clamp_norm(
            desired_acceleration,
            MAX_ACCELERATION
        )

        # --------------------------------------------------
        # DIRECT FORCE CONTROL
        # --------------------------------------------------

        # MuJoCo gravity is already included in the model.
        # The applied force compensates gravity and adds
        # the requested acceleration.
        desired_force = mass * (
            desired_acceleration - gravity_vector
        )

        desired_force = clamp_norm(
            desired_force,
            MAX_FORCE
        )

        # --------------------------------------------------
        # ATTITUDE STABILIZATION
        # --------------------------------------------------

        desired_angles = np.zeros(3)

        angle_error = desired_angles - angles

        desired_torque = (
            KP_ATTITUDE * angle_error
            - KD_ATTITUDE * angular_velocity
        )

        desired_torque = np.clip(
            desired_torque,
            -MAX_TORQUE,
            MAX_TORQUE
        )

        # --------------------------------------------------
        # APPLY FORCE AND TORQUE
        # --------------------------------------------------

        data.xfrc_applied[drone_id, 0:3] = desired_force
        data.xfrc_applied[drone_id, 3:6] = desired_torque

        # --------------------------------------------------
        # STEP SIMULATION
        # --------------------------------------------------

        mujoco.mj_step(model, data)
        viewer.sync()

        simulation_time += dt

        # --------------------------------------------------
        # STATUS PRINT
        # --------------------------------------------------

        if simulation_time - last_print_time >= 0.5:

            last_print_time = simulation_time

            final_position = data.qpos[0:3].copy()

            waypoint_error = np.linalg.norm(
                target - final_position
            )

            goal_error = np.linalg.norm(
                GOAL - final_position
            )

            print(
                f"t={simulation_time:6.2f}s | "
                f"WP={waypoint_index:3d}/"
                f"{len(waypoints)-1:3d} | "
                f"pos=["
                f"{final_position[0]:6.2f}, "
                f"{final_position[1]:6.2f}, "
                f"{final_position[2]:6.2f}] | "
                f"wp_err={waypoint_error:5.2f}m | "
                f"goal_err={goal_error:5.2f}m | "
                f"speed={np.linalg.norm(data.qvel[0:3]):5.2f}m/s"
            )

        # --------------------------------------------------
        # GOAL CHECK
        # --------------------------------------------------

        final_position = data.qpos[0:3].copy()

        final_distance = np.linalg.norm(
            GOAL - final_position
        )

        if (
            waypoint_index == len(waypoints) - 1
            and final_distance < GOAL_TOLERANCE
            and np.linalg.norm(data.qvel[0:3]) < 0.25
        ):
            goal_reached = True

            print()
            print("=" * 60)
            print("FINAL TARGET REACHED")
            print("=" * 60)
            print("Final position:", final_position)
            print("Goal:", GOAL)
            print(f"Final error: {final_distance:.4f} m")
            break

        # --------------------------------------------------
        # TIME LIMIT
        # --------------------------------------------------

        if simulation_time >= MAX_SIMULATION_TIME:

            print()
            print("=" * 60)
            print("SIMULATION TIME LIMIT REACHED")
            print("=" * 60)
            print("Final position:", final_position)
            print("Goal:", GOAL)
            print(f"Final error: {final_distance:.4f} m")
            break

        # --------------------------------------------------
        # REAL-TIME PACING
        # --------------------------------------------------

        elapsed = time.perf_counter() - loop_start
        remaining = dt - elapsed

        if remaining > 0:
            time.sleep(remaining)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print()
print("=" * 60)
print("FINAL SIMULATION SUMMARY")
print("=" * 60)
print(f"Targets planned: {len(TARGETS)}")
print(f"Waypoints: {len(waypoints)}")
print(f"Simulation time: {simulation_time:.3f} s")
print("Final position:", final_position)
print("Goal:", GOAL)
print("Goal reached:", goal_reached)
print("=" * 60)