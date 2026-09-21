
import time
import numpy as np
import mujoco
import mujoco.viewer

from control.position_controller import PositionController
from control.pid_controller import PIDController

from navigation.multi_obstacle_bspline import (
    START,
    TARGETS,
    OBSTACLES,
    SAFETY_DISTANCE,
    grid_route,
    simplify_route,
    smooth_route,
)

from navigation.obstacle_planner import path_is_collision_free

from simulation.mixer import mix_roll_pitch
from simulation.rotor_model import apply_rotor_forces


# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_PATH = "simulation/drone_multi_obstacle.xml"

TRAJECTORY_DURATION = 40.0

# Approximate extra allowance around the drone body.
# The planner's SAFETY_DISTANCE is used for route checking.
ARM_LENGTH = 0.45

MAX_TILT_DEG = 20.0
MAX_HORIZONTAL_ACCELERATION = 2.5


# ==========================================================
# LOAD MUJOCO MODEL
# ==========================================================

print("\nLoading MuJoCo model...")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)


# ==========================================================
# PHYSICAL PARAMETERS
# ==========================================================

mass = model.body("drone").mass[0]
Ixx = model.body("drone").inertia[0]
gravity = abs(model.opt.gravity[2])

dt = model.opt.timestep
hover_thrust = mass * gravity

print(f"Mass: {mass:.4f} kg")
print(f"Ixx: {Ixx:.6f} kg m^2")
print(f"Gravity: {gravity:.4f} m/s^2")
print(f"Time step: {dt:.4f} s")


# ==========================================================
# PID GAIN DESIGN
# ==========================================================

settling_time = 1.0
zeta = 1.0

wn = 4.0 / (zeta * settling_time)
p3 = 5.0 * wn

kp = Ixx * (
    wn**2 + 2.0 * zeta * wn * p3
)

ki = Ixx * (
    wn**2 * p3
)

kd = Ixx * (
    2.0 * zeta * wn + p3
)

roll_pid = PIDController(
    kp=kp,
    ki=ki,
    kd=kd,
    dt=dt,
)

pitch_pid = PIDController(
    kp=kp,
    ki=ki,
    kd=kd,
    dt=dt,
)


# ==========================================================
# POSITION CONTROLLER
# ==========================================================

position_controller = PositionController(
    gravity=gravity,
    horizontal_gain=2.0,
    horizontal_damping=2.5,
    vertical_gain=1.5,
    vertical_damping=2.5,
    max_tilt_rad=np.deg2rad(MAX_TILT_DEG),
    max_horizontal_acceleration=MAX_HORIZONTAL_ACCELERATION,
)


# ==========================================================
# MULTI-OBSTACLE TRAJECTORY PLANNING
# ==========================================================

print("\n" + "=" * 60)
print("MULTI-OBSTACLE B-SPLINE PLANNING")
print("=" * 60)

current = START.copy()

all_waypoints = [current.copy()]
all_trajectory = []

for target_number, target in enumerate(TARGETS, start=1):

    print(f"\nPlanning to target {target_number}: {target}")

    # A* route around all obstacles.
    route = grid_route(
        current,
        target,
        OBSTACLES,
        SAFETY_DISTANCE,
    )

    # Remove unnecessary intermediate waypoints.
    route = simplify_route(
        route,
        OBSTACLES,
        SAFETY_DISTANCE,
    )

    # Generate checked B-spline segments.
    segment = smooth_route(
        route,
        OBSTACLES,
        SAFETY_DISTANCE,
    )

    # Validate the segment before adding it.
    if not path_is_collision_free(
        segment,
        OBSTACLES,
        SAFETY_DISTANCE,
    ):
        raise RuntimeError(
            f"Trajectory to target {target_number} "
            "failed collision validation."
        )

    # Avoid duplicate points where segments join.
    if all_trajectory:
        segment = segment[1:]

    all_trajectory.extend(segment)
    all_waypoints.extend(route[1:])

    current = target.copy()

    print(f"Route points: {len(route)}")
    print(f"Segment samples: {len(segment)}")
    print("Segment collision check: PASS")


# Convert the complete path to a NumPy array.
trajectory = np.asarray(all_trajectory, dtype=float)

number_of_samples = len(trajectory)

if number_of_samples < 2:
    raise RuntimeError(
        "The planner generated too few trajectory samples."
    )


# ==========================================================
# FINAL TRAJECTORY VALIDATION
# ==========================================================

collision_free = path_is_collision_free(
    trajectory,
    OBSTACLES,
    SAFETY_DISTANCE,
)

if not collision_free:
    raise RuntimeError(
        "Combined trajectory failed collision validation. "
        "MuJoCo will not start."
    )


# ==========================================================
# TRAJECTORY TIMING AND VELOCITY
# ==========================================================

# This first integration maps the trajectory samples uniformly
# over the specified duration.
#
# The planner's samples are not necessarily equally spaced by
# distance, so this is a simple initial timing model, not a
# dynamically optimized trajectory.

trajectory_duration = TRAJECTORY_DURATION

sample_period = (
    trajectory_duration / (number_of_samples - 1)
)

desired_velocities = np.gradient(
    trajectory,
    sample_period,
    axis=0,
)


# ==========================================================
# DISPLAY PLANNING RESULTS
# ==========================================================

path_length = np.sum(
    np.linalg.norm(
        np.diff(trajectory, axis=0),
        axis=1,
    )
)

print("\n" + "=" * 60)
print("PLANNING RESULTS")
print("=" * 60)

print(f"Targets: {len(TARGETS)}")
print(f"Waypoints: {len(all_waypoints)}")
print(f"Trajectory samples: {number_of_samples}")
print(f"Duration: {trajectory_duration:.2f} s")
print(f"Path length: {path_length:.4f} m")
print(
    f"Average path speed: "
    f"{path_length / trajectory_duration:.4f} m/s"
)
print(f"Safety distance: {SAFETY_DISTANCE:.3f} m")
print(f"Collision validation: {collision_free}")

print("\nWaypoints:")
for i, waypoint in enumerate(all_waypoints):
    print(f"{i}: {waypoint}")

print("\nFirst trajectory point:", trajectory[0])
print("Final trajectory point:", trajectory[-1])


# ==========================================================
# INITIALIZE MUJOCO STATE
# ==========================================================

data.qpos[0:3] = START

data.qpos[3:7] = np.array([
    1.0,
    0.0,
    0.0,
    0.0,
])

data.qvel[:] = 0.0

data.xfrc_applied[:] = 0.0

mujoco.mj_forward(model, data)


# ==========================================================
# MUJOCO VIEWER
# ==========================================================

print("\n" + "=" * 60)
print("STARTING MULTI-OBSTACLE MUJOCO SIMULATION")
print("=" * 60)
print("Close the viewer window to stop early.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:

    simulation_time = 0.0
    last_print = 0.0

    while viewer.is_running():

        loop_start = time.perf_counter()

        # --------------------------------------------------
        # ACTUAL POSITION AND VELOCITY
        # --------------------------------------------------

        actual_position = data.qpos[0:3].copy()
        actual_velocity = data.qvel[0:3].copy()

        # --------------------------------------------------
        # ACTUAL ORIENTATION
        # --------------------------------------------------

        quaternion = data.qpos[3:7]

        rotation_matrix = np.zeros(9)

        mujoco.mju_quat2Mat(
            rotation_matrix,
            quaternion,
        )

        rotation_matrix = rotation_matrix.reshape(3, 3)

        # --------------------------------------------------
        # ROLL AND PITCH
        # --------------------------------------------------

        roll = np.arctan2(
            rotation_matrix[2, 1],
            rotation_matrix[2, 2],
        )

        pitch = np.arcsin(
            np.clip(
                -rotation_matrix[2, 0],
                -1.0,
                1.0,
            )
        )

        # --------------------------------------------------
        # ANGULAR VELOCITIES
        # --------------------------------------------------

        roll_rate = data.qvel[3]
        pitch_rate = data.qvel[4]

        # --------------------------------------------------
        # SELECT CURRENT TRAJECTORY SAMPLE
        # --------------------------------------------------

        normalized_time = np.clip(
            simulation_time / trajectory_duration,
            0.0,
            1.0,
        )

        trajectory_index = int(
            normalized_time * (number_of_samples - 1)
        )

        trajectory_index = int(
            np.clip(
                trajectory_index,
                0,
                number_of_samples - 1,
            )
        )

        target_position = trajectory[trajectory_index]
        target_velocity = desired_velocities[trajectory_index]

        # --------------------------------------------------
        # POSITION CONTROLLER
        # --------------------------------------------------

        (
            desired_roll,
            desired_pitch,
            desired_z_acceleration,
        ) = position_controller.compute(
            desired_position=target_position,
            actual_position=actual_position,
            actual_velocity=actual_velocity,
            desired_velocity=target_velocity,
        )

        # --------------------------------------------------
        # ROLL PID
        # --------------------------------------------------

        roll_torque = roll_pid.compute(
            desired_roll,
            roll,
            roll_rate,
        )

        # --------------------------------------------------
        # PITCH PID
        # --------------------------------------------------

        pitch_torque = pitch_pid.compute(
            desired_pitch,
            pitch,
            pitch_rate,
        )

        # --------------------------------------------------
        # TOTAL THRUST
        # --------------------------------------------------

        total_thrust = mass * (
            gravity + desired_z_acceleration
        )

        total_thrust = max(
            total_thrust,
            0.0,
        )

        # --------------------------------------------------
        # TILT COMPENSATION
        # --------------------------------------------------

        denominator = (
            np.cos(desired_roll)
            * np.cos(desired_pitch)
        )

        if denominator > 0.1:
            total_thrust /= denominator

        # --------------------------------------------------
        # MIXER
        # --------------------------------------------------

        thrusts = mix_roll_pitch(
            total_thrust=total_thrust,
            roll_torque=roll_torque,
            pitch_torque=pitch_torque,
            arm_length=ARM_LENGTH,
        )

        # This simple mixer has no motor saturation model.
        # Negative thrust is clipped here, as in the baseline.
        thrusts = np.maximum(
            thrusts,
            0.0,
        )

        # --------------------------------------------------
        # APPLY ROTOR FORCES
        # --------------------------------------------------

        apply_rotor_forces(
            model,
            data,
            thrusts,
        )

        # --------------------------------------------------
        # STEP MUJOCO
        # --------------------------------------------------

        mujoco.mj_step(
            model,
            data,
        )

        viewer.sync()

        simulation_time += dt

        # --------------------------------------------------
        # PERIODIC STATUS
        # --------------------------------------------------

        if simulation_time - last_print >= 0.5:

            last_print = simulation_time

            current_position = data.qpos[0:3].copy()

            target_error = np.linalg.norm(
                target_position - current_position
            )

            final_goal_distance = np.linalg.norm(
                TARGETS[-1] - current_position
            )

            print(
                f"t={simulation_time:6.2f}s | "
                f"sample={trajectory_index:4d}/"
                f"{number_of_samples - 1:4d} | "
                f"pos=["
                f"{current_position[0]:6.2f}, "
                f"{current_position[1]:6.2f}, "
                f"{current_position[2]:6.2f}] | "
                f"target_err={target_error:6.3f}m | "
                f"final_goal={final_goal_distance:6.3f}m | "
                f"roll={np.rad2deg(roll):7.2f} deg | "
                f"pitch={np.rad2deg(pitch):7.2f} deg"
            )

        # --------------------------------------------------
        # STOP WHEN TRAJECTORY ENDS
        # --------------------------------------------------

        if simulation_time >= trajectory_duration:

            final_position = data.qpos[0:3].copy()

            final_distance = np.linalg.norm(
                TARGETS[-1] - final_position
            )

            print("\n" + "=" * 60)
            print("TRAJECTORY TIME COMPLETE")
            print("=" * 60)

            print("Final position:", final_position)
            print("Final target:", TARGETS[-1])
            print(f"Final distance: {final_distance:.4f} m")

            if final_distance < 0.3:
                print("Final target threshold: REACHED")
            else:
                print("Final target threshold: NOT REACHED")

            print(
                "Note: reaching the final threshold does not "
                "prove every obstacle was avoided physically."
            )

            break

        # --------------------------------------------------
        # REAL-TIME PACING
        # --------------------------------------------------

        elapsed = time.perf_counter() - loop_start
        remaining = dt - elapsed

        if remaining > 0:
            time.sleep(remaining)


print("\nMuJoCo viewer closed.")