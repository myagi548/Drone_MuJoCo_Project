
import time
import numpy as np
import mujoco
import mujoco.viewer

from control.position_controller import PositionController
from control.pid_controller import PIDController

from navigation.planned_bspline import (
    find_bspline_safe_path,
    calculate_path_length,
    validate_bspline_trajectory,
)

from navigation.obstacle_planner import BoxObstacle

from simulation.mixer import mix_roll_pitch
from simulation.rotor_model import apply_rotor_forces


# ==========================================================
# LOAD MUJOCO MODEL
# ==========================================================

MODEL_PATH = "simulation/drone_model.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)


# ==========================================================
# PHYSICAL PARAMETERS
# ==========================================================

mass = model.body("drone").mass[0]
Ixx = model.body("drone").inertia[0]
gravity = abs(model.opt.gravity[2])

arm_length = 0.45
dt = model.opt.timestep

hover_thrust = mass * gravity


# ==========================================================
# PID DESIGN
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
    dt=dt
)

pitch_pid = PIDController(
    kp=kp,
    ki=ki,
    kd=kd,
    dt=dt
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
    max_tilt_rad=np.deg2rad(20.0),
    max_horizontal_acceleration=2.5
)


# ==========================================================
# START AND GOAL
# ==========================================================

start = np.array([0.0, 0.0, 1.0])
goal = np.array([4.0, 0.0, 1.0])


# ==========================================================
# OBSTACLE
# ==========================================================

# MuJoCo XML obstacle:
# center = [2, 0, 1]
# half-sizes = [0.4, 0.4, 1.0]
#
# Therefore its full dimensions are:
# [0.8, 0.8, 2.0]

obstacle = BoxObstacle(
    center=[2.0, 0.0, 1.0],
    size=[0.8, 0.8, 2.0]
)

obstacles = [obstacle]


# ==========================================================
# PLANNER PARAMETERS
# ==========================================================

# Drone arm length is 0.45 m.
# Add 0.25 m desired clearance around the drone.
#
# This is a conservative axis-aligned footprint allowance.

desired_body_clearance = 0.25

safety_distance = (
    arm_length + desired_body_clearance
)

initial_clearance = 0.05
clearance_step = 0.05
maximum_clearance = 1.00

degree = 3
number_of_samples = 401


# ==========================================================
# TRAJECTORY DURATION
# ==========================================================

trajectory_duration = 25.0


# ==========================================================
# FIND COLLISION-FREE B-SPLINE
# ==========================================================

print()
print("==================================================")
print("SEARCHING FOR B-SPLINE-SAFE PATH")
print("==================================================")

(
    waypoints,
    trajectory,
    selected_clearance,
    waypoint_path_length
) = find_bspline_safe_path(
    start=start,
    goal=goal,
    obstacles=obstacles,
    safety_distance=safety_distance,
    initial_clearance=initial_clearance,
    clearance_step=clearance_step,
    maximum_clearance=maximum_clearance,
    degree=degree,
    number_of_samples=number_of_samples
)


# ==========================================================
# FINAL COLLISION VALIDATION
# ==========================================================

collision_free = validate_bspline_trajectory(
    trajectory,
    obstacles,
    safety_distance
)

if not collision_free:
    raise RuntimeError(
        "B-spline failed collision validation. "
        "Simulation will not start."
    )


# ==========================================================
# PATH LENGTH AND DESIRED VELOCITY
# ==========================================================

bspline_path_length = calculate_path_length(
    trajectory
)

sample_period = (
    trajectory_duration / (number_of_samples - 1)
)

# Estimate the desired velocity at each trajectory sample.
# This gives the controller a velocity target to follow.

desired_velocities = np.gradient(
    trajectory,
    sample_period,
    axis=0
)


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

print()
print("==================================================")
print("POSITION + B-SPLINE + PID + MUJOCO")
print("==================================================")

print()
print("PHYSICAL PARAMETERS")
print("----------------------------------------------")
print(f"Mass                  = {mass:.6f} kg")
print(f"Ixx                   = {Ixx:.8f} kg m^2")
print(f"Gravity               = {gravity:.6f} m/s^2")
print(f"MuJoCo dt             = {dt:.6f} s")
print(f"Hover thrust          = {hover_thrust:.6f} N")
print(f"Rotor arm length      = {arm_length:.3f} m")

print()
print("PLANNING CLEARANCE")
print("----------------------------------------------")
print(
    f"Desired body clearance = "
    f"{desired_body_clearance:.3f} m"
)
print(
    f"Planner safety distance = "
    f"{safety_distance:.3f} m"
)

print()
print("PID GAINS")
print("----------------------------------------------")
print(f"Kp = {kp:.8f}")
print(f"Ki = {ki:.8f}")
print(f"Kd = {kd:.8f}")

print()
print("PATH PLANNING")
print("----------------------------------------------")
print(f"Selected clearance   = {selected_clearance:.3f} m")
print(f"Waypoint path length = {waypoint_path_length:.6f} m")
print(f"B-spline path length = {bspline_path_length:.6f} m")
print(f"Collision free       = {collision_free}")

print()
print("WAYPOINTS")
print("----------------------------------------------")

for i, waypoint in enumerate(waypoints):
    print(
        f"Waypoint {i}: "
        f"[{waypoint[0]:.3f}, "
        f"{waypoint[1]:.3f}, "
        f"{waypoint[2]:.3f}]"
    )

print()
print("TRAJECTORY")
print("----------------------------------------------")
print(f"Duration = {trajectory_duration:.2f} s")
print(
    f"Average path speed = "
    f"{bspline_path_length / trajectory_duration:.4f} m/s"
)
print("First point  =", trajectory[0])
print("Middle point =", trajectory[len(trajectory) // 2])
print("Final point  =", trajectory[-1])


# ==========================================================
# INITIAL MUJOCO STATE
# ==========================================================

data.qpos[0:3] = start

data.qpos[3:7] = np.array([
    1.0,
    0.0,
    0.0,
    0.0
])

data.qvel[:] = 0.0

mujoco.mj_forward(model, data)


# ==========================================================
# VIEWER
# ==========================================================

print()
print("==================================================")
print("STARTING MUJOCO")
print("==================================================")
print("Close the MuJoCo window to stop early.")
print()

with mujoco.viewer.launch_passive(model, data) as viewer:

    simulation_time = 0.0
    last_print = 0.0

    while viewer.is_running():

        loop_start = time.perf_counter()

        # --------------------------------------------------
        # CURRENT POSITION AND VELOCITY
        # --------------------------------------------------

        actual_position = data.qpos[0:3].copy()
        actual_velocity = data.qvel[0:3].copy()

        # --------------------------------------------------
        # CURRENT ORIENTATION
        # --------------------------------------------------

        quaternion = data.qpos[3:7]

        rotation_matrix = np.zeros(9)

        mujoco.mju_quat2Mat(
            rotation_matrix,
            quaternion
        )

        rotation_matrix = rotation_matrix.reshape(3, 3)

        # --------------------------------------------------
        # ROLL AND PITCH
        # --------------------------------------------------

        roll = np.arctan2(
            rotation_matrix[2, 1],
            rotation_matrix[2, 2]
        )

        pitch = np.arcsin(
            np.clip(
                -rotation_matrix[2, 0],
                -1.0,
                1.0
            )
        )

        # --------------------------------------------------
        # ANGULAR VELOCITIES
        # --------------------------------------------------

        roll_rate = data.qvel[3]
        pitch_rate = data.qvel[4]

        # --------------------------------------------------
        # SELECT TRAJECTORY SAMPLE
        # --------------------------------------------------

        normalized_time = np.clip(
            simulation_time / trajectory_duration,
            0.0,
            1.0
        )

        trajectory_index = int(
            normalized_time * (number_of_samples - 1)
        )

        trajectory_index = int(
            np.clip(
                trajectory_index,
                0,
                number_of_samples - 1
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
            desired_z_acceleration
        ) = position_controller.compute(
            desired_position=target_position,
            actual_position=actual_position,
            actual_velocity=actual_velocity,
            desired_velocity=target_velocity
        )

        # --------------------------------------------------
        # ROLL PID
        # --------------------------------------------------

        roll_torque = roll_pid.compute(
            desired_roll,
            roll,
            roll_rate
        )

        # --------------------------------------------------
        # PITCH PID
        # --------------------------------------------------

        pitch_torque = pitch_pid.compute(
            desired_pitch,
            pitch,
            pitch_rate
        )

        # --------------------------------------------------
        # VERTICAL THRUST
        # --------------------------------------------------

        total_thrust = mass * (
            gravity + desired_z_acceleration
        )

        # Prevent negative total thrust.
        total_thrust = max(
            total_thrust,
            0.0
        )

        # --------------------------------------------------
        # TILT COMPENSATION
        # --------------------------------------------------

        denominator = (
            np.cos(desired_roll)
            * np.cos(desired_pitch)
        )

        if denominator > 0.1:
            total_thrust = total_thrust / denominator

        # --------------------------------------------------
        # MOTOR MIXING
        # --------------------------------------------------

        thrusts = mix_roll_pitch(
            total_thrust=total_thrust,
            roll_torque=roll_torque,
            pitch_torque=pitch_torque,
            arm_length=arm_length
        )

        thrusts = np.maximum(
            thrusts,
            0.0
        )

        # --------------------------------------------------
        # APPLY ROTOR FORCES
        # --------------------------------------------------

        apply_rotor_forces(
            model,
            data,
            thrusts
        )

        # --------------------------------------------------
        # STEP MUJOCO
        # --------------------------------------------------

        mujoco.mj_step(
            model,
            data
        )

        viewer.sync()

        simulation_time += dt

        # --------------------------------------------------
        # PRINT STATUS
        # --------------------------------------------------

        if simulation_time - last_print >= 0.5:

            last_print = simulation_time

            current_position = data.qpos[0:3].copy()

            target_error = np.linalg.norm(
                target_position - current_position
            )

            goal_distance = np.linalg.norm(
                goal - current_position
            )

            print(
                f"t={simulation_time:6.2f}s | "
                f"sample={trajectory_index:3d}/"
                f"{number_of_samples - 1:3d} | "
                f"pos=["
                f"{current_position[0]:6.2f}, "
                f"{current_position[1]:6.2f}, "
                f"{current_position[2]:6.2f}] | "
                f"target_err={target_error:6.3f}m | "
                f"goal={goal_distance:6.3f}m | "
                f"roll={np.rad2deg(roll):7.2f} deg | "
                f"pitch={np.rad2deg(pitch):7.2f} deg"
            )

        # --------------------------------------------------
        # STOP WHEN TRAJECTORY ENDS
        # --------------------------------------------------

        if simulation_time >= trajectory_duration:

            final_position = data.qpos[0:3].copy()

            final_distance = np.linalg.norm(
                goal - final_position
            )

            print()
            print("==================================================")
            print("B-SPLINE TRAJECTORY COMPLETE")
            print("==================================================")
            print(f"Final position = {final_position}")
            print(f"Goal position  = {goal}")
            print(f"Final distance = {final_distance:.6f} m")
            print()

            break

        # --------------------------------------------------
        # REAL-TIME PACING
        # --------------------------------------------------

        elapsed = time.perf_counter() - loop_start
        remaining = dt - elapsed

        if remaining > 0:
            time.sleep(remaining)


print()
print("==================================================")
print("MUJOCO VIEWER CLOSED")
print("==================================================")