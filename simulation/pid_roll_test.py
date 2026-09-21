import numpy as np
import mujoco
import mujoco.viewer

from .rotor_model import apply_rotor_forces
from .mixer import mix_roll_torque
from control.pid_controller import PIDController


# ==========================================================
# 1. LOAD MUJOCO MODEL
# ==========================================================

model = mujoco.MjModel.from_xml_path(
    "simulation/drone_model.xml"
)

data = mujoco.MjData(model)

drone_id = model.body("drone").id


# ==========================================================
# 2. READ PHYSICAL PARAMETERS FROM MUJOCO
# ==========================================================

mass = model.body_mass[drone_id]

inertia = model.body_inertia[drone_id]

Ixx = inertia[0]

gravity = 9.81

arm_length = 0.45


# ==========================================================
# 3. PID GAINS
# ==========================================================
#
# These values were calculated using:
#
# Ixx = 0.02296667 kg m^2
# settling time = 1.0 s
# overshoot = 5 %
# third-pole factor = 5
#
# They were NOT randomly chosen.
#
# ==========================================================

kp = 6.09636903
ki = 22.36141179
kd = 0.84933098


# MuJoCo simulation timestep
dt = model.opt.timestep


pid = PIDController(
    kp=kp,
    ki=ki,
    kd=kd,
    dt=dt
)


# ==========================================================
# 4. INITIAL ROLL ANGLE
# ==========================================================
#
# We physically rotate the drone by +10 degrees.
#
# Quaternion for a rotation around X:
#
# w = cos(phi/2)
# x = sin(phi/2)
# y = 0
# z = 0
#
# ==========================================================

initial_roll_deg = 10.0

initial_roll_rad = np.radians(
    initial_roll_deg
)


data.qpos[3] = np.cos(
    initial_roll_rad / 2.0
)

data.qpos[4] = np.sin(
    initial_roll_rad / 2.0
)

data.qpos[5] = 0.0
data.qpos[6] = 0.0


# Update MuJoCo after changing the initial attitude
mujoco.mj_forward(
    model,
    data
)


# ==========================================================
# 5. DESIRED ROLL
# ==========================================================

desired_roll_rad = 0.0


# ==========================================================
# 6. HOVER THRUST
# ==========================================================
#
# Total thrust required to counter gravity:
#
#       T = m*g
#
# ==========================================================

total_hover_thrust = mass * gravity


# ==========================================================
# 7. PRINT EXPERIMENT INFORMATION
# ==========================================================

print("==============================================")
print("CLOSED-LOOP PID ROLL TEST")
print("==============================================")

print()
print("PHYSICAL PARAMETERS")
print("----------------------------------------------")

print(
    f"Mass = {mass:.6f} kg"
)

print(
    f"Ixx = {Ixx:.8f} kg m^2"
)

print(
    f"Arm length = {arm_length:.3f} m"
)

print(
    f"Hover thrust = "
    f"{total_hover_thrust:.6f} N"
)


print()
print("PID GAINS")
print("----------------------------------------------")

print(
    f"Kp = {kp:.8f}"
)

print(
    f"Ki = {ki:.8f}"
)

print(
    f"Kd = {kd:.8f}"
)


print()
print("INITIAL CONDITION")
print("----------------------------------------------")

print(
    f"Initial roll = "
    f"{initial_roll_deg:.2f} deg"
)

print(
    "Desired roll = 0.00 deg"
)

print()
print("Starting simulation...")

print("==============================================")


# ==========================================================
# 8. CLOSED-LOOP SIMULATION
# ==========================================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    next_print = 0.1

    while (
        viewer.is_running()
        and data.time < 5.0
    ):

        # --------------------------------------------------
        # Read drone quaternion
        # --------------------------------------------------

        quat = data.qpos[3:7]

        w, x, y, z = quat


        # --------------------------------------------------
        # Calculate roll angle
        # --------------------------------------------------

        roll_rad = np.arctan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (
                x * x + y * y
            )
        )


        # --------------------------------------------------
        # Read roll angular velocity
        # --------------------------------------------------

        roll_rate_rad = data.qvel[3]


        # --------------------------------------------------
        # PID CONTROLLER
        # --------------------------------------------------

        roll_torque = pid.compute(
            desired_angle_rad=desired_roll_rad,
            actual_angle_rad=roll_rad,
            angular_rate_rad=roll_rate_rad
        )


        # --------------------------------------------------
        # CONVERT TORQUE TO ROTOR THRUSTS
        # --------------------------------------------------

        thrusts = mix_roll_torque(
            total_thrust=total_hover_thrust,
            roll_torque=roll_torque,
            arm_length=arm_length
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
        # ADVANCE MUJOCO
        # --------------------------------------------------

        mujoco.mj_step(
            model,
            data
        )


        # --------------------------------------------------
        # PRINT RESULTS EVERY 0.1 SECOND
        # --------------------------------------------------

        if data.time >= next_print:

            roll_deg = np.degrees(
                roll_rad
            )

            roll_rate_deg = np.degrees(
                roll_rate_rad
            )


            print(
                f"Time = {data.time:.2f} s | "
                f"Roll = {roll_deg:.3f} deg | "
                f"Roll rate = {roll_rate_deg:.3f} deg/s | "
                f"Torque = {roll_torque:.4f} N m | "
                f"T3 = {thrusts[2]:.4f} N | "
                f"T4 = {thrusts[3]:.4f} N"
            )


            next_print += 0.1


        # Update viewer
        viewer.sync()


# ==========================================================
# 9. FINISH
# ==========================================================

print("==============================================")
print("PID ROLL TEST FINISHED")
print("==============================================")