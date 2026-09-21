import time
import numpy as np
import mujoco
import mujoco.viewer

from control.pid_controller import PIDController
from simulation.mixer import mix_roll_pitch
from simulation.rotor_model import apply_rotor_forces


# ==========================================================
# LOAD MODEL
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

wn = 4.0 / (
    zeta * settling_time
)

p3 = 5.0 * wn


# ==========================================================
# PID GAINS
# ==========================================================

kp = Ixx * (
    wn**2
    + 2.0 * zeta * wn * p3
)

ki = Ixx * (
    wn**2 * p3
)

kd = Ixx * (
    2.0 * zeta * wn
    + p3
)


# ==========================================================
# PID CONTROLLERS
# ==========================================================

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
# INITIAL DRONE STATE
# ==========================================================

data.qpos[0:3] = np.array(
    [
        0.0,
        0.0,
        1.0
    ]
)

data.qpos[3:7] = np.array(
    [
        1.0,
        0.0,
        0.0,
        0.0
    ]
)

data.qvel[:] = 0.0

mujoco.mj_forward(
    model,
    data
)


# ==========================================================
# DESIRED ATTITUDE
# ==========================================================

desired_roll = np.deg2rad(2.0)

desired_pitch = 0.0


# ==========================================================
# START VIEWER
# ==========================================================

print()
print("==================================================")
print("VISUAL PID + MIXER + ROTOR + MUJOCO TEST")
print("==================================================")
print()
print(f"Mass          = {mass:.6f} kg")
print(f"Ixx           = {Ixx:.8f} kg m^2")
print(f"Hover thrust  = {hover_thrust:.6f} N")
print(f"dt            = {dt:.6f} s")
print()
print("PID GAINS")
print("----------------------------------------------")
print(f"Kp = {kp:.8f}")
print(f"Ki = {ki:.8f}")
print(f"Kd = {kd:.8f}")
print()
print("Desired roll  = 2 degrees")
print("Desired pitch = 0 degrees")
print()
print("Close the MuJoCo window to finish.")
print()


with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    start_time = time.perf_counter()

    while viewer.is_running():

        loop_start = time.perf_counter()


        # ==================================================
        # CURRENT ORIENTATION
        # ==================================================

        quaternion = data.qpos[3:7]

        rotation_matrix = np.zeros(9)

        mujoco.mju_quat2Mat(
            rotation_matrix,
            quaternion
        )

        rotation_matrix = rotation_matrix.reshape(
            3,
            3
        )


        # ==================================================
        # CURRENT ROLL
        # ==================================================

        roll = np.arctan2(
            rotation_matrix[2, 1],
            rotation_matrix[2, 2]
        )


        # ==================================================
        # CURRENT PITCH
        # ==================================================

        pitch = np.arcsin(
            np.clip(
                -rotation_matrix[2, 0],
                -1.0,
                1.0
            )
        )


        # ==================================================
        # ANGULAR RATES
        # ==================================================

        roll_rate = data.qvel[3]

        pitch_rate = data.qvel[4]


        # ==================================================
        # PID CONTROL
        # ==================================================

        roll_torque = roll_pid.compute(
            desired_roll,
            roll,
            roll_rate
        )

        pitch_torque = pitch_pid.compute(
            desired_pitch,
            pitch,
            pitch_rate
        )


        # ==================================================
        # TOTAL THRUST
        # ==================================================

        total_thrust = hover_thrust


        # ==================================================
        # MOTOR MIXING
        # ==================================================

        thrusts = mix_roll_pitch(
            total_thrust=total_thrust,
            roll_torque=roll_torque,
            pitch_torque=pitch_torque,
            arm_length=arm_length
        )


        # ==================================================
        # SAFETY LIMIT
        # ==================================================

        thrusts = np.maximum(
            thrusts,
            0.0
        )


        # ==================================================
        # APPLY ROTOR FORCES
        # ==================================================

        apply_rotor_forces(
            model,
            data,
            thrusts
        )


        # ==================================================
        # STEP MUJOCO
        # ==================================================

        mujoco.mj_step(
            model,
            data
        )


        # ==================================================
        # UPDATE VIEWER
        # ==================================================

        viewer.sync()


        # ==================================================
        # REAL-TIME PACING
        # ==================================================

        elapsed = (
            time.perf_counter()
            - loop_start
        )

        remaining = dt - elapsed

        if remaining > 0:

            time.sleep(
                remaining
            )


        # ==================================================
        # PRINT STATUS EVERY 0.5 SECONDS
        # ==================================================

        current_time = (
            time.perf_counter()
            - start_time
        )

        if (
            int(current_time * 2)
            != int((current_time - dt) * 2)
        ):

            print(
                f"t = {current_time:5.2f} s | "
                f"roll = {np.rad2deg(roll):7.3f} deg | "
                f"pitch = {np.rad2deg(pitch):7.3f} deg | "
                f"torque = {roll_torque:8.4f} N m"
            )


print()
print("==================================================")
print("MUJOCO VIEWER CLOSED")
print("==================================================")