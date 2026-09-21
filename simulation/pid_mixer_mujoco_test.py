import numpy as np
import mujoco

from control.pid_controller import PIDController
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

drone_id = model.body("drone").id

mass = model.body("drone").mass[0]

Ixx = model.body("drone").inertia[0]

Iyy = model.body("drone").inertia[1]

gravity = abs(model.opt.gravity[2])

arm_length = 0.45

dt = model.opt.timestep


# ==========================================================
# HOVER THRUST
# ==========================================================

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
# SIMULATION SETTINGS
# ==========================================================

simulation_time = 5.0

number_of_steps = int(
    simulation_time / dt
)


# ==========================================================
# DESIRED ATTITUDE
# ==========================================================

desired_roll = np.deg2rad(2.0)

desired_pitch = np.deg2rad(0.0)


# ==========================================================
# INITIAL POSITION
# ==========================================================

data.qpos[0:3] = np.array(
    [
        0.0,
        0.0,
        1.0
    ]
)


# ==========================================================
# INITIAL ORIENTATION
# ==========================================================

data.qpos[3:7] = np.array(
    [
        1.0,
        0.0,
        0.0,
        0.0
    ]
)


# ==========================================================
# INITIAL VELOCITY
# ==========================================================

data.qvel[:] = 0.0

mujoco.mj_forward(
    model,
    data
)


# ==========================================================
# DATA STORAGE
# ==========================================================

time_history = []

roll_history = []

pitch_history = []

roll_torque_history = []

pitch_torque_history = []


# ==========================================================
# SIMULATION LOOP
# ==========================================================

for step in range(number_of_steps):

    current_time = step * dt


    # ------------------------------------------------------
    # CURRENT ORIENTATION
    # ------------------------------------------------------

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


    # ------------------------------------------------------
    # CURRENT ROLL
    # ------------------------------------------------------

    roll = np.arctan2(
        rotation_matrix[2, 1],
        rotation_matrix[2, 2]
    )


    # ------------------------------------------------------
    # CURRENT PITCH
    # ------------------------------------------------------

    pitch = np.arcsin(
        np.clip(
            -rotation_matrix[2, 0],
            -1.0,
            1.0
        )
    )


    # ------------------------------------------------------
    # CURRENT ANGULAR VELOCITIES
    # ------------------------------------------------------
    #
    # MuJoCo free joint:
    #
    # qvel[3] = angular velocity x
    # qvel[4] = angular velocity y
    # qvel[5] = angular velocity z
    #
    # For roll PID:
    # angular rate = x-axis angular velocity
    #
    # For pitch PID:
    # angular rate = y-axis angular velocity
    # ------------------------------------------------------

    roll_rate = data.qvel[3]

    pitch_rate = data.qvel[4]


    # ------------------------------------------------------
    # ROLL PID
    # ------------------------------------------------------

    roll_torque = roll_pid.compute(
        desired_roll,
        roll,
        roll_rate
    )


    # ------------------------------------------------------
    # PITCH PID
    # ------------------------------------------------------

    pitch_torque = pitch_pid.compute(
        desired_pitch,
        pitch,
        pitch_rate
    )


    # ------------------------------------------------------
    # TOTAL THRUST
    # ------------------------------------------------------

    total_thrust = hover_thrust


    # ------------------------------------------------------
    # MIXER
    # ------------------------------------------------------

    thrusts = mix_roll_pitch(
        total_thrust=total_thrust,
        roll_torque=roll_torque,
        pitch_torque=pitch_torque,
        arm_length=arm_length
    )


    # ------------------------------------------------------
    # PREVENT NEGATIVE THRUST
    # ------------------------------------------------------

    thrusts = np.maximum(
        thrusts,
        0.0
    )


    # ------------------------------------------------------
    # APPLY ROTOR FORCES
    # ------------------------------------------------------

    apply_rotor_forces(
        model,
        data,
        thrusts
    )


    # ------------------------------------------------------
    # STORE DATA
    # ------------------------------------------------------

    time_history.append(
        current_time
    )

    roll_history.append(
        np.rad2deg(roll)
    )

    pitch_history.append(
        np.rad2deg(pitch)
    )

    roll_torque_history.append(
        roll_torque
    )

    pitch_torque_history.append(
        pitch_torque
    )


    # ------------------------------------------------------
    # ADVANCE MUJOCO
    # ------------------------------------------------------

    mujoco.mj_step(
        model,
        data
    )


# ==========================================================
# CONVERT DATA
# ==========================================================

time_history = np.array(
    time_history
)

roll_history = np.array(
    roll_history
)

pitch_history = np.array(
    pitch_history
)

roll_torque_history = np.array(
    roll_torque_history
)

pitch_torque_history = np.array(
    pitch_torque_history
)


# ==========================================================
# FINAL VALUES
# ==========================================================

desired_roll_deg = np.rad2deg(
    desired_roll
)

desired_pitch_deg = np.rad2deg(
    desired_pitch
)

final_roll = roll_history[-1]

final_pitch = pitch_history[-1]

roll_error = (
    desired_roll_deg
    - final_roll
)

pitch_error = (
    desired_pitch_deg
    - final_pitch
)


# ==========================================================
# RESULTS
# ==========================================================

print()
print("==================================================")
print("PID + MIXER + ROTOR + MUJOCO TEST")
print("==================================================")


print()
print("PHYSICAL PARAMETERS")
print("----------------------------------------------")

print(
    f"Mass          = {mass:.8f} kg"
)

print(
    f"Ixx           = {Ixx:.8f} kg m^2"
)

print(
    f"Iyy           = {Iyy:.8f} kg m^2"
)

print(
    f"Gravity       = {gravity:.8f} m/s^2"
)

print(
    f"MuJoCo dt     = {dt:.6f} s"
)

print(
    f"Hover thrust  = {hover_thrust:.8f} N"
)

print(
    f"Arm length    = {arm_length:.8f} m"
)


print()
print("PID DESIGN")
print("----------------------------------------------")

print(
    f"Settling target = {settling_time:.2f} s"
)

print(
    f"Damping ratio   = {zeta:.2f}"
)

print(
    f"Natural freq.   = {wn:.2f} rad/s"
)

print(
    f"Third pole      = {p3:.2f} rad/s"
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
print("COMMAND")
print("----------------------------------------------")

print(
    f"Desired roll  = "
    f"{desired_roll_deg:.6f} deg"
)

print(
    f"Desired pitch = "
    f"{desired_pitch_deg:.6f} deg"
)


print()
print("FINAL ATTITUDE")
print("----------------------------------------------")

print(
    f"Final roll    = "
    f"{final_roll:.6f} deg"
)

print(
    f"Final pitch   = "
    f"{final_pitch:.6f} deg"
)


print()
print("FINAL ERROR")
print("----------------------------------------------")

print(
    f"Roll error    = "
    f"{roll_error:.6f} deg"
)

print(
    f"Pitch error   = "
    f"{pitch_error:.6f} deg"
)


print()
print("MAXIMUM ATTITUDE")
print("----------------------------------------------")

print(
    f"Maximum |roll|  = "
    f"{np.max(np.abs(roll_history)):.6f} deg"
)

print(
    f"Maximum |pitch| = "
    f"{np.max(np.abs(pitch_history)):.6f} deg"
)


print()
print("MAXIMUM TORQUE")
print("----------------------------------------------")

print(
    f"Maximum |roll torque|  = "
    f"{np.max(np.abs(roll_torque_history)):.6f} N m"
)

print(
    f"Maximum |pitch torque| = "
    f"{np.max(np.abs(pitch_torque_history)):.6f} N m"
)


print()
print("==================================================")
print("RESULT: INTEGRATED PID TEST COMPLETED")
print("==================================================")