import mujoco
import numpy as np
import os


# ==========================================================
# 1. LOAD MODEL
# ==========================================================

MODEL_PATH = "simulation/drone_model.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

drone_id = model.body("drone").id


# ==========================================================
# 2. PHYSICAL PARAMETERS
# ==========================================================

mass = model.body_mass[drone_id]

Ixx = model.body_inertia[drone_id][0]
Iyy = model.body_inertia[drone_id][1]
Izz = model.body_inertia[drone_id][2]

g = abs(model.opt.gravity[2])

hover_thrust = mass * g


# ==========================================================
# 3. SIMULATION PARAMETERS
# ==========================================================

dt = model.opt.timestep

simulation_time = 10.0

number_of_steps = int(
    simulation_time / dt
)


# ==========================================================
# 4. PID PARAMETERS
# ==========================================================
#
# These are the already-designed gains from our
# pole-placement calculation.
#
# Roll/pitch:
#
# Kp = 4.04213392
# Ki = 7.34933440
# Kd = 0.64306676
#
# For yaw we use a simple PD controller because the current
# MuJoCo rotor model does not yet contain a physical
# rotor reaction-torque model.
# ==========================================================

Kp_roll = 4.04213392
Ki_roll = 7.34933440
Kd_roll = 0.64306676

Kp_pitch = 4.04213392
Ki_pitch = 7.34933440
Kd_pitch = 0.64306676

Kp_yaw = 0.50
Kd_yaw = 0.10


# ==========================================================
# 5. PID INTEGRAL STATES
# ==========================================================

roll_integral = 0.0
pitch_integral = 0.0


# ==========================================================
# 6. DATA ARRAYS
# ==========================================================

states = []
inputs = []


# ==========================================================
# 7. RESET
# ==========================================================

mujoco.mj_resetData(model, data)


data.qpos[0] = 0.0
data.qpos[1] = 0.0
data.qpos[2] = 1.0

data.qvel[:] = 0.0


# ==========================================================
# 8. DATA COLLECTION LOOP
# ==========================================================

for step in range(number_of_steps):

    t = step * dt


    # ======================================================
    # CURRENT ROTATION
    # ======================================================

    quaternion = data.qpos[3:7]

    rotation_flat = np.zeros(9)

    mujoco.mju_quat2Mat(
        rotation_flat,
        quaternion
    )

    R = rotation_flat.reshape(3, 3)


    # ======================================================
    # EULER ANGLES
    # ======================================================

    roll = np.arctan2(
        R[2, 1],
        R[2, 2]
    )

    pitch = np.arctan2(
        -R[2, 0],
        np.sqrt(
            R[2, 1] ** 2
            + R[2, 2] ** 2
        )
    )

    yaw = np.arctan2(
        R[1, 0],
        R[0, 0]
    )


    # ======================================================
    # POSITION AND VELOCITY
    # ======================================================

    x = data.qpos[0]
    y = data.qpos[1]
    z = data.qpos[2]

    vx = data.qvel[0]
    vy = data.qvel[1]
    vz = data.qvel[2]


    # ======================================================
    # BODY ANGULAR VELOCITIES
    # ======================================================

    p = data.qvel[3]
    q = data.qvel[4]
    r = data.qvel[5]


    # ======================================================
    # POSITION HOLD
    # ======================================================
    #
    # Desired position:
    #
    # x = 0
    # y = 0
    # z = 1
    #
    # Small position feedback creates restoring pitch,
    # roll and thrust commands.
    # ======================================================

    x_error = 0.0 - x
    y_error = 0.0 - y
    z_error = 1.0 - z


    # Desired horizontal accelerations.

    desired_ax = (
        0.8 * x_error
        - 0.8 * vx
    )

    desired_ay = (
        0.8 * y_error
        - 0.8 * vy
    )


    # ======================================================
    # SMALL-ANGLE POSITION -> ATTITUDE
    # ======================================================
    #
    # Around hover:
    #
    # ax ≈ g * theta
    #
    # ay ≈ -g * phi
    #
    # Therefore:
    #
    # theta_desired = ax / g
    #
    # phi_desired = -ay / g
    # ======================================================

    desired_pitch = desired_ax / g

    desired_roll = -desired_ay / g


    # Limit desired angles to remain in the
    # small-angle hover region.

    desired_roll = np.clip(
        desired_roll,
        -0.10,
        0.10
    )

    desired_pitch = np.clip(
        desired_pitch,
        -0.10,
        0.10
    )


    # ======================================================
    # ATTITUDE PID
    # ======================================================

    roll_error = (
        desired_roll - roll
    )

    pitch_error = (
        desired_pitch - pitch
    )


    roll_integral += (
        roll_error * dt
    )

    pitch_integral += (
        pitch_error * dt
    )


    tau_x_control = (
        Kp_roll * roll_error
        + Ki_roll * roll_integral
        - Kd_roll * p
    )

    tau_y_control = (
        Kp_pitch * pitch_error
        + Ki_pitch * pitch_integral
        - Kd_pitch * q
    )


    # ======================================================
    # YAW CONTROL
    # ======================================================
    #
    # Desired yaw = 0.
    #
    # Current model does not have a physical rotor
    # reaction-torque model, so this is only an external
    # torque used for keeping the experimental operating
    # point near zero yaw.
    # ======================================================

    yaw_error = -yaw

    tau_z_control = (
        Kp_yaw * yaw_error
        - Kd_yaw * r
    )


    # ======================================================
    # EXTERNAL EXCITATION
    # ======================================================
    #
    # These are experimental excitation signals.
    #
    # They are NOT physical constants.
    #
    # The amplitudes are deliberately small so the system
    # remains near hover.
    # ======================================================

    delta_T_excitation = (
        0.02
        * np.sin(2.0 * np.pi * 0.5 * t)
    )

    tau_x_excitation = (
        0.001
        * np.sin(2.0 * np.pi * 0.7 * t)
    )

    tau_y_excitation = (
        0.001
        * np.sin(2.0 * np.pi * 0.9 * t)
    )

    tau_z_excitation = (
        0.0005
        * np.sin(2.0 * np.pi * 0.6 * t)
    )


    # ======================================================
    # TOTAL CONTROL INPUT
    # ======================================================

    delta_T = (
        delta_T_excitation
        + mass * (
            z_error
            - 0.8 * vz
        )
    )


    tau_x = (
        tau_x_control
        + tau_x_excitation
    )

    tau_y = (
        tau_y_control
        + tau_y_excitation
    )

    tau_z = (
        tau_z_control
        + tau_z_excitation
    )


    # ======================================================
    # TOTAL THRUST
    # ======================================================

    total_thrust = (
        hover_thrust
        + delta_T
    )


    # ======================================================
    # APPLY FORCE
    # ======================================================

    force_body = np.array(
        [
            0.0,
            0.0,
            total_thrust
        ]
    )

    force_world = (
        R @ force_body
    )

    data.xfrc_applied[
        drone_id,
        0:3
    ] = force_world


    # ======================================================
    # APPLY TORQUE
    # ======================================================

    torque_body = np.array(
        [
            tau_x,
            tau_y,
            tau_z
        ]
    )

    torque_world = (
        R @ torque_body
    )

    data.xfrc_applied[
        drone_id,
        3:6
    ] = torque_world


    # ======================================================
    # BUILD STATE VECTOR
    # ======================================================

    state = np.array(
        [
            x,
            y,
            z,

            vx,
            vy,
            vz,

            roll,
            pitch,
            yaw,

            p,
            q,
            r
        ]
    )


    # ======================================================
    # BUILD INPUT VECTOR
    # ======================================================

    control_input = np.array(
        [
            delta_T,
            tau_x,
            tau_y,
            tau_z
        ]
    )


    states.append(state)
    inputs.append(control_input)


    # ======================================================
    # STEP SIMULATION
    # ======================================================

    mujoco.mj_step(
        model,
        data
    )


# ==========================================================
# 9. CONVERT TO NUMPY
# ==========================================================

states = np.array(states)

inputs = np.array(inputs)


# ==========================================================
# 10. SAVE DATA
# ==========================================================

os.makedirs(
    "data",
    exist_ok=True
)

np.save(
    "data/dmdc_states.npy",
    states
)

np.save(
    "data/dmdc_inputs.npy",
    inputs
)


# ==========================================================
# 11. DATA RANGE CHECK
# ==========================================================

position_min = states[:, 0:3].min(axis=0)
position_max = states[:, 0:3].max(axis=0)

angle_min = np.degrees(
    states[:, 6:9].min(axis=0)
)

angle_max = np.degrees(
    states[:, 6:9].max(axis=0)
)


# ==========================================================
# 12. PRINT RESULTS
# ==========================================================

print()
print("==================================================")
print("CONTROLLED DMDc DATA COLLECTION")
print("==================================================")

print()
print("PHYSICAL PARAMETERS")
print("----------------------------------------------")

print(f"Mass = {mass:.8f} kg")
print(f"Ixx  = {Ixx:.8f} kg m^2")
print(f"Iyy  = {Iyy:.8f} kg m^2")
print(f"Izz  = {Izz:.8f} kg m^2")
print(f"g    = {g:.8f} m/s^2")

print()
print("HOVER")
print("----------------------------------------------")

print(
    f"Hover thrust = "
    f"{hover_thrust:.8f} N"
)

print()
print("DATA")
print("----------------------------------------------")

print(
    f"dt = {dt:.6f} s"
)

print(
    f"Samples = {len(states)}"
)

print(
    f"States shape = {states.shape}"
)

print(
    f"Inputs shape = {inputs.shape}"
)

print()
print("POSITION RANGE")
print("----------------------------------------------")

print(
    f"x: {position_min[0]:.6f} "
    f"to {position_max[0]:.6f} m"
)

print(
    f"y: {position_min[1]:.6f} "
    f"to {position_max[1]:.6f} m"
)

print(
    f"z: {position_min[2]:.6f} "
    f"to {position_max[2]:.6f} m"
)

print()
print("ATTITUDE RANGE")
print("----------------------------------------------")

print(
    f"roll : {angle_min[0]:.6f} "
    f"to {angle_max[0]:.6f} deg"
)

print(
    f"pitch: {angle_min[1]:.6f} "
    f"to {angle_max[1]:.6f} deg"
)

print(
    f"yaw  : {angle_min[2]:.6f} "
    f"to {angle_max[2]:.6f} deg"
)

print()
print("FIRST STATE")
print("----------------------------------------------")

print(states[0])

print()
print("LAST STATE")
print("----------------------------------------------")

print(states[-1])

print()
print("FILES")
print("----------------------------------------------")

print("data/dmdc_states.npy")
print("data/dmdc_inputs.npy")

print()
print("==================================================")