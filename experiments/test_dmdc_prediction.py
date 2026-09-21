import mujoco
import numpy as np
import os


# ==========================================================
# 1. LOAD MUJOCO MODEL
# ==========================================================

MODEL_PATH = "simulation/drone_model.xml"

model = mujoco.MjModel.from_xml_path(
    MODEL_PATH
)

data = mujoco.MjData(model)

drone_id = model.body("drone").id


# ==========================================================
# 2. LOAD DMDc MODEL
# ==========================================================

A_d = np.load(
    "data/dmdc_A.npy"
)

B_d = np.load(
    "data/dmdc_B.npy"
)


# ==========================================================
# 3. PHYSICAL PARAMETERS
# ==========================================================

mass = model.body_mass[drone_id]

g = abs(model.opt.gravity[2])

hover_thrust = mass * g

dt = model.opt.timestep


# ==========================================================
# 4. VALIDATION SETTINGS
# ==========================================================

validation_time = 2.0

number_of_steps = int(
    validation_time / dt
)


# ==========================================================
# 5. STATE EXTRACTION
# ==========================================================

def get_state(data):

    position = data.qpos[0:3].copy()

    velocity = data.qvel[0:3].copy()

    quaternion = data.qpos[3:7]

    rotation_flat = np.zeros(9)

    mujoco.mju_quat2Mat(
        rotation_flat,
        quaternion
    )

    R = rotation_flat.reshape(3, 3)


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


    angular_velocity = data.qvel[3:6]


    return np.array(
        [
            position[0],
            position[1],
            position[2],

            velocity[0],
            velocity[1],
            velocity[2],

            roll,
            pitch,
            yaw,

            angular_velocity[0],
            angular_velocity[1],
            angular_velocity[2]
        ]
    )


# ==========================================================
# 6. INPUT GENERATOR
# ==========================================================

def get_input(t, state):

    x = state[0]
    y = state[1]
    z = state[2]

    vx = state[3]
    vy = state[4]
    vz = state[5]

    roll = state[6]
    pitch = state[7]
    yaw = state[8]

    p = state[9]
    q = state[10]
    r = state[11]


    # ------------------------------------------------------
    # Position stabilization
    # ------------------------------------------------------

    x_error = -x
    y_error = -y
    z_error = 1.0 - z


    desired_ax = (
        0.8 * x_error
        - 0.8 * vx
    )

    desired_ay = (
        0.8 * y_error
        - 0.8 * vy
    )


    desired_pitch = (
        desired_ax / g
    )

    desired_roll = (
        -desired_ay / g
    )


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


    # ------------------------------------------------------
    # Attitude control
    # ------------------------------------------------------

    tau_x = (
        4.04213392
        * (desired_roll - roll)
        - 0.64306676 * p
    )

    tau_y = (
        4.04213392
        * (desired_pitch - pitch)
        - 0.64306676 * q
    )

    tau_z = (
        0.50 * (-yaw)
        - 0.10 * r
    )


    # ------------------------------------------------------
    # New validation excitation
    # ------------------------------------------------------

    delta_T_excitation = (
        0.015
        * np.sin(
            2.0 * np.pi * 0.35 * t
        )
    )

    tau_x += (
        0.0008
        * np.sin(
            2.0 * np.pi * 0.55 * t
        )
    )

    tau_y += (
        0.0008
        * np.sin(
            2.0 * np.pi * 0.75 * t
        )
    )

    tau_z += (
        0.0004
        * np.sin(
            2.0 * np.pi * 0.45 * t
        )
    )


    # ------------------------------------------------------
    # Total thrust perturbation
    # ------------------------------------------------------

    delta_T = (
        delta_T_excitation
        + mass * (
            z_error
            - 0.8 * vz
        )
    )


    return np.array(
        [
            delta_T,
            tau_x,
            tau_y,
            tau_z
        ]
    )


# ==========================================================
# 7. RESET MUJOCO
# ==========================================================

mujoco.mj_resetData(
    model,
    data
)


data.qpos[0] = 0.0
data.qpos[1] = 0.0
data.qpos[2] = 1.0

data.qvel[:] = 0.0


# ==========================================================
# 8. INITIAL STATE
# ==========================================================

actual_state = get_state(data)

predicted_state = actual_state.copy()


# ==========================================================
# 9. STORAGE
# ==========================================================

actual_states = []

predicted_states = []

validation_inputs = []


# ==========================================================
# 10. FREE-RUNNING PREDICTION LOOP
# ==========================================================

for step in range(number_of_steps):

    t = step * dt


    # ------------------------------------------------------
    # Read actual MuJoCo state
    # ------------------------------------------------------

    actual_state = get_state(data)


    # ------------------------------------------------------
    # Generate input using actual experiment state
    #
    # The input is known to both MuJoCo and DMDc.
    # ------------------------------------------------------

    u = get_input(
        t,
        actual_state
    )


    actual_states.append(
        actual_state.copy()
    )

    predicted_states.append(
        predicted_state.copy()
    )

    validation_inputs.append(
        u.copy()
    )


    # ======================================================
    # DMDc FREE-RUNNING PREDICTION
    # ======================================================
    #
    # IMPORTANT:
    #
    # We use predicted_state here.
    #
    # NOT actual_state.
    #
    # Therefore:
    #
    # predicted x(k+1)
    #       ↓
    # becomes
    # predicted x(k+1) for the next step
    #
    # This allows prediction errors to accumulate.
    # ======================================================

    predicted_state = (
        A_d @ predicted_state
        + B_d @ u
    )


    # ======================================================
    # APPLY SAME INPUT TO MUJOCO
    # ======================================================

    quaternion = data.qpos[3:7]

    rotation_flat = np.zeros(9)

    mujoco.mju_quat2Mat(
        rotation_flat,
        quaternion
    )

    R = rotation_flat.reshape(3, 3)


    delta_T = u[0]

    tau_x = u[1]
    tau_y = u[2]
    tau_z = u[3]


    total_thrust = (
        hover_thrust
        + delta_T
    )


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
        0:3
    ] = force_world


    data.xfrc_applied[
        drone_id,
        3:6
    ] = torque_world


    mujoco.mj_step(
        model,
        data
    )


# ==========================================================
# 11. CONVERT TO ARRAYS
# ==========================================================

actual_states = np.array(
    actual_states
)

predicted_states = np.array(
    predicted_states
)

validation_inputs = np.array(
    validation_inputs
)


# ==========================================================
# 12. CALCULATE ERRORS
# ==========================================================

errors = (
    actual_states
    - predicted_states
)


rmse = np.sqrt(
    np.mean(
        errors ** 2,
        axis=0
    )
)


overall_rmse = np.sqrt(
    np.mean(
        errors ** 2
    )
)


# ==========================================================
# 13. FINAL ERROR
# ==========================================================

final_error = errors[-1]


# ==========================================================
# 14. SAVE RESULTS
# ==========================================================

os.makedirs(
    "results",
    exist_ok=True
)


np.save(
    "results/dmdc_free_run_actual.npy",
    actual_states
)

np.save(
    "results/dmdc_free_run_predicted.npy",
    predicted_states
)

np.save(
    "results/dmdc_free_run_inputs.npy",
    validation_inputs
)


# ==========================================================
# 15. PRINT RESULTS
# ==========================================================

state_names = [
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "roll",
    "pitch",
    "yaw",
    "p",
    "q",
    "r"
]


print()
print("==================================================")
print("DMDc FREE-RUNNING PREDICTION")
print("==================================================")


print()
print("VALIDATION")
print("----------------------------------------------")

print(
    f"Simulation time = "
    f"{validation_time:.2f} s"
)

print(
    f"dt = {dt:.6f} s"
)

print(
    f"Samples = {number_of_steps}"
)


print()
print("RMSE")
print("----------------------------------------------")

print(
    f"Overall RMSE = "
    f"{overall_rmse:.12e}"
)


print()
print("STATE-BY-STATE RMSE")
print("----------------------------------------------")


for name, value in zip(
    state_names,
    rmse
):

    print(
        f"{name:>5s} : "
        f"{value:.12e}"
    )


print()
print("FINAL ACTUAL STATE")
print("----------------------------------------------")

print(
    actual_states[-1]
)


print()
print("FINAL PREDICTED STATE")
print("----------------------------------------------")

print(
    predicted_states[-1]
)


print()
print("FINAL ERROR")
print("----------------------------------------------")

print(
    final_error
)


print()
print("FILES SAVED")
print("----------------------------------------------")

print(
    "results/dmdc_free_run_actual.npy"
)

print(
    "results/dmdc_free_run_predicted.npy"
)

print(
    "results/dmdc_free_run_inputs.npy"
)


print()
print("==================================================")