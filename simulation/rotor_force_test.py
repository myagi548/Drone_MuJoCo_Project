import numpy as np
import mujoco

from simulation.rotor_model import apply_rotor_forces


# ==========================================================
# LOAD MUJOCO MODEL
# ==========================================================

MODEL_PATH = "simulation/drone_model.xml"

model = mujoco.MjModel.from_xml_path(
    MODEL_PATH
)

data = mujoco.MjData(model)


# ==========================================================
# IMPORTANT:
# CALCULATE MUJOCO'S CURRENT BODY TRANSFORMS
# ==========================================================

mujoco.mj_forward(
    model,
    data
)


# ==========================================================
# GET DRONE BODY ID
# ==========================================================

drone_id = model.body("drone").id


print()
print("==================================================")
print("MUJOCO ROTOR FORCE / TORQUE TEST")
print("==================================================")


print()
print("DRONE BODY")
print("----------------------------------------------")
print(f"Drone body ID = {drone_id}")


# ==========================================================
# ROTOR THRUSTS
# ==========================================================
#
# These values came from the verified mixer test.
#
# Rotor 1 = 2.385833 N
# Rotor 2 = 2.519167 N
# Rotor 3 = 2.563611 N
# Rotor 4 = 2.341389 N
#
# Total thrust ≈ 9.81 N
#
# ==========================================================

thrusts = np.array(
    [
        2.385833,
        2.519167,
        2.563611,
        2.341389
    ],
    dtype=float
)


print()
print("ROTOR THRUSTS")
print("----------------------------------------------")

for i, thrust in enumerate(
    thrusts,
    start=1
):
    print(
        f"Rotor {i} = {thrust:.6f} N"
    )


# ==========================================================
# APPLY ROTOR FORCES
# ==========================================================

apply_rotor_forces(
    model,
    data,
    thrusts
)


# ==========================================================
# READ APPLIED FORCE AND TORQUE
# ==========================================================

applied = data.xfrc_applied[
    drone_id
].copy()


applied_force = applied[0:3]

applied_torque = applied[3:6]


print()
print("xfrc_applied")
print("----------------------------------------------")

print(
    "Raw applied wrench =",
    applied
)


print()
print("APPLIED FORCE")
print("----------------------------------------------")

print(
    f"Fx = {applied_force[0]:.6f} N"
)

print(
    f"Fy = {applied_force[1]:.6f} N"
)

print(
    f"Fz = {applied_force[2]:.6f} N"
)


print()
print("APPLIED TORQUE")
print("----------------------------------------------")

print(
    f"Tx = {applied_torque[0]:.6f} N m"
)

print(
    f"Ty = {applied_torque[1]:.6f} N m"
)

print(
    f"Tz = {applied_torque[2]:.6f} N m"
)


# ==========================================================
# EXPECTED VALUES
# ==========================================================
#
# Arm length comes directly from drone_model.xml:
#
# Rotor 1 = (+0.45, 0, 0.05)
# Rotor 2 = (-0.45, 0, 0.05)
# Rotor 3 = (0, +0.45, 0.05)
# Rotor 4 = (0, -0.45, 0.05)
#
# Therefore:
#
# Total thrust = T1 + T2 + T3 + T4
#
# Roll torque:
#     Tx = L(T3 - T4)
#
# Pitch torque:
#     Ty = L(T2 - T1)
#
# ==========================================================

arm_length = 0.45


expected_force_z = np.sum(
    thrusts
)


expected_roll_torque = (
    arm_length
    * (
        thrusts[2]
        - thrusts[3]
    )
)


expected_pitch_torque = (
    arm_length
    * (
        thrusts[1]
        - thrusts[0]
    )
)


print()
print("EXPECTED VALUES")
print("----------------------------------------------")

print(
    f"Total thrust = "
    f"{expected_force_z:.6f} N"
)

print(
    f"Roll torque = "
    f"{expected_roll_torque:.6f} N m"
)

print(
    f"Pitch torque = "
    f"{expected_pitch_torque:.6f} N m"
)


# ==========================================================
# CALCULATE ERRORS
# ==========================================================

force_error = (
    applied_force[2]
    - expected_force_z
)


roll_error = (
    applied_torque[0]
    - expected_roll_torque
)


pitch_error = (
    applied_torque[1]
    - expected_pitch_torque
)


print()
print("VERIFICATION ERRORS")
print("----------------------------------------------")

print(
    f"Force Z error = "
    f"{force_error:.12e} N"
)

print(
    f"Roll torque error = "
    f"{roll_error:.12e} N m"
)

print(
    f"Pitch torque error = "
    f"{pitch_error:.12e} N m"
)


# ==========================================================
# FINAL VERIFICATION
# ==========================================================

tolerance = 1e-6


passed = (
    abs(force_error) < tolerance
    and abs(roll_error) < tolerance
    and abs(pitch_error) < tolerance
)


print()
print("==================================================")


if passed:

    print(
        "RESULT: MUJOCO ROTOR FORCE/TORQUE TEST PASSED"
    )

else:

    print(
        "RESULT: MUJOCO ROTOR FORCE/TORQUE TEST FAILED"
    )


print("==================================================")