import mujoco
import numpy as np


# ==========================================================
# 1. LOAD ACTUAL MUJOCO MODEL
# ==========================================================

MODEL_PATH = "simulation/drone_model.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)

drone_id = model.body("drone").id


# ==========================================================
# 2. READ PHYSICAL PARAMETERS
# ==========================================================

mass = model.body_mass[drone_id]

inertia = model.body_inertia[drone_id]

Ixx = inertia[0]
Iyy = inertia[1]
Izz = inertia[2]

gravity_vector = model.opt.gravity

g = abs(gravity_vector[2])


# ==========================================================
# 3. NUMBER OF STATES AND INPUTS
# ==========================================================

number_of_states = 12
number_of_inputs = 4


# ==========================================================
# 4. STATE VECTOR
# ==========================================================
#
# x =
#
# [x, y, z,
#  vx, vy, vz,
#  phi, theta, psi,
#  p, q, r]^T
#

print()
print("==================================================")
print("REALISTIC SENSOR OBSERVABILITY ANALYSIS")
print("==================================================")


print()
print("STATE VECTOR")
print("----------------------------------------------")

print(
    "x = [x, y, z, vx, vy, vz, "
    "phi, theta, psi, p, q, r]^T"
)


# ==========================================================
# 5. INPUT VECTOR
# ==========================================================
#
# u =
#
# [delta_T, tau_x, tau_y, tau_z]^T
#

print()
print("INPUT VECTOR")
print("----------------------------------------------")

print(
    "u = [delta_T, tau_x, tau_y, tau_z]^T"
)


# ==========================================================
# 6. PHYSICAL MODEL
# ==========================================================

A = np.zeros(
    (
        number_of_states,
        number_of_states
    )
)


# Position derivatives

A[0, 3] = 1.0
A[1, 4] = 1.0
A[2, 5] = 1.0


# Horizontal acceleration around hover

A[3, 7] = g
A[4, 6] = -g


# Attitude derivatives

A[6, 9] = 1.0
A[7, 10] = 1.0
A[8, 11] = 1.0


# ==========================================================
# 7. B MATRIX
# ==========================================================

B = np.zeros(
    (
        number_of_states,
        number_of_inputs
    )
)


# Vertical acceleration

B[5, 0] = 1.0 / mass


# Roll acceleration

B[9, 1] = 1.0 / Ixx


# Pitch acceleration

B[10, 2] = 1.0 / Iyy


# Yaw acceleration

B[11, 3] = 1.0 / Izz


# ==========================================================
# 8. REALISTIC SENSOR OUTPUT
# ==========================================================
#
# We assume the following quantities are measured:
#
# y1 = x
# y2 = y
# y3 = z
# y4 = phi
# y5 = theta
# y6 = psi
#
# We do NOT directly measure:
#
# vx, vy, vz, p, q, r
#
# Therefore C selects states:
#
# x, y, z, phi, theta, psi
#

C = np.zeros(
    (
        6,
        number_of_states
    )
)


# Position measurements

C[0, 0] = 1.0
C[1, 1] = 1.0
C[2, 2] = 1.0


# Attitude measurements

C[3, 6] = 1.0
C[4, 7] = 1.0
C[5, 8] = 1.0


# No direct input-to-output feedthrough

D = np.zeros(
    (
        6,
        number_of_inputs
    )
)


# ==========================================================
# 9. PRINT SENSOR OUTPUT
# ==========================================================

print()
print("MEASURED OUTPUTS")
print("----------------------------------------------")

print("y1 = x       [m]")
print("y2 = y       [m]")
print("y3 = z       [m]")
print("y4 = phi     [rad]")
print("y5 = theta   [rad]")
print("y6 = psi     [rad]")


print()
print("UNMEASURED STATES")
print("----------------------------------------------")

print("vx = x velocity")
print("vy = y velocity")
print("vz = z velocity")
print("p  = roll rate")
print("q  = pitch rate")
print("r  = yaw rate")


# ==========================================================
# 10. PRINT C MATRIX
# ==========================================================

print()
print("C MATRIX")
print("----------------------------------------------")

print(C)


# ==========================================================
# 11. OBSERVABILITY MATRIX
# ==========================================================
#
# For n = 12:
#
# O =
#
# [ C    ]
# [ CA   ]
# [ CA^2 ]
# [ ...  ]
# [ CA^11]
#
# ==========================================================

observability_blocks = []

current_block = C.copy()

for i in range(number_of_states):

    observability_blocks.append(
        current_block
    )

    current_block = current_block @ A


observability_matrix = np.vstack(
    observability_blocks
)


# ==========================================================
# 12. OBSERVABILITY RANK
# ==========================================================

observability_rank = np.linalg.matrix_rank(
    observability_matrix
)


print()
print("OBSERVABILITY MATRIX")
print("----------------------------------------------")

print(
    f"Shape = {observability_matrix.shape}"
)

print(
    f"Rank = {observability_rank}"
)

print(
    f"Number of states = {number_of_states}"
)


# ==========================================================
# 13. FINAL OBSERVABILITY RESULT
# ==========================================================

print()
print("==================================================")
print("OBSERVABILITY RESULT")
print("==================================================")


if observability_rank == number_of_states:

    print(
        "RESULT: All 12 states are observable "
        "from the selected outputs."
    )

else:

    print(
        "RESULT: The system is NOT fully observable "
        "from the selected outputs."
    )


print()
print(
    f"Observability rank : "
    f"{observability_rank}/{number_of_states}"
)


print(
    "Fully observable   :",
    observability_rank == number_of_states
)


# ==========================================================
# 14. PHYSICAL PARAMETERS
# ==========================================================

print()
print("PHYSICAL PARAMETERS")
print("----------------------------------------------")

print(
    f"Mass = {mass:.8f} kg"
)

print(
    f"Ixx  = {Ixx:.8f} kg m^2"
)

print(
    f"Iyy  = {Iyy:.8f} kg m^2"
)

print(
    f"Izz  = {Izz:.8f} kg m^2"
)

print(
    f"g    = {g:.8f} m/s^2"
)


print()
print("==================================================")