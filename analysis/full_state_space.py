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
# 3. DEFINE STATE VECTOR
# ==========================================================
#
# x =
#
# [ position_x
#   position_y
#   position_z
#   velocity_x
#   velocity_y
#   velocity_z
#   roll
#   pitch
#   yaw
#   roll_rate
#   pitch_rate
#   yaw_rate ]
#
# Therefore:
#
# x1  = x
# x2  = y
# x3  = z
# x4  = vx
# x5  = vy
# x6  = vz
# x7  = phi
# x8  = theta
# x9  = psi
# x10 = p
# x11 = q
# x12 = r
#

number_of_states = 12


# ==========================================================
# 4. DEFINE INPUT VECTOR
# ==========================================================
#
# u =
#
# [ delta_T
#   tau_x
#   tau_y
#   tau_z ]
#
# delta_T = change in total thrust from hover
# tau_x   = roll torque
# tau_y   = pitch torque
# tau_z   = yaw torque
#

number_of_inputs = 4


# ==========================================================
# 5. CREATE A MATRIX
# ==========================================================
#
# State equation:
#
#       x_dot = A*x + B*u
#
# Start with zeros because each element must come
# from the physical equations.
#

A = np.zeros(
    (
        number_of_states,
        number_of_states
    )
)


# ==========================================================
# 6. POSITION EQUATIONS
# ==========================================================
#
# x_dot  = vx
# y_dot  = vy
# z_dot  = vz
#
# Therefore:
#
# A[0,3] = 1
# A[1,4] = 1
# A[2,5] = 1
#

A[0, 3] = 1.0
A[1, 4] = 1.0
A[2, 5] = 1.0


# ==========================================================
# 7. HORIZONTAL ACCELERATION
# ==========================================================
#
# Around hover and using the small-angle approximation:
#
# x_ddot = g * theta
#
# y_ddot = -g * phi
#
# Therefore:
#
# A[3,7] = g
# A[4,6] = -g
#

A[3, 7] = g
A[4, 6] = -g


# ==========================================================
# 8. ATTITUDE KINEMATICS
# ==========================================================
#
# Around hover:
#
# phi_dot   = p
# theta_dot = q
# psi_dot   = r
#
# Therefore:
#
# A[6,9]  = 1
# A[7,10] = 1
# A[8,11] = 1
#

A[6, 9] = 1.0
A[7, 10] = 1.0
A[8, 11] = 1.0


# ==========================================================
# 9. CREATE B MATRIX
# ==========================================================
#
# B describes how the inputs affect the states.
#
# Inputs:
#
# u1 = delta_T
# u2 = tau_x
# u3 = tau_y
# u4 = tau_z
#

B = np.zeros(
    (
        number_of_states,
        number_of_inputs
    )
)


# ==========================================================
# 10. VERTICAL ACCELERATION
# ==========================================================
#
# m * z_ddot = delta_T
#
# Therefore:
#
# z_ddot = delta_T / m
#
# Hence:
#
# B[5,0] = 1/m
#

B[5, 0] = 1.0 / mass


# ==========================================================
# 11. ROLL DYNAMICS
# ==========================================================
#
# Ixx * phi_ddot = tau_x
#
# Therefore:
#
# phi_ddot = tau_x / Ixx
#

B[9, 1] = 1.0 / Ixx


# ==========================================================
# 12. PITCH DYNAMICS
# ==========================================================
#
# Iyy * theta_ddot = tau_y
#
# Therefore:
#
# theta_ddot = tau_y / Iyy
#

B[10, 2] = 1.0 / Iyy


# ==========================================================
# 13. YAW DYNAMICS
# ==========================================================
#
# Izz * psi_ddot = tau_z
#
# Therefore:
#
# psi_ddot = tau_z / Izz
#

B[11, 3] = 1.0 / Izz


# ==========================================================
# 14. OUTPUT MATRIX
# ==========================================================
#
# For this first full-state analysis we assume that
# the simulation provides all 12 states as measurable/
# available outputs.
#
# Therefore:
#
# C = identity matrix
#
# This means:
#
# y = x
#
# Every state is available to the controller/estimator.
#

C = np.eye(number_of_states)


# D matrix:
#
# There is no direct input-to-output feedthrough
# in this state-space model.
#

D = np.zeros(
    (
        number_of_states,
        number_of_inputs
    )
)


# ==========================================================
# 15. PRINT PHYSICAL PARAMETERS
# ==========================================================

print()
print("==================================================")
print("FULL DRONE STATE-SPACE MODEL")
print("==================================================")


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


# ==========================================================
# 16. PRINT STATE AND INPUT DEFINITIONS
# ==========================================================

print()
print("STATE VECTOR")
print("----------------------------------------------")

print(
    "x = [x, y, z, vx, vy, vz, "
    "phi, theta, psi, p, q, r]^T"
)


print()
print("INPUT VECTOR")
print("----------------------------------------------")

print(
    "u = [delta_T, tau_x, tau_y, tau_z]^T"
)


# ==========================================================
# 17. PRINT A MATRIX
# ==========================================================

print()
print("A MATRIX")
print("----------------------------------------------")

print(A)


# ==========================================================
# 18. PRINT B MATRIX
# ==========================================================

print()
print("B MATRIX")
print("----------------------------------------------")

print(B)


# ==========================================================
# 19. PRINT C MATRIX
# ==========================================================

print()
print("C MATRIX")
print("----------------------------------------------")

print(C)


# ==========================================================
# 20. PRINT D MATRIX
# ==========================================================

print()
print("D MATRIX")
print("----------------------------------------------")

print(D)


# ==========================================================
# 21. CONTROLLABILITY MATRIX
# ==========================================================
#
# For n states:
#
# Mc = [B  AB  A^2B ... A^(n-1)B]
#
# Here n = 12.
#
# We construct it manually using repeated multiplication.
#

controllability_blocks = []

current_block = B.copy()

for i in range(number_of_states):

    controllability_blocks.append(
        current_block
    )

    current_block = A @ current_block


controllability_matrix = np.hstack(
    controllability_blocks
)


# ==========================================================
# 22. CONTROLLABILITY RANK
# ==========================================================

controllability_rank = np.linalg.matrix_rank(
    controllability_matrix
)


print()
print("CONTROLLABILITY")
print("----------------------------------------------")

print(
    f"Controllability matrix shape = "
    f"{controllability_matrix.shape}"
)

print(
    f"Controllability rank = "
    f"{controllability_rank}"
)

print(
    f"Number of states = "
    f"{number_of_states}"
)


if controllability_rank == number_of_states:

    print(
        "RESULT: The full model is CONTROLLABLE."
    )

else:

    print(
        "RESULT: The full model is NOT fully controllable."
    )


# ==========================================================
# 23. OBSERVABILITY MATRIX
# ==========================================================
#
# For n states:
#
# Mo =
#
# [ C    ]
# [ CA   ]
# [ CA^2 ]
# [ ...  ]
# [ CA^(n-1) ]
#
# We construct it using repeated multiplication.
#

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
# 24. OBSERVABILITY RANK
# ==========================================================

observability_rank = np.linalg.matrix_rank(
    observability_matrix
)


print()
print("OBSERVABILITY")
print("----------------------------------------------")

print(
    f"Observability matrix shape = "
    f"{observability_matrix.shape}"
)

print(
    f"Observability rank = "
    f"{observability_rank}"
)

print(
    f"Number of states = "
    f"{number_of_states}"
)


if observability_rank == number_of_states:

    print(
        "RESULT: The full model is OBSERVABLE."
    )

else:

    print(
        "RESULT: The full model is NOT fully observable."
    )


# ==========================================================
# 25. FINAL SUMMARY
# ==========================================================

print()
print("==================================================")
print("FINAL RESULT")
print("==================================================")

print(
    f"Controllability rank : "
    f"{controllability_rank}/{number_of_states}"
)

print(
    f"Observability rank   : "
    f"{observability_rank}/{number_of_states}"
)

print()

print(
    "Controllable :",
    controllability_rank == number_of_states
)

print(
    "Observable   :",
    observability_rank == number_of_states
)

print()
print("==================================================")