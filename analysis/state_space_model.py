import numpy as np


# ==========================================================
# 1. PHYSICAL PARAMETER
# ==========================================================
#
# This value comes from the MuJoCo drone model.
#
# Ixx = roll-axis moment of inertia
#

Ixx = 0.02296667


# ==========================================================
# 2. STATE DEFINITION
# ==========================================================
#
# x1 = roll angle, phi       [rad]
# x2 = roll angular velocity [rad/s]
#
# Therefore:
#
# x = [phi, phi_dot]^T
#
# Input:
#
# u = roll torque [N m]
#

print()
print("==================================================")
print("ROLL STATE-SPACE ANALYSIS")
print("==================================================")


print()
print("STATE DEFINITION")
print("----------------------------------------------")

print("x1 = roll angle phi [rad]")
print("x2 = roll angular velocity phi_dot [rad/s]")
print("u  = roll torque tau_x [N m]")


# ==========================================================
# 3. PHYSICAL EQUATION
# ==========================================================
#
# Ixx * phi_ddot = tau_x
#
# Therefore:
#
# phi_ddot = (1/Ixx) * tau_x
#

inverse_Ixx = 1.0 / Ixx


print()
print("PHYSICAL MODEL")
print("----------------------------------------------")

print(
    f"Ixx = {Ixx:.8f} kg m^2"
)

print(
    f"1/Ixx = {inverse_Ixx:.8f}"
)


# ==========================================================
# 4. STATE-SPACE MATRICES
# ==========================================================
#
# x_dot = A*x + B*u
#
# A =
#
# [ 0   1 ]
# [ 0   0 ]
#
# B =
#
# [   0   ]
# [ 1/Ixx ]
#

A = np.array(
    [
        [0.0, 1.0],
        [0.0, 0.0]
    ]
)

B = np.array(
    [
        [0.0],
        [inverse_Ixx]
    ]
)


# ==========================================================
# 5. OUTPUT EQUATION
# ==========================================================
#
# We assume roll angle is measured:
#
# y = phi
#
# Therefore:
#
# C = [1  0]
#
# D = [0]
#

C = np.array(
    [
        [1.0, 0.0]
    ]
)

D = np.array(
    [
        [0.0]
    ]
)


print()
print("STATE-SPACE MATRICES")
print("----------------------------------------------")

print("A =")
print(A)

print()

print("B =")
print(B)

print()

print("C =")
print(C)

print()

print("D =")
print(D)


# ==========================================================
# 6. CONTROLLABILITY MATRIX
# ==========================================================
#
# For a system with n = 2 states:
#
# Controllability matrix:
#
# Mc = [B  AB]
#
# ----------------------------------------------------------
#
# First calculate AB.
#

AB = A @ B


print()
print("CONTROLLABILITY CALCULATION")
print("----------------------------------------------")

print("AB =")
print(AB)


# Construct:
#
# [ B | AB ]
#

controllability_matrix = np.hstack(
    [
        B,
        AB
    ]
)


print()
print("Controllability matrix =")
print(controllability_matrix)


# ==========================================================
# 7. CONTROLLABILITY RANK
# ==========================================================

controllability_rank = np.linalg.matrix_rank(
    controllability_matrix
)

number_of_states = A.shape[0]


print()
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
        "RESULT: The roll system is CONTROLLABLE."
    )

else:

    print(
        "RESULT: The roll system is NOT fully controllable."
    )


# ==========================================================
# 8. OBSERVABILITY MATRIX
# ==========================================================
#
# For a system with n = 2 states:
#
# Observability matrix:
#
# Mo =
#
# [ C  ]
# [ CA ]
#
# ----------------------------------------------------------
#
# Calculate CA.
#

CA = C @ A


print()
print("OBSERVABILITY CALCULATION")
print("----------------------------------------------")

print("CA =")
print(CA)


# Construct:
#
# [ C  ]
# [ CA ]
#

observability_matrix = np.vstack(
    [
        C,
        CA
    ]
)


print()
print("Observability matrix =")
print(observability_matrix)


# ==========================================================
# 9. OBSERVABILITY RANK
# ==========================================================

observability_rank = np.linalg.matrix_rank(
    observability_matrix
)


print()
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
        "RESULT: The roll system is OBSERVABLE."
    )

else:

    print(
        "RESULT: The roll system is NOT fully observable."
    )


# ==========================================================
# 10. DETERMINANTS
# ==========================================================
#
# A full-rank square matrix has a non-zero determinant.
#
# This gives an additional mathematical check.
#

controllability_determinant = np.linalg.det(
    controllability_matrix
)

observability_determinant = np.linalg.det(
    observability_matrix
)


print()
print("ADDITIONAL CHECK")
print("----------------------------------------------")

print(
    f"det(Mc) = "
    f"{controllability_determinant:.8f}"
)

print(
    f"det(Mo) = "
    f"{observability_determinant:.8f}"
)


# ==========================================================
# 11. FINAL SUMMARY
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

print("Controllable :",
      controllability_rank == number_of_states)

print("Observable   :",
      observability_rank == number_of_states)

print()
print("==================================================")