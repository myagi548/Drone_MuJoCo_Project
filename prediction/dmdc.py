import numpy as np


# ==========================================================
# 1. LOAD COLLECTED MUJOCO DATA
# ==========================================================

states = np.load(
    "data/dmdc_states.npy"
)

inputs = np.load(
    "data/dmdc_inputs.npy"
)


# ==========================================================
# 2. BUILD DMDc MATRICES
# ==========================================================
#
# We have:
#
# X  = [x0  x1  x2  ... x(N-1)]
#
# X' = [x1  x2  x3  ... xN]
#
# U  = [u0  u1  u2  ... u(N-1)]
#
# DMDc model:
#
#       x(k+1) = A_d x(k) + B_d u(k)
#
# Therefore:
#
#       X' = [A_d  B_d] [X]
#                        [U]
#
# Define:
#
#       Z = [X]
#           [U]
#
# Then:
#
#       X' = G Z
#
# where:
#
#       G = [A_d  B_d]
#
# ==========================================================


# Current states

X = states[:-1].T


# Next states

X_next = states[1:].T


# Inputs corresponding to current states

U = inputs[:-1].T


# Combined data matrix

Z = np.vstack(
    [
        X,
        U
    ]
)


# ==========================================================
# 3. LEAST-SQUARES DMDc
# ==========================================================
#
# We want:
#
#       X_next = G Z
#
# The least-squares solution is:
#
#       G = X_next Z^+
#
# where Z^+ is the Moore-Penrose pseudoinverse.
#
# NumPy's pinv uses SVD internally.
#
# SVD is numerical linear algebra used to solve the
# least-squares problem robustly.
# ==========================================================

G = (
    X_next
    @ np.linalg.pinv(Z)
)


# ==========================================================
# 4. SEPARATE A AND B
# ==========================================================
#
# State dimension = 12
# Input dimension = 4
#
# Therefore:
#
# G = [ A_d | B_d ]
#
# A_d -> 12 x 12
#
# B_d -> 12 x 4
# ==========================================================

number_of_states = states.shape[1]

number_of_inputs = inputs.shape[1]


A_d = G[
    :,
    :number_of_states
]


B_d = G[
    :,
    number_of_states:
]


# ==========================================================
# 5. PREDICTION ERROR
# ==========================================================

X_next_predicted = (
    A_d @ X
    + B_d @ U
)


prediction_error = (
    X_next
    - X_next_predicted
)


rmse = np.sqrt(
    np.mean(
        prediction_error ** 2
    )
)


# ==========================================================
# 6. STATE-BY-STATE RMSE
# ==========================================================

state_rmse = np.sqrt(
    np.mean(
        prediction_error ** 2,
        axis=1
    )
)


# ==========================================================
# 7. SAVE MATRICES
# ==========================================================

np.save(
    "data/dmdc_A.npy",
    A_d
)

np.save(
    "data/dmdc_B.npy",
    B_d
)


# ==========================================================
# 8. PRINT RESULTS
# ==========================================================

print()
print("==================================================")
print("DMDc SYSTEM IDENTIFICATION")
print("==================================================")


print()
print("DATA")
print("----------------------------------------------")

print(
    f"State samples = {states.shape[0]}"
)

print(
    f"State dimension = {number_of_states}"
)

print(
    f"Input dimension = {number_of_inputs}"
)


print()
print("DMDc MATRICES")
print("----------------------------------------------")

print(
    f"A_d shape = {A_d.shape}"
)

print(
    f"B_d shape = {B_d.shape}"
)


print()
print("A_d")
print("----------------------------------------------")

np.set_printoptions(
    precision=8,
    suppress=True
)

print(A_d)


print()
print("B_d")
print("----------------------------------------------")

print(B_d)


print()
print("PREDICTION ERROR")
print("----------------------------------------------")

print(
    f"Overall RMSE = {rmse:.12e}"
)


print()
print("STATE-BY-STATE RMSE")
print("----------------------------------------------")

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


for name, error in zip(
    state_names,
    state_rmse
):

    print(
        f"{name:>5s} : "
        f"{error:.12e}"
    )


print()
print("FILES SAVED")
print("----------------------------------------------")

print(
    "data/dmdc_A.npy"
)

print(
    "data/dmdc_B.npy"
)


print()
print("==================================================")