import numpy as np
import mujoco


MODEL_PATH = "simulation/drone_model.xml"


# ==========================================================
# LOAD MODEL
# ==========================================================

model = mujoco.MjModel.from_xml_path(
    MODEL_PATH
)

data = mujoco.MjData(model)


# ==========================================================
# GET DRONE BODY
# ==========================================================

drone_id = model.body("drone").id


print()
print("==================================================")
print("MUJOCO DIRECT FORCE ASSIGNMENT TEST")
print("==================================================")

print()
print("Drone body ID")
print("----------------------------------------------")
print(drone_id)


# ==========================================================
# DIRECTLY WRITE A FORCE AND TORQUE
# ==========================================================
#
# This is deliberately NOT using rotor_model.py.
#
# We are testing MuJoCo's xfrc_applied array itself.
#
# Force:
#     [0, 0, 9.81] N
#
# Torque:
#     [0.10, 0.06, 0] N m
#
# ==========================================================

data.xfrc_applied[drone_id, 0:3] = np.array(
    [0.0, 0.0, 9.81]
)

data.xfrc_applied[drone_id, 3:6] = np.array(
    [0.10, 0.06, 0.0]
)


# ==========================================================
# READ IT BACK
# ==========================================================

print()
print("xfrc_applied AFTER ASSIGNMENT")
print("----------------------------------------------")

print(
    data.xfrc_applied[drone_id]
)


print()
print("FORCE")
print("----------------------------------------------")

print(
    data.xfrc_applied[drone_id, 0:3]
)


print()
print("TORQUE")
print("----------------------------------------------")

print(
    data.xfrc_applied[drone_id, 3:6]
)


# ==========================================================
# VERIFICATION
# ==========================================================

expected = np.array(
    [0.0, 0.0, 9.81, 0.10, 0.06, 0.0]
)

actual = data.xfrc_applied[
    drone_id
].copy()


error = actual - expected


print()
print("ERROR")
print("----------------------------------------------")

print(error)


print()
print("==================================================")

if np.allclose(actual, expected):
    print(
        "RESULT: DIRECT MUJOCO ASSIGNMENT PASSED"
    )
else:
    print(
        "RESULT: DIRECT MUJOCO ASSIGNMENT FAILED"
    )

print("==================================================")