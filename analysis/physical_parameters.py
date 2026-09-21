import mujoco
import numpy as np


# ==========================================================
# 1. LOAD THE ACTUAL MUJOCO MODEL
# ==========================================================

MODEL_PATH = "simulation/drone_model.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)


# ==========================================================
# 2. FIND THE DRONE BODY
# ==========================================================

drone_id = model.body("drone").id


# ==========================================================
# 3. READ PHYSICAL PARAMETERS
# ==========================================================

mass = model.body_mass[drone_id]

inertia = model.body_inertia[drone_id]

Ixx = inertia[0]
Iyy = inertia[1]
Izz = inertia[2]


# Gravity is defined in the MuJoCo XML as:
#
# gravity="0 0 -9.81"
#

gravity_vector = model.opt.gravity

g = abs(gravity_vector[2])


# ==========================================================
# 4. PRINT RESULTS
# ==========================================================

print()
print("==================================================")
print("MUJOCO PHYSICAL PARAMETERS")
print("==================================================")


print()
print("DRONE BODY")
print("----------------------------------------------")

print(
    f"Body ID       = {drone_id}"
)

print(
    f"Mass          = {mass:.8f} kg"
)


print()
print("MOMENTS OF INERTIA")
print("----------------------------------------------")

print(
    f"Ixx           = {Ixx:.8f} kg m^2"
)

print(
    f"Iyy           = {Iyy:.8f} kg m^2"
)

print(
    f"Izz           = {Izz:.8f} kg m^2"
)


print()
print("GRAVITY")
print("----------------------------------------------")

print(
    f"Gravity vector = {gravity_vector}"
)

print(
    f"g              = {g:.8f} m/s^2"
)


# ==========================================================
# 5. HOVER THRUST
# ==========================================================
#
# At hover:
#
#       T = mg
#
# Therefore:
#
#       T_hover = mass * g
#

hover_thrust = mass * g


print()
print("HOVER CONDITION")
print("----------------------------------------------")

print(
    f"Total hover thrust = "
    f"{hover_thrust:.8f} N"
)


print(
    f"Hover thrust/rotor = "
    f"{hover_thrust / 4.0:.8f} N"
)


# ==========================================================
# 6. FINAL SUMMARY
# ==========================================================

print()
print("==================================================")
print("PARAMETER SUMMARY")
print("==================================================")

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

print(
    f"T_hover = {hover_thrust:.8f} N"
)

print()
print("==================================================")
