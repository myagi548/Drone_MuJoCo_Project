import numpy as np

from control.position_controller import PositionController


# ==========================================================
# PHYSICAL PARAMETER
# ==========================================================

gravity = 9.81


# ==========================================================
# CONTROLLER DESIGN PARAMETERS
# ==========================================================
#
# These are controller parameters, not physical constants.
#
# horizontal_gain:
#   converts position error into desired acceleration.
#
# vertical_gain:
#   converts vertical position error into desired
#   vertical acceleration.
#
# max_tilt:
#   limits the requested roll/pitch angle.
#
# ==========================================================

horizontal_gain = 1.0

vertical_gain = 1.0

max_tilt_rad = np.deg2rad(20.0)


# ==========================================================
# CREATE CONTROLLER
# ==========================================================

controller = PositionController(
    gravity=gravity,
    horizontal_gain=horizontal_gain,
    vertical_gain=vertical_gain,
    max_tilt_rad=max_tilt_rad
)


# ==========================================================
# TEST CASE 1
# ==========================================================
#
# Drone is behind the desired X position.
#
# Desired:
# x = 1 m
#
# Actual:
# x = 0 m
#
# Therefore:
#
# error_x = +1 m
#
# Positive x acceleration is required.
#
# From:
#
# x_ddot = g * theta
#
# theta = x_ddot / g
#
# the desired pitch should be positive.
#
# ==========================================================

desired_position = np.array(
    [1.0, 0.0, 1.0]
)

actual_position = np.array(
    [0.0, 0.0, 1.0]
)


roll, pitch, z_acceleration = (
    controller.compute(
        desired_position,
        actual_position
    )
)


print()
print("==================================================")
print("POSITION CONTROLLER TEST")
print("==================================================")


print()
print("TEST 1: POSITIVE X ERROR")
print("----------------------------------------------")

print(
    "Desired position =",
    desired_position
)

print(
    "Actual position  =",
    actual_position
)

print(
    "Position error   =",
    desired_position - actual_position
)

print(
    "Desired roll     =",
    np.rad2deg(roll),
    "deg"
)

print(
    "Desired pitch    =",
    np.rad2deg(pitch),
    "deg"
)

print(
    "Desired Z accel  =",
    z_acceleration,
    "m/s^2"
)


# ==========================================================
# TEST CASE 2
# ==========================================================
#
# Drone is to the left of the desired Y position.
#
# Desired:
# y = 1 m
#
# Actual:
# y = 0 m
#
# Therefore:
#
# error_y = +1 m
#
# Positive Y acceleration is required.
#
# From:
#
# y_ddot = -g * phi
#
# phi = -y_ddot / g
#
# desired roll should therefore be negative.
#
# ==========================================================

desired_position = np.array(
    [0.0, 1.0, 1.0]
)

actual_position = np.array(
    [0.0, 0.0, 1.0]
)


roll, pitch, z_acceleration = (
    controller.compute(
        desired_position,
        actual_position
    )
)


print()
print("TEST 2: POSITIVE Y ERROR")
print("----------------------------------------------")

print(
    "Desired position =",
    desired_position
)

print(
    "Actual position  =",
    actual_position
)

print(
    "Position error   =",
    desired_position - actual_position
)

print(
    "Desired roll     =",
    np.rad2deg(roll),
    "deg"
)

print(
    "Desired pitch    =",
    np.rad2deg(pitch),
    "deg"
)

print(
    "Desired Z accel  =",
    z_acceleration,
    "m/s^2"
)


# ==========================================================
# TEST CASE 3
# ==========================================================
#
# Drone is below the desired altitude.
#
# Desired z = 2 m
# Actual z  = 1 m
#
# error_z = +1 m
#
# Therefore a positive vertical acceleration is requested.
#
# ==========================================================

desired_position = np.array(
    [0.0, 0.0, 2.0]
)

actual_position = np.array(
    [0.0, 0.0, 1.0]
)


roll, pitch, z_acceleration = (
    controller.compute(
        desired_position,
        actual_position
    )
)


print()
print("TEST 3: POSITIVE Z ERROR")
print("----------------------------------------------")

print(
    "Desired position =",
    desired_position
)

print(
    "Actual position  =",
    actual_position
)

print(
    "Position error   =",
    desired_position - actual_position
)

print(
    "Desired roll     =",
    np.rad2deg(roll),
    "deg"
)

print(
    "Desired pitch    =",
    np.rad2deg(pitch),
    "deg"
)

print(
    "Desired Z accel  =",
    z_acceleration,
    "m/s^2"
)


# ==========================================================
# TEST CASE 4
# ==========================================================
#
# Drone and target are at the same position.
#
# Therefore all errors should be zero.
#
# ==========================================================

desired_position = np.array(
    [2.0, 3.0, 1.5]
)

actual_position = np.array(
    [2.0, 3.0, 1.5]
)


roll, pitch, z_acceleration = (
    controller.compute(
        desired_position,
        actual_position
    )
)


print()
print("TEST 4: ZERO POSITION ERROR")
print("----------------------------------------------")

print(
    "Desired position =",
    desired_position
)

print(
    "Actual position  =",
    actual_position
)

print(
    "Position error   =",
    desired_position - actual_position
)

print(
    "Desired roll     =",
    np.rad2deg(roll),
    "deg"
)

print(
    "Desired pitch    =",
    np.rad2deg(pitch),
    "deg"
)

print(
    "Desired Z accel  =",
    z_acceleration,
    "m/s^2"
)


print()
print("==================================================")
print("POSITION CONTROLLER TEST COMPLETE")
print("==================================================")