import numpy as np


# ==========================================================
# ROLL PLANT
# ==========================================================
#
#     Ixx * phi_ddot = torque
#
# Therefore:
#
#     phi_ddot = torque / Ixx
#
# Ixx comes from the MuJoCo drone model.
#

Ixx = 0.02296667       # kg m^2


# ==========================================================
# DESIGN REQUIREMENTS
# ==========================================================
#
# These are controller design choices, not physical
# properties of the drone.
#

settling_time = 1.0    # seconds

zeta = 1.0             # critical damping


# ==========================================================
# NATURAL FREQUENCY
# ==========================================================
#
# Using the approximate 2% settling-time relationship:
#
#     Ts = 4 / (zeta * wn)
#
# Therefore:
#
#     wn = 4 / (zeta * Ts)
#

wn = 4.0 / (
    zeta * settling_time
)


# ==========================================================
# THIRD POLE
# ==========================================================
#
# The third pole is placed at:
#
#     p3 = third_pole_factor * wn
#
# The factor 5 is a design choice.
#

third_pole_factor = 5.0

p3 = (
    third_pole_factor
    * wn
)


# ==========================================================
# PID GAINS
# ==========================================================
#
# Desired characteristic polynomial:
#
# (s^2 + 2*zeta*wn*s + wn^2)(s + p3)
#
# Matching coefficients gives:
#
# Kd = Ixx * (2*zeta*wn + p3)
#
# Kp = Ixx * (wn^2 + 2*zeta*wn*p3)
#
# Ki = Ixx * wn^2 * p3
#

Kd = Ixx * (
    2.0 * zeta * wn
    + p3
)

Kp = Ixx * (
    wn**2
    + 2.0 * zeta * wn * p3
)

Ki = Ixx * (
    wn**2
    * p3
)


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

print("==============================================")
print("PID GAIN CALCULATION")
print("==============================================")


print()
print("PHYSICAL PARAMETER")
print("----------------------------------------------")

print(
    f"Ixx = {Ixx:.8f} kg m^2"
)


print()
print("DESIGN REQUIREMENTS")
print("----------------------------------------------")

print(
    f"Settling-time target = "
    f"{settling_time:.2f} s"
)

print(
    f"Damping ratio (zeta) = "
    f"{zeta:.2f}"
)

print(
    "Damping condition = "
    "Critical damping"
)


print()
print("CALCULATED PARAMETERS")
print("----------------------------------------------")

print(
    f"Natural frequency (wn) = "
    f"{wn:.6f} rad/s"
)

print(
    f"Third-pole factor = "
    f"{third_pole_factor:.2f}"
)

print(
    f"Third pole (p3) = "
    f"{p3:.6f} rad/s"
)


print()
print("PID GAINS")
print("----------------------------------------------")

print(
    f"Kp = {Kp:.8f}"
)

print(
    f"Ki = {Ki:.8f}"
)

print(
    f"Kd = {Kd:.8f}"
)


print()
print("==============================================")