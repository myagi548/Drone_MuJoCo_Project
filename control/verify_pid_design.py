import numpy as np


# ==========================================================
# 1. PHYSICAL PARAMETER
# ==========================================================

Ixx = 0.02296667


# ==========================================================
# 2. CURRENT PID GAINS
# ==========================================================

Kp = 4.04213392

Ki = 7.34933440

Kd = 0.64306676


# ==========================================================
# 3. DESIGN PARAMETERS
# ==========================================================

zeta = 1.0

settling_time_target = 1.0

wn = 4.0 / (
    zeta * settling_time_target
)

third_pole_factor = 5.0

p3 = (
    third_pole_factor
    * wn
)


# ==========================================================
# 4. ACTUAL CLOSED-LOOP POLYNOMIAL
# ==========================================================
#
# Ixx*s^3 + Kd*s^2 + Kp*s + Ki = 0
#
# Divide by Ixx:
#
# s^3
# + (Kd/Ixx)s^2
# + (Kp/Ixx)s
# + Ki/Ixx = 0
#

actual_coefficients = np.array(
    [
        Ixx,
        Kd,
        Kp,
        Ki
    ]
)


# ==========================================================
# 5. CALCULATE ACTUAL POLES
# ==========================================================

actual_poles = np.roots(
    actual_coefficients
)


# ==========================================================
# 6. INTENDED SECOND-ORDER POLES
# ==========================================================
#
# s^2 + 2*zeta*wn*s + wn^2
#
# For zeta = 1:
#
# (s + wn)^2
#

second_order_poles = np.roots(
    [
        1.0,
        2.0 * zeta * wn,
        wn**2
    ]
)


# ==========================================================
# 7. INTENDED THIRD POLE
# ==========================================================

intended_poles = np.array(
    [
        -wn,
        -wn,
        -p3
    ]
)


# ==========================================================
# 8. RECONSTRUCT POLYNOMIAL FROM INTENDED POLES
# ==========================================================

normalized_intended_coefficients = np.poly(
    intended_poles
)


physical_intended_coefficients = (
    Ixx
    * normalized_intended_coefficients
)


# ==========================================================
# 9. COEFFICIENT ERRORS
# ==========================================================

coefficient_errors = (
    actual_coefficients
    - physical_intended_coefficients
)


# ==========================================================
# 10. DISPLAY
# ==========================================================

print()

print("==============================================")
print("PID CLOSED-LOOP MATHEMATICAL VERIFICATION")
print("==============================================")

print()

print("PHYSICAL PARAMETER")
print("----------------------------------------------")

print(
    f"Ixx = {Ixx:.8f} kg m^2"
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

print("DESIGN PARAMETERS")
print("----------------------------------------------")

print(
    f"zeta = {zeta:.6f}"
)

print(
    f"wn = {wn:.6f} rad/s"
)

print(
    f"Third-pole factor = "
    f"{third_pole_factor:.6f}"
)

print(
    f"p3 = {p3:.6f} rad/s"
)


print()

print("ACTUAL CLOSED-LOOP POLYNOMIAL")
print("----------------------------------------------")

print(
    "Ixx*s^3 + Kd*s^2 + Kp*s + Ki = 0"
)

print()

print(
    f"{Ixx:.8f}s^3 "
    f"+ {Kd:.8f}s^2 "
    f"+ {Kp:.8f}s "
    f"+ {Ki:.8f} = 0"
)


print()

print("INTENDED POLES")
print("----------------------------------------------")

for pole in intended_poles:

    print(
        f"{pole:.8f}"
    )


print()

print("ACTUAL POLES")
print("----------------------------------------------")

for pole in actual_poles:

    print(
        f"{pole.real:.8f}"
        f" {pole.imag:+.8f}j"
    )


print()

print("INTENDED PHYSICAL COEFFICIENTS")
print("----------------------------------------------")

labels = [
    "s^3 coefficient",
    "s^2 coefficient",
    "s coefficient",
    "constant coefficient"
]

for label, value in zip(
    labels,
    physical_intended_coefficients
):

    print(
        f"{label} = {value:.8f}"
    )


print()

print("COEFFICIENT DIFFERENCE")
print("----------------------------------------------")

for label, value in zip(
    labels,
    coefficient_errors
):

    print(
        f"{label} = {value:.10f}"
    )


print()

print("SECOND-ORDER DESIGN POLES")
print("----------------------------------------------")

for pole in second_order_poles:

    print(
        f"{pole:.8f}"
    )


print()

print("==============================================")