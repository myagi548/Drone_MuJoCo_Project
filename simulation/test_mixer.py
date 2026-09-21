from mixer import mix_roll_torque


mass = 1.0
gravity = 9.81

total_thrust = mass * gravity

arm_length = 0.45

roll_torque = -1.071823


thrusts = mix_roll_torque(
    total_thrust,
    roll_torque,
    arm_length
)


T1, T2, T3, T4 = thrusts


calculated_total = (
    T1 + T2 + T3 + T4
)

calculated_roll_torque = (
    arm_length * (T3 - T4)
)


print("==============================================")
print("ROLL MIXER TEST")
print("==============================================")

print(f"Required total thrust = {total_thrust:.6f} N")

print(f"Required roll torque = {roll_torque:.6f} N m")

print()
print("ROTOR THRUSTS")
print("----------------------------------------------")

print(f"T1 = {T1:.6f} N")
print(f"T2 = {T2:.6f} N")
print(f"T3 = {T3:.6f} N")
print(f"T4 = {T4:.6f} N")

print()
print("VERIFICATION")
print("----------------------------------------------")

print(
    f"Calculated total thrust = "
    f"{calculated_total:.6f} N"
)

print(
    f"Calculated roll torque = "
    f"{calculated_roll_torque:.6f} N m"
)

print("==============================================")