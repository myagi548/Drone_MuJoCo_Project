import numpy as np


def mix_roll_pitch(
    total_thrust,
    roll_torque,
    pitch_torque,
    arm_length
):
    """
    Convert total thrust, roll torque and pitch torque
    into four rotor thrusts.

    Rotor configuration:

                  Rotor 3
                     |
                     |
        Rotor 2 ---- + ---- Rotor 1
                     |
                     |
                  Rotor 4

    Rotor positions:

        Rotor 1 = (+L,  0, 0)
        Rotor 2 = (-L,  0, 0)
        Rotor 3 = ( 0, +L, 0)
        Rotor 4 = ( 0, -L, 0)

    Each rotor produces an upward force:

        F_i = [0, 0, T_i]

    Therefore:

        tau_x = L(T3 - T4)

        tau_y = L(T2 - T1)

    and:

        total_thrust = T1 + T2 + T3 + T4
    """

    if arm_length <= 0:
        raise ValueError(
            "Arm length must be greater than zero."
        )

    # ------------------------------------------------------
    # Equal thrust distribution
    # ------------------------------------------------------

    hover_per_rotor = (
        total_thrust / 4.0
    )

    # ------------------------------------------------------
    # Thrust difference required for roll torque
    # ------------------------------------------------------

    roll_difference = (
        roll_torque
        / (2.0 * arm_length)
    )

    # ------------------------------------------------------
    # Thrust difference required for pitch torque
    # ------------------------------------------------------

    pitch_difference = (
        pitch_torque
        / (2.0 * arm_length)
    )

    # ------------------------------------------------------
    # Rotor thrusts
    # ------------------------------------------------------

    rotor_1 = (
        hover_per_rotor
        - pitch_difference
    )

    rotor_2 = (
        hover_per_rotor
        + pitch_difference
    )

    rotor_3 = (
        hover_per_rotor
        + roll_difference
    )

    rotor_4 = (
        hover_per_rotor
        - roll_difference
    )

    return np.array(
        [
            rotor_1,
            rotor_2,
            rotor_3,
            rotor_4
        ]
    )


def calculate_resulting_force_and_torque(
    thrusts,
    arm_length
):
    """
    Calculate the total thrust, roll torque and
    pitch torque produced by the four rotor thrusts.

    This is used to verify the mixer mathematically.
    """

    thrusts = np.asarray(
        thrusts,
        dtype=float
    )

    if thrusts.shape != (4,):
        raise ValueError(
            "Thrusts must contain exactly four values."
        )

    if arm_length <= 0:
        raise ValueError(
            "Arm length must be greater than zero."
        )

    total_thrust = np.sum(thrusts)

    roll_torque = (
        arm_length
        * (thrusts[2] - thrusts[3])
    )

    pitch_torque = (
        arm_length
        * (thrusts[1] - thrusts[0])
    )

    return (
        total_thrust,
        roll_torque,
        pitch_torque
    )


if __name__ == "__main__":

    print()
    print("==================================================")
    print("ROLL + PITCH MIXER")
    print("==================================================")


    # ------------------------------------------------------
    # Test parameters
    # ------------------------------------------------------
    #
    # These are deliberately simple test inputs.
    # They are not claimed to be optimal controller values.
    #
    # ------------------------------------------------------

    total_thrust = 9.81

    roll_torque = 0.10

    pitch_torque = 0.06

    arm_length = 0.45


    # ------------------------------------------------------
    # Calculate rotor thrusts
    # ------------------------------------------------------

    thrusts = mix_roll_pitch(
        total_thrust,
        roll_torque,
        pitch_torque,
        arm_length
    )


    print()
    print("INPUT")
    print("----------------------------------------------")

    print(
        f"Total thrust = "
        f"{total_thrust:.6f} N"
    )

    print(
        f"Roll torque  = "
        f"{roll_torque:.6f} N m"
    )

    print(
        f"Pitch torque = "
        f"{pitch_torque:.6f} N m"
    )

    print(
        f"Arm length   = "
        f"{arm_length:.6f} m"
    )


    # ------------------------------------------------------
    # Rotor thrusts
    # ------------------------------------------------------

    print()
    print("ROTOR THRUSTS")
    print("----------------------------------------------")

    for i, thrust in enumerate(
        thrusts,
        start=1
    ):

        print(
            f"Rotor {i} = "
            f"{thrust:.6f} N"
        )


    # ------------------------------------------------------
    # Verify the mixer
    # ------------------------------------------------------

    (
        resulting_thrust,
        resulting_roll,
        resulting_pitch
    ) = calculate_resulting_force_and_torque(
        thrusts,
        arm_length
    )


    print()
    print("VERIFICATION")
    print("----------------------------------------------")

    print(
        f"Resulting total thrust = "
        f"{resulting_thrust:.6f} N"
    )

    print(
        f"Resulting roll torque  = "
        f"{resulting_roll:.6f} N m"
    )

    print(
        f"Resulting pitch torque = "
        f"{resulting_pitch:.6f} N m"
    )


    # ------------------------------------------------------
    # Errors
    # ------------------------------------------------------

    thrust_error = (
        resulting_thrust
        - total_thrust
    )

    roll_error = (
        resulting_roll
        - roll_torque
    )

    pitch_error = (
        resulting_pitch
        - pitch_torque
    )


    print()
    print("MIXING ERRORS")
    print("----------------------------------------------")

    print(
        f"Thrust error = "
        f"{thrust_error:.12f} N"
    )

    print(
        f"Roll error   = "
        f"{roll_error:.12f} N m"
    )

    print(
        f"Pitch error  = "
        f"{pitch_error:.12f} N m"
    )


    print()
    print("==================================================")

    if (
        abs(thrust_error) < 1e-10
        and abs(roll_error) < 1e-10
        and abs(pitch_error) < 1e-10
    ):

        print(
            "RESULT: MIXER VERIFICATION PASSED"
        )

    else:

        print(
            "RESULT: MIXER VERIFICATION FAILED"
        )

    print("==================================================")