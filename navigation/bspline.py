import numpy as np


# ==========================================================
# CUBIC B-SPLINE BASIS FUNCTION
# ==========================================================

def basis_function(i, degree, u, knots):
    """
    Calculate the B-spline basis function N(i, degree).

    Parameters
    ----------
    i : int
        Basis-function index.

    degree : int
        Polynomial degree.
        For cubic B-spline, degree = 3.

    u : float
        Current trajectory parameter.

    knots : numpy array
        Knot vector.

    Returns
    -------
    float
        Value of N(i, degree, u).
    """

    # ------------------------------------------------------
    # Degree 0
    # ------------------------------------------------------

    if degree == 0:

        if (
            knots[i] <= u < knots[i + 1]
        ):
            return 1.0

        return 0.0


    # ------------------------------------------------------
    # Recursive Cox-de Boor formula
    # ------------------------------------------------------

    left_denominator = (
        knots[i + degree]
        - knots[i]
    )

    right_denominator = (
        knots[i + degree + 1]
        - knots[i + 1]
    )


    left_term = 0.0
    right_term = 0.0


    if left_denominator != 0:

        left_term = (
            (u - knots[i])
            / left_denominator
        ) * basis_function(
            i,
            degree - 1,
            u,
            knots
        )


    if right_denominator != 0:

        right_term = (
            (knots[i + degree + 1] - u)
            / right_denominator
        ) * basis_function(
            i + 1,
            degree - 1,
            u,
            knots
        )


    return left_term + right_term


# ==========================================================
# CREATE CLAMPED KNOT VECTOR
# ==========================================================

def create_clamped_knot_vector(
    number_of_control_points,
    degree
):
    """
    Create a uniform clamped knot vector.

    For n+1 control points and degree p,
    the knot vector contains n+p+2 values.
    """

    n = (
        number_of_control_points - 1
    )

    number_of_knots = (
        n + degree + 2
    )


    # Number of internal knots
    number_of_internal_knots = (
        number_of_knots
        - 2 * (degree + 1)
    )


    if number_of_internal_knots > 0:

        internal_knots = np.linspace(
            0.0,
            1.0,
            number_of_internal_knots + 2
        )[1:-1]

    else:

        internal_knots = np.array([])


    start_knots = np.zeros(
        degree + 1
    )

    end_knots = np.ones(
        degree + 1
    )


    knots = np.concatenate(
        [
            start_knots,
            internal_knots,
            end_knots
        ]
    )


    return knots


# ==========================================================
# CALCULATE ONE B-SPLINE POINT
# ==========================================================

def calculate_bspline_point(
    u,
    control_points,
    degree,
    knots
):
    """
    Calculate one point on the B-spline curve.

    C(u) = sum N(i,p,u) * P_i
    """

    number_of_control_points = (
        len(control_points)
    )


    point = np.zeros(
        control_points.shape[1]
    )


    for i in range(
        number_of_control_points
    ):

        basis = basis_function(
            i,
            degree,
            u,
            knots
        )

        point += (
            basis
            * control_points[i]
        )


    return point


# ==========================================================
# GENERATE COMPLETE B-SPLINE TRAJECTORY
# ==========================================================

def generate_bspline(
    control_points,
    number_of_samples=101,
    degree=3
):
    """
    Generate a cubic B-spline trajectory.
    """

    control_points = np.asarray(
        control_points,
        dtype=float
    )


    if control_points.ndim != 2:

        raise ValueError(
            "Control points must be a 2D array."
        )


    if len(control_points) < degree + 1:

        raise ValueError(
            "A cubic B-spline requires "
            "at least 4 control points."
        )


    knots = create_clamped_knot_vector(
        len(control_points),
        degree
    )


    # The valid parameter range is
    # from the first knot to the last knot.
    #
    # The final point u=1 needs special
    # handling because the degree-zero
    # basis uses a half-open interval.

    parameters = np.linspace(
        0.0,
        1.0,
        number_of_samples
    )


    trajectory = []


    for index, u in enumerate(
        parameters
    ):

        if index == len(parameters) - 1:

            # Evaluate slightly inside the
            # final knot interval.
            evaluation_u = (
                1.0 - 1e-12
            )

        else:

            evaluation_u = u


        point = calculate_bspline_point(
            evaluation_u,
            control_points,
            degree,
            knots
        )


        trajectory.append(point)


    trajectory = np.array(
        trajectory
    )


    # Explicitly force the final point
    # to the final control point.
    trajectory[-1] = control_points[-1]


    return (
        parameters,
        trajectory,
        knots
    )


# ==========================================================
# MAIN TEST
# ==========================================================

if __name__ == "__main__":

    # ------------------------------------------------------
    # Control points
    #
    # These are deliberately simple test values.
    # They are NOT claimed to be physically derived
    # drone waypoints.
    # ------------------------------------------------------

    control_points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.5, 1.2],
            [2.0, -0.5, 1.5],
            [3.0, 0.5, 1.8],
            [4.0, 0.0, 2.0]
        ]
    )


    degree = 3

    number_of_samples = 101


    # ------------------------------------------------------
    # Generate trajectory
    # ------------------------------------------------------

    parameters, trajectory, knots = (
        generate_bspline(
            control_points,
            number_of_samples,
            degree
        )
    )


    # ------------------------------------------------------
    # Print information
    # ------------------------------------------------------

    print()
    print("==================================================")
    print("CUBIC B-SPLINE TRAJECTORY TEST")
    print("==================================================")


    print()
    print("DEGREE")
    print("----------------------------------------------")

    print(
        f"Degree = {degree}"
    )


    print()
    print("CONTROL POINTS")
    print("----------------------------------------------")

    print(
        control_points
    )


    print()
    print("KNOT VECTOR")
    print("----------------------------------------------")

    print(
        knots
    )


    print()
    print("NUMBER OF SAMPLES")
    print("----------------------------------------------")

    print(
        f"{number_of_samples}"
    )


    print()
    print("FIRST TRAJECTORY POINT")
    print("----------------------------------------------")

    print(
        trajectory[0]
    )


    print()
    print("MIDDLE TRAJECTORY POINT")
    print("----------------------------------------------")

    print(
        trajectory[
            number_of_samples // 2
        ]
    )


    print()
    print("FINAL TRAJECTORY POINT")
    print("----------------------------------------------")

    print(
        trajectory[-1]
    )


    print()
    print("TRAJECTORY SHAPE")
    print("----------------------------------------------")

    print(
        trajectory.shape
    )


    print()
    print("==================================================")
    print("B-SPLINE TEST COMPLETE")
    print("==================================================")