
import numpy as np

from navigation.obstacle_planner import (
    BoxObstacle,
    path_is_collision_free,
    inflate_obstacle
)

from navigation.waypoint_planner import (
    find_shortest_collision_free_path
)

from navigation.bspline import (
    generate_bspline
)


# ==========================================================
# PATH LENGTH
# ==========================================================

def calculate_path_length(path):
    """
    Calculate the total length of a sampled trajectory.
    """

    total_length = 0.0

    for i in range(len(path) - 1):
        segment = path[i + 1] - path[i]
        total_length += np.linalg.norm(segment)

    return total_length


# ==========================================================
# B-SPLINE COLLISION CHECK
# ==========================================================

def validate_bspline_trajectory(
    trajectory,
    obstacles,
    safety_distance
):
    """
    Check every consecutive sampled B-spline segment
    against the obstacles inflated by safety_distance.
    """

    return path_is_collision_free(
        trajectory,
        obstacles,
        safety_distance
    )


# ==========================================================
# MINIMUM CLEARANCE
# ==========================================================

def calculate_minimum_clearance(
    trajectory,
    obstacle
):
    """
    Calculate the minimum distance between sampled
    trajectory points and the supplied obstacle boundary.
    """

    minimum = obstacle.minimum
    maximum = obstacle.maximum

    minimum_distance = float("inf")

    for point in trajectory:

        dx = max(
            minimum[0] - point[0],
            0.0,
            point[0] - maximum[0]
        )

        dy = max(
            minimum[1] - point[1],
            0.0,
            point[1] - maximum[1]
        )

        dz = max(
            minimum[2] - point[2],
            0.0,
            point[2] - maximum[2]
        )

        distance = np.sqrt(
            dx ** 2 + dy ** 2 + dz ** 2
        )

        minimum_distance = min(
            minimum_distance,
            distance
        )

    return minimum_distance


# ==========================================================
# B-SPLINE SEARCH
# ==========================================================

def find_bspline_safe_path(
    start,
    goal,
    obstacles,
    safety_distance,
    initial_clearance,
    clearance_step,
    maximum_clearance,
    degree,
    number_of_samples
):
    """
    Search for a waypoint path whose resulting
    B-spline trajectory is collision free.

    The additional planning clearance is increased
    systematically until a safe B-spline is found.

    Returns:
        waypoints
        trajectory
        clearance
        waypoint_length
    """

    clearance = initial_clearance

    while clearance <= maximum_clearance:

        # --------------------------------------------------
        # Generate waypoint path using current clearance.
        # --------------------------------------------------

        waypoints, waypoint_length = (
            find_shortest_collision_free_path(
                start,
                goal,
                obstacles,
                safety_distance,
                clearance
            )
        )

        # --------------------------------------------------
        # Generate B-spline.
        # --------------------------------------------------

        parameters, trajectory, knots = generate_bspline(
            waypoints,
            number_of_samples=number_of_samples,
            degree=degree
        )

        # --------------------------------------------------
        # Check B-spline against obstacles.
        # --------------------------------------------------

        collision_free = validate_bspline_trajectory(
            trajectory,
            obstacles,
            safety_distance
        )

        print(
            f"Clearance = {clearance:.3f} m | "
            f"B-spline safe = {collision_free}"
        )

        # --------------------------------------------------
        # Return first safe solution.
        # --------------------------------------------------

        if collision_free:

            return (
                waypoints,
                trajectory,
                clearance,
                waypoint_length
            )

        # --------------------------------------------------
        # Increase clearance.
        # --------------------------------------------------

        clearance += clearance_step

    raise RuntimeError(
        "No B-spline-safe path was found within "
        "the specified clearance range."
    )


# ==========================================================
# MAIN TEST
# ==========================================================

if __name__ == "__main__":

    print()
    print("==================================================")
    print("B-SPLINE SAFE PATH SEARCH")
    print("==================================================")

    # ======================================================
    # TEST SCENARIO
    # ======================================================

    start = np.array([0.0, 0.0, 1.0])

    goal = np.array([4.0, 0.0, 1.0])

    # ======================================================
    # OBSTACLE
    # ======================================================
    #
    # MuJoCo XML:
    #   pos  = [2.0, 0.0, 1.0]
    #   size = [0.4, 0.4, 1.0]
    #
    # MuJoCo box size is half-size.
    # BoxObstacle expects full dimensions.
    #
    # Therefore:
    #   full dimensions = [0.8, 0.8, 2.0]
    #
    # ======================================================

    obstacle = BoxObstacle(
        center=[2.0, 0.0, 1.0],
        size=[0.8, 0.8, 2.0]
    )

    # ======================================================
    # SAFETY DISTANCE
    # ======================================================

    safety_distance = 0.25

    # ======================================================
    # CLEARANCE SEARCH SETTINGS
    # ======================================================

    initial_clearance = 0.05
    clearance_step = 0.05
    maximum_clearance = 1.00

    # ======================================================
    # B-SPLINE SETTINGS
    # ======================================================

    degree = 3
    number_of_samples = 401

    # ======================================================
    # DISPLAY INPUTS
    # ======================================================

    print()
    print("START")
    print("----------------------------------------------")
    print(start)

    print()
    print("GOAL")
    print("----------------------------------------------")
    print(goal)

    print()
    print("OBSTACLE")
    print("----------------------------------------------")
    print("Center =", obstacle.center)
    print("Full size =", obstacle.size)
    print("Minimum =", obstacle.minimum)
    print("Maximum =", obstacle.maximum)

    inflated_obstacle = inflate_obstacle(
        obstacle,
        safety_distance
    )

    print()
    print("INFLATED OBSTACLE")
    print("----------------------------------------------")
    print("Safety distance =", safety_distance, "m")
    print("Minimum =", inflated_obstacle.minimum)
    print("Maximum =", inflated_obstacle.maximum)

    # ======================================================
    # SEARCH
    # ======================================================

    print()
    print("B-SPLINE CLEARANCE SEARCH")
    print("----------------------------------------------")

    (
        safe_waypoints,
        safe_trajectory,
        selected_clearance,
        waypoint_length
    ) = find_bspline_safe_path(
        start,
        goal,
        [obstacle],
        safety_distance,
        initial_clearance,
        clearance_step,
        maximum_clearance,
        degree,
        number_of_samples
    )

    # ======================================================
    # FINAL RESULTS
    # ======================================================

    bspline_length = calculate_path_length(
        safe_trajectory
    )

    minimum_clearance = calculate_minimum_clearance(
        safe_trajectory,
        inflated_obstacle
    )

    print()
    print("==================================================")
    print("SAFE WAYPOINTS")
    print("==================================================")

    for i, waypoint in enumerate(safe_waypoints):

        print(
            f"Waypoint {i}: "
            f"[{waypoint[0]:.6f}, "
            f"{waypoint[1]:.6f}, "
            f"{waypoint[2]:.6f}]"
        )

    print()
    print("SELECTED CLEARANCE")
    print("----------------------------------------------")
    print(f"{selected_clearance:.6f} m")

    print()
    print("WAYPOINT PATH LENGTH")
    print("----------------------------------------------")
    print(f"{waypoint_length:.6f} m")

    print()
    print("B-SPLINE PATH LENGTH")
    print("----------------------------------------------")
    print(f"{bspline_length:.6f} m")

    print()
    print("TRAJECTORY")
    print("----------------------------------------------")
    print("Number of samples =", len(safe_trajectory))
    print("First point =", safe_trajectory[0])
    print("Final point =", safe_trajectory[-1])

    # ======================================================
    # FINAL COLLISION CHECK
    # ======================================================

    final_check = validate_bspline_trajectory(
        safe_trajectory,
        [obstacle],
        safety_distance
    )

    print()
    print("FINAL COLLISION CHECK")
    print("----------------------------------------------")
    print("B-spline collision free:", final_check)

    print()
    print("MINIMUM CLEARANCE")
    print("----------------------------------------------")
    print(f"{minimum_clearance:.6f} m")

    print()
    print("==================================================")

    if final_check:
        print("RESULT: SAFE B-SPLINE FOUND")
    else:
        print("RESULT: B-SPLINE IS NOT SAFE")

    print("==================================================")