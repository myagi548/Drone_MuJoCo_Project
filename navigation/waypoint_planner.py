import numpy as np

from navigation.obstacle_planner import (
    BoxObstacle,
    inflate_obstacle,
    path_is_collision_free
)


# ==========================================================
# PATH LENGTH
# ==========================================================

def calculate_path_length(path):
    """
    Calculate the total Euclidean length of a path.

    For every consecutive pair:

        d_i = ||P_(i+1) - P_i||

    Total:

        L = sum(d_i)
    """

    path = np.asarray(
        path,
        dtype=float
    )

    if len(path) < 2:
        return 0.0

    total_length = 0.0

    for i in range(len(path) - 1):

        segment = path[i + 1] - path[i]

        distance = np.linalg.norm(segment)

        total_length += distance

    return total_length


# ==========================================================
# DETOUR CANDIDATES
# ==========================================================

def generate_detour_paths(
    start,
    goal,
    obstacle,
    clearance
):
    """
    Generate candidate paths around the six faces
    of an axis-aligned box.

    The candidate points are placed OUTSIDE the obstacle
    by the specified clearance.

    Six possible detour directions are considered:

        +X
        -X
        +Y
        -Y
        +Z
        -Z

    For each direction, two waypoints are generated:

        Start -> waypoint_1 -> waypoint_2 -> Goal
    """

    minimum = obstacle.minimum
    maximum = obstacle.maximum

    candidates = []


    # ======================================================
    # X-MIN / X-MAX DETOURS
    # ======================================================

    # Around the +Z face
    candidates.append(
        np.array([
            start,
            [
                start[0],
                start[1],
                maximum[2] + clearance
            ],
            [
                goal[0],
                goal[1],
                maximum[2] + clearance
            ],
            goal
        ])
    )


    # Around the -Z face
    candidates.append(
        np.array([
            start,
            [
                start[0],
                start[1],
                minimum[2] - clearance
            ],
            [
                goal[0],
                goal[1],
                minimum[2] - clearance
            ],
            goal
        ])
    )


    # ======================================================
    # +Y DETOUR
    # ======================================================

    candidates.append(
        np.array([
            start,
            [
                start[0],
                maximum[1] + clearance,
                start[2]
            ],
            [
                goal[0],
                maximum[1] + clearance,
                goal[2]
            ],
            goal
        ])
    )


    # ======================================================
    # -Y DETOUR
    # ======================================================

    candidates.append(
        np.array([
            start,
            [
                start[0],
                minimum[1] - clearance,
                start[2]
            ],
            [
                goal[0],
                minimum[1] - clearance,
                goal[2]
            ],
            goal
        ])
    )


    # ======================================================
    # +X DETOUR
    # ======================================================

    candidates.append(
        np.array([
            start,
            [
                maximum[0] + clearance,
                start[1],
                start[2]
            ],
            [
                maximum[0] + clearance,
                goal[1],
                goal[2]
            ],
            goal
        ])
    )


    # ======================================================
    # -X DETOUR
    # ======================================================

    candidates.append(
        np.array([
            start,
            [
                minimum[0] - clearance,
                start[1],
                start[2]
            ],
            [
                minimum[0] - clearance,
                goal[1],
                goal[2]
            ],
            goal
        ])
    )


    return candidates


# ==========================================================
# PLANNER
# ==========================================================

def find_shortest_collision_free_path(
    start,
    goal,
    obstacles,
    safety_distance,
    planning_clearance
):
    """
    Find the shortest collision-free candidate path.

    safety_distance:
        Distance used to inflate physical obstacles.

    planning_clearance:
        Additional geometric clearance used to place
        generated waypoints outside the inflated obstacle.

    Both values are planner parameters.
    They are not physical constants.
    """

    start = np.asarray(
        start,
        dtype=float
    )

    goal = np.asarray(
        goal,
        dtype=float
    )


    # ======================================================
    # DIRECT PATH
    # ======================================================

    direct_path = np.array(
        [
            start,
            goal
        ]
    )


    if path_is_collision_free(
        direct_path,
        obstacles,
        safety_distance
    ):

        return (
            direct_path,
            calculate_path_length(
                direct_path
            )
        )


    # ======================================================
    # GENERATE CANDIDATES
    # ======================================================

    all_candidates = []


    for obstacle in obstacles:

        inflated_obstacle = inflate_obstacle(
            obstacle,
            safety_distance
        )


        candidates = generate_detour_paths(
            start,
            goal,
            inflated_obstacle,
            planning_clearance
        )


        all_candidates.extend(
            candidates
        )


    # ======================================================
    # CHECK CANDIDATES
    # ======================================================

    valid_paths = []


    for path in all_candidates:

        collision_free = path_is_collision_free(
            path,
            obstacles,
            safety_distance
        )


        if collision_free:

            path_length = calculate_path_length(
                path
            )


            valid_paths.append(
                (
                    path_length,
                    path
                )
            )


    # ======================================================
    # NO SOLUTION
    # ======================================================

    if len(valid_paths) == 0:

        raise RuntimeError(
            "No collision-free candidate path was found."
        )


    # ======================================================
    # SHORTEST VALID PATH
    # ======================================================

    valid_paths.sort(
        key=lambda item: item[0]
    )


    best_length = valid_paths[0][0]

    best_path = valid_paths[0][1]


    return (
        best_path,
        best_length
    )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print()
    print("==================================================")
    print("WAYPOINT PLANNER TEST")
    print("==================================================")


    # ======================================================
    # TEST SCENARIO
    # ======================================================
    #
    # These coordinates define the simulation scenario.
    #
    # They are NOT physical constants.
    #
    # Start:
    #       (0, 0, 1)
    #
    # Goal:
    #       (4, 0, 1)
    #
    # The obstacle lies directly between them.
    #
    # ======================================================

    start = np.array(
        [
            0.0,
            0.0,
            1.0
        ]
    )


    goal = np.array(
        [
            4.0,
            0.0,
            1.0
        ]
    )


    obstacle = BoxObstacle(
        center=[
            2.0,
            0.0,
            1.0
        ],
        size=[
            1.0,
            1.0,
            1.0
        ]
    )


    # ======================================================
    # PLANNER PARAMETERS
    # ======================================================

    safety_distance = 0.25

    planning_clearance = 0.05


    # ======================================================
    # DISPLAY SCENARIO
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

    print(
        "Center =",
        obstacle.center
    )

    print(
        "Size   =",
        obstacle.size
    )


    # ======================================================
    # INFLATED OBSTACLE
    # ======================================================

    inflated_obstacle = inflate_obstacle(
        obstacle,
        safety_distance
    )


    print()
    print("INFLATED OBSTACLE")
    print("----------------------------------------------")

    print(
        "Minimum =",
        inflated_obstacle.minimum
    )

    print(
        "Maximum =",
        inflated_obstacle.maximum
    )


    # ======================================================
    # DIRECT PATH
    # ======================================================

    direct_path = np.array(
        [
            start,
            goal
        ]
    )


    direct_is_safe = path_is_collision_free(
        direct_path,
        [obstacle],
        safety_distance
    )


    print()
    print("DIRECT PATH")
    print("----------------------------------------------")

    print(
        "Collision free:",
        direct_is_safe
    )


    # ======================================================
    # RUN PLANNER
    # ======================================================

    best_path, best_length = (
        find_shortest_collision_free_path(
            start,
            goal,
            [obstacle],
            safety_distance,
            planning_clearance
        )
    )


    # ======================================================
    # DISPLAY WAYPOINTS
    # ======================================================

    print()
    print("PLANNED WAYPOINTS")
    print("----------------------------------------------")


    for i, waypoint in enumerate(best_path):

        print(
            f"Waypoint {i}: "
            f"[{waypoint[0]:.3f}, "
            f"{waypoint[1]:.3f}, "
            f"{waypoint[2]:.3f}]"
        )


    # ======================================================
    # PATH INFORMATION
    # ======================================================

    print()
    print("NUMBER OF WAYPOINTS")
    print("----------------------------------------------")

    print(
        len(best_path)
    )


    print()
    print("TOTAL PATH LENGTH")
    print("----------------------------------------------")

    print(
        f"{best_length:.6f} m"
    )


    # ======================================================
    # FINAL SAFETY CHECK
    # ======================================================

    final_check = path_is_collision_free(
        best_path,
        [obstacle],
        safety_distance
    )


    print()
    print("FINAL COLLISION CHECK")
    print("----------------------------------------------")

    print(
        "Collision free:",
        final_check
    )


    print()
    print("==================================================")
    print("WAYPOINT PLANNER TEST COMPLETE")
    print("==================================================")