
import heapq
import itertools
import numpy as np

from navigation.obstacle_planner import (
    BoxObstacle,
    path_is_collision_free,
)

from navigation.bspline import generate_bspline


# ==========================================================
# SCENARIO
# ==========================================================

START = np.array([0.0, 0.0, 1.0])

TARGETS = [
    np.array([3.0, 0.0, 1.0]),
    np.array([5.0, 0.0, 1.0]),
    np.array([7.0, 0.0, 1.0]),
    np.array([10.0, 0.0, 1.0]),
]

OBSTACLES = [
    BoxObstacle([2.0, 0.0, 1.0], [0.8, 0.8, 2.0]),
    BoxObstacle([4.0, 1.5, 1.0], [0.8, 0.8, 2.0]),
    BoxObstacle([6.0, -1.5, 1.0], [0.8, 0.8, 2.0]),
    BoxObstacle([8.0, 0.0, 1.0], [0.8, 0.8, 2.0]),
]

SAFETY_DISTANCE = 0.25
GRID_STEP = 0.25
SPLINE_SAMPLES_PER_SEGMENT = 101
SPLINE_DEGREE = 3


# ==========================================================
# GRID SEARCH
# ==========================================================

def point_is_safe(point, obstacles, safety_distance):
    return path_is_collision_free(
        np.array([point, point]),
        obstacles,
        safety_distance,
    )


def grid_route(start, goal, obstacles, safety_distance):
    """
    A* search on a 3D grid.

    Returns a collision-checked sequence of grid points.
    """

    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    step = GRID_STEP

    def to_index(point):
        return tuple(np.rint(point / step).astype(int))

    def to_point(index):
        return np.asarray(index, dtype=float) * step

    start_idx = to_index(start)
    goal_idx = to_index(goal)

    lower = np.array([-1.0, -4.0, 0.25])
    upper = np.array([11.0, 4.0, 4.0])

    lower_idx = to_index(lower)
    upper_idx = to_index(upper)

    def inside_bounds(index):
        return all(
            lower_idx[i] <= index[i] <= upper_idx[i]
            for i in range(3)
        )

    def heuristic(index):
        return np.linalg.norm(
            to_point(index) - to_point(goal_idx)
        )

    # 26-connected neighborhood
    moves = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]

    counter = itertools.count()

    frontier = []
    heapq.heappush(
        frontier,
        (heuristic(start_idx), next(counter), start_idx),
    )

    came_from = {start_idx: None}
    cost_so_far = {start_idx: 0.0}

    while frontier:
        _, _, current = heapq.heappop(frontier)

        if current == goal_idx:
            break

        current_point = to_point(current)

        for move in moves:
            neighbor = tuple(
                current[i] + move[i]
                for i in range(3)
            )

            if not inside_bounds(neighbor):
                continue

            neighbor_point = to_point(neighbor)

            edge = np.array([
                current_point,
                neighbor_point,
            ])

            if not path_is_collision_free(
                edge,
                obstacles,
                safety_distance,
            ):
                continue

            move_cost = np.linalg.norm(
                neighbor_point - current_point
            )

            new_cost = cost_so_far[current] + move_cost

            if (
                neighbor not in cost_so_far
                or new_cost < cost_so_far[neighbor]
            ):
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current

                priority = new_cost + heuristic(neighbor)

                heapq.heappush(
                    frontier,
                    (priority, next(counter), neighbor),
                )

    if goal_idx not in came_from:
        raise RuntimeError(
            f"A* could not find a route from {start} to {goal}."
        )

    # Reconstruct the route
    indices = []
    current = goal_idx

    while current is not None:
        indices.append(current)
        current = came_from[current]

    indices.reverse()

    route = np.array([
        to_point(index)
        for index in indices
    ])

    # Preserve exact requested endpoints
    route[0] = start
    route[-1] = goal

    if not path_is_collision_free(
        route,
        obstacles,
        safety_distance,
    ):
        raise RuntimeError(
            "Reconstructed route failed collision validation."
        )

    return route


# ==========================================================
# REMOVE REDUNDANT WAYPOINTS
# ==========================================================

def simplify_route(route, obstacles, safety_distance):
    """
    Skip intermediate points when the direct shortcut
    remains collision-free.
    """

    if len(route) <= 2:
        return route

    simplified = [route[0]]
    current = 0

    while current < len(route) - 1:
        next_index = len(route) - 1

        while next_index > current + 1:
            shortcut = np.array([
                route[current],
                route[next_index],
            ])

            if path_is_collision_free(
                shortcut,
                obstacles,
                safety_distance,
            ):
                break

            next_index -= 1

        simplified.append(route[next_index])
        current = next_index

    return np.asarray(simplified)


# ==========================================================
# B-SPLINE SEGMENT
# ==========================================================

def smooth_route(route, obstacles, safety_distance):
    """
    Generate a cubic B-spline for each consecutive
    waypoint pair.

    If a sampled spline segment is unsafe, use the
    already-validated straight segment instead.
    """

    complete_trajectory = []

    for i in range(len(route) - 1):
        start = route[i]
        end = route[i + 1]

        # Four collinear control points for a cubic segment
        control_points = np.array([
            start,
            start + (end - start) / 3.0,
            start + 2.0 * (end - start) / 3.0,
            end,
        ])

        _, segment, _ = generate_bspline(
            control_points,
            number_of_samples=SPLINE_SAMPLES_PER_SEGMENT,
            degree=SPLINE_DEGREE,
        )

        if not path_is_collision_free(
            segment,
            obstacles,
            safety_distance,
        ):
            print(
                f"Segment {i} spline unsafe; "
                "using validated straight segment."
            )

            segment = np.linspace(
                start,
                end,
                SPLINE_SAMPLES_PER_SEGMENT,
            )

        # Avoid duplicate points between segments
        if i > 0:
            segment = segment[1:]

        complete_trajectory.extend(segment)

    return np.asarray(complete_trajectory)


# ==========================================================
# BUILD COMPLETE MULTI-TARGET TRAJECTORY
# ==========================================================

def build_full_trajectory():
    """
    Build the complete multi-target route and trajectory.

    Returns:
        all_waypoints: simplified route points
        all_trajectory: sampled B-spline path

    This function can be imported by another simulation
    module without running the planner's printout.
    """

    current = START.copy()

    all_waypoints = [current.copy()]
    all_trajectory = []

    for target_index, target in enumerate(TARGETS, start=1):
        print(
            f"Planning route to target {target_index}: {target}"
        )

        route = grid_route(
            current,
            target,
            OBSTACLES,
            SAFETY_DISTANCE,
        )

        route = simplify_route(
            route,
            OBSTACLES,
            SAFETY_DISTANCE,
        )

        trajectory = smooth_route(
            route,
            OBSTACLES,
            SAFETY_DISTANCE,
        )

        if not path_is_collision_free(
            trajectory,
            OBSTACLES,
            SAFETY_DISTANCE,
        ):
            raise RuntimeError(
                f"Final trajectory to target {target_index} "
                "failed collision validation."
            )

        # Avoid duplicate connecting points
        if all_trajectory:
            trajectory = trajectory[1:]

        all_trajectory.extend(trajectory)
        all_waypoints.extend(route[1:])

        current = target.copy()

        print("Route waypoints:", len(route))
        print("Trajectory samples:", len(trajectory))
        print("Collision check: PASS")

    all_waypoints = np.asarray(all_waypoints)
    all_trajectory = np.asarray(all_trajectory)

    if len(all_trajectory) == 0:
        raise RuntimeError("Generated trajectory is empty.")

    if not path_is_collision_free(
        all_trajectory,
        OBSTACLES,
        SAFETY_DISTANCE,
    ):
        raise RuntimeError(
            "Combined trajectory failed collision validation."
        )

    return all_waypoints, all_trajectory


# ==========================================================
# MAIN
# ==========================================================

def main():
    print()
    print("=" * 60)
    print("MULTI-OBSTACLE B-SPLINE PLANNER")
    print("=" * 60)

    all_waypoints, all_trajectory = build_full_trajectory()

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print("Targets visited:", len(TARGETS))
    print("Route waypoints:", len(all_waypoints))
    print("Trajectory samples:", len(all_trajectory))

    print("Start:", all_trajectory[0])
    print("Final point:", all_trajectory[-1])

    print(
        "Final collision check:",
        path_is_collision_free(
            all_trajectory,
            OBSTACLES,
            SAFETY_DISTANCE,
        ),
    )

    print("RESULT: TRAJECTORY VALIDATION PASSED")


if __name__ == "__main__":
    main()