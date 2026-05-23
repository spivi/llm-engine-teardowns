"""Reference TSP-with-time-windows solver via OR-Tools RoutingModel.

Single vehicle, depot=0, returns to depot. Travel time = euclidean(coords[i], coords[j]) / speed,
rounded to integer. Service time added when departing each non-depot node.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Sequence, Tuple

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def _build_time_matrix(coords: Sequence[Sequence[float]], speed: float) -> List[List[int]]:
    n = len(coords)
    m = [[0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = coords[j]
            d = math.hypot(xi - xj, yi - yj)
            m[i][j] = int(round(d / speed))
    return m


def solve(
    coords: Sequence[Sequence[float]],
    service_time: int,
    time_windows: Sequence[Tuple[int, int]],
    speed: float = 50.0,
    time_limit_s: int = 30,
) -> Dict[str, Any]:
    """Solve a single-vehicle TSP-TW."""
    n = len(coords)
    if len(time_windows) != n:
        return {
            "status": "error",
            "tour": None,
            "arrivals": None,
            "total_time": None,
            "solve_ms": 0.0,
            "message": f"time_windows length {len(time_windows)} != n_cities {n}",
        }

    time_matrix = _build_time_matrix(coords, speed)
    horizon = max(int(w[1]) for w in time_windows) + service_time * n + 1

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # Service time charged when leaving a non-depot node.
        srv = service_time if from_node != 0 else 0
        return time_matrix[from_node][to_node] + srv

    transit_idx = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    routing.AddDimension(
        transit_idx,
        horizon,        # slack: allow waiting before a window opens
        horizon,        # capacity: max time on a route
        False,          # don't force start cumul to zero (depot window handles it)
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Apply time windows.
    for node in range(n):
        lo, hi = int(time_windows[node][0]), int(time_windows[node][1])
        if node == 0:
            depot_start = routing.Start(0)
            depot_end = routing.End(0)
            time_dim.CumulVar(depot_start).SetRange(lo, hi)
            time_dim.CumulVar(depot_end).SetRange(lo, hi)
        else:
            idx = manager.NodeToIndex(node)
            time_dim.CumulVar(idx).SetRange(lo, hi)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = int(time_limit_s)
    search_params.solution_limit = 100
    # Determinism: fixed random seed.
    try:
        search_params.random_seed = 42
    except AttributeError:
        pass  # Older OR-Tools versions; field not present.

    t0 = time.perf_counter()
    solution = routing.SolveWithParameters(search_params)
    solve_ms = (time.perf_counter() - t0) * 1000.0

    if solution is None:
        return {
            "status": "infeasible",
            "tour": None,
            "arrivals": None,
            "total_time": None,
            "solve_ms": solve_ms,
            "message": "OR-Tools returned no solution; instance may be infeasible.",
        }

    tour: List[int] = []
    arrivals: List[int] = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        tour.append(node)
        arrivals.append(int(solution.Value(time_dim.CumulVar(index))))
        index = solution.Value(routing.NextVar(index))
    # Append depot return.
    end_node = manager.IndexToNode(index)
    tour.append(end_node)
    arrivals.append(int(solution.Value(time_dim.CumulVar(index))))

    total_time = arrivals[-1] - arrivals[0]

    return {
        "status": "optimal",
        "tour": tour,
        "arrivals": arrivals,
        "total_time": total_time,
        "solve_ms": solve_ms,
        "message": "",
    }
