#!/usr/bin/env python3

import sys
import csv
from typing import List, Tuple, Dict, Set

from route_planner import load_graph, dijkstra, reconstruct_path

PRIORITY_ORDER = ["HIGH", "MEDIUM", "LOW"]


def shortest_distance_and_path(graph, a: int, b: int) -> Tuple[float, List[int]]:
    dist, prev, _ = dijkstra(graph, a, b)
    if dist[b] == float('inf'):
        return float('inf'), []
    path = reconstruct_path(prev, a, b)
    return dist[b], path


def path_distance(graph, path: List[int]) -> float:
    if not path or len(path) == 1:
        return 0.0
    total = 0.0
    for i in range(1, len(path)):
        u = path[i-1]
        v = path[i]
        found = False
        for edge in graph.adj_list.get(u, []):
            if edge.to == v:
                total += edge.weight
                found = True
                break
        if not found:
            # No direct edge along reconstructed path (shouldn't happen)
            total += float('inf')
    return total


def nearest_neighbor(graph, start: int, targets: Set[int]) -> Tuple[List[int], float]:
    """Visit all nodes in targets starting from start using nearest-neighbor
    Returns full node visit sequence (including start) and total distance."""
    remaining = set(targets)
    current = start
    full_route = [start]
    total_distance = 0.0

    while remaining:
        best = None
        best_dist = float('inf')
        best_path = []
        for t in remaining:
            d, p = shortest_distance_and_path(graph, current, t)
            if d < best_dist:
                best_dist = d
                best = t
                best_path = p
        if best is None or best_dist == float('inf'):
            # Some target unreachable
            return full_route + list(remaining), float('inf')
        # Append path from current to best, but avoid repeating current
        if best_path:
            # best_path starts with current and ends with best
            full_route.extend(best_path[1:])
        total_distance += best_dist
        current = best
        remaining.remove(best)

    return full_route, total_distance


def build_prioritized_route(graph, start: int, dests: List[Tuple[int,str]]) -> Tuple[List[int], float]:
    """Build route by visiting priority groups in order"""
    priorities: Dict[str, Set[int]] = {p: set() for p in PRIORITY_ORDER}
    for node, prio in dests:
        if prio.upper() not in priorities:
            prio = 'LOW'
        priorities[prio.upper()].add(node)

    current = start
    full_route = [start]
    total_distance = 0.0

    for prio in PRIORITY_ORDER:
        targets = priorities[prio]
        if not targets:
            continue
        route_segment, dist_seg = nearest_neighbor(graph, current, targets)
        if dist_seg == float('inf'):
            return full_route + route_segment[1:], float('inf')
        # route_segment begins with current; append the rest
        full_route.extend(route_segment[1:])
        total_distance += dist_seg
        current = full_route[-1]

    return full_route, total_distance


def detect_priority_violations(order: List[int], dests_with_prio: List[Tuple[int,str]]) -> List[int]:
    """Return a list of node ids that violate priority ordering in the visit order."""
    pr_map = {node: pr.upper() for node, pr in dests_with_prio}
    # Build index map
    idx = {node: i for i, node in enumerate(order)}
    violations = []

    # For each pair of nodes (u,v) where priority(u) > priority(v) (higher index),
    # if v is visited before u, that's a violation for v.
    # Simpler: For each priority level, find the last index where any node of that level appears.
    last_index_by_pr = {}
    for pr in PRIORITY_ORDER:
        last = -1
        for node, p in pr_map.items():
            if p == pr and node in idx:
                last = max(last, idx[node])
        last_index_by_pr[pr] = last

    # If any node of lower priority appears at an index < last index of any higher priority, it's a violation
    for node, p in pr_map.items():
        pr_idx = PRIORITY_ORDER.index(p) if p in PRIORITY_ORDER else len(PRIORITY_ORDER)-1
        # check any higher priority
        for higher_pr in PRIORITY_ORDER[:pr_idx]:
            if last_index_by_pr[higher_pr] > -1 and idx.get(node, float('inf')) < last_index_by_pr[higher_pr]:
                violations.append(node)
                break

    return violations

# Reads the destinations, and priorities from a CSV file.
def read_destinations_csv(path: str) -> List[Tuple[int,str]]:
    out = []
    with open(path, 'r') as f:
        r = csv.DictReader(f)
        # support both destination,priority header or id,priority
        for row in r:
            # find dest column
            if 'destination' in row:
                dest = int(row['destination'])
            elif 'id' in row:
                dest = int(row['id'])
            else:
                # try first column
                keys = list(row.keys())
                dest = int(row[keys[0]])
            pr = row.get('priority') or row.get('Priority') or row.get('prio') or ''
            out.append((dest, pr))
    return out


def main():
    if len(sys.argv) < 5:
        print("Usage: python priority_router.py <nodes.csv> <edges.csv> <destinations.csv> <start_node> [threshold_percent]")
        sys.exit(1)

    nodes_file = sys.argv[1]
    edges_file = sys.argv[2]
    dests_file = sys.argv[3]
    start_node = int(sys.argv[4])
    threshold = float(sys.argv[5]) if len(sys.argv) > 5 else 0.20

    # Load graph and destinations
    graph = load_graph(nodes_file, edges_file)
    dests = read_destinations_csv(dests_file)

    if not dests:
        print("No destinations provided")
        sys.exit(1)

    # Build prioritized route
    prio_route, prio_dist = build_prioritized_route(graph, start_node, dests)

    # Build unconstrained greedy route for comparison (visit all destinations ignoring priority)
    targets = {d for d, _ in dests}
    nn_route, nn_dist = nearest_neighbor(graph, start_node, targets)

    print("=== Prioritized Routing ===")
    print(f"Priority route distance: {prio_dist:.2f}")
    print(f"Unconstrained route distance: {nn_dist:.2f}\n")

    allowed = (1.0 + threshold) * nn_dist if nn_dist != float('inf') else float('inf')
    if prio_dist <= allowed:
        print("Using priority route")
        route = prio_route
        dist = prio_dist
        violations = detect_priority_violations(route, dests)
    else:
        print("Priority route exceeds allowed threshold.")
        route = nn_route
        dist = nn_dist
        violations = detect_priority_violations(route, dests)

    print("\nRoute (node sequence):")
    print(' -> '.join(str(n) for n in route))
    print(f"Total route distance: {dist:.2f}")

    if violations:
        print(f"Priority violations ({len(violations)}): {violations}")
    else:
        print("Priority violations: 0")


if __name__ == '__main__':
    main()
