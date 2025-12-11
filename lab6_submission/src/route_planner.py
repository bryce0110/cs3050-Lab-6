#!/usr/bin/env python3
"""
Route Planner with Dijkstra, A*, and Bellman-Ford algorithms
"""

import sys
import csv
import heapq
import math
from typing import Dict, List, Tuple, Optional

EARTH_RADIUS = 6371.0  # km


class Node:
    """Represents a node in the graph"""
    def __init__(self, node_id: int, lat: float, lon: float, earliest: Optional[float] = None, latest: Optional[float] = None):
        self.id = node_id
        self.lat = lat
        self.lon = lon
        self.earliest = earliest
        self.latest = latest

class Edge:
    """Represents an edge in the graph"""
    def __init__(self, to: int, weight: float):
        self.to = to
        self.weight = weight


class Graph:
    """Graph data structure with adjacency list"""
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.adj_list: Dict[int, List[Edge]] = {}
    
    def add_node(self, node_id: int, lat: float, lon: float, earliest: Optional[float] = None, latest: Optional[float] = None):
        """Add a node to the graph"""
        self.nodes[node_id] = Node(node_id, lat, lon, earliest, latest)
        if node_id not in self.adj_list:
            self.adj_list[node_id] = []
    
    def add_edge(self, from_id: int, to_id: int, weight: float):
        """Add an edge to the graph"""
        if from_id not in self.adj_list:
            self.adj_list[from_id] = []
        self.adj_list[from_id].append(Edge(to_id, weight))


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points"""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return EARTH_RADIUS * c


def dijkstra(graph: Graph, start: int, end: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]], int]:
    """
    Dijkstra's algorithm for shortest path
    Returns: (distances, previous nodes, nodes explored)
    """
    dist = {node_id: float('inf') for node_id in graph.nodes}
    prev = {node_id: None for node_id in graph.nodes}
    dist[start] = 0
    
    pq = [(0, start)]
    nodes_explored = 0
    visited = set()
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        if u in visited:
            continue
        
        visited.add(u)
        nodes_explored += 1
        
        if u == end:
            break
        
        if current_dist > dist[u]:
            continue
        
        for edge in graph.adj_list.get(u, []):
            v = edge.to
            alt = dist[u] + edge.weight
            
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))
    
    return dist, prev, nodes_explored

def dijkstra_time(graph: Graph, start: int, end: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]], int, bool]:
    """
    Dijkstra's algorithm for shortest path with time constraints
    Returns: (distances, previous nodes, nodes explored, success)
    """
    dist = {node_id: float('inf') for node_id in graph.nodes}
    prev = {node_id: None for node_id in graph.nodes}
    dist[start] = 0
    
    pq = [(0, start)]
    nodes_explored = 0
    visited = set()
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        if u in visited:
            continue
        
        visited.add(u)
        nodes_explored += 1
        
        # Time constraints
        earliest = graph.nodes[u].earliest
        latest = graph.nodes[u].latest

        if earliest is not None and current_dist < earliest:
            current_dist = earliest  # Wait until earliest time
        if latest is not None and current_dist > latest:
            continue  # Cannot arrive within time window

        if u == end:
            return dist, prev, nodes_explored, True
        
        if current_dist > dist[u]:
            continue
        
        for edge in graph.adj_list.get(u, []):
            v = edge.to
            arrival = current_dist + edge.weight
            
            if arrival < dist[v]:
                dist[v] = arrival
                prev[v] = u
                heapq.heappush(pq, (arrival, v))
    
    return dist, prev, nodes_explored, False

def astar(graph: Graph, start: int, end: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]], int]:
    """
    A* algorithm for shortest path
    Returns: (distances, previous nodes, nodes explored)
    """
    dist = {node_id: float('inf') for node_id in graph.nodes}
    prev = {node_id: None for node_id in graph.nodes}
    dist[start] = 0
    
    end_node = graph.nodes[end]
    
    def heuristic(node_id: int) -> float:
        node = graph.nodes[node_id]
        return haversine(node.lat, node.lon, end_node.lat, end_node.lon)
    
    pq = [(heuristic(start), start)]
    nodes_explored = 0
    visited = set()
    
    while pq:
        _, u = heapq.heappop(pq)
        
        if u in visited:
            continue
        
        visited.add(u)
        nodes_explored += 1
        
        if u == end:
            break
        
        for edge in graph.adj_list.get(u, []):
            v = edge.to
            alt = dist[u] + edge.weight
            
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                f_score = alt + heuristic(v)
                heapq.heappush(pq, (f_score, v))
    
    return dist, prev, nodes_explored


def bellman_ford(graph: Graph, start: int, end: int) -> Tuple[Optional[Dict[int, float]], Optional[Dict[int, Optional[int]]], int]:
    """
    Bellman-Ford algorithm for shortest path
    Can handle negative weights and detect negative cycles
    Returns: (distances, previous nodes, nodes explored) or (None, None, 0) if negative cycle detected
    """
    dist = {node_id: float('inf') for node_id in graph.nodes}
    prev = {node_id: None for node_id in graph.nodes}
    dist[start] = 0
    
    nodes_explored = 0
    node_count = len(graph.nodes)
    
    # Relax edges |V| - 1 times
    for i in range(node_count - 1):
        updated = False
        for u in graph.nodes:
            if dist[u] == float('inf'):
                continue
            
            for edge in graph.adj_list.get(u, []):
                v = edge.to
                if dist[u] + edge.weight < dist[v]:
                    dist[v] = dist[u] + edge.weight
                    prev[v] = u
                    updated = True
        
        nodes_explored += 1
        if not updated:
            break
    
    # Check for negative cycles
    for u in graph.nodes:
        if dist[u] == float('inf'):
            continue
        
        for edge in graph.adj_list.get(u, []):
            v = edge.to
            if dist[u] + edge.weight < dist[v]:
                return None, None, 0  # Negative cycle detected
    
    return dist, prev, nodes_explored


def reconstruct_path(prev: Dict[int, Optional[int]], start: int, end: int) -> Optional[List[int]]:
    """Reconstruct path from start to end using previous nodes"""
    if prev[end] is None and start != end:
        return None
    
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = prev[current]
    
    path.reverse()
    return path


def compute_constraint_violations(graph: Graph, prev: Dict[int, Optional[int]], start: int, end: int) -> Tuple[int, List[Tuple[int, float, Optional[float], Optional[float], str]]]:
    """
    Compute time constraint violations along the path
    Returns a tuple of (violation_count, violations_list (List)).

    Each entry in violations_list is a tuple: (node_id, arrival_time, earliest, latest, violation_type)

    violation_type is either 'early' or 'late'.
    """
    path = reconstruct_path(prev, start, end)
    if path is None:
        return 0, []

    violations: List[Tuple[int, float, Optional[float], Optional[float], str]] = []
    arrival = 0.0

    for i, node_id in enumerate(path):
        if i == 0:
            arrival = 0.0
        else:
            from_id = path[i - 1]
            to_id = node_id
            # find the edge weight from from_id -> to_id
            weight = None
            for edge in graph.adj_list.get(from_id, []):
                if edge.to == to_id:
                    weight = edge.weight
                    break

            if weight is None:
                # If there's no explicit edge between consecutive nodes in the
                # reconstructed path, stop processing further arrivals.
                break

            arrival += weight

        node = graph.nodes.get(node_id)
        if node is None:
            continue

        earliest = node.earliest
        latest = node.latest

        if earliest is not None and arrival < earliest:
            violations.append((node_id, arrival, earliest, latest, 'early'))
        elif latest is not None and arrival > latest:
            violations.append((node_id, arrival, earliest, latest, 'late'))

    return len(violations), violations


def print_path(graph: Graph, prev: Dict[int, Optional[int]], start: int, end: int, distance: float):
    """Print the path from start to end"""
    path = reconstruct_path(prev, start, end)
    
    if path is None:
        print("No path found")
        return
    
    path_str = " -> ".join(str(node) for node in path)
    print(f"Path from {start} to {end}: {path_str}")
    print(f"Total distance: {distance:.2f} km")


def load_graph(nodes_file: str, edges_file: str) -> Graph:
    """Load graph from CSV files"""
    graph = Graph()
    
    # Load nodes
    with open(nodes_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = int(row['id'])
            lat = float(row['lat'])
            lon = float(row['lon'])
            earliest = None
            latest = None
            if 'earliest' in row and row['earliest'] not in '':
                try:
                    earliest = float(row['earliest'])
                except ValueError:
                    earliest = None
            if 'latest' in row and row['latest'] not in '':
                try:
                    latest = float(row['latest'])
                except ValueError:
                    latest = None

            graph.add_node(node_id, lat, lon, earliest, latest)
    
    # Load edges
    with open(edges_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            from_id = int(row['from'])
            to_id = int(row['to'])
            distance = float(row['distance'])
            graph.add_edge(from_id, to_id, distance)
    
    return graph


def main():
    if len(sys.argv) != 6:
        print(f"Usage: {sys.argv[0]} <nodes.csv> <edges.csv> <start_node> <end_node> <algorithm>")
        print("Algorithms: dijkstra, astar, bellman-ford")
        sys.exit(1)
    
    nodes_file = sys.argv[1]
    edges_file = sys.argv[2]
    start_node = int(sys.argv[3])
    end_node = int(sys.argv[4])
    algorithm = sys.argv[5]
    
    # Load graph
    graph = load_graph(nodes_file, edges_file)
    
    # Validate nodes
    if start_node not in graph.nodes or end_node not in graph.nodes:
        print("Invalid start or end node")
        sys.exit(1)
    
    # Run selected algorithm
    if algorithm == "dijkstra":
        print("=== Dijkstra's Algorithm ===")
        # First try a time-aware Dijkstra (find a feasible path respecting windows)
        dist, prev, nodes_explored, success = dijkstra_time(graph, start_node, end_node)

        # If no feasible path found
        if not success:
            dist, prev, nodes_explored = dijkstra(graph, start_node, end_node)

            # Report violations on the unconstrained shortest path
            uncon_viol_count, uncon_violations = compute_constraint_violations(graph, prev, start_node, end_node)
            if uncon_viol_count != 0:
                print(f"Time violations: {uncon_viol_count}")
                for node_id, arrival, earliest, latest, vtype in uncon_violations:
                    e_str = f"earliest={earliest}" if earliest is not None else "earliest=None"
                    l_str = f"latest={latest}" if latest is not None else "latest=None"
                    print(f" - Node {node_id}: arrival={arrival:.2f}, {e_str}, {l_str}, violation={vtype}")

                print("No feasible path found within time constraints.")
    elif algorithm == "astar":
        print("=== A* Algorithm ===")
        dist, prev, nodes_explored = astar(graph, start_node, end_node)
    elif algorithm == "bellman-ford":
        print("=== Bellman-Ford Algorithm ===")
        dist, prev, nodes_explored = bellman_ford(graph, start_node, end_node)
        if dist is None:
            print("Negative cycle detected!")
            sys.exit(1)
    else:
        print(f"Unknown algorithm: {algorithm}")
        print("Available algorithms: dijkstra, astar, bellman-ford")
        sys.exit(1)
    
    # Print results
    print_path(graph, prev, start_node, end_node, dist[end_node])
    print(f"Nodes explored: {nodes_explored}")

if __name__ == "__main__":
    main()