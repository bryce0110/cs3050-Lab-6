#!/usr/bin/env python3
"""
Performance Analysis for dijkstra_time Algorithm
Generates random graphs and measures runtime vs. graph size
"""

import sys
import time
import random
import csv
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Add python directory to path
sys.path.insert(0, '/mnt/Important Shit/Documents/Projects/CS3050/cs3050-Lab-6/python')
from route_planner import Graph, dijkstra_time, dijkstra

def generate_random_graph(num_nodes: int, edge_density: float = 3.0) -> Tuple[Graph, List[Tuple[int, int]]]:
    """
    Generate a random connected graph with time windows
    
    Args:
        num_nodes: Number of nodes in graph
        edge_density: Average number of edges per node
    
    Returns:
        (graph, list of (start, end) test pairs)
    """
    graph = Graph()
    
    # Add nodes with random time windows
    for i in range(1, num_nodes + 1):
        lat = random.uniform(38.0, 39.0)
        lon = random.uniform(-78.0, -77.0)
        
        # Random time windows
        if random.random() < 0.7:  # 70% of nodes have time windows
            earliest = random.uniform(0, 50)
            latest = earliest + random.uniform(10, 100)
        else:
            earliest = 0
            latest = 1000  # Effectively no constraint
        
        graph.add_node(i, lat, lon, earliest, latest)
    
    # Generate edges - ensure connectivity first
    edges_created = 0
    target_edges = int(num_nodes * edge_density)
    
    # Create a spanning tree for connectivity
    for i in range(2, num_nodes + 1):
        parent = random.randint(1, i - 1)
        weight = random.uniform(1, 20)
        graph.add_edge(parent, i, weight)
        graph.add_edge(i, parent, weight)  # Undirected
        edges_created += 2
    
    # Add random edges
    while edges_created < target_edges:
        u = random.randint(1, num_nodes)
        v = random.randint(1, num_nodes)
        if u != v:
            weight = random.uniform(1, 20)
            graph.add_edge(u, v, weight)
            edges_created += 1
    
    # Generate test pairs
    test_pairs = []
    for _ in range(min(10, num_nodes - 1)):
        start = random.randint(1, num_nodes)
        end = random.randint(1, num_nodes)
        while end == start:
            end = random.randint(1, num_nodes)
        test_pairs.append((start, end))
    
    return graph, test_pairs

def measure_runtime(graph: Graph, test_pairs: List[Tuple[int, int]], num_trials: int = 5) -> float:
    """
    Measure average runtime for dijkstra_time on test pairs
    
    Args:
        graph: The graph to test
        test_pairs: List of (start, end) node pairs
        num_trials: Number of trials to average
    
    Returns:
        Average runtime in milliseconds
    """
    total_time = 0.0
    successful_runs = 0
    
    for _ in range(num_trials):
        for start, end in test_pairs:
            start_time = time.perf_counter()
            try:
                dist, prev, nodes_explored, success = dijkstra_time(graph, start, end)
                end_time = time.perf_counter()
                total_time += (end_time - start_time) * 1000  # Convert to ms
                successful_runs += 1
            except Exception as e:
                print(f"Error on ({start}, {end}): {e}")
                continue
    
    if successful_runs == 0:
        return 0.0
    
    return total_time / successful_runs

def run_experiments():
    """Run performance experiments and generate plots"""
    
    print("=" * 70)
    print("Performance Analysis: dijkstra_time Algorithm")
    print("=" * 70)
    print()
    
    # Test configurations
    graph_sizes = [10, 25, 50, 75, 100, 200, 500]
    edge_densities = [3.0, 5.0]  # Test both sparse and denser graphs
    
    results = {
        'sparse': {'sizes': [], 'runtimes': [], 'edges': []},
        'dense': {'sizes': [], 'runtimes': [], 'edges': []}
    }
    
    for density_name, density in [('sparse', 3.0), ('dense', 5.0)]:
        print(f"\n{'='*70}")
        print(f"Testing {density_name.upper()} graphs (edge_density={density})")
        print(f"{'='*70}\n")
        
        for size in graph_sizes:
            print(f"Graph size |V| = {size}...", end=" ", flush=True)
            
            # Generate graph
            graph, test_pairs = generate_random_graph(size, density)
            
            # Count actual edges
            num_edges = sum(len(edges) for edges in graph.adj_list.values())
            
            # Measure runtime
            avg_runtime = measure_runtime(graph, test_pairs, num_trials=3)
            
            # Store results
            results[density_name]['sizes'].append(size)
            results[density_name]['runtimes'].append(avg_runtime)
            results[density_name]['edges'].append(num_edges)
            
            print(f"|E| = {num_edges}, avg runtime = {avg_runtime:.4f} ms")
    
    # Create plots
    print(f"\n{'='*70}")
    print("Generating plots...")
    print(f"{'='*70}\n")
    
    create_plots(results)
    
    # Print summary
    print_summary(results)
    
    return results

def create_plots(results):
    """Create visualization plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('dijkstra_time Performance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Runtime vs. Number of Nodes
    ax1 = axes[0, 0]
    ax1.plot(results['sparse']['sizes'], results['sparse']['runtimes'], 
             'o-', label='Sparse (E≈3V)', linewidth=2, markersize=8)
    ax1.plot(results['dense']['sizes'], results['dense']['runtimes'], 
             's-', label='Dense (E≈5V)', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Nodes (|V|)', fontsize=12)
    ax1.set_ylabel('Average Runtime (ms)', fontsize=12)
    ax1.set_title('Runtime vs. Graph Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Runtime vs. Number of Edges
    ax2 = axes[0, 1]
    ax2.plot(results['sparse']['edges'], results['sparse']['runtimes'], 
             'o-', label='Sparse', linewidth=2, markersize=8)
    ax2.plot(results['dense']['edges'], results['dense']['runtimes'], 
             's-', label='Dense', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Edges (|E|)', fontsize=12)
    ax2.set_ylabel('Average Runtime (ms)', fontsize=12)
    ax2.set_title('Runtime vs. Number of Edges')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Log-Log plot (check for power-law relationship)
    ax3 = axes[1, 0]
    ax3.loglog(results['sparse']['sizes'], results['sparse']['runtimes'], 
               'o-', label='Sparse', linewidth=2, markersize=8)
    ax3.loglog(results['dense']['sizes'], results['dense']['runtimes'], 
               's-', label='Dense', linewidth=2, markersize=8)
    
    # Add theoretical O(E log V) line for reference (sparse)
    V = np.array(results['sparse']['sizes'])
    E = np.array(results['sparse']['edges'])
    # Normalize to match empirical data at V=100
    idx_100 = list(V).index(100) if 100 in V else len(V)//2
    theoretical = (E * np.log(V)) / (E[idx_100] * np.log(V[idx_100])) * results['sparse']['runtimes'][idx_100]
    ax3.loglog(V, theoretical, '--', label='O(E log V)', linewidth=2, alpha=0.7)
    
    ax3.set_xlabel('Number of Nodes (|V|) [log scale]', fontsize=12)
    ax3.set_ylabel('Runtime (ms) [log scale]', fontsize=12)
    ax3.set_title('Log-Log Plot: Runtime vs. Graph Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')
    
    # Plot 4: Normalized runtime (runtime / (E log V))
    ax4 = axes[1, 1]
    sparse_normalized = np.array(results['sparse']['runtimes']) / (np.array(results['sparse']['edges']) * np.log(np.array(results['sparse']['sizes']) + 1))
    dense_normalized = np.array(results['dense']['runtimes']) / (np.array(results['dense']['edges']) * np.log(np.array(results['dense']['sizes']) + 1))
    
    ax4.plot(results['sparse']['sizes'], sparse_normalized, 
             'o-', label='Sparse', linewidth=2, markersize=8)
    ax4.plot(results['dense']['sizes'], dense_normalized, 
             's-', label='Dense', linewidth=2, markersize=8)
    ax4.set_xlabel('Number of Nodes (|V|)', fontsize=12)
    ax4.set_ylabel('Runtime / (E log V)', fontsize=12)
    ax4.set_title('Normalized Runtime')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = '/mnt/Important Shit/Documents/Projects/CS3050/cs3050-Lab-6/lab6_submission/performance_plots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to: {output_path}")
    
    # Also save as PDF for report
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ PDF saved to: {pdf_path}")

def print_summary(results):
    """Print summary statistics"""
    
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}\n")
    
    for density_name in ['sparse', 'dense']:
        print(f"{density_name.upper()} GRAPHS:")
        print(f"{'-'*70}")
        print(f"{'|V|':>6} {'|E|':>8} {'Runtime (ms)':>15} {'E log V':>12} {'Runtime/(E log V)':>18}")
        print(f"{'-'*70}")
        
        data = results[density_name]
        for i in range(len(data['sizes'])):
            V = data['sizes'][i]
            E = data['edges'][i]
            runtime = data['runtimes'][i]
            e_log_v = E * np.log(V) if V > 0 else 1
            normalized = runtime / e_log_v if e_log_v > 0 else 0
            
            print(f"{V:>6} {E:>8} {runtime:>15.4f} {e_log_v:>12.2f} {normalized:>18.6f}")
        print()

if __name__ == "__main__":
    print("\nChecking for required packages...")
    try:
        import matplotlib
        import numpy
        print("✓ All required packages available\n")
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("\nInstall with: pip install matplotlib numpy")
        sys.exit(1)
    
    results = run_experiments()
    
    print(f"\n{'='*70}")
    print("Performance analysis complete!")
    print(f"{'='*70}\n")
