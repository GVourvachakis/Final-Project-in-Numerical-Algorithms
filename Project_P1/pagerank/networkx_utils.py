import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from .pagerank import compute_pagerank  # Assuming previous implementation is in pagerank_power.py

def read_and_create_graph(filename: str):
    """
    Read a graph from file and create a NetworkX DiGraph object
    """
    G = nx.DiGraph()
    file_path = os.path.join(os.path.dirname(__file__), '..', filename)
    
    with open(file_path, 'r') as f:
        for line in f:
            source, target = map(int, line.strip().split())
            G.add_edge(source, target)
            
    return G

def compare_pageranks(filename, damping_factor=0.85):
    """
    Compare PageRank results between NetworkX and Power method implementations
    """
    # Get NetworkX PageRank
    G = read_and_create_graph(filename)
    nx_pagerank = nx.pagerank(G, alpha=damping_factor)
    
    # Get Power method PageRank
    power_pagerank, eigenvalue, iterations = compute_pagerank(filename, d=damping_factor)
    
    # Convert power method results to dictionary format
    power_dict = {node: rank for node, rank in power_pagerank}
    
    # Calculate difference statistics
    differences = []
    print("\nPageRank Comparison:")
    print("Node\tNetworkX\t\tPower Method\tDifference")
    print("-" * 50)
    
    for node in sorted(nx_pagerank.keys()):
        diff = abs(nx_pagerank[node] - power_dict[node])
        differences.append(diff)
        print(f"{node}\t{nx_pagerank[node]:.10f}\t{power_dict[node]:.10f}\t{diff:.10f}")
    
    print("\nComparison Statistics:")
    print(f"Maximum difference: {max(differences):.10f}")
    print(f"Average difference: {np.mean(differences):.10f}")
    print(f"Power method eigenvalue: {eigenvalue:.10f}")
    print(f"Power method iterations: {iterations}")
    
    return G, nx_pagerank

def visualize_pagerank(G, pagerank, filename):
    """
    Create a visualization of the graph with node sizes proportional to PageRank
    """
    
    plt.figure(figsize=(12, 8))
    
    # Normalize node sizes for better visualization
    # Multiply by scaling factor to make differences more visible
    node_sizes = [pagerank[node] * 5000 for node in G.nodes()]
    
    # Create layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw the graph
    nx.draw(G, pos,
            node_color='red',
            node_size=node_sizes,
            with_labels=True,
            font_size=10,
            font_weight='bold',
            arrows=True,
            edge_color='gray',
            arrowsize=20)
    
    plt.title(f'PageRank Visualization for {filename}\nNode size proportional to PageRank value')
    
    # Save the visualization
    output_file = os.path.join(os.path.dirname(__file__), '..', f"{filename.split('.')[0]}_visualization.png")
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\nVisualization saved as {output_file}")

# def main():
#     filenames = ['graph0.txt', 'graph1.txt', 'graph2.txt']
    
#     for filename in filenames:
#         try:
#             print(f"\nProcessing {filename}:")
#             G, pagerank = compare_pageranks(filename)
#             #visualize_pagerank(G, pagerank, filename)
            
#         except FileNotFoundError:
#             print(f"Error: {filename} not found")
#             continue

# if __name__ == "__main__":
#     main()