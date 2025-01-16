import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_pagerank_comparison(filename: str, norms: List[int] = [1, 2, np.inf], d: float = 0.85):
    from .graph_utils import read_graph, construct_adjacency_matrix, construct_pagerank_matrix
    from .pagerank import power_method

    n_nodes, edges = read_graph(filename)
    A = construct_adjacency_matrix(n_nodes, edges)
    M = construct_pagerank_matrix(A, d)

    plt.figure(figsize=(10, 6))

    # Plot for each norm with markers for each node
    for norm in norms:
        ranks, _, _ = power_method(M, norm=norm)
        plt.plot(range(1, n_nodes + 1), ranks, label=f'Norm-{norm}', marker='o')
    
    # Set x-axis to show only integer values
    #plt.xticks(np.arange(1, n_nodes + 1, 1))  # Ticks for every node (integer values)
    
    # Adding title, labels, and legend
    plt.title(f"PageRank Comparison for Different Norms (File: {filename})")
    plt.xlabel("Node")
    plt.ylabel("Rank")
    plt.legend()
    plt.grid(True)
    plt.show()