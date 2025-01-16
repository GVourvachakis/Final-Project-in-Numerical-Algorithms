import numpy as np
from typing import List, Tuple
import os

def read_graph(filename: str) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Read a graph from a file and return the number of nodes and edge list.
    Each line in the file should be of format: source_node target_node
    """
    file_path = os.path.join(os.path.dirname(__file__), '..', filename)
    edges = []
    nodes = set()
    with open(file_path, 'r') as f:
        for line in f:
            source, target = map(int, line.strip().split())
            edges.append((source, target))
            nodes.add(source)
            nodes.add(target)
    return len(nodes), edges

def construct_adjacency_matrix(n_nodes: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Construct the normalized adjacency matrix A where
    a_ij = 1/L(j) if there is a link from j to i, 0 otherwise
    L(j) is the number of outgoing links from node j
    """
    outgoing_counts = np.zeros(n_nodes)
    for source, _ in edges:
        outgoing_counts[source - 1] += 1
    
    A = np.zeros((n_nodes, n_nodes))
    for source, target in edges:
        A[target - 1, source - 1] = 1 / outgoing_counts[source - 1]
    
    return A

def construct_pagerank_matrix(A: np.ndarray, d: float = 0.85) -> np.ndarray:
    """
    Construct the PageRank matrix M = d*A + ((1-d)/N)*B
    where B is a matrix of ones
    """
    N = A.shape[0]
    B = np.ones((N, N))
    return d * A + ((1 - d) / N) * B
