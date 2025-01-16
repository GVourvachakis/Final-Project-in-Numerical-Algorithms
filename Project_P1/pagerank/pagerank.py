import numpy as np
from typing import Tuple, List
from .graph_utils import read_graph, construct_adjacency_matrix, construct_pagerank_matrix
    
def power_method(A: np.ndarray, norm: int = 2, tol: float = 1e-10, 
                 max_iter: int = 1_000) -> Tuple[np.ndarray, float, int]:
    n = A.shape[0]
    x = np.ones(n) / n # Initial vector with equal probabilities
    for k in range(max_iter):
        x_new = A @ x
        x_new = x_new / np.linalg.norm(x_new, norm)

        if np.linalg.norm(x_new - x, norm) < tol:
            lambda_k = x_new.T @ (A @ x_new)
            return x_new, lambda_k, k + 1
        
        x = x_new

    lambda_k = x.T @ (A @ x)
    return x, lambda_k, max_iter

def compute_pagerank(filename: str, d: float = 0.85) -> Tuple[List[Tuple[int, float]], float, int]:
    """
    Compute PageRank for a graph stored in a file
    """

    n_nodes, edges = read_graph(filename)

    # Construct matrices
    A = construct_adjacency_matrix(n_nodes, edges)
    M = construct_pagerank_matrix(A, d)

    # Perform power method
    ranks, eigenvalue, iterations = power_method(M, norm=np.inf)

    # Create ranking with node indices (1-based)
    ranking = [(i + 1, rank) for i, rank in enumerate(ranks)]
    ranking.sort(key=lambda x: x[1], reverse=True)
    
    return ranking, eigenvalue, iterations