import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Type

def compare_methods(    
                        LineSearch: Type,
                        objective_fn: Callable,
                        gradient_fn: Callable,
                        x0: np.ndarray,
                        title: str,
                        x_range: Tuple[float, float],
                        y_range: Tuple[float, float] = None,
                        n_points: int = 200
                    ) -> Tuple[Tuple[np.ndarray, float, int, str],Tuple[np.ndarray, float, int, str]]:
    
    # Initialize both methods
    gd_backtrack = LineSearch(
        objective_fn,
        gradient_fn,
        alpha0=1.0,
        tau=0.5,
        c=0.0001
    )
    
    gd_fixed = LineSearch(
        objective_fn,
        gradient_fn,
        alpha0=0.1,  # Fixed step size
        tau=1.0,  # No backtracking
        c=0.0
    )
    
    # Run optimization
    x_opt_back, f_opt_back, iter_back, msg_back = gd_backtrack.optimize(x0)
    x_opt_fixed, f_opt_fixed, iter_fixed, msg_fixed = gd_fixed.optimize(x0)
    
    # Create visualization
    if len(x0) == 1:  # 1D case
        plot_1d_comparison(
            objective_fn,
            gd_backtrack,
            gd_fixed,
            x_range,
            title
        )
    else:  # 2D case
        plot_2d_comparison(
            objective_fn,
            gd_backtrack,
            gd_fixed,
            x_range,
            y_range,
            n_points,
            title
        )
    
    # Plot alpha values for backtracking method
    # plt.figure(figsize=(10, 4))
    # plt.plot(gd_backtrack.alphas, 'g.-', label='Step sizes')
    # plt.xlabel('Iteration')
    # plt.ylabel('Step size (α)')
    # plt.title('Backtracking Step Sizes Over Iterations')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    return (x_opt_back, f_opt_back, iter_back, msg_back), \
           (x_opt_fixed, f_opt_fixed, iter_fixed, msg_fixed)

def plot_1d_comparison(
    objective_fn: Callable,
    gd_backtrack: Type,
    gd_fixed: Type,
    x_range: Tuple[float, float],
    title: str
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title)
    
    # Create x values for plotting
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = np.array([objective_fn(np.array([xi])) for xi in x])
    
    # Plot paths for both methods
    for ax, gd, method_name in [
        (ax1, gd_backtrack, 'Backtracking'),
        (ax2, gd_fixed, 'Fixed Step')
    ]:
        ax.plot(x, y, 'b-', label='Function', alpha=0.5)
        
        path = np.array(gd.path)
        path_y = np.array([objective_fn(np.array([xi])) for xi in path])
        
        ax.plot(path, path_y, 'r.-', linewidth=0.8, markersize=2, label=f'{method_name} path')
        ax.plot(path[0], path_y[0], 'go', label='Start')
        ax.plot(path[-1], path_y[-1], 'ro', label='End')
        
        ax.set_title(f'{method_name} Method\nIterations: {len(gd.path)-1}')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(True)
        ax.legend(fontsize='small')
        
        # Set reasonable y-axis limits
        ax.set_ylim(-0.5, 2)
    
    plt.tight_layout()
    plt.show()
    
    # Convergence comparison
    plt.figure(figsize=(10, 5))
    plt.semilogy(gd_backtrack.function_values, 'r.-', label='Backtracking', linewidth=1)
    plt.semilogy(gd_fixed.function_values, 'b.-', label='Fixed Step', linewidth=1)
    plt.xlabel('Iterations')
    plt.ylabel('Objective Value (log scale)')
    plt.title('Convergence Comparison')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_2d_comparison(
    objective_fn: Callable,
    gd_backtrack: Type,
    gd_fixed: Type,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    n_points: int,
    title: str
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title)
    
    # Create mesh grid for contour plots
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0] if y_range else x_range[0], 
                    y_range[1] if y_range else x_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(n_points):
        for j in range(n_points):
            Z[i, j] = objective_fn(np.array([X[i, j], Y[i, j]]))
    
    # Plot paths for both methods
    for ax, gd, method_name in [
        (ax1, gd_backtrack, 'Backtracking'),
        (ax2, gd_fixed, 'Fixed Step')
    ]:
        contour = ax.contour(X, Y, Z, levels=50)
        path = np.array(gd.path)
        
        ax.plot(path[:, 0], path[:, 1], 'r.-', linewidth=0.8, markersize=1, label=f'{method_name} path')
        ax.plot(path[0, 0], path[0, 1], 'go', label='Start')
        ax.plot(path[-1, 0], path[-1, 1], 'ro', label='End')
        
        ax.set_title(f'{method_name} Method\nIterations: {len(gd.path)-1}')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.grid(True)
        ax.legend(fontsize='small')
    
    plt.tight_layout()
    plt.show()
    
    # Convergence comparison
    plt.figure(figsize=(10, 5))
    plt.semilogy(gd_backtrack.function_values, 'r.-',linewidth=0.5, markersize=1, label='Backtracking')
    plt.semilogy(gd_fixed.function_values, 'b.-', linewidth=0.5, markersize=1,label='Fixed Step')
    plt.xlabel('Iterations')
    plt.ylabel('Objective Value (log scale)')
    plt.title('Convergence Comparison')
    plt.grid(True)
    plt.legend()
    plt.show()