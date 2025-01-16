import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List, Type

def plot_multiple_learning_rates(
                                    GD: Type,
                                    objective_fn: Callable,
                                    gradient_fn: Callable,
                                    x0: np.ndarray,
                                    learning_rates: List[float],
                                    x_range: Tuple[float, float],
                                    y_range: Tuple[float, float],
                                    title: str,
                                    tau: float = 1.0,
                                    c: float = 0.0,
                                    n_points: int = 200,
                                    max_iterations: int = 1000,
                                    epsilon: float = 1e-6
                                ):
    
    n_rates = len(learning_rates)
    fig, axes = plt.subplots(2, n_rates, figsize=(5*n_rates, 10))
    fig.suptitle(title, fontsize=16, y=1.02)
    
    # Create mesh grid for contour plots
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(n_points):
        for j in range(n_points):
            Z[i, j] = objective_fn(np.array([X[i, j], Y[i, j]]))
    
    for idx, alpha in enumerate(learning_rates):
        # Initialize and run GD
        gd = GD(
            objective_fn,
            gradient_fn,
            alpha0=alpha,
            tau = tau,
            c = c,
            epsilon=epsilon,
            max_iterations=max_iterations
        )
        x_opt, f_opt, iterations, message = gd.optimize(x0)
        
        # Contour plot
        ax_contour = axes[0, idx]
        contour = ax_contour.contour(X, Y, Z, levels=50)
        path = np.array(gd.path)
        ax_contour.plot(path[:, 0], path[:, 1], 'r.-', linewidth=0.8, markersize=1, label='GD path')
        ax_contour.plot(path[0, 0], path[0, 1], 'go', label='Start')
        ax_contour.plot(path[-1, 0], path[-1, 1], 'ro', label='End')
        
        ax_contour.set_title(f'α = {alpha}')
        ax_contour.set_xlabel('x₁')
        ax_contour.set_ylabel('x₂')
        ax_contour.grid(True)
        ax_contour.legend(fontsize='small')
        
        # Convergence plot
        ax_conv = axes[1, idx]
        ax_conv.semilogy(range(len(gd.function_values)), gd.function_values)
        ax_conv.set_xlabel('Iterations')
        ax_conv.set_ylabel('Objective Value (log scale)')
        ax_conv.grid(True)
        
        # Add final values to plot
        ax_conv.text(0.02, 0.98, f'Iterations: {iterations}\nFinal value: {f_opt:.2e}',
                    transform=ax_conv.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def plot_1d_optimization(
                            GD: Type,
                            objective_fn: Callable,
                            gradient_fn: Callable,
                            x0: np.ndarray,
                            learning_rates: List[float],
                            x_range: Tuple[float, float],
                            tau: float = 1.0,
                            c: float = 0.0,
                            max_iterations: int = 1000,
                            epsilon: float = 1e-6
                        ):
    
    n_rates = len(learning_rates)
    fig, axes = plt.subplots(2, n_rates, figsize=(5*n_rates, 10))
    fig.suptitle("1D Griewank Function Optimization with Different Learning Rates", fontsize=16, y=1.02)
    
    # Create x values for plotting
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = np.array([objective_fn(np.array([xi])) for xi in x])
    
    for idx, alpha in enumerate(learning_rates):
        # Initialize and run GD
        gd = GD(
            objective_fn,
            gradient_fn,
            alpha0=alpha,
            tau=c,
            c=c,
            epsilon=epsilon,
            max_iterations=max_iterations
        )
        x_opt, f_opt, iterations, message = gd.optimize(x0)
        
        # Function plot with path
        ax_func = axes[0, idx]
        ax_func.plot(x, y, 'b-', label='Function', alpha=0.5)
        
        # Plot optimization path
        path = np.array(gd.path)
        path_y = np.array([objective_fn(np.array([xi])) for xi in path])
        ax_func.plot(path, path_y, 'r.-', linewidth=0.8, markersize=2, label='GD path')
        ax_func.plot(path[0], path_y[0], 'go', label='Start')
        ax_func.plot(path[-1], path_y[-1], 'ro', label='End')
        
        ax_func.set_title(f'α = {alpha}')
        ax_func.set_xlabel('x')
        ax_func.set_ylabel('f(x)')
        ax_func.grid(True)
        ax_func.legend(fontsize='small')
        
        # Set reasonable y-axis limits to focus on the interesting region
        ax_func.set_ylim(-0.5, 2)
        
        # Convergence plot
        ax_conv = axes[1, idx]
        ax_conv.semilogy(range(len(gd.function_values)), gd.function_values)
        ax_conv.set_xlabel('Iterations')
        ax_conv.set_ylabel('Objective Value (log scale)')
        ax_conv.grid(True)
        
        # Add final values to plot
        ax_conv.text(0.02, 0.98, f'Iterations: {iterations}\nFinal value: {f_opt:.2e}\nFinal x: {x_opt[0]:.2e}',
                    transform=ax_conv.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()