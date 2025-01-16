import numpy as np
from typing import Callable, Tuple, List
from numpy.linalg import norm

from lr_playground import plot_multiple_learning_rates, plot_1d_optimization
from comparison_utils import compare_methods, plot_1d_comparison, plot_2d_comparison
from trial_objectives import rosenbrock, rosenbrock_gradient, matyas, matyas_gradient, \
                             griewank, griewank_gradient

class GradientDescentBacktracking:
    """
    * Optimization with configurable parameters (learning rate, epsilon, max iterations, tau, c, alpha0)

    * Multiple stopping criteria as specified 

    * Armijo's Backtracking Line Searck (if tau = 0 -> plain GD (constant learning rate))
    
    * Path tracking for visualization
    
    * Both contour plot with optimization path and convergence plot
    """

    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], float],
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        alpha0: float = 1.0,
        tau: float = 0.5,  # Reduction factor for backtracking
        c: float = 0.0001,  # Parameter for Armijo condition
        epsilon: float = 1e-6,
        max_iterations: int = 1000
    ):
        self.objective_fn = objective_fn
        self.gradient_fn = gradient_fn
        self.alpha0 = alpha0
        self.tau = tau
        self.c = c
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.path: List[np.ndarray] = []
        self.function_values: List[float] = []
        #self.alphas: List[float] = []  # Store alpha values for analysis
        
    def backtracking_line_search(self, x: np.ndarray, gradient: np.ndarray) -> float:
        alpha = self.alpha0
        p = -gradient  # Steepest descent direction
        m = gradient.T @ p  # Gradient along the search direction (- ||Grad(f)||_2)
        t = self.c * m  # Target reduction
        fx = self.objective_fn(x)
        
        j = 0
        while self.objective_fn(x + alpha * p) > fx + alpha * t:
            alpha *= self.tau
            j += 1
            if j > 50:  # Prevent infinite loops
                break
                
        return alpha
        
    def optimize(self, x0: np.ndarray) -> Tuple[np.ndarray, float, int, str]:
        x = x0
        self.path = [x]
        self.function_values = [self.objective_fn(x)]
        #self.alphas = []
        
        for k in range(self.max_iterations):
            gradient = self.gradient_fn(x)
            
            if norm(gradient) <= self.epsilon:
                return x, self.objective_fn(x), k + 1, "Convergence: ||∇f(x_k)|| ≤ ε"
                
            # Compute step size using backtracking
            alpha = self.backtracking_line_search(x, gradient)
            #self.alphas.append(alpha)

            x_new = x - alpha * gradient
            
            self.path.append(x_new)
            self.function_values.append(self.objective_fn(x_new))
            
            if norm(x_new - x) <= self.epsilon:
                return x_new, self.objective_fn(x_new), k + 1, "Convergence: ||x_{k+1} - x_k|| ≤ ε"
                
            x = x_new
            
        return x, self.objective_fn(x), self.max_iterations, "Maximum iterations reached"

if __name__ == "__main__":
    # Test all functions with both methods
    
    # 1D Griewank
    # np.random.seed(1)
    # x0_1d = np.random.rand(1)
    # compare_methods(
    #     griewank,
    #     griewank_gradient,
    #     x0_1d,
    #     "1D Griewank Function Optimization Comparison",
    #     (-5, 5)
    # )
    
    # 2D Griewank
    # np.random.seed(1)
    # x0_2d = np.random.rand(2)
    # compare_methods(
    #     griewank,
    #     griewank_gradient,
    #     x0_2d,
    #     "2D Griewank Function Optimization Comparison",
    #     (-3, 3),
    #     (-3, 3)
    # )
    
    # Rosenbrock (set alpha0 = 0.001 (not the default alpha0=0.1) 
    # to circumvent overflow in gradient computation)
    # np.random.seed(1)
    # x0_rosenbrock = np.random.rand(2) #np.array([1.0, 1.0])
    # compare_methods(
    #    GradientDescentBacktracking,
    #    rosenbrock,
    #    rosenbrock_gradient,
    #    x0_rosenbrock,
    #    "Rosenbrock Function Optimization Comparison",
    #    (-2, 2),
    #    (-2, 2)
    # )

    #=============================================================
    # Test Rosenbrock function with different learning rates (example usage)
    # learning_rates_rosenbrock = [0.001, 0.002, 0.005, 0.0001]
    
    # np.random.seed(1)
    # x0_rosenbrock = np.random.rand(2) #Choose a random x0 ∈ R^n
    # plot_multiple_learning_rates(
    #     GradientDescentBacktracking,
    #     rosenbrock,
    #     rosenbrock_gradient,
    #     x0_rosenbrock,
    #     learning_rates_rosenbrock,
    #     (-2, 2), (-1, 3),
    #     "Rosenbrock Function Optimization with Different Learning Rates"\
    #     #,tau=1.1, c=0.0
    # )
    #=============================================================

    # Matyas
    np.random.seed(1)
    x0_matyas = np.random.rand(2) #np.array([1.0, 1.0])
    compare_methods(
       GradientDescentBacktracking,
       matyas,
       matyas_gradient,
       x0_matyas,
       "Matyas Function Optimization Comparison",
       (-2, 2),
       (-2, 2)
    )