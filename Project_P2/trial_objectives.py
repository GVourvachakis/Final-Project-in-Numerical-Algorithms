import numpy as np

# Adding Rosenbrock function and its gradient
def rosenbrock(x: np.ndarray) -> float:
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_gradient(x: np.ndarray) -> np.ndarray:
    dx1 = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]) # df/dx_1
    dx2 = 200 * (x[1] - x[0]**2) # df/dx_2
    return np.array([dx1, dx2]) # df/dx = (df/dx_1)*x_1^ + (df/dx_2)*x_2^ 

# Adding Matyas function and its gradient
def matyas(x: np.ndarray) -> float:
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

def matyas_gradient(x: np.ndarray) -> np.ndarray:
    dx1 = 0.52 * x[0] - 0.48 * x[1]
    dx2 = 0.52 * x[1] - 0.48 * x[0]
    return np.array([dx1, dx2])

# Adding Griewank function and its gradient
def griewank(x: np.ndarray) -> float:
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum_term - prod_term

def griewank_gradient(x: np.ndarray) -> np.ndarray:
    n = len(x)
    gradient = np.zeros(n)
    
    # Derivative of sum term
    sum_term = x / 2000
    
    # Derivative of product term
    for i in range(n):
        prod_term = 1
        for j in range(n):
            if j != i:
                prod_term *= np.cos(x[j] / np.sqrt(j + 1))
        prod_term *= -np.sin(x[i] / np.sqrt(i + 1)) / np.sqrt(i + 1)
        gradient[i] = sum_term[i] - prod_term
    
    return gradient 