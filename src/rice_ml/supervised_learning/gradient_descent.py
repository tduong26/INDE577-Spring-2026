"""
Gradient Descent Optimization Techniques
This module builds simple gradient descent optimizers from the ground up
for both single-variable and multi-variable objective functions.
The implementations are kept deliberately straightforward and aimed at
demonstrating the fundamental mechanics of gradient-based optimization.
Design Goals
------------
- Prioritize readability and teaching value over speed or flexibility
- Direct update rules with minimal layers of abstraction
- Compatible with both scalar and vector parameter spaces
- Visible convergence behavior through recorded parameter trajectories
Implemented Optimizers
----------------------
- GradientDescent1D:
    Gradient descent for scalar parameters w ∈ ℝ, requiring an
    explicit derivative df/dw supplied by the caller.
- GradientDescentND:
    Gradient descent for vector parameters w ∈ ℝⁿ, requiring a
    gradient function ∇f(w) supplied by the caller.
Key Characteristics
-------------------
- Constant learning rate (no adaptive or scheduled updates)
- Basic stopping condition derived from the magnitude of parameter change
- Full optimization path retained for plotting and post-analysis
- Pure NumPy implementation with no third-party dependencies
Implementation Notes
--------------------
- These optimizers operate independently of any particular model class
- They are meant for learning, prototyping, and classroom demonstration
  rather than production-scale or high-performance optimization
- Features such as numerical safeguards and advanced methods
  (momentum, Adam, etc.) are left out on purpose to keep things clear
This module lays the groundwork for understanding how gradient-based
learning works at its most basic level.
"""

import numpy as np
from typing import Callable

class GradientDescent1D:
    """Gradient descent for 1D functions."""
    
    def __init__(self, df: Callable[[float], float], alpha: float = 0.1, tol: float = 1e-6, max_iter: int = 1000):
        """
        Parameters:
            df : derivative of the function f(w)
            alpha : learning rate
            tol : tolerance for stopping
            max_iter : maximum number of iterations
        """
        self.df = df
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.history = []

    def fit(self, w0: float) -> list[float]:
        """Run gradient descent starting from w0."""
        w = w0
        self.history = [w]
        
        for i in range(self.max_iter):
            grad = self.df(w)
            w_new = w - self.alpha * grad
            self.history.append(w_new)
            if abs(w_new - w) < self.tol:
                break
            w = w_new
        
        return self.history


class GradientDescentND:
    """Gradient descent for N-dimensional functions."""
    
    def __init__(self, grad_f: Callable[[np.ndarray], np.ndarray], alpha: float = 0.1, tol: float = 1e-6, max_iter: int = 1000):
        """
        Parameters:
            grad_f : gradient function ∇f(w)
            alpha : learning rate
            tol : tolerance for stopping
            max_iter : maximum number of iterations
        """
        self.grad_f = grad_f
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.history = []

    def fit(self, w0: np.ndarray) -> list[np.ndarray]:
        """Run gradient descent starting from w0."""
        w = np.array(w0, dtype=float)
        self.history = [w.copy()]
        
        for i in range(self.max_iter):
            grad = self.grad_f(w)
            w_new = w - self.alpha * grad
            self.history.append(w_new.copy())
            if np.linalg.norm(w_new - w) < self.tol:
                break
            w = w_new
        
        return self.history