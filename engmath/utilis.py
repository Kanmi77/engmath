import numpy as np

def trapezoidal(f, a, b, n=1000):
    """
    Approximate ∫ f(x) dx from a to b using trapezoidal rule.
    f : function
    a, b : integration limits
    n : number of subintervals
    """
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    return (h/2) * (y[0] + 2*np.sum(y[1:-1]) + y[-1])

def simpson(f, a, b, n=1000):
    """
    Approximate ∫ f(x) dx from a to b using Simpson’s rule.
    n must be even.
    """
    if n % 2 == 1:
        n += 1  # force even
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    return (h/3) * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])

def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    """
    Newton-Raphson method for solving f(x)=0.
    f : function
    df : derivative of f
    x0 : initial guess
    """
    x = x0
    for _ in range(max_iter):
        x_new = x - f(x)/df(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise ValueError("Newton-Raphson did not converge")

def bisection(f, a, b, tol=1e-6, max_iter=100):
    """
    Bisection method for solving f(x)=0.
    f : function
    a, b : interval endpoints (f(a) and f(b) must have opposite signs)
    """
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    for _ in range(max_iter):
        c = (a + b) / 2
        if abs(f(c)) < tol or (b - a) / 2 < tol:
            return c
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    raise ValueError("Bisection method did not converge")

def linear_interpolation(x_points, y_points, x):
    """
    Linear interpolation for given data points.
    """
    for i in range(len(x_points)-1):
        if x_points[i] <= x <= x_points[i+1]:
            slope = (y_points[i+1] - y_points[i]) / (x_points[i+1] - x_points[i])
            return y_points[i] + slope * (x - x_points[i])
    raise ValueError("x out of bounds")
def compute_inverse_z_transform(self, X, n):
    N = len(n)
    x = np.zeros(N, dtype=complex)
    for k in range(N):
        sum_val = 0
        for m in range(N):
            sum_val += X * (n[m] ** k)
        x[k] = sum_val / N
    return x

def divergence(self, point):
    return np.divergence(self._vector_field, axis=0)[point]