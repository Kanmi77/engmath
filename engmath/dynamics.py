import numpy as np

class SystemODESolver:
    def __init__(self, f):
        """
        f: function f(t, y) defining dy/dt, where y is a vector
        """
        self.f = f

    def runge_kutta(self, y0, t0, t_end, h):
        """Runge-Kutta 4th order method for systems"""
        t_values = np.arange(t0, t_end+h, h)
        y_values = [np.array(y0, dtype=float)]
        y = np.array(y0, dtype=float)

        for t in t_values[:-1]:
            k1 = h * self.f(t, y)
            k2 = h * self.f(t + h/2, y + k1/2)
            k3 = h * self.f(t + h/2, y + k2/2)
            k4 = h * self.f(t + h, y + k3)
            y = y + (k1 + 2*k2 + 2*k3 + k4)/6
            y_values.append(y)

        return t_values, np.array(y_values)