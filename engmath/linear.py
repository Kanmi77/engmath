class MatrixSolver:
    def __init__(self, matrix):
        self._matrix = matrix
    def determinant(self):
        return np.linalg.det(self._matrix)
    def inverse(self):
        return np.linalg.inv(self._matrix)
    def eigen(self):
        return np.linalg.eig(self._matrix)
    def transpose(self):
        return np.transpose(self._matrix)
    def rank(self):
        return np.linalg.matrix_rank(self._matrix)
    def solve(self, b):
        return np.linalg.solve(self._matrix, b) 


class DifferentialEquationSolver:
    def __init__(self, f):
        """
        f: function f(t, y) defining dy/dt
        """

        self._f = f
    def euler(self, y0, t0, t_end, h):
        " Euler's Method"
        t_values = np.arange(t0, t_end, h)
        y_values = [y0]
        y = y0
        for t in t_values[:-1]:
            y += h * self._f(t, y)
            y_values.append(y)
        return t_values, np.array(y_values)
    def runge_kutta(self, y0, t0, t_end, h):
        """4th Order Runge-Kutta Method"""
        t_values = np.arange(t0, t_end, h)
        y_values = [y0]
        y = y0
        for t in t_values[:-1]:
            k1 = h * self._f(t, y)
            k2 = h * self._f(t + h / 2, y + k1 / 2)
            k3 = h * self._f(t + h / 2, y + k2 / 2)
            k4 = h * self._f(t + h, y + k3)
            y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            y_values.append(y)
        return t_values, np.array(y_values)
    def finite_difference(self, y0, y1, t0, t_end, h):
        """Finite Difference Method for second-order ODEs"""
        t_values = np.arange(t0, t_end, h)
        y_values = [y0, y1]
        for i in range(1, len(t_values) - 1):
            y_next = 2 * y_values[i] - y_values[i - 1] + h**2 * self._f(t_values[i], y_values[i])
            y_values.append(y_next)
        return t_values, np.array(y_values)
    def __str__(self):
        return "DifferentialEquationSolver with function f(t, y)"
    

class FourierTransform:
    def __init__(self, signal, sampling_rate):
        self._signal = signal
        self._sampling_rate = sampling_rate
    def compute_fft(self):
        n = len(self._signal)
        freq = np.fft.fftfreq(n, d=1/self._sampling_rate)
        fft_values = np.fft.fft(self._signal)
        return freq, fft_values
    def compute_ifft(self, fft_values):
        return np.fft.ifft(fft_values)
    def __str__(self):
        return "FourierTransform with given signal and sampling rate"

class LaplaceTransform:
    def __init__(self, f, t):
        self._f = f
        self._t = t
    def compute_laplace(self, s):
        dt = self._t[1] - self._t[0]
        integral = np.trapz(self._f * np.exp(-s * self._t), dx=dt)
        return integral
    def compute_inverse_laplace(self, F, t):
        dt = t[1] - t[0]
        integral = np.trapz(F * np.exp(s * t), dx=dt) / (2 * np.pi * 1j)
        return integral
    def __str__(self):
        return "LaplaceTransform with given function and time array"
class ZTransform:
    def __init__(self, x, n):
        self._x = x
        self._n = n
    def compute_z_transform(self, z):
        return sum(self._x[k] * (z ** -self._n[k]) for k in range(len(self._x)))
    def compute_inverse_z_transform(self, X, n):
        N = len(n)
        x = np.zeros(N, dtype=complex)
        for k in range(N):
            integral = 0
            for m in range(N):
                integral += X[m] * (n[k] ** m)
            x[k] = integral / N
        return x
    def __str__(self):
        return "ZTransform with given sequence and index array"

class VectorCalculus:
    def __init__(self, vector_field):
        self._vector_field = vector_field
    def gradient(self, point):
        return np.gradient(self._vector_field, axis=0)[point]
    def divergence(self, point):
        div = 0
        for i in range(len(self._vector_field)):
            div += np.gradient(self._vector_field[i], axis=i)[point]
        return div
    def curl(self, point):
        curl = np.zeros(3)
        curl[0] = np.gradient(self._vector_field[2], axis=1)[point] - np.gradient(self._vector_field[1], axis=2)[point]
        curl[1] = np.gradient(self._vector_field[0], axis=2)[point] - np.gradient(self._vector_field[2], axis=0)[point]
        curl[2] = np.gradient(self._vector_field[1], axis=0)[point] - np.gradient(self._vector_field[0], axis=1)[point]
        return curl
    def __str__(self):
        return "VectorCalculus with given vector field"
class PartialDifferentialEquationSolver:
    def __init__(self, f, x, t):
        self._f = f
        self._x = x
        self._t = t
    def finite_difference(self, u0, u1, dx, dt):
        nx = len(self._x)
        nt = len(self._t)
        u = np.zeros((nt, nx))
        u[0, :] = u0
        u[1, :] = u1
        for n in range(1, nt - 1):
            for i in range(1, nx - 1):
                u[n + 1, i] = (2 * u[n, i] - u[n - 1, i] +
                               (dt**2) * self._f(self._x[i], self._t[n]) *
                               (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1]) / (dx**2))
        return u
    def __str__(self):
        return "PartialDifferentialEquationSolver with given function, spatial and temporal arrays"
    