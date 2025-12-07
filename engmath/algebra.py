import math
class ComplexNumber:
    def __init__(self, real: float, imag: float):
        self._real = real
        self._imag = imag

    def __str__(self):
        f"{self._real} + {self._imag}i"

    def add(self, other):
        return ComplexNumber(self._real + other.real, self._imag + other.imag)
    def subtract(self, other):
        return ComplexNumber(self._real - other.real, self._imag - other.imag)
    def multiply(self, other):  
        real_part = self._real * other.real - self._imag * other.imag
        imag_part = self._real * other.imag + self._imag * other.real
        return ComplexNumber(real_part, imag_part)
    def divide(self, other):
        denom = other.real ** 2 + other.imag ** 2
        real_part = (self._real * other.real + self._imag * other.imag) / denom
        imag_part = (self._imag * other.real - self._real * other.imag) / denom
        return ComplexNumber(real_part, imag_part)
    def conjugate(self):
        return ComplexNumber(self._real, -self._imag)
    def power(self, n: int):
        r = self.modulus()
        theta = math.atan2(self._imag, self._real)
        real_= r**n * math.cos(n * theta)
        imag_= r**n * math.sin(n * theta)
        return ComplexNumber(real_, imag_)
    def modulus(self):
        return math.sqrt(self._real**2 + self._imag**2)
    def __eq__(self, other):
        return self.real == other.real and self.imag == other.imag

    def __abs__(self):
        return math.sqrt(self.real**2 + self.imag**2)
    