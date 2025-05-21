"""Physical constants and lattice helpers for mosaic simulations."""
import math

# Wavelength of Cu-Kα radiation in meters
λ = 1.5406e-10

# Hexagonal lattice constants for Bi₂Se₃
a_hex = 4.143e-10
c_hex = 28.636e-10

# Magnitude of the incident wavevector |k|
K_MAG = 2 * math.pi / λ


def d_hex(h: int, k: int, l: int, a: float = a_hex, c: float = c_hex) -> float:
    """Return the d-spacing for (h k l) using hexagonal parameters."""
    return 1.0 / math.sqrt((4/3) * (h*h + h*k + k*k) / a**2 + (l / c) ** 2)

__all__ = ["λ", "a_hex", "c_hex", "K_MAG", "d_hex"]
