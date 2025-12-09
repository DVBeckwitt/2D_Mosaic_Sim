"""Physical constants and lattice helpers for mosaic simulations."""
import math

# Wavelength of Cu-Kα radiation in meters
CU_K_ALPHA_WAVELENGTH = 1.5406e-10

# Hexagonal lattice constants for Bi₂Se₃
a_hex = 4.143e-10
c_hex = 28.636e-10

# Hexagonal lattice constants for 2H PbI₂ (in meters)
PBI2_2H_A_HEX = 4.557e-10
PBI2_2H_C_HEX = 6.979e-10

# Magnitude of the incident wavevector |k|
K_MAG = 2 * math.pi / CU_K_ALPHA_WAVELENGTH


def d_hex(h: int, k: int, l: int, a: float = a_hex, c: float = c_hex) -> float:
    """Return the d-spacing for (h k l) using hexagonal parameters."""
    return 1.0 / math.sqrt((4/3) * (h*h + h*k + k*k) / a**2 + (l / c) ** 2)

# Backwards compatibility alias for existing scripts
λ = CU_K_ALPHA_WAVELENGTH

__all__ = [
    "CU_K_ALPHA_WAVELENGTH",
    "PBI2_2H_A_HEX",
    "PBI2_2H_C_HEX",
    "λ",
    "a_hex",
    "c_hex",
    "K_MAG",
    "d_hex",
]
