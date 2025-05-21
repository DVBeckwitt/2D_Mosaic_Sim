"""Basic geometry helpers shared by the mosaic simulations."""
import math
import numpy as np

__all__ = ["sphere", "rot_x", "intersection_circle"]

def sphere(R: float, phi: np.ndarray, theta: np.ndarray, center=(0, 0, 0)):
    cx, cy, cz = center
    return (
        cx + R * np.sin(phi) * np.cos(theta),
        cy + R * np.sin(phi) * np.sin(theta),
        cz + R * np.cos(phi),
    )


def rot_x(x: np.ndarray, y: np.ndarray, z: np.ndarray, ang: float):
    c, s = np.cos(ang), np.sin(ang)
    return x, c * y - s * z, s * y + c * z


def intersection_circle(Rg: float, Re: float, d: float):
    """Return x, y, z arrays for the Bragg/Ewald intersection circle."""
    y0 = (d * d - Re * Re + Rg * Rg) / (2 * d)
    r = math.sqrt(max(Rg * Rg - y0 * y0, 0.0))
    t = np.linspace(0.0, 2 * math.pi, 400)
    return r * np.cos(t), np.full_like(t, y0), r * np.sin(t)
