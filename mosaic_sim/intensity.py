"""Mosaic intensity kernels."""
import math
import numpy as np
from numba import njit

__all__ = ["cap_intensity", "belt_intensity", "mosaic_intensity"]

@njit
def cap_intensity(Qx, Qy, Qz, sigma):
    """Gaussian mosaic cap centred on +z.

    Works with inputs of any dimensionality by iterating over the flattened
    arrays. The output has the same shape as ``Qx``.
    """
    I = np.empty_like(Qx)
    for n in range(Qx.size):
        qx, qy, qz = Qx.flat[n], Qy.flat[n], Qz.flat[n]
        Qmag = math.sqrt(qx*qx + qy*qy + qz*qz) or 1e-14
        alpha = math.acos(max(-1.0, min(1.0, qz / Qmag)))
        I.flat[n] = math.exp(-0.5 * (alpha / sigma) ** 2)
    return I / I.max()

@njit
def belt_intensity(Qx, Qy, Qz, Gx, Gy, Gz, sigma, gamma, eta):
    """Pseudo-Voigt belt intensity around the vector ``(Gx, Gy, Gz)``."""
    I = np.empty_like(Qx)
    Gmag = math.sqrt(Gx * Gx + Gy * Gy + Gz * Gz)
    nu_c = math.acos(max(-1.0, min(1.0, Gz / Gmag)))
    for n in range(Qx.size):
        qx, qy, qz = Qx.flat[n], Qy.flat[n], Qz.flat[n]
        Qmag = math.sqrt(qx*qx + qy*qy + qz*qz) or 1e-14
        nu_p = math.acos(max(-1.0, min(1.0, qz / Qmag)))
        dnu = abs(nu_p - nu_c)
        I.flat[n] = (1 - eta) * math.exp(-dnu*dnu / (2 * sigma * sigma)) + eta / (1 + (dnu / gamma) ** 2)

    return I / I.max()

def mosaic_intensity(Qx, Qy, Qz, H, K, L, sigma, gamma, eta):
    if H == 0 and K == 0:
        return cap_intensity(Qx, Qy, Qz, sigma)
    return belt_intensity(Qx, Qy, Qz, H, K, L, sigma, gamma, eta)
