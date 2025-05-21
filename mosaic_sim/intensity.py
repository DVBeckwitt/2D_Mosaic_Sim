"""Mosaic intensity kernels."""
import math
import numpy as np
from numba import njit

__all__ = ["cap_intensity", "belt_intensity", "mosaic_intensity"]

@njit
def cap_intensity(Qx, Qy, Qz, sigma):
    I = np.empty_like(Qx)
    for i in range(Qx.shape[0]):
        for j in range(Qx.shape[1]):
            qx, qy, qz = Qx[i, j], Qy[i, j], Qz[i, j]
            Qmag = math.sqrt(qx*qx + qy*qy + qz*qz) or 1e-14
            alpha = math.acos(max(-1.0, min(1.0, qz / Qmag)))
            I[i, j] = math.exp(-0.5 * (alpha / sigma) ** 2)
    return I / I.max()

@njit
def belt_intensity(Qx, Qy, Qz, Gx, Gy, Gz, sigma, gamma, eta):
    I = np.empty_like(Qx)
    Gmag = math.sqrt(Gx*Gx + Gy*Gy + Gz*Gz)
    nu_c = math.acos(max(-1.0, min(1.0, Gz / Gmag)))
    for i in range(Qx.shape[0]):
        for j in range(Qx.shape[1]):
            qz = Qz[i, j]
            Qmag = math.sqrt(Qx[i, j]**2 + Qy[i, j]**2 + qz*qz) or 1e-14
            nu_p = math.acos(max(-1.0, min(1.0, qz / Qmag)))
            dnu = abs(nu_p - nu_c)
            I[i, j] = (1 - eta) * math.exp(-dnu*dnu / (2*sigma*sigma)) + eta / (1 + (dnu / gamma)**2)
    return I / I.max()

def mosaic_intensity(Qx, Qy, Qz, H, K, L, sigma, gamma, eta):
    if H == 0 and K == 0:
        return cap_intensity(Qx, Qy, Qz, sigma)
    return belt_intensity(Qx, Qy, Qz, H, K, L, sigma, gamma, eta)
