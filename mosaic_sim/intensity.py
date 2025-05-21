"""Mosaic intensity kernels."""
import math
import numpy as np
from numba import njit

__all__ = ["cap_intensity", "belt_intensity", "mosaic_intensity"]

@njit
def cap_intensity(Qx, Qy, Qz, sigma):
    """Gaussian cap centred on +z.

    Works for arrays of any shape (1-D ring or 2-D sphere mesh).
    """
    I = np.empty_like(Qx)
    # Flatten views for generic dimensionality
    qx1 = Qx.ravel()
    qy1 = Qy.ravel()
    qz1 = Qz.ravel()
    I1 = I.ravel()
    for idx in range(qx1.size):
        qx = qx1[idx]
        qy = qy1[idx]
        qz = qz1[idx]
        Qmag = math.sqrt(qx*qx + qy*qy + qz*qz) or 1e-14
        alpha = math.acos(max(-1.0, min(1.0, qz / Qmag)))
        I1[idx] = math.exp(-0.5 * (alpha / sigma) ** 2)
    maxval = I1.max()
    if maxval != 0.0:
        I1 /= maxval
    return I

@njit
def belt_intensity(Qx, Qy, Qz, Gx, Gy, Gz, sigma, gamma, eta):
    """Pseudo-Voigt belt at polar angle of vector (Gx, Gy, Gz)."""
    I = np.empty_like(Qx)
    Gmag = math.sqrt(Gx*Gx + Gy*Gy + Gz*Gz)
    nu_c = math.acos(max(-1.0, min(1.0, Gz / Gmag)))
    qx1 = Qx.ravel()
    qy1 = Qy.ravel()
    qz1 = Qz.ravel()
    I1 = I.ravel()
    for idx in range(qx1.size):
        qz = qz1[idx]
        Qmag = math.sqrt(qx1[idx]*qx1[idx] + qy1[idx]*qy1[idx] + qz*qz) or 1e-14
        nu_p = math.acos(max(-1.0, min(1.0, qz / Qmag)))
        dnu = abs(nu_p - nu_c)
        I1[idx] = (1 - eta) * math.exp(-dnu*dnu / (2*sigma*sigma)) + eta / (1 + (dnu / gamma)**2)
    maxval = I1.max()
    if maxval != 0.0:
        I1 /= maxval
    return I


def mosaic_intensity(Qx, Qy, Qz, H, K, L, sigma, gamma, eta):
    if H == 0 and K == 0:
        return cap_intensity(Qx, Qy, Qz, sigma)
    return belt_intensity(Qx, Qy, Qz, H, K, L, sigma, gamma, eta)
