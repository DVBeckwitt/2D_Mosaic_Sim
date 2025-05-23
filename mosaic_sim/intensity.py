"""Intensity kernels describing the mosaic spread of reflections."""
import math
import numpy as np
from numba import njit

__all__ = ["cap_intensity", "belt_intensity", "mosaic_intensity"]

@njit
def cap_intensity(Qx: np.ndarray, Qy: np.ndarray, Qz: np.ndarray,
                  sigma: float, gamma: float, eta: float) -> np.ndarray:
    """Pseudo-Voigt mosaic cap centred on +z.

    Parameters mirror :func:`belt_intensity` so that a Gaussian width
    ``sigma``, Lorentzian width ``gamma`` and mixing factor ``eta`` can be
    provided.  ``eta=0`` yields a pure Gaussian while ``eta=1`` produces a
    Lorentzian cap.

    Works with inputs of any dimensionality by iterating over the flattened
    arrays. The output has the same shape as ``Qx``.
    """
    I = np.empty_like(Qx)
    for n in range(Qx.size):
        qx, qy, qz = Qx.flat[n], Qy.flat[n], Qz.flat[n]
        Qmag = math.sqrt(qx*qx + qy*qy + qz*qz) or 1e-14
        nu_p = math.acos(max(-1.0, min(1.0, qz / Qmag)))
        I.flat[n] = ((1 - eta) * math.exp(-nu_p * nu_p / (2 * sigma * sigma)) +
                     eta / (1 + (nu_p / gamma) ** 2))
    return I / I.max()

@njit
def belt_intensity(Qx: np.ndarray, Qy: np.ndarray, Qz: np.ndarray,
                   Gx: float, Gy: float, Gz: float,
                   sigma: float, gamma: float, eta: float) -> np.ndarray:
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

def mosaic_intensity(Qx: np.ndarray, Qy: np.ndarray, Qz: np.ndarray,
                     H: int, K: int, L: int,
                     sigma: float, gamma: float, eta: float) -> np.ndarray:
    """Dispatch to either :func:`cap_intensity` or :func:`belt_intensity`.

    The function chooses between the cap or belt model depending on the Miller
    indices.  ``H=K=0`` corresponds to a cap around the ``L`` direction, while
    any in-plane component switches to the belt model.
    """

    if H == 0 and K == 0:
        return cap_intensity(Qx, Qy, Qz, sigma, gamma, eta)
    return belt_intensity(Qx, Qy, Qz, H, K, L, sigma, gamma, eta)
