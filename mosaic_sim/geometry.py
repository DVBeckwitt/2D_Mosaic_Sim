"""Basic geometry helpers shared by the mosaic simulations.

This module groups together small utilities for constructing points on
geometric primitives and rotating them.  They are primarily used by the
visualisation scripts to build the Bragg and Ewald spheres.
"""
import math
import numpy as np

__all__ = ["sphere", "rot_x", "intersection_circle"]

def sphere(R: float, phi: np.ndarray, theta: np.ndarray,
           center: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the Cartesian coordinates of points on a sphere.

    Parameters
    ----------
    R:
        Radius of the sphere.
    phi, theta:
        Arrays of polar and azimuthal angles in radians.
    center:
        Optional ``(x, y, z)`` centre of the sphere.

    Returns
    -------
    tuple of ``numpy.ndarray``
        Arrays ``(x, y, z)`` describing the points on the sphere.
    """

    cx, cy, cz = center
    return (
        cx + R * np.sin(phi) * np.cos(theta),
        cy + R * np.sin(phi) * np.sin(theta),
        cz + R * np.cos(phi),
    )


def rot_x(x: np.ndarray, y: np.ndarray, z: np.ndarray, ang: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate arrays ``(x, y, z)`` about the *x*-axis by ``ang`` radians."""
    c, s = np.cos(ang), np.sin(ang)
    return x, c * y - s * z, s * y + c * z


def intersection_circle(Rg: float, Re: float, d: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordinates of the circle formed by two intersecting spheres.

    Parameters
    ----------
    Rg:
        Radius of the Bragg sphere.
    Re:
        Radius of the Ewald sphere.
    d:
        Distance between the centres of the two spheres.

    Returns
    -------
    tuple of ``numpy.ndarray``
        Arrays ``(x, y, z)`` forming the intersection circle in reciprocal
        space.
    """

    y0 = (d * d - Re * Re + Rg * Rg) / (2 * d)
    r = math.sqrt(max(Rg * Rg - y0 * y0, 0.0))
    t = np.linspace(0.0, 2 * math.pi, 400)
    return r * np.cos(t), np.full_like(t, y0), r * np.sin(t)
