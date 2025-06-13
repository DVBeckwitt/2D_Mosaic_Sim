"""Basic geometry helpers shared by the mosaic simulations.

This module groups together small utilities for constructing points on
geometric primitives and rotating them.  They are primarily used by the
visualisation scripts to build the Bragg and Ewald spheres.
"""
import math
import numpy as np

__all__ = [
    "sphere",
    "rot_x",
    "intersection_circle",
    "intersection_cylinder_sphere",
]

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


def intersection_cylinder_sphere(
    rc: float,
    Re: float,
    dy: float,
    dz: float = 0.0,
    npts: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points where a cylinder intersects a sphere.

    The cylinder is assumed to run along the ``qz`` axis and to be centred on
    the origin. ``rc`` gives its radius.  The sphere of radius ``Re`` is offset
    along ``qy`` by ``dy`` and optionally along ``qz`` by ``dz``.

    Parameters
    ----------
    rc:
        Radius of the cylinder.
    Re:
        Radius of the sphere.
    dy:
        Offset of the sphere centre along ``qy``.
    dz:
        Optional offset of the sphere centre along ``qz``.
    npts:
        Number of points used to sample the intersection curve.

    Returns
    -------
    tuple of ``numpy.ndarray``
        Arrays ``(x, y, z)`` describing the intersection curve.  The arrays are
        empty if the cylinder and sphere do not intersect.
    """

    t = np.linspace(0.0, 2 * math.pi, npts)
    x = rc * np.cos(t)
    y = rc * np.sin(t)
    zsq = Re * Re - rc * rc - dy * dy + 2 * rc * dy * np.sin(t)
    mask = zsq >= 0
    if not np.any(mask):
        return np.array([]), np.array([]), np.array([])

    z = np.sqrt(zsq[mask])
    x = x[mask]
    y = y[mask]
    return (
        np.concatenate([x, x[::-1]]),
        np.concatenate([y, y[::-1]]),
        np.concatenate([dz + z, dz - z[::-1]]),
    )
