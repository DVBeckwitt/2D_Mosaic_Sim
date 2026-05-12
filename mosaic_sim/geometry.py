"""Basic geometry helpers shared by the mosaic simulations.

This module groups together small utilities for constructing points on
geometric primitives and rotating them.  They are primarily used by the
visualisation scripts to build the Bragg and Ewald spheres.
"""
import math
from dataclasses import dataclass
import numpy as np

__all__ = [
    "EWALD_BANDWIDTH_LAYER_COUNT",
    "EWALD_LAYER_MIN_OPACITY",
    "EWALD_LAYER_MAX_OPACITY",
    "WAVELENGTH_BANDWIDTH_CONTROL_MAX_PCT",
    "EwaldLayer",
    "ewald_bandwidth_k_bounds",
    "ewald_bandwidth_layers",
    "normalize_wavelength_bandwidth_pct",
    "sphere",
    "rot_x",
    "intersection_circle",
    "intersection_cylinder_sphere",
]

EWALD_BANDWIDTH_LAYER_COUNT = 7
EWALD_LAYER_MIN_OPACITY = 0.04
EWALD_LAYER_MAX_OPACITY = 0.30
WAVELENGTH_BANDWIDTH_CONTROL_MAX_PCT = 100.0


def normalize_wavelength_bandwidth_pct(value: float | int | None, default: float = 0.0) -> float:
    """Normalize the full fractional wavelength bandwidth percentage."""

    raw_value = default if value is None else value
    try:
        bandwidth_pct = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("wavelength_bandwidth_pct must be a finite number") from exc

    if not math.isfinite(bandwidth_pct):
        raise ValueError("wavelength_bandwidth_pct must be a finite number")
    if bandwidth_pct < 0.0 or bandwidth_pct >= 200.0:
        raise ValueError("wavelength_bandwidth_pct must be >= 0.0 and < 200.0")
    return bandwidth_pct


def ewald_bandwidth_k_bounds(k0: float, wavelength_bandwidth_pct: float) -> tuple[float, float]:
    """Return ordered Ewald ``k`` radii for a full fractional wavelength bandwidth."""

    k0 = float(k0)
    if not math.isfinite(k0) or k0 <= 0.0:
        raise ValueError("k0 must be a finite positive number")

    bandwidth_pct = normalize_wavelength_bandwidth_pct(wavelength_bandwidth_pct)
    if bandwidth_pct == 0.0:
        return k0, k0

    half_bandwidth = bandwidth_pct / 200.0
    return k0 / (1.0 + half_bandwidth), k0 / (1.0 - half_bandwidth)


@dataclass(frozen=True)
class EwaldLayer:
    """Wavelength-specific Ewald geometry and rendering weight."""

    relative_wavelength_offset: float
    k_mag: float
    opacity: float


def ewald_bandwidth_layers(
    k0: float,
    wavelength_bandwidth_pct: float,
    *,
    layer_count: int = EWALD_BANDWIDTH_LAYER_COUNT,
    min_opacity: float = EWALD_LAYER_MIN_OPACITY,
    max_opacity: float = EWALD_LAYER_MAX_OPACITY,
) -> tuple[EwaldLayer, ...]:
    """Return sampled wavelength-dependent Ewald layers around central ``k0``."""

    k_min, _ = ewald_bandwidth_k_bounds(k0, wavelength_bandwidth_pct)
    bandwidth_pct = normalize_wavelength_bandwidth_pct(wavelength_bandwidth_pct)
    min_opacity = float(min_opacity)
    max_opacity = float(max_opacity)
    if not math.isfinite(min_opacity) or not math.isfinite(max_opacity):
        raise ValueError("opacity bounds must be finite numbers")
    if min_opacity < 0.0 or max_opacity < min_opacity:
        raise ValueError("opacity bounds must satisfy 0.0 <= min_opacity <= max_opacity")

    if bandwidth_pct == 0.0:
        return (EwaldLayer(0.0, k_min, max_opacity),)

    count = int(layer_count)
    if count < 1:
        raise ValueError("layer_count must be at least 1")
    if count % 2 == 0:
        count += 1

    half_bandwidth = bandwidth_pct / 200.0
    layers: list[EwaldLayer] = []
    for offset in np.linspace(-half_bandwidth, half_bandwidth, count):
        relative_offset = float(offset)
        weight = 1.0 - abs(relative_offset) / half_bandwidth
        opacity = min_opacity + (max_opacity - min_opacity) * max(weight, 0.0)
        layers.append(
            EwaldLayer(
                relative_wavelength_offset=relative_offset,
                k_mag=k0 / (1.0 + relative_offset),
                opacity=opacity,
            )
        )
    return tuple(layers)


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


def intersection_circle(
    Rg: float,
    Re: float,
    d: float,
    *,
    npts: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordinates of the circle formed by two intersecting spheres.

    Parameters
    ----------
    Rg:
        Radius of the Bragg sphere.
    Re:
        Radius of the Ewald sphere.
    d:
        Distance between the centres of the two spheres.

    npts:
        Number of points used to sample the intersection circle.

    Returns
    -------
    tuple of ``numpy.ndarray``
        Arrays ``(x, y, z)`` forming the intersection circle in reciprocal
        space. Empty arrays are returned when the spheres do not intersect.
    """

    if d > Rg + Re or d < abs(Rg - Re):
        empty = np.array([], dtype=float)
        return empty, empty, empty

    y0 = (d * d - Re * Re + Rg * Rg) / (2 * d)
    r = math.sqrt(max(Rg * Rg - y0 * y0, 0.0))
    t = np.linspace(0.0, 2 * math.pi, npts)
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
