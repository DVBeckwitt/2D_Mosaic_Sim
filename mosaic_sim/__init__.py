"""Public API for :mod:`mosaic_sim`.

Importing from this package exposes the key helpers for constructing geometry,
intensity kernels and interactive Plotly figures used in the examples.
"""
from .constants import λ, a_hex, c_hex, K_MAG, d_hex
from .geometry import sphere, rot_x, intersection_circle, intersection_cylinder_sphere
from .intensity import cap_intensity, belt_intensity, mosaic_intensity
from .detector import build_detector_figure
from .animation import build_animation
from .cylinder import build_cylinder_figure

__all__ = [
    "λ", "a_hex", "c_hex", "K_MAG", "d_hex",
    "sphere", "rot_x", "intersection_circle", "intersection_cylinder_sphere",
    "cap_intensity", "belt_intensity", "mosaic_intensity",
    "build_detector_figure", "build_animation", "build_cylinder_figure",
]

