#!/usr/bin/env python3
"""Standalone beam -> sample -> detector specular reflection simulator.

This module keeps all geometry in one lab frame and intentionally models only
the pieces needed for a simplified specular-reflection experiment:

- a family of incoming rays with transverse width and divergence,
- a finite rectangular sample,
- explicit sample rotations ``theta_i``, ``delta``, ``alpha``, and ``psi``,
- explicit detector rotations ``beta``, ``gamma``, and ``chi``,
- line-plane intersections for the sample and detector, and
- mirror reflection from the current sample normal.

The implementation does not solve reciprocal-lattice or refraction physics.
It is a geometry-first simplification of the fuller diffraction model.

Rotation conventions
--------------------
The lab frame is right-handed:

- ``+y``: nominal incident-beam direction,
- ``+x``: in-plane transverse direction,
- ``+z``: nominal sample normal for a flat sample.

Sample rotations use column vectors with the following explicit order:

``R_sample = R_y(psi) @ R_z(alpha) @ R_x(delta) @ R_x(theta_i)``

So the ideal incidence setting happens first, followed by an extra substrate
pitch ``delta`` about lab ``x``, a yaw ``alpha`` about lab ``z``, and a roll
``psi`` about lab ``y``.

Detector rotations follow the order given in the prompt:

``R_tilt = R_z(gamma) @ R_x(beta)``

applied to the untilted detector basis, then ``chi`` is applied as an
in-plane rotation inside the detector plane.
"""

from __future__ import annotations

import argparse
import math
import tempfile
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


LAB_X = np.array([1.0, 0.0, 0.0], dtype=float)
LAB_Y = np.array([0.0, 1.0, 0.0], dtype=float)
LAB_Z = np.array([0.0, 0.0, 1.0], dtype=float)
EPSILON = 1e-12
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8051
SPECULAR_CAMERA_UIREVISION = "specular-camera"


@dataclass(frozen=True)
class BeamConfig:
    """Beam-family parameters."""

    ray_count: int = 600
    source_y: float = -150.0
    width_x: float = 0.15
    width_z: float = 0.15
    divergence_x_deg: float = 0.03
    divergence_z_deg: float = 0.03
    z_offset: float = 0.0
    seed: int = 7
    display_rays: int = 80


@dataclass(frozen=True)
class SampleConfig:
    """Finite sample geometry and rigid-body pose."""

    width: float = 20.0
    height: float = 80.0
    theta_i_deg: float = 10.0
    delta_deg: float = 0.0
    alpha_deg: float = 0.0
    psi_deg: float = 0.0
    z_offset: float = 0.0


@dataclass(frozen=True)
class DetectorConfig:
    """Finite detector plane and detector-frame calibration."""

    distance: float = 200.0
    width: float = 180.0
    height: float = 180.0
    beta_deg: float = 0.0
    gamma_deg: float = 0.0
    chi_deg: float = 0.0
    pixel_u: float = 0.1
    pixel_v: float = 0.1
    i0: float = 1024.0
    j0: float = 1024.0


@dataclass(frozen=True)
class MathLabel:
    """Math-style label with an optional subscript and suffix."""

    symbol: str
    subscript: str | None = None
    suffix: str = ""


@dataclass(frozen=True)
class ControlSpec:
    """GUI metadata for one numeric control."""

    name: str
    label: str | MathLabel
    config_group: str
    attr_name: str
    step: int | float
    min_value: int | float
    max_value: int | float


CONTROL_SECTIONS: tuple[tuple[str, tuple[ControlSpec, ...]], ...] = (
    (
        "Beam",
        (
            ControlSpec("rays", "Rays", "beam", "ray_count", 1, 1, 5000),
            ControlSpec("seed", "Seed", "beam", "seed", 1, 0, 999999),
            ControlSpec("display_rays", "Display rays", "beam", "display_rays", 1, 1, 400),
            ControlSpec("source_y", MathLabel("y", "src"), "beam", "source_y", 1.0, -1000.0, 1000.0),
            ControlSpec("beam_width_x", MathLabel("w", "x"), "beam", "width_x", 0.01, 0.0, 5.0),
            ControlSpec("beam_width_z", MathLabel("w", "z"), "beam", "width_z", 0.01, 0.0, 5.0),
            ControlSpec("divergence_x", MathLabel("Δθ", "x", " (deg)"), "beam", "divergence_x_deg", 0.01, 0.0, 5.0),
            ControlSpec("divergence_z", MathLabel("Δθ", "z", " (deg)"), "beam", "divergence_z_deg", 0.01, 0.0, 5.0),
            ControlSpec("z_beam", MathLabel("z", "B"), "beam", "z_offset", 0.01, -50.0, 50.0),
        ),
    ),
    (
        "Sample",
        (
            ControlSpec("sample_width", "W", "sample", "width", 0.5, 0.001, 400.0),
            ControlSpec("sample_height", "H", "sample", "height", 0.5, 0.001, 400.0),
            ControlSpec("theta_i", MathLabel("θ", "i", " (deg)"), "sample", "theta_i_deg", 0.1, -89.0, 89.0),
            ControlSpec("delta", "δ (deg)", "sample", "delta_deg", 0.05, -45.0, 45.0),
            ControlSpec("alpha", "α (deg)", "sample", "alpha_deg", 0.05, -45.0, 45.0),
            ControlSpec("psi", "ψ (deg)", "sample", "psi_deg", 0.05, -45.0, 45.0),
            ControlSpec("z_sample", MathLabel("z", "S"), "sample", "z_offset", 0.01, -50.0, 50.0),
        ),
    ),
    (
        "Detector",
        (
            ControlSpec("distance", "D", "detector", "distance", 1.0, 0.001, 1000.0),
            ControlSpec("detector_width", MathLabel("W", "D"), "detector", "width", 0.5, 0.001, 1000.0),
            ControlSpec("detector_height", MathLabel("H", "D"), "detector", "height", 0.5, 0.001, 1000.0),
            ControlSpec("beta", "β (deg)", "detector", "beta_deg", 0.05, -45.0, 45.0),
            ControlSpec("gamma", "γ (deg)", "detector", "gamma_deg", 0.05, -45.0, 45.0),
            ControlSpec("chi", "χ (deg)", "detector", "chi_deg", 0.05, -180.0, 180.0),
            ControlSpec("pixel_u", MathLabel("p", "u"), "detector", "pixel_u", 0.01, 0.000001, 5.0),
            ControlSpec("pixel_v", MathLabel("p", "v"), "detector", "pixel_v", 0.01, 0.000001, 5.0),
            ControlSpec("i0", MathLabel("i", "0"), "detector", "i0", 1.0, 0.0, 4096.0),
            ControlSpec("j0", MathLabel("j", "0"), "detector", "j0", 1.0, 0.0, 4096.0),
        ),
    ),
)


@dataclass(frozen=True)
class Frame3D:
    """Orthonormal frame embedded in the lab frame."""

    origin: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    normal: np.ndarray


@dataclass(frozen=True)
class RayBundle:
    """Incident beam rays."""

    origins: np.ndarray
    directions: np.ndarray


@dataclass(frozen=True)
class SimulationResult:
    """Outputs from the specular ray trace."""

    beam: RayBundle
    sample: Frame3D
    detector: Frame3D
    sample_hit_indices: np.ndarray
    hit_points: np.ndarray
    reflected_dirs: np.ndarray
    sample_uv: np.ndarray
    plane_hit_indices: np.ndarray
    plane_points: np.ndarray
    plane_uv: np.ndarray
    plane_pixels: np.ndarray
    active_detector_mask: np.ndarray
    direct_beam_point: np.ndarray | None
    direct_beam_uv: np.ndarray | None
    direct_beam_pixels: np.ndarray | None

    @property
    def sample_hit_count(self) -> int:
        return int(self.sample_hit_indices.size)

    @property
    def detector_plane_hit_count(self) -> int:
        return int(self.plane_hit_indices.size)

    @property
    def detector_hit_count(self) -> int:
        return int(np.count_nonzero(self.active_detector_mask))

    @property
    def detector_points(self) -> np.ndarray:
        return self.plane_points[self.active_detector_mask]

    @property
    def detector_uv(self) -> np.ndarray:
        return self.plane_uv[self.active_detector_mask]

    @property
    def detector_pixels(self) -> np.ndarray:
        return self.plane_pixels[self.active_detector_mask]


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Return a unit vector."""

    arr = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= EPSILON:
        raise ValueError("Cannot normalize a zero-length vector")
    return arr / norm


def normalize_rows(array: np.ndarray) -> np.ndarray:
    """Normalize each row of a 2-D array."""

    arr = np.asarray(array, dtype=float)
    if arr.ndim != 2:
        raise ValueError("normalize_rows expects a 2-D array")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms <= EPSILON):
        raise ValueError("Cannot normalize one or more zero-length rows")
    return arr / norms


def rotation_x(angle_rad: float) -> np.ndarray:
    """Right-handed rotation about lab x."""

    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=float,
    )


def rotation_y(angle_rad: float) -> np.ndarray:
    """Right-handed rotation about lab y."""

    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=float,
    )


def rotation_z(angle_rad: float) -> np.ndarray:
    """Right-handed rotation about lab z."""

    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def validate_configs(
    beam: BeamConfig,
    sample: SampleConfig,
    detector: DetectorConfig,
) -> None:
    """Validate basic geometry inputs."""

    if beam.ray_count < 1:
        raise ValueError("beam.ray_count must be at least 1")
    if beam.display_rays < 1:
        raise ValueError("beam.display_rays must be at least 1")
    if sample.width <= 0.0 or sample.height <= 0.0:
        raise ValueError("sample.width and sample.height must be positive")
    if detector.distance <= 0.0:
        raise ValueError("detector.distance must be positive")
    if detector.width <= 0.0 or detector.height <= 0.0:
        raise ValueError("detector.width and detector.height must be positive")
    if detector.pixel_u <= 0.0 or detector.pixel_v <= 0.0:
        raise ValueError("detector pixel sizes must be positive")


def coerce_value(value: Any, default: int | float, cast: type[int] | type[float]) -> int | float:
    """Return a validated numeric value, falling back to ``default`` for ``None``."""

    if value is None:
        return default
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def configs_from_values(
    *,
    rays: Any = None,
    seed: Any = None,
    display_rays: Any = None,
    source_y: Any = None,
    beam_width_x: Any = None,
    beam_width_z: Any = None,
    divergence_x: Any = None,
    divergence_z: Any = None,
    z_beam: Any = None,
    sample_width: Any = None,
    sample_height: Any = None,
    theta_i: Any = None,
    delta: Any = None,
    alpha: Any = None,
    psi: Any = None,
    z_sample: Any = None,
    distance: Any = None,
    detector_width: Any = None,
    detector_height: Any = None,
    beta: Any = None,
    gamma: Any = None,
    chi: Any = None,
    pixel_u: Any = None,
    pixel_v: Any = None,
    i0: Any = None,
    j0: Any = None,
    default_beam: BeamConfig | None = None,
    default_sample: SampleConfig | None = None,
    default_detector: DetectorConfig | None = None,
) -> tuple[BeamConfig, SampleConfig, DetectorConfig]:
    """Build strongly typed configs from CLI or GUI values."""

    beam_defaults = default_beam or BeamConfig()
    sample_defaults = default_sample or SampleConfig()
    detector_defaults = default_detector or DetectorConfig()

    beam = BeamConfig(
        ray_count=int(coerce_value(rays, beam_defaults.ray_count, int)),
        source_y=float(coerce_value(source_y, beam_defaults.source_y, float)),
        width_x=float(coerce_value(beam_width_x, beam_defaults.width_x, float)),
        width_z=float(coerce_value(beam_width_z, beam_defaults.width_z, float)),
        divergence_x_deg=float(
            coerce_value(divergence_x, beam_defaults.divergence_x_deg, float)
        ),
        divergence_z_deg=float(
            coerce_value(divergence_z, beam_defaults.divergence_z_deg, float)
        ),
        z_offset=float(coerce_value(z_beam, beam_defaults.z_offset, float)),
        seed=int(coerce_value(seed, beam_defaults.seed, int)),
        display_rays=int(coerce_value(display_rays, beam_defaults.display_rays, int)),
    )
    sample = SampleConfig(
        width=float(coerce_value(sample_width, sample_defaults.width, float)),
        height=float(coerce_value(sample_height, sample_defaults.height, float)),
        theta_i_deg=float(coerce_value(theta_i, sample_defaults.theta_i_deg, float)),
        delta_deg=float(coerce_value(delta, sample_defaults.delta_deg, float)),
        alpha_deg=float(coerce_value(alpha, sample_defaults.alpha_deg, float)),
        psi_deg=float(coerce_value(psi, sample_defaults.psi_deg, float)),
        z_offset=float(coerce_value(z_sample, sample_defaults.z_offset, float)),
    )
    detector = DetectorConfig(
        distance=float(coerce_value(distance, detector_defaults.distance, float)),
        width=float(coerce_value(detector_width, detector_defaults.width, float)),
        height=float(coerce_value(detector_height, detector_defaults.height, float)),
        beta_deg=float(coerce_value(beta, detector_defaults.beta_deg, float)),
        gamma_deg=float(coerce_value(gamma, detector_defaults.gamma_deg, float)),
        chi_deg=float(coerce_value(chi, detector_defaults.chi_deg, float)),
        pixel_u=float(coerce_value(pixel_u, detector_defaults.pixel_u, float)),
        pixel_v=float(coerce_value(pixel_v, detector_defaults.pixel_v, float)),
        i0=float(coerce_value(i0, detector_defaults.i0, float)),
        j0=float(coerce_value(j0, detector_defaults.j0, float)),
    )
    validate_configs(beam, sample, detector)
    return beam, sample, detector


def extract_scene_camera(relayout_data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a Plotly scene camera extracted from Dash relayout data."""

    if not isinstance(relayout_data, dict):
        return None

    nested_camera = relayout_data.get("scene.camera")
    if isinstance(nested_camera, dict):
        return dict(nested_camera)

    camera: dict[str, Any] = {}
    for component_name in ("center", "eye", "up"):
        component_value = relayout_data.get(f"scene.camera.{component_name}")
        component = dict(component_value) if isinstance(component_value, dict) else {}
        for axis_name in ("x", "y", "z"):
            axis_value = relayout_data.get(f"scene.camera.{component_name}.{axis_name}")
            if axis_value is not None:
                component[axis_name] = float(axis_value)
        if component:
            camera[component_name] = component

    projection_value = relayout_data.get("scene.camera.projection")
    projection = dict(projection_value) if isinstance(projection_value, dict) else {}
    projection_type = relayout_data.get("scene.camera.projection.type")
    if projection_type is not None:
        projection["type"] = str(projection_type)
    if projection:
        camera["projection"] = projection

    return camera or None


def build_specular_error_figure(message: str) -> go.Figure:
    """Return a compact figure surfacing invalid GUI inputs."""

    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=18, color="crimson"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        title="Specular Beam-Sample-Detector Simulation",
        margin=dict(l=40, r=40, b=40, t=80),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def sample_rotation_matrix(config: SampleConfig) -> np.ndarray:
    """Return the explicit sample rotation matrix."""

    theta_i = math.radians(config.theta_i_deg)
    delta = math.radians(config.delta_deg)
    alpha = math.radians(config.alpha_deg)
    psi = math.radians(config.psi_deg)
    return rotation_y(psi) @ rotation_z(alpha) @ rotation_x(delta) @ rotation_x(theta_i)


def build_sample_frame(config: SampleConfig) -> Frame3D:
    """Return the sample frame embedded in the lab frame."""

    rotation = sample_rotation_matrix(config)
    axis_u = normalize_vector(rotation @ LAB_X)
    axis_v = normalize_vector(rotation @ LAB_Y)
    normal = normalize_vector(rotation @ LAB_Z)
    return Frame3D(
        origin=np.array([0.0, 0.0, config.z_offset], dtype=float),
        axis_u=axis_u,
        axis_v=axis_v,
        normal=normal,
    )


def build_detector_frame(config: DetectorConfig) -> Frame3D:
    """Return the detector frame embedded in the lab frame."""

    center = np.array([0.0, config.distance, 0.0], dtype=float)
    tilt = rotation_z(math.radians(config.gamma_deg)) @ rotation_x(
        math.radians(config.beta_deg)
    )
    u1 = tilt @ LAB_X
    v1 = tilt @ LAB_Z
    chi = math.radians(config.chi_deg)
    axis_u = normalize_vector(math.cos(chi) * u1 + math.sin(chi) * v1)
    axis_v = normalize_vector(-math.sin(chi) * u1 + math.cos(chi) * v1)
    normal = normalize_vector(np.cross(axis_u, axis_v))
    return Frame3D(origin=center, axis_u=axis_u, axis_v=axis_v, normal=normal)


def generate_beam_rays(config: BeamConfig) -> RayBundle:
    """Generate the incoming beam as a family of rays."""

    rng = np.random.default_rng(config.seed)
    x_offsets = rng.normal(0.0, config.width_x, config.ray_count)
    z_offsets = config.z_offset + rng.normal(0.0, config.width_z, config.ray_count)
    origins = np.column_stack(
        [
            x_offsets,
            np.full(config.ray_count, config.source_y, dtype=float),
            z_offsets,
        ]
    )

    eps_x = rng.normal(0.0, math.radians(config.divergence_x_deg), config.ray_count)
    eps_z = rng.normal(0.0, math.radians(config.divergence_z_deg), config.ray_count)
    directions = normalize_rows(np.column_stack([eps_x, np.ones(config.ray_count), eps_z]))
    return RayBundle(origins=origins, directions=directions)


def point_plane_coordinates(point: np.ndarray, frame: Frame3D) -> np.ndarray:
    """Return the local in-plane coordinates of a lab-frame point."""

    rel = np.asarray(point, dtype=float) - frame.origin
    return np.array(
        [float(np.dot(rel, frame.axis_u)), float(np.dot(rel, frame.axis_v))],
        dtype=float,
    )


def detector_pixels_from_uv(uv: np.ndarray, detector: DetectorConfig) -> np.ndarray:
    """Convert detector-plane coordinates to detector pixels."""

    u = float(uv[0])
    v = float(uv[1])
    return np.array(
        [detector.i0 + u / detector.pixel_u, detector.j0 - v / detector.pixel_v],
        dtype=float,
    )


def intersect_ray_with_plane(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray | None:
    """Intersect a single ray with a plane, returning ``None`` if invalid."""

    origin_arr = np.asarray(origin, dtype=float)
    direction_arr = normalize_vector(direction)
    plane_origin_arr = np.asarray(plane_origin, dtype=float)
    plane_normal_arr = normalize_vector(plane_normal)

    denominator = float(np.dot(plane_normal_arr, direction_arr))
    if abs(denominator) <= EPSILON:
        return None

    distance = float(np.dot(plane_normal_arr, plane_origin_arr - origin_arr) / denominator)
    if distance <= 0.0:
        return None

    return origin_arr + distance * direction_arr


def project_ray_to_detector(
    origin: np.ndarray,
    direction: np.ndarray,
    detector_frame: Frame3D,
    detector_config: DetectorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Project a single ray onto the detector plane."""

    point = intersect_ray_with_plane(
        origin,
        direction,
        detector_frame.origin,
        detector_frame.normal,
    )
    if point is None:
        return None

    uv = point_plane_coordinates(point, detector_frame)
    pixels = detector_pixels_from_uv(uv, detector_config)
    return point, uv, pixels


def reflect_directions(directions: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Mirror incoming directions about a plane normal."""

    incoming = normalize_rows(directions)
    surface_normal = normalize_vector(normal)
    dot_terms = incoming @ surface_normal
    reflected = incoming - 2.0 * dot_terms[:, None] * surface_normal[None, :]
    return normalize_rows(reflected)


def nominal_specular_direction(sample_frame: Frame3D) -> np.ndarray:
    """Return the nominal specular direction for a perfect incident beam."""

    return reflect_directions(LAB_Y[None, :], sample_frame.normal)[0]


def trace_specular_simulation(
    beam_config: BeamConfig | None = None,
    sample_config: SampleConfig | None = None,
    detector_config: DetectorConfig | None = None,
) -> SimulationResult:
    """Trace the beam family from source to sample to detector."""

    beam = beam_config or BeamConfig()
    sample = sample_config or SampleConfig()
    detector = detector_config or DetectorConfig()
    validate_configs(beam, sample, detector)

    rays = generate_beam_rays(beam)
    sample_frame = build_sample_frame(sample)
    detector_frame = build_detector_frame(detector)

    sample_denominator = rays.directions @ sample_frame.normal
    sample_numerator = (sample_frame.origin - rays.origins) @ sample_frame.normal
    sample_intersection_mask = np.abs(sample_denominator) > EPSILON

    sample_distance = np.empty_like(sample_denominator)
    sample_distance.fill(np.nan)
    sample_distance[sample_intersection_mask] = (
        sample_numerator[sample_intersection_mask]
        / sample_denominator[sample_intersection_mask]
    )
    sample_intersection_mask &= sample_distance > 0.0

    candidate_indices = np.flatnonzero(sample_intersection_mask)
    candidate_points = (
        rays.origins[candidate_indices]
        + sample_distance[candidate_indices, None] * rays.directions[candidate_indices]
    )
    rel_sample = candidate_points - sample_frame.origin
    sample_u = rel_sample @ sample_frame.axis_u
    sample_v = rel_sample @ sample_frame.axis_v
    inside_sample = (
        (np.abs(sample_u) <= 0.5 * sample.width)
        & (np.abs(sample_v) <= 0.5 * sample.height)
    )

    sample_hit_indices = candidate_indices[inside_sample]
    hit_points = candidate_points[inside_sample]
    sample_uv = np.column_stack([sample_u[inside_sample], sample_v[inside_sample]])
    reflected_dirs = reflect_directions(
        rays.directions[sample_hit_indices],
        sample_frame.normal,
    )

    detector_denominator = reflected_dirs @ detector_frame.normal
    detector_numerator = (detector_frame.origin - hit_points) @ detector_frame.normal
    plane_hit_mask = np.abs(detector_denominator) > EPSILON

    plane_distance = np.empty_like(detector_denominator)
    plane_distance.fill(np.nan)
    plane_distance[plane_hit_mask] = (
        detector_numerator[plane_hit_mask] / detector_denominator[plane_hit_mask]
    )
    plane_hit_mask &= plane_distance > 0.0

    plane_hit_indices = np.flatnonzero(plane_hit_mask)
    plane_points = (
        hit_points[plane_hit_indices]
        + plane_distance[plane_hit_indices, None] * reflected_dirs[plane_hit_indices]
    )
    rel_detector = plane_points - detector_frame.origin
    plane_u = rel_detector @ detector_frame.axis_u
    plane_v = rel_detector @ detector_frame.axis_v
    plane_uv = np.column_stack([plane_u, plane_v])
    plane_pixels = np.column_stack(
        [
            detector.i0 + plane_u / detector.pixel_u,
            detector.j0 - plane_v / detector.pixel_v,
        ]
    )
    active_detector_mask = (
        (np.abs(plane_u) <= 0.5 * detector.width)
        & (np.abs(plane_v) <= 0.5 * detector.height)
    )

    direct_beam_projection = project_ray_to_detector(
        np.zeros(3, dtype=float),
        LAB_Y,
        detector_frame,
        detector,
    )
    if direct_beam_projection is None:
        direct_beam_point = None
        direct_beam_uv = None
        direct_beam_pixels = None
    else:
        direct_beam_point, direct_beam_uv, direct_beam_pixels = direct_beam_projection

    return SimulationResult(
        beam=rays,
        sample=sample_frame,
        detector=detector_frame,
        sample_hit_indices=sample_hit_indices,
        hit_points=hit_points,
        reflected_dirs=reflected_dirs,
        sample_uv=sample_uv,
        plane_hit_indices=plane_hit_indices,
        plane_points=plane_points,
        plane_uv=plane_uv,
        plane_pixels=plane_pixels,
        active_detector_mask=active_detector_mask,
        direct_beam_point=direct_beam_point,
        direct_beam_uv=direct_beam_uv,
        direct_beam_pixels=direct_beam_pixels,
    )


def plane_patch(
    frame: Frame3D,
    half_u: float,
    half_v: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a 2x2 surface patch for a plane."""

    p00 = frame.origin - half_u * frame.axis_u - half_v * frame.axis_v
    p10 = frame.origin + half_u * frame.axis_u - half_v * frame.axis_v
    p01 = frame.origin - half_u * frame.axis_u + half_v * frame.axis_v
    p11 = frame.origin + half_u * frame.axis_u + half_v * frame.axis_v

    x = np.array([[p00[0], p10[0]], [p01[0], p11[0]]], dtype=float)
    y = np.array([[p00[1], p10[1]], [p01[1], p11[1]]], dtype=float)
    z = np.array([[p00[2], p10[2]], [p01[2], p11[2]]], dtype=float)
    return x, y, z


def plane_outline(frame: Frame3D, half_u: float, half_v: float) -> np.ndarray:
    """Return a closed rectangular plane outline."""

    return np.array(
        [
            frame.origin - half_u * frame.axis_u - half_v * frame.axis_v,
            frame.origin + half_u * frame.axis_u - half_v * frame.axis_v,
            frame.origin + half_u * frame.axis_u + half_v * frame.axis_v,
            frame.origin - half_u * frame.axis_u + half_v * frame.axis_v,
            frame.origin - half_u * frame.axis_u - half_v * frame.axis_v,
        ],
        dtype=float,
    )


def segmented_line_points(
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack many line segments into NaN-separated coordinate arrays."""

    if starts.size == 0 or ends.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    x_values: list[float] = []
    y_values: list[float] = []
    z_values: list[float] = []
    for start, end in zip(starts, ends, strict=True):
        x_values.extend([float(start[0]), float(end[0]), math.nan])
        y_values.extend([float(start[1]), float(end[1]), math.nan])
        z_values.extend([float(start[2]), float(end[2]), math.nan])
    return (
        np.array(x_values, dtype=float),
        np.array(y_values, dtype=float),
        np.array(z_values, dtype=float),
    )


def choose_display_indices(count: int, max_count: int) -> np.ndarray:
    """Evenly subsample indices for plotting."""

    if count <= 0:
        return np.array([], dtype=int)
    if count <= max_count:
        return np.arange(count, dtype=int)
    return np.linspace(0, count - 1, max_count, dtype=int)


def nominal_scattering_angle_deg(sample_frame: Frame3D) -> float:
    """Return the scattering angle between the nominal reflected ray and +y."""

    direction = nominal_specular_direction(sample_frame)
    cos_angle = float(np.clip(np.dot(direction, LAB_Y), -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def build_specular_figure(
    result: SimulationResult,
    beam_config: BeamConfig,
    sample_config: SampleConfig,
    detector_config: DetectorConfig,
    *,
    camera: dict[str, Any] | None = None,
) -> go.Figure:
    """Build a 3-panel Plotly figure for the specular ray trace."""

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "xy"}, {"type": "xy"}]],
        column_widths=[0.56, 0.22, 0.22],
        subplot_titles=("Lab Geometry", "Sample Footprint", "Detector Plane"),
    )

    sample_half_u = 0.5 * sample_config.width
    sample_half_v = 0.5 * sample_config.height
    detector_half_u = 0.5 * detector_config.width
    detector_half_v = 0.5 * detector_config.height

    sample_x, sample_y, sample_z = plane_patch(result.sample, sample_half_u, sample_half_v)
    detector_x, detector_y, detector_z = plane_patch(
        result.detector,
        detector_half_u,
        detector_half_v,
    )
    sample_outline = plane_outline(result.sample, sample_half_u, sample_half_v)
    detector_outline = plane_outline(result.detector, detector_half_u, detector_half_v)

    fig.add_trace(
        go.Surface(
            x=sample_x,
            y=sample_y,
            z=sample_z,
            opacity=0.72,
            showscale=False,
            colorscale=[[0.0, "#4f7cac"], [1.0, "#4f7cac"]],
            name="Sample",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=sample_outline[:, 0],
            y=sample_outline[:, 1],
            z=sample_outline[:, 2],
            mode="lines",
            line=dict(color="#1d3557", width=5),
            name="Sample edge",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Surface(
            x=detector_x,
            y=detector_y,
            z=detector_z,
            opacity=0.38,
            showscale=False,
            colorscale=[[0.0, "#f4a261"], [1.0, "#f4a261"]],
            name="Detector",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=detector_outline[:, 0],
            y=detector_outline[:, 1],
            z=detector_outline[:, 2],
            mode="lines",
            line=dict(color="#bc6c25", width=5),
            name="Detector edge",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    plane_display = choose_display_indices(
        result.detector_plane_hit_count,
        beam_config.display_rays,
    )
    if plane_display.size:
        sample_display = result.plane_hit_indices[plane_display]
        incident_starts = result.beam.origins[result.sample_hit_indices[sample_display]]
        incident_ends = result.hit_points[sample_display]
        reflected_starts = result.hit_points[sample_display]
        reflected_ends = result.plane_points[plane_display]

        inc_x, inc_y, inc_z = segmented_line_points(incident_starts, incident_ends)
        ref_x, ref_y, ref_z = segmented_line_points(reflected_starts, reflected_ends)

        fig.add_trace(
            go.Scatter3d(
                x=inc_x,
                y=inc_y,
                z=inc_z,
                mode="lines",
                line=dict(color="#3a86ff", width=5),
                name="Incident rays",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=ref_x,
                y=ref_y,
                z=ref_z,
                mode="lines",
                line=dict(color="#d62828", width=5),
                name="Specular rays",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    if result.sample_hit_count:
        fig.add_trace(
            go.Scatter3d(
                x=result.hit_points[:, 0],
                y=result.hit_points[:, 1],
                z=result.hit_points[:, 2],
                mode="markers",
                marker=dict(size=4, color="#111111"),
                name="Sample hits",
                hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if result.detector_plane_hit_count:
        hit_colors = np.where(result.active_detector_mask, "#d62828", "#bdbdbd")
        fig.add_trace(
            go.Scatter3d(
                x=result.plane_points[:, 0],
                y=result.plane_points[:, 1],
                z=result.plane_points[:, 2],
                mode="markers",
                marker=dict(size=4, color=hit_colors),
                name="Detector intersections",
                hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    normal_scale = 0.18 * max(
        sample_config.width,
        sample_config.height,
        detector_config.width,
    )
    sample_normal_end = result.sample.origin + normal_scale * result.sample.normal
    detector_normal_end = result.detector.origin + normal_scale * result.detector.normal

    for start, end, color, name in (
        (result.sample.origin, sample_normal_end, "#1d3557", "Sample normal"),
        (result.detector.origin, detector_normal_end, "#bc6c25", "Detector normal"),
    ):
        fig.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line=dict(color=color, width=8),
                name=name,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    axis_length = max(
        abs(beam_config.source_y),
        detector_config.distance,
        sample_config.height,
    ) * 0.28
    for label, axis, color in (
        ("x", LAB_X, "#457b9d"),
        ("y", LAB_Y, "#2a9d8f"),
        ("z", LAB_Z, "#e76f51"),
    ):
        endpoint = axis_length * axis
        fig.add_trace(
            go.Scatter3d(
                x=[0.0, endpoint[0]],
                y=[0.0, endpoint[1]],
                z=[0.0, endpoint[2]],
                mode="lines+text",
                text=[None, label],
                textposition="top center",
                line=dict(color=color, width=7),
                name=f"lab {label}",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    if result.sample_hit_count:
        fig.add_trace(
            go.Scatter(
                x=result.sample_uv[:, 0],
                y=result.sample_uv[:, 1],
                mode="markers",
                marker=dict(size=7, color="#111111", opacity=0.75),
                name="Sample footprint",
                hovertemplate="u_s=%{x:.3f}<br>v_s=%{y:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    footprint_x = np.array(
        [-sample_half_u, sample_half_u, sample_half_u, -sample_half_u, -sample_half_u],
        dtype=float,
    )
    footprint_y = np.array(
        [-sample_half_v, -sample_half_v, sample_half_v, sample_half_v, -sample_half_v],
        dtype=float,
    )
    fig.add_trace(
        go.Scatter(
            x=footprint_x,
            y=footprint_y,
            mode="lines",
            line=dict(color="#1d3557", width=3),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    if result.detector_plane_hit_count:
        inactive = ~result.active_detector_mask
        if np.any(inactive):
            fig.add_trace(
                go.Scatter(
                    x=result.plane_uv[inactive, 0],
                    y=result.plane_uv[inactive, 1],
                    mode="markers",
                    marker=dict(size=7, color="#bdbdbd", opacity=0.6),
                    name="Off-detector intersections",
                    hovertemplate=(
                        "u=%{x:.3f}<br>v=%{y:.3f}<br>i=%{customdata[0]:.2f}"
                        "<br>j=%{customdata[1]:.2f}<extra></extra>"
                    ),
                    customdata=result.plane_pixels[inactive],
                    showlegend=False,
                ),
                row=1,
                col=3,
            )

        if result.detector_hit_count:
            fig.add_trace(
                go.Scatter(
                    x=result.detector_uv[:, 0],
                    y=result.detector_uv[:, 1],
                    mode="markers",
                    marker=dict(size=8, color="#d62828", opacity=0.85),
                    name="Detector hits",
                    hovertemplate=(
                        "u=%{x:.3f}<br>v=%{y:.3f}<br>i=%{customdata[0]:.2f}"
                        "<br>j=%{customdata[1]:.2f}<extra></extra>"
                    ),
                    customdata=result.detector_pixels,
                    showlegend=False,
                ),
                row=1,
                col=3,
            )

    detector_outline_u = np.array(
        [
            -detector_half_u,
            detector_half_u,
            detector_half_u,
            -detector_half_u,
            -detector_half_u,
        ],
        dtype=float,
    )
    detector_outline_v = np.array(
        [
            -detector_half_v,
            -detector_half_v,
            detector_half_v,
            detector_half_v,
            -detector_half_v,
        ],
        dtype=float,
    )
    fig.add_trace(
        go.Scatter(
            x=detector_outline_u,
            y=detector_outline_v,
            mode="lines",
            line=dict(color="#bc6c25", width=3),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    if result.direct_beam_uv is not None:
        fig.add_trace(
            go.Scatter(
                x=[result.direct_beam_uv[0]],
                y=[result.direct_beam_uv[1]],
                mode="markers",
                marker=dict(size=11, color="#3a86ff", symbol="x"),
                name="Direct beam center",
                hovertemplate=(
                    "direct beam<br>u=%{x:.3f}<br>v=%{y:.3f}<br>"
                    f"i={result.direct_beam_pixels[0]:.2f}<br>"
                    f"j={result.direct_beam_pixels[1]:.2f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=3,
        )

    nominal_two_theta = nominal_scattering_angle_deg(result.sample)
    title = (
        "Specular Beam-Sample-Detector Simulation"
        f" | sample hits {result.sample_hit_count}/{beam_config.ray_count}"
        f" | detector hits {result.detector_hit_count}"
        f" | nominal 2θ = {nominal_two_theta:.2f}°"
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=90, b=20),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        uirevision=SPECULAR_CAMERA_UIREVISION,
    )
    fig.update_scenes(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="z",
        aspectmode="data",
        bgcolor="rgba(0,0,0,0)",
        camera=dict(eye=dict(x=1.5, y=1.25, z=0.9)),
        uirevision=SPECULAR_CAMERA_UIREVISION,
        row=1,
        col=1,
    )
    fig.update_xaxes(title="u_s", scaleanchor="y", row=1, col=2)
    fig.update_yaxes(title="v_s", row=1, col=2)
    fig.update_xaxes(title="u", scaleanchor="y", row=1, col=3)
    fig.update_yaxes(title="v", row=1, col=3)

    fig.add_annotation(
        text=(
            f"Sample rotations: θᵢ={sample_config.theta_i_deg:.2f}°, "
            f"δ={sample_config.delta_deg:.2f}°, "
            f"α={sample_config.alpha_deg:.2f}°, "
            f"ψ={sample_config.psi_deg:.2f}°<br>"
            f"Detector rotations: β={detector_config.beta_deg:.2f}°, "
            f"γ={detector_config.gamma_deg:.2f}°, "
            f"χ={detector_config.chi_deg:.2f}°"
        ),
        x=0.5,
        y=1.05,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=12),
    )

    if camera:
        fig.update_layout(scene_camera=camera)

    return fig


def write_figure_html(fig: go.Figure, output_html: str | None = None) -> Path:
    """Write the figure to an HTML file and return the path."""

    if output_html is None:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".html",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(fig.to_html(include_plotlyjs="cdn", full_html=True))
            return Path(handle.name)

    path = Path(output_html).resolve()
    path.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True), encoding="utf-8")
    return path


def simulation_summary(
    result: SimulationResult,
    sample_config: SampleConfig,
    detector_config: DetectorConfig,
) -> str:
    """Return a compact text summary for CLI use."""

    nominal_direction = nominal_specular_direction(result.sample)
    nominal_two_theta = nominal_scattering_angle_deg(result.sample)

    if result.detector_hit_count:
        detector_centroid = np.mean(result.detector_uv, axis=0)
        pixel_centroid = np.mean(result.detector_pixels, axis=0)
        centroid_text = (
            f"detector centroid (u, v) = ({detector_centroid[0]:.4f}, "
            f"{detector_centroid[1]:.4f}), "
            f"pixel centroid (i, j) = ({pixel_centroid[0]:.2f}, "
            f"{pixel_centroid[1]:.2f})"
        )
    else:
        centroid_text = "detector centroid unavailable (no active detector hits)"

    direct_beam_text = "direct beam center unavailable"
    if result.direct_beam_uv is not None and result.direct_beam_pixels is not None:
        direct_beam_text = (
            "direct beam detector center "
            f"(u, v) = ({result.direct_beam_uv[0]:.4f}, {result.direct_beam_uv[1]:.4f}), "
            f"pixels = ({result.direct_beam_pixels[0]:.2f}, {result.direct_beam_pixels[1]:.2f})"
        )

    return "\n".join(
        [
            "Specular reflection summary",
            f"  sample hits: {result.sample_hit_count}/{result.beam.origins.shape[0]}",
            f"  detector-plane intersections: {result.detector_plane_hit_count}",
            f"  active detector hits: {result.detector_hit_count}",
            "  nominal sample normal: "
            f"({result.sample.normal[0]:.6f}, {result.sample.normal[1]:.6f}, {result.sample.normal[2]:.6f})",
            "  nominal reflected direction: "
            f"({nominal_direction[0]:.6f}, {nominal_direction[1]:.6f}, {nominal_direction[2]:.6f})",
            f"  nominal 2θ from +y: {nominal_two_theta:.6f} deg",
            f"  sample size (W x H): {sample_config.width:.3f} x {sample_config.height:.3f}",
            f"  detector size (W x H): {detector_config.width:.3f} x {detector_config.height:.3f}",
            f"  {centroid_text}",
            f"  {direct_beam_text}",
        ]
    )


def build_specular_outputs(
    beam_config: BeamConfig,
    sample_config: SampleConfig,
    detector_config: DetectorConfig,
    *,
    camera: dict[str, Any] | None = None,
) -> tuple[go.Figure, str]:
    """Return the live figure and summary text for the current configs."""

    result = trace_specular_simulation(beam_config, sample_config, detector_config)
    figure = build_specular_figure(
        result,
        beam_config,
        sample_config,
        detector_config,
        camera=camera,
    )
    summary = simulation_summary(result, sample_config, detector_config)
    return figure, summary


def build_number_control(
    control_name: str,
    label: str | MathLabel,
    value: int | float,
    *,
    step: int | float,
    min_value: int | float,
    max_value: int | float,
):
    """Return a compact slider plus numeric-entry control."""

    from dash import dcc, html

    slider_id = {"type": "specular-slider", "name": control_name}
    input_id = {"type": "specular-input", "name": control_name}
    if isinstance(label, MathLabel):
        label_content: Any = html.Span(
            [
                label.symbol,
                html.Sub(label.subscript) if label.subscript is not None else None,
                label.suffix,
            ]
        )
    else:
        label_content = label

    return html.Div(
        [
            html.Label(label_content, htmlFor=str(slider_id), style={"fontWeight": 600}),
            html.Div(
                [
                    dcc.Slider(
                        id=slider_id,
                        min=min_value,
                        max=max_value,
                        step=step,
                        value=value,
                        marks=None,
                        included=False,
                        updatemode="drag",
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    dcc.Input(
                        id=input_id,
                        type="number",
                        value=value,
                        step=step,
                        min=min_value,
                        max=max_value,
                        debounce=False,
                        style={"width": "64px", "minWidth": "64px"},
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "minmax(0, 1fr) 64px",
                    "gap": "0.6rem",
                    "alignItems": "center",
                },
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "0.45rem",
            "minWidth": "130px",
        },
    )


def build_specular_app(
    initial_beam: BeamConfig | None = None,
    initial_sample: SampleConfig | None = None,
    initial_detector: DetectorConfig | None = None,
):
    """Return a Dash app exposing all beam, sample, and detector parameters."""

    import dash
    from dash import ALL, MATCH, ctx, dcc, html
    from dash.dependencies import Input, Output, State
    from dash.exceptions import PreventUpdate

    beam_defaults = initial_beam or BeamConfig()
    sample_defaults = initial_sample or SampleConfig()
    detector_defaults = initial_detector or DetectorConfig()
    initial_figure, initial_summary = build_specular_outputs(
        beam_defaults,
        sample_defaults,
        detector_defaults,
    )

    app = dash.Dash(__name__)
    app.title = "Specular Reflection Simulator"

    config_by_group = {
        "beam": beam_defaults,
        "sample": sample_defaults,
        "detector": detector_defaults,
    }

    def section(title: str, children: list[Any]):
        return html.Div(
            [
                html.H3(title, style={"margin": "0 0 0.75rem 0"}),
                html.Div(
                    children,
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(135px, 1fr))",
                        "gap": "0.75rem",
                    },
                ),
            ],
            style={
                "border": "1px solid #d8dee9",
                "borderRadius": "10px",
                "padding": "0.9rem",
                "backgroundColor": "#fbfcfd",
            },
        )

    def controls_for_section(specs: tuple[ControlSpec, ...]) -> list[Any]:
        controls: list[Any] = []
        for spec in specs:
            config_obj = config_by_group[spec.config_group]
            controls.append(
                build_number_control(
                    spec.name,
                    spec.label,
                    getattr(config_obj, spec.attr_name),
                    step=spec.step,
                    min_value=spec.min_value,
                    max_value=spec.max_value,
                )
            )
        return controls

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H2("Specular Beam-Sample-Detector Simulation", style={"margin": "0"}),
                    html.Div(
                        "All beam, sample, and detector parameters update the ray trace live in one lab frame.",
                        style={"color": "#4a5568"},
                    ),
                ],
                style={"paddingBottom": "0.9rem"},
            ),
            html.Div(
                [
                    section(
                        "Beam",
                        controls_for_section(CONTROL_SECTIONS[0][1]),
                    ),
                    section(
                        "Sample",
                        controls_for_section(CONTROL_SECTIONS[1][1]),
                    ),
                    section(
                        "Detector",
                        controls_for_section(CONTROL_SECTIONS[2][1]),
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
                    "gap": "1rem",
                    "paddingBottom": "1rem",
                },
            ),
            dcc.Graph(
                id="specular-fig",
                figure=initial_figure,
                style={"height": "78vh"},
                config={"responsive": True},
            ),
            html.Pre(
                id="specular-summary",
                children=initial_summary,
                style={
                    "whiteSpace": "pre-wrap",
                    "backgroundColor": "#f7fafc",
                    "border": "1px solid #e2e8f0",
                    "borderRadius": "10px",
                    "padding": "0.85rem",
                    "marginTop": "0.9rem",
                    "fontFamily": "Consolas, Monaco, monospace",
                    "fontSize": "0.92rem",
                },
            ),
        ],
        style={"padding": "1rem"},
    )

    @app.callback(
        Output({"type": "specular-slider", "name": MATCH}, "value"),
        Output({"type": "specular-input", "name": MATCH}, "value"),
        Input({"type": "specular-slider", "name": MATCH}, "value"),
        Input({"type": "specular-input", "name": MATCH}, "value"),
        State({"type": "specular-slider", "name": MATCH}, "min"),
        State({"type": "specular-slider", "name": MATCH}, "max"),
        prevent_initial_call=True,
    )
    def sync_slider_and_input(slider_value, input_value, slider_min, slider_max):  # pragma: no cover - UI callback
        trigger = ctx.triggered_id
        if not isinstance(trigger, dict):
            raise PreventUpdate

        if trigger.get("type") == "specular-input":
            if input_value is None:
                raise PreventUpdate
            clamped_value = max(float(slider_min), min(float(slider_max), float(input_value)))
            return clamped_value, clamped_value

        if slider_value is None:
            raise PreventUpdate
        return slider_value, slider_value

    @app.callback(
        Output("specular-fig", "figure"),
        Output("specular-summary", "children"),
        Input({"type": "specular-slider", "name": ALL}, "value"),
        State({"type": "specular-slider", "name": ALL}, "id"),
        State("specular-fig", "relayoutData"),
    )
    def update_specular_figure(
        slider_values,
        slider_ids,
        relayout_data,
    ):  # pragma: no cover - UI callback
        value_by_name = {
            control_id["name"]: control_value
            for control_id, control_value in zip(slider_ids, slider_values, strict=True)
        }
        try:
            beam_config, sample_config, detector_config = configs_from_values(
                rays=value_by_name.get("rays"),
                seed=value_by_name.get("seed"),
                display_rays=value_by_name.get("display_rays"),
                source_y=value_by_name.get("source_y"),
                beam_width_x=value_by_name.get("beam_width_x"),
                beam_width_z=value_by_name.get("beam_width_z"),
                divergence_x=value_by_name.get("divergence_x"),
                divergence_z=value_by_name.get("divergence_z"),
                z_beam=value_by_name.get("z_beam"),
                sample_width=value_by_name.get("sample_width"),
                sample_height=value_by_name.get("sample_height"),
                theta_i=value_by_name.get("theta_i"),
                delta=value_by_name.get("delta"),
                alpha=value_by_name.get("alpha"),
                psi=value_by_name.get("psi"),
                z_sample=value_by_name.get("z_sample"),
                distance=value_by_name.get("distance"),
                detector_width=value_by_name.get("detector_width"),
                detector_height=value_by_name.get("detector_height"),
                beta=value_by_name.get("beta"),
                gamma=value_by_name.get("gamma"),
                chi=value_by_name.get("chi"),
                pixel_u=value_by_name.get("pixel_u"),
                pixel_v=value_by_name.get("pixel_v"),
                i0=value_by_name.get("i0"),
                j0=value_by_name.get("j0"),
                default_beam=beam_defaults,
                default_sample=sample_defaults,
                default_detector=detector_defaults,
            )
        except ValueError as exc:
            return build_specular_error_figure(str(exc)), f"Error: {exc}"

        return build_specular_outputs(
            beam_config,
            sample_config,
            detector_config,
            camera=extract_scene_camera(relayout_data),
        )

    return app


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Standalone beam/sample/detector specular reflection simulator"
    )
    parser.add_argument("--rays", type=int, default=BeamConfig.ray_count, help="Number of incident rays")
    parser.add_argument("--seed", type=int, default=BeamConfig.seed, help="Random seed for ray sampling")
    parser.add_argument(
        "--display-rays",
        type=int,
        default=BeamConfig.display_rays,
        help="Maximum rays drawn in the 3D view",
    )
    parser.add_argument("--source-y", type=float, default=BeamConfig.source_y, help="Beam launch y position")
    parser.add_argument("--beam-width-x", type=float, default=BeamConfig.width_x, help="1σ beam width along lab x")
    parser.add_argument("--beam-width-z", type=float, default=BeamConfig.width_z, help="1σ beam width along lab z")
    parser.add_argument(
        "--divergence-x",
        type=float,
        default=BeamConfig.divergence_x_deg,
        help="1σ divergence about lab x in degrees",
    )
    parser.add_argument(
        "--divergence-z",
        type=float,
        default=BeamConfig.divergence_z_deg,
        help="1σ divergence about lab z in degrees",
    )
    parser.add_argument("--z-beam", type=float, default=BeamConfig.z_offset, help="Beam vertical offset z_B")
    parser.add_argument("--sample-width", type=float, default=SampleConfig.width, help="Finite sample width W along a_s")
    parser.add_argument("--sample-height", type=float, default=SampleConfig.height, help="Finite sample height H along b_s")
    parser.add_argument("--theta-i", type=float, default=SampleConfig.theta_i_deg, help="Incidence angle θᵢ in degrees")
    parser.add_argument("--delta", type=float, default=SampleConfig.delta_deg, help="Residual substrate tilt δ in degrees")
    parser.add_argument("--alpha", type=float, default=SampleConfig.alpha_deg, help="Sample yaw α in degrees")
    parser.add_argument("--psi", type=float, default=SampleConfig.psi_deg, help="Sample roll ψ in degrees")
    parser.add_argument("--z-sample", type=float, default=SampleConfig.z_offset, help="Sample vertical offset z_S")
    parser.add_argument("--distance", type=float, default=DetectorConfig.distance, help="Detector distance D")
    parser.add_argument("--detector-width", type=float, default=DetectorConfig.width, help="Active detector width")
    parser.add_argument("--detector-height", type=float, default=DetectorConfig.height, help="Active detector height")
    parser.add_argument("--beta", type=float, default=DetectorConfig.beta_deg, help="Detector tilt β about lab x in degrees")
    parser.add_argument("--gamma", type=float, default=DetectorConfig.gamma_deg, help="Detector tilt γ about lab z in degrees")
    parser.add_argument("--chi", type=float, default=DetectorConfig.chi_deg, help="Detector in-plane rotation χ in degrees")
    parser.add_argument("--pixel-u", type=float, default=DetectorConfig.pixel_u, help="Detector pixel size along u")
    parser.add_argument("--pixel-v", type=float, default=DetectorConfig.pixel_v, help="Detector pixel size along v")
    parser.add_argument("--i0", type=float, default=DetectorConfig.i0, help="Detector beam-center pixel i0")
    parser.add_argument("--j0", type=float, default=DetectorConfig.j0, help="Detector beam-center pixel j0")
    parser.add_argument(
        "--output-html",
        type=str,
        default=None,
        help="Optional path for exporting the initial figure HTML before the GUI starts",
    )
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help=f"Dash host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Dash port (default: {DEFAULT_PORT})")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the Dash GUI in a browser")
    return parser.parse_args()


def configs_from_args(
    args: argparse.Namespace,
) -> tuple[BeamConfig, SampleConfig, DetectorConfig]:
    """Build strongly typed configs from CLI arguments."""

    beam, sample, detector = configs_from_values(
        rays=args.rays,
        seed=args.seed,
        display_rays=args.display_rays,
        source_y=args.source_y,
        beam_width_x=args.beam_width_x,
        beam_width_z=args.beam_width_z,
        divergence_x=args.divergence_x,
        divergence_z=args.divergence_z,
        z_beam=args.z_beam,
        sample_width=args.sample_width,
        sample_height=args.sample_height,
        theta_i=args.theta_i,
        delta=args.delta,
        alpha=args.alpha,
        psi=args.psi,
        z_sample=args.z_sample,
        distance=args.distance,
        detector_width=args.detector_width,
        detector_height=args.detector_height,
        beta=args.beta,
        gamma=args.gamma,
        chi=args.chi,
        pixel_u=args.pixel_u,
        pixel_v=args.pixel_v,
        i0=args.i0,
        j0=args.j0,
    )
    return beam, sample, detector


def main() -> None:
    """Run the standalone specular simulation GUI."""

    args = parse_args()
    beam_config, sample_config, detector_config = configs_from_args(args)
    figure, summary = build_specular_outputs(beam_config, sample_config, detector_config)
    if args.output_html is not None:
        output_path = write_figure_html(figure, args.output_html)
        print(f"Initial figure HTML exported to: {output_path}")
    print(summary)

    app = build_specular_app(beam_config, sample_config, detector_config)
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open_new(url)).start()
    app.run(debug=False, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
