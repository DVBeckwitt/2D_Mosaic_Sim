#!/usr/bin/env python3
"""Standalone beam -> sample -> detector specular diffraction simulator.

This module keeps all geometry in one lab frame and models the pieces needed
for a simplified specular-diffraction experiment:

- a family of incoming rays with transverse width and divergence,
- a finite rectangular sample,
- explicit sample rotations ``theta_i``, ``delta``, ``alpha``, and ``psi``,
- explicit detector rotations ``beta``, ``gamma``, and ``chi``,
- HKL/mosaic controls that generate exit-ray diffraction families, and
- line-plane intersections for the sample and detector.

The implementation is a geometry-first diffraction model that emphasizes the
beam, sample, detector, and detector-hit relationships.

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
from copy import deepcopy
from functools import lru_cache
import math
import tempfile
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
DIFFRACTION_SAMPLES = 180
ProgressCallback = Callable[[str], None]


class TerminalProgressReporter:
    """Emit concise elapsed-time status lines for CLI startup work."""

    def __init__(self, label: str = "specular") -> None:
        self._label = label
        self._start = time.perf_counter()
        self._step = 0

    def emit(self, message: str) -> None:
        self._step += 1
        elapsed = time.perf_counter() - self._start
        print(
            f"[{self._label}] +{elapsed:6.2f}s | step {self._step:02d} | {message}",
            flush=True,
        )


def _emit_progress(progress: ProgressCallback | None, message: str) -> None:
    """Send a progress update if a callback is available."""

    if progress is not None:
        progress(message)


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
    width: float = 1000.0
    height: float = 1000.0
    beta_deg: float = 0.0
    gamma_deg: float = 0.0
    chi_deg: float = 0.0
    pixel_u: float = 0.1
    pixel_v: float = 0.1
    i0: float = 1024.0
    j0: float = 1024.0


@dataclass(frozen=True)
class DiffractionConfig:
    """HKL and mosaic controls for the specular diffraction family."""

    H: int = 1
    K: int = 1
    L: int = 1
    sigma_deg: float = 0.8
    mosaic_gamma_deg: float = 5.0
    eta: float = 0.5


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
    updatemode: str = "mouseup"


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
            ControlSpec(
                "theta_i",
                MathLabel("θ", "i", " (deg)"),
                "sample",
                "theta_i_deg",
                0.1,
                -89.0,
                89.0,
                updatemode="drag",
            ),
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
    (
        "Diffraction",
        (
            ControlSpec("H", "H", "diffraction", "H", 1, -12, 12),
            ControlSpec("K", "K", "diffraction", "K", 1, -12, 12),
            ControlSpec("L", "L", "diffraction", "L", 1, 0, 40),
            ControlSpec(
                "sigma_deg",
                MathLabel("σ", None, " (deg)"),
                "diffraction",
                "sigma_deg",
                0.05,
                0.05,
                10.0,
            ),
            ControlSpec(
                "mosaic_gamma_deg",
                MathLabel("Γ", None, " (deg)"),
                "diffraction",
                "mosaic_gamma_deg",
                0.1,
                0.1,
                30.0,
            ),
            ControlSpec("eta", "η", "diffraction", "eta", 0.01, 0.0, 1.0),
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
    """Outputs from the HKL-driven specular diffraction trace."""

    beam: RayBundle
    sample: Frame3D
    detector: Frame3D
    sample_hit_indices: np.ndarray
    hit_points: np.ndarray
    exit_parent_indices: np.ndarray
    exit_dirs: np.ndarray
    exit_weights: np.ndarray
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
    def diffraction_ray_count(self) -> int:
        return int(self.exit_parent_indices.size)

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
    def plane_parent_indices(self) -> np.ndarray:
        return self.exit_parent_indices[self.plane_hit_indices]

    @property
    def detector_parent_indices(self) -> np.ndarray:
        return self.plane_parent_indices[self.active_detector_mask]

    @property
    def detector_uv(self) -> np.ndarray:
        return self.plane_uv[self.active_detector_mask]

    @property
    def detector_pixels(self) -> np.ndarray:
        return self.plane_pixels[self.active_detector_mask]

    @property
    def plane_weights(self) -> np.ndarray:
        return self.exit_weights[self.plane_hit_indices]

    @property
    def detector_weights(self) -> np.ndarray:
        return self.plane_weights[self.active_detector_mask]


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


@lru_cache(maxsize=1)
def _hkl_engine():
    """Load the HKL helpers lazily to avoid import-time package cycles."""

    from mosaic_sim.common import normalize_peak_params
    from mosaic_sim.constants import K_MAG, d_hex
    from mosaic_sim.geometry import intersection_circle
    from mosaic_sim.intensity import mosaic_intensity

    return normalize_peak_params, K_MAG, d_hex, intersection_circle, mosaic_intensity


def normalized_diffraction_params(
    config: DiffractionConfig,
) -> tuple[int, int, int, float, float, float]:
    """Return validated HKL/mosaic parameters with angular widths in radians."""

    normalize_peak_params, _, _, _, _ = _hkl_engine()
    params = normalize_peak_params(
        config.H,
        config.K,
        config.L,
        config.sigma_deg,
        config.mosaic_gamma_deg,
        config.eta,
        defaults=(
            DiffractionConfig.H,
            DiffractionConfig.K,
            DiffractionConfig.L,
            DiffractionConfig.sigma_deg,
            DiffractionConfig.mosaic_gamma_deg,
            DiffractionConfig.eta,
        ),
    )
    return params.as_tuple()


def diffraction_magnitude_angstrom(diffraction: DiffractionConfig) -> float:
    """Return ``|G|`` in reciprocal angstroms for the selected HKL."""

    _, _, d_hex, _, _ = _hkl_engine()
    d_spacing = d_hex(diffraction.H, diffraction.K, diffraction.L)
    return (2.0 * math.pi / d_spacing) * 1e-10


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


def skew_symmetric(vector: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix for ``vector``."""

    x, y, z = np.asarray(vector, dtype=float)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=float,
    )


def rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a rotation matrix that maps ``source`` to ``target``."""

    source_unit = normalize_vector(source)
    target_unit = normalize_vector(target)
    cross = np.cross(source_unit, target_unit)
    sin_angle = float(np.linalg.norm(cross))
    cos_angle = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))

    if sin_angle <= EPSILON:
        if cos_angle > 0.0:
            return np.eye(3, dtype=float)

        axis = np.cross(source_unit, LAB_X)
        if float(np.linalg.norm(axis)) <= EPSILON:
            axis = np.cross(source_unit, LAB_Z)
        axis = normalize_vector(axis)
        skew = skew_symmetric(axis)
        return np.eye(3, dtype=float) + 2.0 * (skew @ skew)

    axis = cross / sin_angle
    skew = skew_symmetric(axis)
    return (
        np.eye(3, dtype=float)
        + sin_angle * skew
        + (1.0 - cos_angle) * (skew @ skew)
    )


def validate_configs(
    beam: BeamConfig,
    sample: SampleConfig,
    detector: DetectorConfig,
    diffraction: DiffractionConfig | None = None,
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
    if diffraction is not None:
        normalized_diffraction_params(diffraction)


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
    H: Any = None,
    K: Any = None,
    L: Any = None,
    h_index: Any = None,
    k_index: Any = None,
    l_index: Any = None,
    sigma_deg: Any = None,
    mosaic_gamma_deg: Any = None,
    eta: Any = None,
    default_beam: BeamConfig | None = None,
    default_sample: SampleConfig | None = None,
    default_detector: DetectorConfig | None = None,
    default_diffraction: DiffractionConfig | None = None,
) -> tuple[BeamConfig, SampleConfig, DetectorConfig, DiffractionConfig]:
    """Build strongly typed configs from CLI or GUI values."""

    beam_defaults = default_beam or BeamConfig()
    sample_defaults = default_sample or SampleConfig()
    detector_defaults = default_detector or DetectorConfig()
    diffraction_defaults = default_diffraction or DiffractionConfig()

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
    diffraction = DiffractionConfig(
        H=int(coerce_value(h_index if h_index is not None else H, diffraction_defaults.H, int)),
        K=int(coerce_value(k_index if k_index is not None else K, diffraction_defaults.K, int)),
        L=int(coerce_value(l_index if l_index is not None else L, diffraction_defaults.L, int)),
        sigma_deg=float(coerce_value(sigma_deg, diffraction_defaults.sigma_deg, float)),
        mosaic_gamma_deg=float(
            coerce_value(
                mosaic_gamma_deg,
                diffraction_defaults.mosaic_gamma_deg,
                float,
            )
        ),
        eta=float(coerce_value(eta, diffraction_defaults.eta, float)),
    )
    validate_configs(beam, sample, detector, diffraction)
    return beam, sample, detector, diffraction


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


def project_rays_to_detector(
    origins: np.ndarray,
    directions: np.ndarray,
    detector_frame: Frame3D,
    detector_config: DetectorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project many rays onto the detector plane."""

    if origins.size == 0 or directions.size == 0:
        empty_points = np.empty((0, 3), dtype=float)
        empty_uv = np.empty((0, 2), dtype=float)
        empty_pixels = np.empty((0, 2), dtype=float)
        return np.zeros(0, dtype=bool), empty_points, empty_uv, empty_pixels

    origin_arr = np.asarray(origins, dtype=float)
    direction_arr = normalize_rows(np.asarray(directions, dtype=float))
    plane_normal = normalize_vector(detector_frame.normal)
    denominator = direction_arr @ plane_normal
    mask = np.abs(denominator) > EPSILON

    distances = np.empty(direction_arr.shape[0], dtype=float)
    distances.fill(np.nan)
    numerator = (detector_frame.origin - origin_arr) @ plane_normal
    distances[mask] = numerator[mask] / denominator[mask]
    mask &= distances > 0.0

    if not np.any(mask):
        empty_points = np.empty((0, 3), dtype=float)
        empty_uv = np.empty((0, 2), dtype=float)
        empty_pixels = np.empty((0, 2), dtype=float)
        return mask, empty_points, empty_uv, empty_pixels

    points = origin_arr[mask] + distances[mask, None] * direction_arr[mask]
    rel = points - detector_frame.origin
    u = rel @ detector_frame.axis_u
    v = rel @ detector_frame.axis_v
    uv = np.column_stack([u, v])
    pixels = np.column_stack(
        [
            detector_config.i0 + u / detector_config.pixel_u,
            detector_config.j0 - v / detector_config.pixel_v,
        ]
    )
    return mask, points, uv, pixels


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
    diffraction_config: DiffractionConfig | None = None,
    progress: ProgressCallback | None = None,
) -> SimulationResult:
    """Trace the beam family from source to sample to detector."""

    beam = beam_config or BeamConfig()
    sample = sample_config or SampleConfig()
    detector = detector_config or DetectorConfig()
    diffraction = diffraction_config or DiffractionConfig()
    _emit_progress(progress, "Validating beam, sample, detector, and diffraction inputs")
    validate_configs(beam, sample, detector, diffraction)

    _emit_progress(progress, f"Generating {beam.ray_count} incident beam rays")
    rays = generate_beam_rays(beam)
    _emit_progress(progress, "Building sample and detector frames")
    sample_frame = build_sample_frame(sample)
    detector_frame = build_detector_frame(detector)
    sample_basis = np.vstack(
        [sample_frame.axis_u, sample_frame.axis_v, sample_frame.normal]
    )

    _emit_progress(progress, "Intersecting incident rays with the sample plane")
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
    _emit_progress(
        progress,
        f"Sample hits on finite sample: {sample_hit_indices.size}/{beam.ray_count}",
    )
    (
        H,
        K,
        L,
        sigma,
        Gamma,
        eta,
    ) = normalized_diffraction_params(diffraction)
    _, reciprocal_k_mag, d_hex, intersection_circle, mosaic_intensity = _hkl_engine()
    g_mag = 2.0 * math.pi / d_hex(H, K, L)
    _emit_progress(
        progress,
        (
            "Building HKL diffraction ring for "
            f"({H}, {K}, {L}) with {DIFFRACTION_SAMPLES} azimuthal samples"
        ),
    )
    ring_x, ring_y, ring_z = intersection_circle(
        g_mag,
        reciprocal_k_mag,
        reciprocal_k_mag,
        npts=DIFFRACTION_SAMPLES,
    )

    exit_parent_indices = np.empty(0, dtype=int)
    exit_dirs = np.empty((0, 3), dtype=float)
    exit_weights = np.empty(0, dtype=float)
    plane_hit_indices = np.empty(0, dtype=int)
    plane_points = np.empty((0, 3), dtype=float)
    plane_uv = np.empty((0, 2), dtype=float)
    plane_pixels = np.empty((0, 2), dtype=float)
    active_detector_mask = np.zeros(0, dtype=bool)

    if sample_hit_indices.size and ring_x.size:
        canonical_q = np.column_stack([ring_x, ring_y, ring_z])
        incident_dirs_local = rays.directions[sample_hit_indices] @ sample_basis.T
        exit_parent_parts: list[np.ndarray] = []
        exit_dir_parts: list[np.ndarray] = []
        exit_weight_parts: list[np.ndarray] = []
        total_hits = incident_dirs_local.shape[0]
        progress_interval = max(1, total_hits // 5)

        for parent_index, incident_dir_local in enumerate(incident_dirs_local):
            rotation = rotation_between_vectors(LAB_Y, -incident_dir_local)
            q_local = canonical_q @ rotation.T
            k_i_local = reciprocal_k_mag * incident_dir_local
            exit_dirs_local = normalize_rows(q_local + k_i_local[None, :])
            exit_parent_parts.append(
                np.full(exit_dirs_local.shape[0], parent_index, dtype=int)
            )
            exit_dir_parts.append(exit_dirs_local @ sample_basis)
            exit_weight_parts.append(
                np.asarray(
                    mosaic_intensity(
                        q_local[:, 0],
                        q_local[:, 1],
                        q_local[:, 2],
                        H,
                        K,
                        L,
                        sigma,
                        Gamma,
                        eta,
                    ),
                    dtype=float,
                )
            )
            processed_hits = parent_index + 1
            if processed_hits == 1 or processed_hits == total_hits or processed_hits % progress_interval == 0:
                _emit_progress(
                    progress,
                    f"Expanding diffraction families from sample hits: {processed_hits}/{total_hits}",
                )

        exit_parent_indices = np.concatenate(exit_parent_parts)
        exit_dirs = np.vstack(exit_dir_parts)
        exit_weights = np.concatenate(exit_weight_parts)
        _emit_progress(
            progress,
            (
                f"Generated {exit_parent_indices.size} diffraction rays from "
                f"{sample_hit_indices.size} sample hits"
            ),
        )

        exit_origins = hit_points[exit_parent_indices]
        _emit_progress(
            progress,
            f"Projecting {exit_dirs.shape[0]} diffraction rays onto detector plane",
        )
        projection_mask, plane_points, plane_uv, plane_pixels = project_rays_to_detector(
            exit_origins,
            exit_dirs,
            detector_frame,
            detector,
        )
        plane_hit_indices = np.flatnonzero(projection_mask)
        active_detector_mask = (
            (np.abs(plane_uv[:, 0]) <= 0.5 * detector.width)
            & (np.abs(plane_uv[:, 1]) <= 0.5 * detector.height)
        )
        _emit_progress(
            progress,
            (
                f"Detector-plane intersections: {plane_hit_indices.size}; "
                f"active detector hits: {int(np.count_nonzero(active_detector_mask))}"
            ),
        )
    elif not sample_hit_indices.size:
        _emit_progress(
            progress,
            "No incident rays hit the sample; skipping diffraction-family expansion",
        )
    else:
        _emit_progress(
            progress,
            "Diffraction ring produced no samples; skipping detector projection",
        )

    _emit_progress(progress, "Tracing direct-beam reference to the detector")
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
    _emit_progress(progress, "Trace complete")

    return SimulationResult(
        beam=rays,
        sample=sample_frame,
        detector=detector_frame,
        sample_hit_indices=sample_hit_indices,
        hit_points=hit_points,
        exit_parent_indices=exit_parent_indices,
        exit_dirs=exit_dirs,
        exit_weights=exit_weights,
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
    diffraction_config: DiffractionConfig,
    *,
    camera: dict[str, Any] | None = None,
) -> go.Figure:
    """Build a 3-panel Plotly figure for the HKL diffraction trace."""

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "xy"}, {"type": "xy"}]],
        column_widths=[0.54, 0.16, 0.30],
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
        display_parent_indices = np.unique(result.plane_parent_indices[plane_display])
        incident_starts = result.beam.origins[result.sample_hit_indices[display_parent_indices]]
        incident_ends = result.hit_points[display_parent_indices]
        diffraction_starts = result.hit_points[result.plane_parent_indices[plane_display]]
        diffraction_ends = result.plane_points[plane_display]
        displayed_exit_dirs = result.exit_dirs[result.plane_hit_indices[plane_display]]

        inc_x, inc_y, inc_z = segmented_line_points(incident_starts, incident_ends)
        diff_x, diff_y, diff_z = segmented_line_points(diffraction_starts, diffraction_ends)

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
                x=diff_x,
                y=diff_y,
                z=diff_z,
                mode="lines",
                line=dict(color="rgba(214, 40, 40, 0.16)", width=4),
                name="Diffraction rays",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        arrow_display = choose_display_indices(plane_display.size, min(24, plane_display.size))
        if arrow_display.size:
            arrow_tips = diffraction_ends[arrow_display]
            arrow_dirs = displayed_exit_dirs[arrow_display]
            axis_scale = max(
                sample_config.width,
                sample_config.height,
                detector_config.width,
                detector_config.height,
                detector_config.distance,
            )
            arrow_length = 0.06 * axis_scale
            arrow_vectors = arrow_length * arrow_dirs
            fig.add_trace(
                go.Cone(
                    x=arrow_tips[:, 0],
                    y=arrow_tips[:, 1],
                    z=arrow_tips[:, 2],
                    u=arrow_vectors[:, 0],
                    v=arrow_vectors[:, 1],
                    w=arrow_vectors[:, 2],
                    anchor="tip",
                    sizemode="absolute",
                    sizeref=max(arrow_length, EPSILON),
                    colorscale=[[0.0, "#d62828"], [1.0, "#d62828"]],
                    showscale=False,
                    opacity=0.75,
                    name="Diffraction direction",
                    hoverinfo="skip",
                    showlegend=False,
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
        fig.add_trace(
            go.Scatter3d(
                x=result.plane_points[:, 0],
                y=result.plane_points[:, 1],
                z=result.plane_points[:, 2],
                mode="markers",
                marker=dict(
                    size=4,
                    color=result.plane_weights,
                    colorscale="Viridis",
                    cmin=0.0,
                    cmax=1.0,
                    showscale=False,
                    opacity=0.55,
                ),
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
                    marker=dict(
                        size=6,
                        color=result.plane_weights[inactive],
                        colorscale="Viridis",
                        cmin=0.0,
                        cmax=1.0,
                        opacity=0.35,
                        showscale=False,
                    ),
                    name="Off-detector intersections",
                    hovertemplate=(
                        "u=%{x:.3f}<br>v=%{y:.3f}<br>i=%{customdata[0]:.2f}"
                        "<br>j=%{customdata[1]:.2f}<br>w=%{marker.color:.3f}<extra></extra>"
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
                marker=dict(
                    size=8,
                    color=result.detector_weights,
                    colorscale="Viridis",
                    cmin=0.0,
                    cmax=1.0,
                    opacity=0.9,
                    showscale=True,
                    colorbar=dict(title="w", len=0.72),
                ),
                name="Detector hits",
                hovertemplate=(
                    "u=%{x:.3f}<br>v=%{y:.3f}<br>i=%{customdata[0]:.2f}"
                    "<br>j=%{customdata[1]:.2f}<br>w=%{marker.color:.3f}<extra></extra>"
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

    hkl_label = (
        f"HKL = ({diffraction_config.H}, {diffraction_config.K}, {diffraction_config.L})"
    )
    g_mag = diffraction_magnitude_angstrom(diffraction_config)
    sample_x_padding = max(0.05 * sample_half_u, 1e-6)
    sample_y_padding = max(0.05 * sample_half_v, 1e-6)
    sample_x_range = [
        -(sample_half_u + sample_x_padding),
        sample_half_u + sample_x_padding,
    ]
    sample_y_range = [
        -(sample_half_v + sample_y_padding),
        sample_half_v + sample_y_padding,
    ]
    detector_axis_extent = max(
        detector_half_u,
        detector_half_v,
        float(np.max(np.abs(result.plane_uv))) if result.detector_plane_hit_count else 0.0,
        float(np.max(np.abs(result.direct_beam_uv))) if result.direct_beam_uv is not None else 0.0,
    )
    detector_axis_padding = max(0.05 * detector_axis_extent, 1e-6)
    detector_axis_range = [
        -(detector_axis_extent + detector_axis_padding),
        detector_axis_extent + detector_axis_padding,
    ]
    title = (
        "Specular Diffraction"
        f" | {hkl_label}"
        f" | |G| = {g_mag:.3f} A^-1"
        f" | sample {result.sample_hit_count}/{beam_config.ray_count}"
        f" | diffraction rays {result.diffraction_ray_count}"
        f" | detector hits {result.detector_hit_count}/{result.detector_plane_hit_count}"
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=64, b=20),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.74)",
            bordercolor="rgba(15, 23, 42, 0.12)",
            borderwidth=1,
            font=dict(size=11),
        ),
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
    fig.update_xaxes(
        title="u_s",
        range=sample_x_range,
        scaleanchor="y",
        scaleratio=1,
        constrain="domain",
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title="v_s",
        range=sample_y_range,
        constrain="domain",
        row=1,
        col=2,
    )
    fig.update_xaxes(
        title="u",
        range=detector_axis_range,
        scaleanchor="y",
        scaleratio=1,
        constrain="domain",
        row=1,
        col=3,
    )
    fig.update_yaxes(
        title="v",
        range=detector_axis_range,
        constrain="domain",
        row=1,
        col=3,
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
    diffraction_config: DiffractionConfig,
) -> str:
    """Return a compact text summary for CLI use."""

    hkl_label = (
        f"({diffraction_config.H}, {diffraction_config.K}, {diffraction_config.L})"
    )
    g_mag = diffraction_magnitude_angstrom(diffraction_config)

    if result.detector_hit_count:
        detector_centroid = np.average(
            result.detector_uv,
            axis=0,
            weights=result.detector_weights,
        )
        pixel_centroid = np.average(
            result.detector_pixels,
            axis=0,
            weights=result.detector_weights,
        )
        centroid_text = (
            f"weighted detector centroid (u, v) = ({detector_centroid[0]:.4f}, "
            f"{detector_centroid[1]:.4f}), "
            f"weighted pixel centroid (i, j) = ({pixel_centroid[0]:.2f}, "
            f"{pixel_centroid[1]:.2f})"
        )
    else:
        centroid_text = "weighted detector centroid unavailable (no active detector hits)"

    plane_centroid_text = "weighted detector-plane centroid unavailable"
    if result.detector_plane_hit_count:
        plane_centroid = np.average(
            result.plane_uv,
            axis=0,
            weights=result.plane_weights,
        )
        plane_centroid_text = (
            f"weighted detector-plane centroid (u, v) = "
            f"({plane_centroid[0]:.4f}, {plane_centroid[1]:.4f})"
        )

    direct_beam_text = "direct beam center unavailable"
    if result.direct_beam_uv is not None and result.direct_beam_pixels is not None:
        direct_beam_text = (
            "direct beam detector center "
            f"(u, v) = ({result.direct_beam_uv[0]:.4f}, {result.direct_beam_uv[1]:.4f}), "
            f"pixels = ({result.direct_beam_pixels[0]:.2f}, {result.direct_beam_pixels[1]:.2f})"
        )

    return "\n".join(
        [
            "Specular diffraction summary",
            f"  HKL = {hkl_label}",
            f"  |G|: {g_mag:.6f} A^-1",
            f"  sample hits: {result.sample_hit_count}/{result.beam.origins.shape[0]}",
            f"  diffraction rays: {result.diffraction_ray_count}",
            f"  detector-plane intersections: {result.detector_plane_hit_count}",
            f"  active detector hits: {result.detector_hit_count}",
            f"  mosaic sigma: {diffraction_config.sigma_deg:.4f} deg",
            f"  mosaic Gamma: {diffraction_config.mosaic_gamma_deg:.4f} deg",
            f"  mosaic eta: {diffraction_config.eta:.4f}",
            f"  sample size (W x H): {sample_config.width:.3f} x {sample_config.height:.3f}",
            f"  detector size (W x H): {detector_config.width:.3f} x {detector_config.height:.3f}",
            f"  {plane_centroid_text}",
            f"  {centroid_text}",
            f"  {direct_beam_text}",
        ]
    )


def build_specular_companion_figure(
    sample_config: SampleConfig,
    diffraction_config: DiffractionConfig,
    *,
    camera: dict[str, Any] | None = None,
) -> go.Figure:
    """Return the companion reciprocal-space/integration view for specular mode."""

    from mosaic_sim.detector import build_detector_figure

    base_figure = build_detector_figure(
        diffraction_config.H,
        diffraction_config.K,
        diffraction_config.L,
        math.radians(diffraction_config.sigma_deg),
        Gamma=math.radians(diffraction_config.mosaic_gamma_deg),
        eta=diffraction_config.eta,
        theta_i=math.radians(sample_config.theta_i_deg),
        camera=camera,
    )
    companion_figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[0.7, 0.3],
        subplot_titles=("Reciprocal space", "Centered integration"),
    )

    for trace in base_figure.data:
        if getattr(trace, "scene", None) == "scene":
            companion_figure.add_trace(deepcopy(trace), row=1, col=1)
        elif getattr(trace, "xaxis", None) == "x2" and getattr(trace, "yaxis", None) == "y2":
            centered_trace = deepcopy(trace)
            if hasattr(centered_trace, "xaxis"):
                centered_trace.xaxis = None
            if hasattr(centered_trace, "yaxis"):
                centered_trace.yaxis = None
            companion_figure.add_trace(centered_trace, row=1, col=2)

    scene_layout = (
        deepcopy(base_figure.layout.scene.to_plotly_json())
        if base_figure.layout.scene is not None
        else {}
    )
    centered_xaxis_layout = (
        deepcopy(base_figure.layout.xaxis2.to_plotly_json())
        if getattr(base_figure.layout, "xaxis2", None) is not None
        else {}
    )
    centered_yaxis_layout = (
        deepcopy(base_figure.layout.yaxis2.to_plotly_json())
        if getattr(base_figure.layout, "yaxis2", None) is not None
        else {}
    )

    companion_figure.update_layout(
        paper_bgcolor=base_figure.layout.paper_bgcolor,
        plot_bgcolor=base_figure.layout.plot_bgcolor,
        showlegend=False,
        uirevision=base_figure.layout.uirevision,
        margin=dict(l=0, r=0, b=0, t=110),
        title=deepcopy(base_figure.layout.title.to_plotly_json()),
    )
    companion_figure.update_scenes(row=1, col=1, **scene_layout)
    companion_figure.update_xaxes(row=1, col=2, **centered_xaxis_layout)
    companion_figure.update_yaxes(row=1, col=2, **centered_yaxis_layout)
    return companion_figure


def build_specular_companion_error_figure(message: str) -> go.Figure:
    """Return a compact error figure for the companion mosaic view."""

    from mosaic_sim.detector import build_detector_error_figure

    figure = build_detector_error_figure(message)
    figure.update_layout(
        title=dict(
            text="Reciprocal Space and Integrated Response",
            x=0.5,
            xanchor="center",
        )
    )
    return figure


def build_specular_dashboard_outputs(
    beam_config: BeamConfig,
    sample_config: SampleConfig,
    detector_config: DetectorConfig,
    diffraction_config: DiffractionConfig,
    *,
    camera: dict[str, Any] | None = None,
    companion_camera: dict[str, Any] | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[go.Figure, go.Figure, str]:
    """Return the main specular figure, companion figure, and textual summary."""

    result = trace_specular_simulation(
        beam_config,
        sample_config,
        detector_config,
        diffraction_config,
        progress=progress,
    )
    _emit_progress(progress, "Building specular geometry figure")
    figure = build_specular_figure(
        result,
        beam_config,
        sample_config,
        detector_config,
        diffraction_config,
        camera=camera,
    )
    _emit_progress(progress, "Building reciprocal-space and integrated-response companion figure")
    companion_figure = build_specular_companion_figure(
        sample_config,
        diffraction_config,
        camera=companion_camera,
    )
    _emit_progress(progress, "Formatting textual summary")
    summary = simulation_summary(
        result,
        sample_config,
        detector_config,
        diffraction_config,
    )
    _emit_progress(progress, "Initial outputs ready")
    return figure, companion_figure, summary


def build_specular_outputs(
    beam_config: BeamConfig,
    sample_config: SampleConfig,
    detector_config: DetectorConfig,
    diffraction_config: DiffractionConfig,
    *,
    camera: dict[str, Any] | None = None,
    companion_camera: dict[str, Any] | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[go.Figure, str]:
    """Return the live figure and summary text for the current configs."""

    figure, _, summary = build_specular_dashboard_outputs(
        beam_config,
        sample_config,
        detector_config,
        diffraction_config,
        camera=camera,
        companion_camera=companion_camera,
        progress=progress,
    )
    return figure, summary


def build_number_control(
    control_name: str,
    label: str | MathLabel,
    value: int | float,
    *,
    step: int | float,
    min_value: int | float,
    max_value: int | float,
    updatemode: str = "mouseup",
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
    marks = {
        float(min_value): f"{min_value:g}",
        float(max_value): f"{max_value:g}",
    }

    return html.Div(
        [
            html.Label(
                label_content,
                htmlFor=str(slider_id),
                className="specular-control-label",
                style={"fontWeight": 600},
            ),
            html.Div(
                [
                    dcc.Slider(
                        id=slider_id,
                        min=min_value,
                        max=max_value,
                        step=step,
                        value=value,
                        marks=marks,
                        included=False,
                        updatemode=updatemode,
                        tooltip={"placement": "bottom", "always_visible": False},
                        className="specular-slider",
                    ),
                    dcc.Input(
                        id=input_id,
                        type="number",
                        value=value,
                        step=step,
                        min=min_value,
                        max=max_value,
                        debounce=True,
                        className="specular-number-input",
                        style={"width": "88px", "minWidth": "88px"},
                    ),
                ],
                className="specular-control-row",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "minmax(0, 1fr) 88px",
                    "gap": "0.6rem",
                    "alignItems": "center",
                },
            ),
        ],
        className="specular-control",
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "0.4rem",
        },
    )


def build_specular_app(
    initial_beam: BeamConfig | None = None,
    initial_sample: SampleConfig | None = None,
    initial_detector: DetectorConfig | None = None,
    initial_diffraction: DiffractionConfig | None = None,
    *,
    initial_figure: go.Figure | None = None,
    initial_companion_figure: go.Figure | None = None,
    initial_summary: str | None = None,
):
    """Return a Dash app exposing all beam, sample, detector, and HKL parameters."""

    import dash
    from dash import ALL, MATCH, ctx, dcc, html
    from dash.dependencies import Input, Output, State
    from dash.exceptions import PreventUpdate

    beam_defaults = initial_beam or BeamConfig()
    sample_defaults = initial_sample or SampleConfig()
    detector_defaults = initial_detector or DetectorConfig()
    diffraction_defaults = initial_diffraction or DiffractionConfig()
    if initial_figure is None or initial_summary is None:
        initial_figure, initial_companion_figure, initial_summary = build_specular_dashboard_outputs(
            beam_defaults,
            sample_defaults,
            detector_defaults,
            diffraction_defaults,
        )
    elif initial_companion_figure is None:
        initial_companion_figure = build_specular_companion_figure(
            sample_defaults,
            diffraction_defaults,
        )

    assets_folder = Path(__file__).resolve().parent / "assets"
    app = dash.Dash(__name__, assets_folder=str(assets_folder))
    app.title = "Specular Diffraction Simulator"

    config_by_group = {
        "beam": beam_defaults,
        "sample": sample_defaults,
        "detector": detector_defaults,
        "diffraction": diffraction_defaults,
    }

    def section(title: str, children: list[Any]):
        return html.Div(
            [
                html.H3(title, className="specular-section-title", style={"margin": "0"}),
                html.Div(
                    children,
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "minmax(0, 1fr)",
                        "gap": "0.75rem",
                    },
                    className="specular-section-grid",
                ),
            ],
            className="specular-section-card",
            style={
                "border": "1px solid #d8dee9",
                "borderRadius": "10px",
                "padding": "0.9rem",
                "backgroundColor": "#fbfcfd",
                "display": "flex",
                "flexDirection": "column",
                "gap": "0.85rem",
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
                    updatemode=spec.updatemode,
                )
            )
        return controls

    def visual_card(
        title: str,
        description: str,
        graph_id: str,
        figure: go.Figure,
        *,
        frame_class_name: str,
    ) -> html.Section:
        return html.Section(
            [
                html.Div(
                    [
                        html.H3(title, style={"margin": "0"}),
                        html.Div(
                            description,
                            style={"color": "#475569", "lineHeight": "1.5"},
                        ),
                    ],
                    className="specular-visual-header",
                    style={"display": "flex", "flexDirection": "column", "gap": "0.35rem"},
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id=graph_id,
                            figure=figure,
                            style={"height": "100%", "minHeight": "0"},
                            config={"responsive": True, "displaylogo": False},
                        ),
                    ],
                    className=frame_class_name,
                ),
            ],
            className="specular-visual-card",
            style={"display": "grid", "gap": "0.75rem", "minHeight": "0"},
        )

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Specular Diffraction", style={"margin": "0"}),
                            html.Div(
                                "Beam, sample, detector, and HKL controls stay visible while the diffraction figure remains in view.",
                                style={"color": "#4a5568"},
                            ),
                        ],
                        className="specular-sidebar-header",
                        style={"display": "flex", "flexDirection": "column", "gap": "0.45rem"},
                    ),
                    html.Pre(
                        id="specular-summary",
                        children=initial_summary,
                        className="specular-summary-card",
                        style={
                            "whiteSpace": "pre-wrap",
                            "backgroundColor": "#f7fafc",
                            "border": "1px solid #e2e8f0",
                            "borderRadius": "10px",
                            "padding": "0.85rem",
                            "fontFamily": "Consolas, Monaco, monospace",
                            "fontSize": "0.92rem",
                        },
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
                            section(
                                "Diffraction",
                                controls_for_section(CONTROL_SECTIONS[3][1]),
                            ),
                        ],
                        className="specular-sections",
                        style={"display": "grid", "gap": "1rem"},
                    ),
                ],
                className="specular-sidebar",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            visual_card(
                                "Specular Geometry",
                                "Lab-frame beam, sample footprint, and detector-hit geometry for the current incident and sample setup.",
                                "specular-fig",
                                initial_figure,
                                frame_class_name="specular-graph-frame specular-graph-frame--primary",
                            ),
                            visual_card(
                                "Reciprocal Space and Integrated Response",
                                "The synchronized reciprocal-space and centered-integration mosaic panels for the current HKL, mosaic widths, and θᵢ.",
                                "specular-companion-fig",
                                initial_companion_figure,
                                frame_class_name="specular-graph-frame specular-graph-frame--secondary",
                            ),
                        ],
                        className="specular-visuals",
                        style={"display": "grid", "gap": "1rem", "minHeight": "0"},
                    ),
                ],
                className="specular-main",
            ),
        ],
        className="specular-workspace",
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
        Output("specular-companion-fig", "figure"),
        Output("specular-summary", "children"),
        Input({"type": "specular-slider", "name": ALL}, "value"),
        State({"type": "specular-slider", "name": ALL}, "id"),
        State("specular-fig", "relayoutData"),
        State("specular-companion-fig", "relayoutData"),
    )
    def update_specular_figure(
        slider_values,
        slider_ids,
        relayout_data,
        companion_relayout_data,
    ):  # pragma: no cover - UI callback
        value_by_name = {
            control_id["name"]: control_value
            for control_id, control_value in zip(slider_ids, slider_values, strict=True)
        }
        try:
            beam_config, sample_config, detector_config, diffraction_config = configs_from_values(
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
                h_index=value_by_name.get("H"),
                k_index=value_by_name.get("K"),
                l_index=value_by_name.get("L"),
                sigma_deg=value_by_name.get("sigma_deg"),
                mosaic_gamma_deg=value_by_name.get("mosaic_gamma_deg"),
                eta=value_by_name.get("eta"),
                default_beam=beam_defaults,
                default_sample=sample_defaults,
                default_detector=detector_defaults,
                default_diffraction=diffraction_defaults,
            )
        except ValueError as exc:
            message = f"Error: {exc}"
            return (
                build_specular_error_figure(str(exc)),
                build_specular_companion_error_figure(str(exc)),
                message,
            )

        return build_specular_dashboard_outputs(
            beam_config,
            sample_config,
            detector_config,
            diffraction_config,
            camera=extract_scene_camera(relayout_data),
            companion_camera=extract_scene_camera(companion_relayout_data),
        )

    return app


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Standalone beam/sample/detector specular diffraction simulator"
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
    parser.add_argument("--h-index", type=int, default=DiffractionConfig.H, help="Miller index H")
    parser.add_argument("--k-index", type=int, default=DiffractionConfig.K, help="Miller index K")
    parser.add_argument("--l-index", type=int, default=DiffractionConfig.L, help="Miller index L")
    parser.add_argument("--sigma-deg", type=float, default=DiffractionConfig.sigma_deg, help="Gaussian mosaic width σ in degrees")
    parser.add_argument(
        "--mosaic-gamma-deg",
        type=float,
        default=DiffractionConfig.mosaic_gamma_deg,
        help="Lorentzian mosaic width Γ in degrees",
    )
    parser.add_argument("--eta", type=float, default=DiffractionConfig.eta, help="Pseudo-Voigt mixing factor η")
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
) -> tuple[BeamConfig, SampleConfig, DetectorConfig, DiffractionConfig]:
    """Build strongly typed configs from CLI arguments."""

    beam, sample, detector, diffraction = configs_from_values(
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
        h_index=args.h_index,
        k_index=args.k_index,
        l_index=args.l_index,
        sigma_deg=args.sigma_deg,
        mosaic_gamma_deg=args.mosaic_gamma_deg,
        eta=args.eta,
    )
    return beam, sample, detector, diffraction


def main() -> None:
    """Run the standalone specular simulation GUI."""

    reporter = TerminalProgressReporter()
    reporter.emit("Parsing CLI arguments")
    args = parse_args()
    reporter.emit("Normalizing beam, sample, detector, and HKL inputs")
    beam_config, sample_config, detector_config, diffraction_config = configs_from_args(args)
    reporter.emit(
        (
            "Preparing initial diffraction trace for "
            f"HKL=({diffraction_config.H}, {diffraction_config.K}, {diffraction_config.L})"
        )
    )
    figure, companion_figure, summary = build_specular_dashboard_outputs(
        beam_config,
        sample_config,
        detector_config,
        diffraction_config,
        progress=reporter.emit,
    )
    if args.output_html is not None:
        reporter.emit(f"Exporting initial figure HTML to {args.output_html}")
        output_path = write_figure_html(figure, args.output_html)
        reporter.emit(f"Initial figure HTML exported to: {output_path}")
    print(summary, flush=True)

    reporter.emit("Building Dash app layout")
    app = build_specular_app(
        beam_config,
        sample_config,
        detector_config,
        diffraction_config,
        initial_figure=figure,
        initial_companion_figure=companion_figure,
        initial_summary=summary,
    )
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        reporter.emit(f"Opening browser at {url}")
        threading.Timer(1.0, lambda: webbrowser.open_new(url)).start()
    else:
        reporter.emit(f"Browser launch disabled; Dash app available at {url}")
    reporter.emit(f"Starting Dash server on {args.host}:{args.port}")
    app.run(debug=False, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
