#!/usr/bin/env python3
"""Command-line entry for a single Ewald-sphere visualisation.

This script exposes a minimal view that only shows the Ewald sphere and allows
interactive control of the incident angle ``theta_i``.
"""

import argparse
import json
import math
import tempfile
import webbrowser

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
from mosaic_sim.geometry import rot_x, sphere


LAMBDA_CU_K_ALPHA = 1.5406  # Å
PB_I2_A_HEX = 4.557  # Å
PB_I2_C_HEX = 6.979  # Å

K_MAG_PLOT = 2 * math.pi / LAMBDA_CU_K_ALPHA  # Å⁻¹
RECIP_A = 2 * math.pi / PB_I2_A_HEX
RECIP_C = 2 * math.pi / PB_I2_C_HEX


def _d_hex(h: int, k: int, l: int, a: float = PB_I2_A_HEX, c: float = PB_I2_C_HEX) -> float:
    return 1.0 / math.sqrt((4 / 3) * (h * h + h * k + k * k) / (a**2) + (l / c) ** 2)


G_002 = 2 * math.pi / _d_hex(0, 0, 2)
THETA_BRAGG_002 = math.degrees(math.asin(G_002 / (2.0 * K_MAG_PLOT)))
THETA_DEFAULT_MIN = 0.0
THETA_DEFAULT_MAX = THETA_BRAGG_002 + 10.0
N_FRAMES_DEFAULT = 60
CAMERA_EYE = np.array([2.0, 2.0, 1.6])
AXIS_RANGE = 5.0
ARC_RADIUS = 0.6


def _ewald_surface(theta_i: float, Ew_x: np.ndarray, Ew_y: np.ndarray, Ew_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return rot_x(Ew_x, Ew_y, Ew_z, theta_i)


def _k_vector(theta_i: float) -> tuple[list[float], list[float], list[float]]:
    """Return the incident wavevector from its tip back to the origin."""

    k_tip = np.array([0.0, K_MAG_PLOT, 0.0])
    tip_x, tip_y, tip_z = rot_x(np.array([k_tip[0]]), np.array([k_tip[1]]), np.array([k_tip[2]]), theta_i)
    return (
        [tip_x[0], 0.0],
        [tip_y[0], 0.0],
        [tip_z[0], 0.0],
    )


def _k_label(x: list[float], y: list[float], z: list[float]) -> go.Scatter3d:
    mid_x = 0.5 * (x[0] + x[1])
    mid_y = 0.5 * (y[0] + y[1])
    mid_z = 0.5 * (z[0] + z[1])
    return go.Scatter3d(
        x=[mid_x],
        y=[mid_y],
        z=[mid_z],
        mode="text",
        text=["kᵢ"],
        textposition="middle left",
        textfont=dict(color="black", size=14),
        showlegend=False,
    )


def _theta_arc(theta: float) -> go.Scatter3d:
    arc_thetas = np.linspace(0.0, theta, 50)
    arc_x = np.zeros_like(arc_thetas)
    arc_y = ARC_RADIUS * np.cos(arc_thetas)
    arc_z = ARC_RADIUS * np.sin(arc_thetas)
    return go.Scatter3d(
        x=arc_x,
        y=arc_y,
        z=arc_z,
        mode="lines",
        line=dict(color="orange", width=4, dash="dash"),
        showlegend=False,
    )


def _theta_arc_label(theta: float) -> go.Scatter3d:
    mid_theta = 0.5 * theta
    y = ARC_RADIUS * math.cos(mid_theta)
    z = ARC_RADIUS * math.sin(mid_theta)
    return go.Scatter3d(
        x=[0.0],
        y=[y],
        z=[z],
        mode="text",
        text=["θᵢ"],
        textfont=dict(color="orange", size=16),
        textposition="middle right",
        showlegend=False,
    )


def build_mono_figure(theta_min: float = math.radians(THETA_DEFAULT_MIN),
                      theta_max: float = math.radians(THETA_DEFAULT_MAX),
                      n_frames: int = N_FRAMES_DEFAULT) -> tuple[go.Figure, dict]:
    """Return a Plotly figure showing only the Ewald sphere.

    A slider controls the incident angle ``theta_i`` by rotating the sphere
    about the *qx* axis.
    """

    phi, theta = np.meshgrid(np.linspace(0, math.pi, 100),
                             np.linspace(0, 2 * math.pi, 200))
    Ew_x, Ew_y, Ew_z = sphere(K_MAG_PLOT, phi, theta, (0, K_MAG_PLOT, 0))

    grid_coords = np.arange(-2, 3)
    gx, gy, gz = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing="ij")
    lattice_indices = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    def _reciprocal_points(indices: np.ndarray) -> np.ndarray:
        scaled = indices.astype(float)
        scaled[:, 0] *= RECIP_A
        scaled[:, 1] *= RECIP_A
        scaled[:, 2] *= RECIP_C
        return scaled

    lattice_points = _reciprocal_points(lattice_indices)
    nonzero_mask = ~np.all(lattice_indices == 0, axis=1)
    lattice_indices = lattice_indices[nonzero_mask]
    lattice_points = lattice_points[nonzero_mask]

    g_magnitudes = np.linalg.norm(lattice_points, axis=1)
    unique_g = np.unique(np.round(g_magnitudes, 6))

    def _intersection_thetas() -> list[float]:
        thetas: list[float] = []
        for (hx, hy, hz), (x, y, z) in zip(lattice_indices, lattice_points, strict=True):
            if hx == hy == hz == 0:
                continue
            rhs = (x * x + y * y + z * z) / (2.0 * K_MAG_PLOT)
            amplitude = math.hypot(y, z)
            if amplitude == 0:
                continue
            ratio = rhs / amplitude
            if abs(ratio) > 1.0:
                continue
            delta = math.atan2(y, z)
            arcsin = math.asin(ratio)
            thetas.extend([arcsin - delta, math.pi - arcsin - delta])
        return thetas

    intersection_thetas = [th for th in _intersection_thetas() if th >= 0.0]
    theta_min = max(theta_min, 0.0)
    theta_max = max(theta_max, 0.0, *intersection_thetas)

    theta_all = np.linspace(theta_min, theta_max, n_frames)
    theta_all = np.unique(np.concatenate([theta_all, [0.0], intersection_thetas]))

    fig = go.Figure()

    Ew_x0, Ew_y0, Ew_z0 = _ewald_surface(theta_all[0], Ew_x, Ew_y, Ew_z)
    fig.add_trace(
        go.Surface(
            x=Ew_x0,
            y=Ew_y0,
            z=Ew_z0,
            opacity=1.0,
            colorscale="Blues",
            showscale=False,
            name="Ewald sphere",
        )
    )
    ewald_idx = len(fig.data) - 1

    kx, ky, kz = _k_vector(theta_all[0])
    fig.add_trace(
        go.Scatter3d(
            x=kx,
            y=ky,
            z=kz,
            mode="lines",
            line=dict(color="black", width=5),
            name="kᵢ",
        )
    )
    fig.add_trace(
        go.Cone(
            x=[kx[1]],
            y=[ky[1]],
            z=[kz[1]],
            u=[kx[1] - kx[0]],
            v=[ky[1] - ky[0]],
            w=[kz[1] - kz[0]],
            anchor="tip",
            sizemode="absolute",
            sizeref=0.2,
            colorscale=[[0, "black"], [1, "black"]],
            showscale=False,
        )
    )
    fig.add_trace(_k_label(kx, ky, kz))
    k_label_idx = len(fig.data) - 1
    cone_idx = k_label_idx - 1

    arc_trace = _theta_arc(theta_all[0])
    fig.add_trace(arc_trace)
    arc_idx = len(fig.data) - 1

    arc_label = _theta_arc_label(theta_all[0])
    fig.add_trace(arc_label)
    arc_label_idx = len(fig.data) - 1

    lattice_max = float(np.max(np.abs(lattice_points)))
    axis_range = max(AXIS_RANGE, 1.2 * max(K_MAG_PLOT, lattice_max))
    R_MAX = axis_range
    first_axis_idx = len(fig.data)
    for xyz in [([-R_MAX, R_MAX], [0, 0], [0, 0]),
                ([0, 0], [-R_MAX, R_MAX], [0, 0]),
                ([0, 0], [0, 0], [-R_MAX, R_MAX])]:
        fig.add_trace(
            go.Scatter3d(
                x=xyz[0],
                y=xyz[1],
                z=xyz[2],
                mode="lines",
                showlegend=False,
                line=dict(color="black", width=2, dash="dash"),
            )
        )

    axis_indices = list(range(first_axis_idx, len(fig.data)))

    def lattice_hits(theta: float) -> tuple[np.ndarray, np.ndarray]:
        center = np.array([0.0, K_MAG_PLOT * math.cos(theta), K_MAG_PLOT * math.sin(theta)])
        distances = np.linalg.norm(lattice_points - center, axis=1)
        mask = np.isclose(distances, K_MAG_PLOT, atol=1e-3)
        return mask, lattice_points[mask]

    def lattice_marker(theta: float) -> go.Scatter3d:
        hit_mask, _ = lattice_hits(theta)
        sizes = np.where(hit_mask, 9.0, 3.0)
        colors = np.where(hit_mask, "orange", "#a0a0a0")
        return go.Scatter3d(
            x=lattice_points[:, 0],
            y=lattice_points[:, 1],
            z=lattice_points[:, 2],
            mode="markers",
            marker=dict(size=sizes, color=colors, opacity=0.95),
            name="Integer lattice",
        )

    def hit_projection(theta: float) -> go.Scatter3d:
        hit_mask, hits = lattice_hits(theta)
        x: list[float] = []
        y: list[float] = []
        z: list[float] = []
        for hx, hy, hz in hits:
            x.extend([hx, hx, np.nan])
            y.extend([hy, hy, np.nan])
            z.extend([hz, 0.0, np.nan])
        return go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(color="orange", width=2, dash="dash"),
            showlegend=False,
        )

    def hit_labels(theta: float) -> go.Scatter3d:
        hit_mask, hits = lattice_hits(theta)
        hit_indices = lattice_indices[hit_mask]
        if hits.size == 0:
            return go.Scatter3d(x=[], y=[], z=[], mode="text", showlegend=False)
        x, y, z = hits[:, 0], hits[:, 1], hits[:, 2]
        labels = [f"({int(hx)}, {int(hy)}, {int(hz)})" for hx, hy, hz in hit_indices]
        distances = np.linalg.norm(hits - CAMERA_EYE, axis=1)
        reference = np.linalg.norm(CAMERA_EYE)
        scale = np.sqrt(np.maximum(distances, 1e-6) / reference)
        sizes = np.clip(14.0 * scale, 12.0, 18.0).tolist()
        return go.Scatter3d(
            x=x,
            y=y,
            z=z + 0.2,
            mode="text",
            text=labels,
            textposition="top center",
            textfont=dict(color="orange", size=sizes),
            showlegend=False,
        )

    lattice_trace = lattice_marker(theta_all[0])
    fig.add_trace(lattice_trace)
    lattice_idx = len(fig.data) - 1

    projection_trace = hit_projection(theta_all[0])
    fig.add_trace(projection_trace)
    projection_idx = len(fig.data) - 1

    label_trace = hit_labels(theta_all[0])
    fig.add_trace(label_trace)
    hit_label_idx = len(fig.data) - 1

    phi_g, theta_g = np.meshgrid(np.linspace(0, math.pi, 40),
                                 np.linspace(0, 2 * math.pi, 80))

    palette = qualitative.Plotly

    def g_magnitude_spheres() -> list[go.Surface]:
        traces: list[go.Surface] = []
        for i, g_val in enumerate(unique_g):
            color = palette[i % len(palette)]
            sx, sy, sz = sphere(g_val, phi_g, theta_g)
            traces.append(
                go.Surface(
                    x=sx,
                    y=sy,
                    z=sz,
                    opacity=0.15,
                    surfacecolor=np.full_like(sx, g_val),
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    name=f"|G| = {g_val:.3f} Å⁻¹",
                    visible=False,
                )
            )
        return traces

    g_sphere_traces = g_magnitude_spheres()
    for trace in g_sphere_traces:
        fig.add_trace(trace)
    g_sphere_indices = list(range(len(fig.data) - len(g_sphere_traces), len(fig.data)))

    def g_magnitude_points() -> list[go.Scatter3d]:
        traces: list[go.Scatter3d] = []
        rounded = np.round(g_magnitudes, 6)
        for i, g_val in enumerate(unique_g):
            mask = np.isclose(rounded, g_val, atol=1e-6)
            pts = lattice_points[mask]
            color = palette[i % len(palette)]
            traces.append(
                go.Scatter3d(
                    x=pts[:, 0] if len(pts) else [],
                    y=pts[:, 1] if len(pts) else [],
                    z=pts[:, 2] if len(pts) else [],
                    mode="markers",
                    marker=dict(color=color, size=5, opacity=0.95),
                    name=f"|G| shell points ({g_val:.3f} Å⁻¹)",
                    visible=False,
                )
            )
        return traces

    g_point_traces = g_magnitude_points()
    for trace in g_point_traces:
        fig.add_trace(trace)
    g_point_indices = list(range(len(fig.data) - len(g_point_traces), len(fig.data)))

    g_r = np.linalg.norm(lattice_points[:, :2], axis=1)
    g_z = lattice_points[:, 2]
    rounded_pairs: dict[tuple[float, float], tuple[float, float]] = {}
    for g_r_val, g_z_val, g_r_round, g_z_round in zip(
        g_r, g_z, np.round(g_r, 6), np.round(g_z, 6), strict=True
    ):
        key = (float(g_r_round), float(g_z_round))
        rounded_pairs.setdefault(key, (float(g_r_val), float(g_z_val)))
    g_ring_specs = list(rounded_pairs.values())

    def g_radial_rings() -> list[go.Scatter3d]:
        rings: list[go.Scatter3d] = []
        t_vals = np.linspace(0.0, 2 * math.pi, 200)
        for i, (g_r_val, g_z_val) in enumerate(g_ring_specs):
            color = palette[i % len(palette)]
            x = g_r_val * np.cos(t_vals)
            y = g_r_val * np.sin(t_vals)
            z = np.full_like(t_vals, g_z_val)
            rings.append(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(color=color, width=4),
                    name=f"|Gᵣ| ring ({g_r_val:.3f} Å⁻¹, G_z = {g_z_val:.3f} Å⁻¹)",
                    visible=False,
                )
            )
        return rings

    g_ring_traces = g_radial_rings()
    for trace in g_ring_traces:
        fig.add_trace(trace)
    g_ring_indices = list(range(len(fig.data) - len(g_ring_traces), len(fig.data)))

    def g_ring_intersection_points(
        theta: float, *, visibility: bool | None = False
    ) -> list[go.Scatter3d]:
        intersections: list[go.Scatter3d] = []
        center_y = K_MAG_PLOT * math.cos(theta)
        center_z = K_MAG_PLOT * math.sin(theta)

        for i, (g_r_val, g_z_val) in enumerate(g_ring_specs):
            color = palette[i % len(palette)]

            if g_r_val == 0:
                dist_sq = center_y * center_y + (g_z_val - center_z) ** 2
                on_sphere = math.isclose(dist_sq, K_MAG_PLOT * K_MAG_PLOT, rel_tol=1e-9, abs_tol=1e-9)
                intersections.append(
                    go.Scatter3d(
                        x=[0.0] if on_sphere else [],
                        y=[0.0] if on_sphere else [],
                        z=[g_z_val] if on_sphere else [],
                        mode="markers",
                        marker=dict(color=color, size=5, symbol="circle"),
                        name=f"|Gᵣ| ∩ Ewald ({g_r_val:.3f} Å⁻¹, G_z = {g_z_val:.3f} Å⁻¹)",
                        visible=visibility,
                        showlegend=False,
                    )
                )
                continue

            denominator = 2.0 * g_r_val * center_y
            numerator = (
                g_r_val * g_r_val
                + center_y * center_y
                + (g_z_val - center_z) * (g_z_val - center_z)
                - K_MAG_PLOT * K_MAG_PLOT
            )

            if abs(denominator) < 1e-12:
                intersections.append(
                    go.Scatter3d(
                        x=[],
                        y=[],
                        z=[],
                        mode="markers",
                        showlegend=False,
                        visible=visibility,
                    )
                )
                continue

            ratio = numerator / denominator
            if abs(ratio) > 1.0:
                intersections.append(
                    go.Scatter3d(
                        x=[],
                        y=[],
                        z=[],
                        mode="markers",
                        showlegend=False,
                        visible=visibility,
                    )
                )
                continue

            t0 = math.asin(ratio)
            candidates = [(t0 % (2 * math.pi)), ((math.pi - t0) % (2 * math.pi))]

            t_unique: list[float] = []
            for t_val in candidates:
                if not any(math.isclose(t_val, seen, rel_tol=1e-9, abs_tol=1e-9) for seen in t_unique):
                    t_unique.append(t_val)

            x_vals = [g_r_val * math.cos(t_val) for t_val in t_unique]
            y_vals = [g_r_val * math.sin(t_val) for t_val in t_unique]
            z_vals = [g_z_val for _ in t_unique]

            intersections.append(
                go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode="markers",
                    marker=dict(color=color, size=5, symbol="circle"),
                    name=f"|Gᵣ| ∩ Ewald ({g_r_val:.3f} Å⁻¹, G_z = {g_z_val:.3f} Å⁻¹)",
                    visible=visibility,
                    showlegend=False,
                )
            )

        return intersections

    g_ring_intersection_traces = g_ring_intersection_points(theta_all[0])
    for trace in g_ring_intersection_traces:
        fig.add_trace(trace)
    g_ring_intersection_indices = list(
        range(len(fig.data) - len(g_ring_intersection_traces), len(fig.data))
    )

    g_ring_groups = list(
        zip(g_ring_indices, g_ring_intersection_indices, strict=True)
    )

    cylinder_values = sorted({float(val) for val in np.round(g_r[g_r > 0], 6)})

    def g_radial_cylinders() -> list[go.Surface]:
        cylinders: list[go.Surface] = []
        theta_vals = np.linspace(0.0, 2 * math.pi, 80)
        z_vals = np.linspace(-axis_range, axis_range, 60)
        theta_grid, z_grid = np.meshgrid(theta_vals, z_vals)
        for i, g_r_val in enumerate(cylinder_values):
            color = palette[i % len(palette)]
            x = g_r_val * np.cos(theta_grid)
            y = g_r_val * np.sin(theta_grid)
            cylinders.append(
                go.Surface(
                    x=x,
                    y=y,
                    z=z_grid,
                    opacity=0.12,
                    surfacecolor=np.full_like(x, g_r_val),
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    name=f"|Gᵣ| cylinder ({g_r_val:.3f} Å⁻¹)",
                    visible=False,
                )
            )
        return cylinders

    g_cylinder_traces = g_radial_cylinders()
    for trace in g_cylinder_traces:
        fig.add_trace(trace)
    g_cylinder_indices = list(
        range(len(fig.data) - len(g_cylinder_traces), len(fig.data))
    )

    def g_cylinder_rings() -> list[go.Scatter3d]:
        traces: list[go.Scatter3d] = []
        rounded = np.round(g_r, 6)
        for i, g_r_val in enumerate(cylinder_values):
            mask = np.isclose(rounded, g_r_val, atol=1e-6)
            z_vals = g_z[mask]
            color = palette[i % len(palette)]

            if len(z_vals) == 0:
                traces.append(
                    go.Scatter3d(
                        x=[], y=[], z=[], mode="lines", showlegend=False, visible=False
                    )
                )
                continue

            z_lookup: dict[float, float] = {}
            for z_actual, z_round in zip(z_vals, np.round(z_vals, 6), strict=True):
                z_lookup.setdefault(float(z_round), float(z_actual))
            unique_z = [z_lookup[key] for key in sorted(z_lookup.keys())]

            t_vals = np.linspace(0.0, 2 * math.pi, 200)
            x_segments: list[np.ndarray] = []
            y_segments: list[np.ndarray] = []
            z_segments: list[np.ndarray] = []

            for pos, z_val in enumerate(unique_z):
                x_ring = g_r_val * np.cos(t_vals)
                y_ring = g_r_val * np.sin(t_vals)
                z_ring = np.full_like(t_vals, z_val)
                x_segments.append(x_ring)
                y_segments.append(y_ring)
                z_segments.append(z_ring)
                if pos < len(unique_z) - 1:
                    nan_pad = np.array([np.nan])
                    x_segments.append(nan_pad)
                    y_segments.append(nan_pad)
                    z_segments.append(nan_pad)

            traces.append(
                go.Scatter3d(
                    x=np.concatenate(x_segments),
                    y=np.concatenate(y_segments),
                    z=np.concatenate(z_segments),
                    mode="lines",
                    line=dict(color=color, width=4),
                    name=f"|Gᵣ| rings ({g_r_val:.3f} Å⁻¹)",
                    visible=False,
                )
            )
        return traces

    g_cylinder_ring_traces = g_cylinder_rings()
    for trace in g_cylinder_ring_traces:
        fig.add_trace(trace)
    g_cylinder_ring_indices = list(
        range(len(fig.data) - len(g_cylinder_ring_traces), len(fig.data))
    )

    def g_cylinder_intersection_curves(
        theta: float, *, visibility: bool | None = False
    ) -> list[go.Scatter3d]:
        curves: list[go.Scatter3d] = []
        center_y = K_MAG_PLOT * math.cos(theta)
        center_z = K_MAG_PLOT * math.sin(theta)
        t_vals = np.linspace(0.0, 2 * math.pi, 400)
        sin_t = np.sin(t_vals)
        cos_t = np.cos(t_vals)

        for i, g_r_val in enumerate(cylinder_values):
            color = palette[i % len(palette)]
            rhs = (
                K_MAG_PLOT * K_MAG_PLOT
                - g_r_val * g_r_val
                - center_y * center_y
                + 2.0 * g_r_val * center_y * sin_t
            )
            valid = rhs >= -1e-12
            if not np.any(valid):
                curves.append(
                    go.Scatter3d(
                        x=[], y=[], z=[], mode="lines", showlegend=False, visible=visibility
                    )
                )
                continue

            sqrt_rhs = np.sqrt(np.maximum(rhs, 0.0))
            x_vals = g_r_val * cos_t
            y_vals = g_r_val * sin_t
            z_top = np.where(valid, center_z + sqrt_rhs, np.nan)
            z_bottom = np.where(valid, center_z - sqrt_rhs, np.nan)
            x_top = x_vals
            y_top = y_vals
            x_bottom = x_vals
            y_bottom = y_vals

            x_combined = np.concatenate([x_top, [np.nan], x_bottom])
            y_combined = np.concatenate([y_top, [np.nan], y_bottom])
            z_combined = np.concatenate([z_top, [np.nan], z_bottom])

            curves.append(
                go.Scatter3d(
                    x=x_combined,
                    y=y_combined,
                    z=z_combined,
                    mode="lines",
                    line=dict(color=color, width=5),
                    showlegend=False,
                    name=f"|Gᵣ| ∩ Ewald ({g_r_val:.3f} Å⁻¹)",
                    visible=visibility,
                )
            )

        return curves

    g_cylinder_intersection_traces = g_cylinder_intersection_curves(theta_all[0])
    for trace in g_cylinder_intersection_traces:
        fig.add_trace(trace)
    g_cylinder_intersection_indices = list(
        range(len(fig.data) - len(g_cylinder_intersection_traces), len(fig.data))
    )

    g_cylinder_groups = list(
        zip(
            g_cylinder_indices,
            g_cylinder_intersection_indices,
            g_cylinder_ring_indices,
            strict=True,
        )
    )

    def g_intersection_circle(
        theta: float, g_val: float, color: str, *, visibility: bool | None = False
    ) -> go.Scatter3d:
        R_ewald = K_MAG_PLOT
        d = K_MAG_PLOT
        if g_val <= 0 or g_val > 2 * R_ewald:
            return go.Scatter3d(x=[], y=[], z=[], mode="lines", showlegend=False)

        a = (g_val * g_val) / (2 * d)
        r_sq = g_val * g_val - a * a
        if r_sq <= 0:
            return go.Scatter3d(x=[], y=[], z=[], mode="lines", showlegend=False)
        r = math.sqrt(r_sq)

        normal = np.array([0.0, math.cos(theta), math.sin(theta)])
        center = normal * a

        u = np.array([0.0, math.sin(theta), -math.cos(theta)])
        v = np.cross(normal, u)

        t_vals = np.linspace(0.0, 2 * math.pi, 200)
        circle = (
            center.reshape(3, 1)
            + r
            * (u.reshape(3, 1) * np.cos(t_vals) + v.reshape(3, 1) * np.sin(t_vals))
        )

        return go.Scatter3d(
            x=circle[0],
            y=circle[1],
            z=circle[2],
            mode="lines",
            line=dict(color=color, width=5),
            showlegend=False,
            name=f"|G| ∩ Ewald ({g_val:.3f} Å⁻¹)",
            visible=visibility,
        )

    g_circle_traces = [
        g_intersection_circle(theta_all[0], g_val, palette[i % len(palette)])
        for i, g_val in enumerate(unique_g)
    ]
    for trace in g_circle_traces:
        fig.add_trace(trace)
    g_circle_indices = list(range(len(fig.data) - len(g_circle_traces), len(fig.data)))

    base_indices = {ewald_idx, cone_idx - 1, cone_idx, k_label_idx,
                    arc_idx, arc_label_idx, *axis_indices}

    def _mode_visibility(
        mode: str,
        g_mask: list[bool] | None = None,
        ring_mask: list[bool] | None = None,
        cylinder_mask: list[bool] | None = None,
    ) -> list[bool]:
        lattice_related = {lattice_idx, projection_idx, hit_label_idx}
        g_related = g_sphere_indices + g_circle_indices + g_point_indices
        ring_related = [idx for pair in g_ring_groups for idx in pair]
        cylinder_related = [idx for group in g_cylinder_groups for idx in group]
        vis: list[bool] = []
        for idx in range(len(fig.data)):
            if idx in base_indices:
                vis.append(True)
            elif idx in lattice_related:
                vis.append(mode == "lattice")
            elif idx in g_related:
                if mode != "g_spheres":
                    vis.append(False)
                else:
                    pos = g_related.index(idx)
                    vis.append(False if g_mask is None else g_mask[pos])
            elif idx in ring_related:
                if mode != "g_rings":
                    vis.append(False)
                else:
                    pos = next(pos for pos, pair in enumerate(g_ring_groups) if idx in pair)
                    vis.append(True if ring_mask is None else ring_mask[pos])
            elif idx in cylinder_related:
                if mode != "g_cylinders":
                    vis.append(False)
                else:
                    pos = next(pos for pos, group in enumerate(g_cylinder_groups) if idx in group)
                    vis.append(True if cylinder_mask is None else cylinder_mask[pos])
            else:
                vis.append(True)
        return vis

    g_mask_default = [i == 0 for i in range(len(g_sphere_indices))]
    g_mask_default.extend([i == 0 for i in range(len(g_circle_indices))])
    g_mask_default.extend([i == 0 for i in range(len(g_point_indices))])
    lattice_visibility = _mode_visibility("lattice")
    g_sphere_visibility = _mode_visibility("g_spheres", g_mask_default)
    g_ring_mask_default = [True for _ in g_ring_indices]
    g_ring_visibility = _mode_visibility("g_rings", ring_mask=g_ring_mask_default)
    g_cylinder_mask_default = [i == 0 for i in range(len(g_cylinder_groups))]
    g_cylinder_visibility = _mode_visibility(
        "g_cylinders", cylinder_mask=g_cylinder_mask_default
    )

    def _two_theta_str(g_mag: float) -> str:
        ratio = g_mag / (2.0 * K_MAG_PLOT)
        if ratio > 1.0:
            return "n/a"
        ratio = min(1.0, max(-1.0, ratio))
        return f"{2.0 * math.degrees(math.asin(ratio)):.2f}°"

    g_two_thetas = [_two_theta_str(val) for val in unique_g]
    g_group_indices = list(zip(g_sphere_indices, g_circle_indices, g_point_indices, strict=True))
    g_cylinder_group_indices = list(
        zip(
            g_cylinder_indices,
            g_cylinder_intersection_indices,
            g_cylinder_ring_indices,
            strict=True,
        )
    )

    frames = []
    for i, th in enumerate(theta_all):
        Bx, By, Bz = _ewald_surface(th, Ew_x, Ew_y, Ew_z)
        kx, ky, kz = _k_vector(th)
        circles = [
            g_intersection_circle(
                th, g_val, palette[j % len(palette)], visibility=None
            )
        for j, g_val in enumerate(unique_g)
    ]
        ring_intersections = g_ring_intersection_points(th, visibility=None)
        cylinder_intersections = g_cylinder_intersection_curves(th, visibility=None)
        frames.append(
            go.Frame(
                name=f"f{i}",
                data=[
                    go.Surface(
                        x=Bx,
                        y=By,
                        z=Bz,
                        opacity=1.0,
                        colorscale="Blues",
                        showscale=False,
                    ),
                    go.Scatter3d(
                        x=kx,
                        y=ky,
                        z=kz,
                        mode="lines",
                        line=dict(color="black", width=5),
                    ),
                    go.Cone(
                        x=[kx[1]],
                        y=[ky[1]],
                        z=[kz[1]],
                        u=[kx[1] - kx[0]],
                        v=[ky[1] - ky[0]],
                        w=[kz[1] - kz[0]],
                        anchor="tip",
                        sizemode="absolute",
                        sizeref=0.2,
                        colorscale=[[0, "black"], [1, "black"]],
                        showscale=False,
                    ),
                    _k_label(kx, ky, kz),
                    _theta_arc(th),
                    _theta_arc_label(th),
                    lattice_marker(th),
                    hit_projection(th),
                    hit_labels(th),
                    *ring_intersections,
                    *cylinder_intersections,
                    *circles,
                ],
                traces=[
                    ewald_idx,
                    cone_idx - 1,
                    cone_idx,
                    k_label_idx,
                    arc_idx,
                    arc_label_idx,
                    lattice_idx,
                    projection_idx,
                    hit_label_idx,
                    *g_ring_intersection_indices,
                    *g_cylinder_intersection_indices,
                    *g_circle_indices,
                ],
            )
        )
    fig.frames = frames

    steps = [dict(method="animate",
                  args=[[f.name],
                        dict(frame=dict(duration=0, redraw=True),
                             transition=dict(duration=0),
                             mode="immediate")],
                  label=f"{math.degrees(th):.1f}°")
             for th, f in zip(theta_all, fig.frames)]
    sliders = [dict(steps=steps,
                    currentvalue=dict(prefix="θᵢ: "),
                    x=0.5, xanchor="center", y=-0.1, len=0.9)]

    buttons = [
        dict(label="Single crystal",
             method="update",
             args=[{"visible": lattice_visibility}, {}]),
        dict(label="3D Powder",
             method="update",
             args=[{"visible": g_sphere_visibility}, {}]),
        dict(label="2D Powder",
             method="update",
             args=[{"visible": g_ring_visibility}, {}]),
        dict(label="Cylinder",
             method="update",
             args=[{"visible": g_cylinder_visibility}, {}]),
    ]

    fig.update_layout(scene=dict(xaxis=dict(title=dict(text="$G_x\\,(\\mathrm{\\AA}^{-1})$"),
                                            range=[-axis_range, axis_range],
                                            autorange=False,
                                            showbackground=False,
                                            showticklabels=False,
                                            zeroline=False),
                                 yaxis=dict(title=dict(text="$G_y\\,(\\mathrm{\\AA}^{-1})$"),
                                            range=[-axis_range, axis_range],
                                            autorange=False,
                                            showbackground=False,
                                            showticklabels=False,
                                            zeroline=False),
                                 zaxis=dict(title=dict(text="$G_z\\,(\\mathrm{\\AA}^{-1})$"),
                                            range=[-axis_range, axis_range],
                                            autorange=False,
                                            showbackground=False,
                                            showticklabels=False,
                                            zeroline=False),
                                 aspectmode="cube",
                                 bgcolor="rgba(0,0,0,0)",
                                 camera=dict(eye=dict(x=CAMERA_EYE[0],
                                                     y=CAMERA_EYE[1],
                                                     z=CAMERA_EYE[2]))),
                      paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, b=0, t=0),
                      sliders=sliders,
                      updatemenus=[dict(type="buttons",
                                        buttons=buttons,
                                        direction="right",
                                        x=0.5,
                                        xanchor="center",
                                        y=1.1,
                                        pad=dict(t=5, r=5))],
                      meta=dict(lattice_visibility=lattice_visibility,
                                g_sphere_visibility=g_sphere_visibility,
                                g_sphere_indices=g_sphere_indices,
                                g_circle_indices=g_circle_indices,
                                g_point_indices=g_point_indices,
                                ewald_idx=ewald_idx,
                                g_ring_indices=g_ring_indices,
                                g_ring_intersection_indices=g_ring_intersection_indices,
                                g_group_indices=g_group_indices,
                                g_ring_groups=g_ring_groups,
                                g_ring_visibility=g_ring_visibility,
                                g_cylinder_indices=g_cylinder_indices,
                                g_cylinder_intersection_indices=g_cylinder_intersection_indices,
                                g_cylinder_ring_indices=g_cylinder_ring_indices,
                                g_cylinder_group_indices=g_cylinder_group_indices,
                                g_cylinder_visibility=g_cylinder_visibility,
                                g_values=unique_g.tolist()))

    context = dict(
        g_values=unique_g.tolist(),
        g_sphere_indices=g_sphere_indices,
        g_circle_indices=g_circle_indices,
        g_point_indices=g_point_indices,
        g_ring_indices=g_ring_indices,
        g_ring_groups=g_ring_groups,
        g_group_indices=g_group_indices,
        g_ring_specs=g_ring_specs,
        lattice_visibility=lattice_visibility,
        g_sphere_visibility=g_sphere_visibility,
        g_ring_visibility=g_ring_visibility,
        g_cylinder_visibility=g_cylinder_visibility,
        g_cylinder_specs=cylinder_values,
        g_cylinder_groups=g_cylinder_group_indices,
        g_two_thetas=g_two_thetas,
    )

    return fig, context


def _selector_checkbox_html(g_values: list[float], group_indices: list[tuple[int, int, int]], two_thetas: list[str]) -> str:
    lines = ["<div id=\"g-selector\">"]
    for i, (g_val, (sphere_idx, circle_idx, points_idx), two_theta) in enumerate(
        zip(g_values, group_indices, two_thetas, strict=True)
    ):
        checked = "checked" if i == 0 else ""
        lines.append(
            f'<label class="g-option"><input class="g-toggle" type="checkbox" '
            f'data-pos="{i}" data-traces="{sphere_idx},{circle_idx},{points_idx}" {checked}>'
            f'2θ ≈ {two_theta}</label>'
        )
    lines.append("</div>")
    return "\n".join(lines)


def _ring_selector_checkbox_html(
    ring_specs: list[tuple[float, float]], ring_groups: list[tuple[int, int]]
) -> str:
    lines = ["<div id=\"g-selector\">"]
    for i, ((g_r_val, g_z_val), (ring_idx, intersection_idx)) in enumerate(
        zip(ring_specs, ring_groups, strict=True)
    ):
        lines.append(
            f'<label class="g-option"><input class="g-toggle" type="checkbox" '
            f'data-pos="{i}" data-traces="{ring_idx},{intersection_idx}" checked>'
            f'|Gᵣ| ≈ {g_r_val:.3f} Å⁻¹, G_z ≈ {g_z_val:.3f} Å⁻¹</label>'
        )
    lines.append("</div>")
    return "\n".join(lines)


def _cylinder_selector_checkbox_html(
    cylinder_specs: list[float], cylinder_groups: list[tuple[int, int, int]]
) -> str:
    lines = ["<div id=\"g-selector\">"]
    for i, (g_r_val, (cyl_idx, intersection_idx, ring_idx)) in enumerate(
        zip(cylinder_specs, cylinder_groups, strict=True)
    ):
        checked = "checked" if i == 0 else ""
        lines.append(
            f'<label class="g-option"><input class="g-toggle" type="checkbox" '
            f'data-pos="{i}" data-traces="{cyl_idx},{intersection_idx},{ring_idx}" {checked}>'
            f'|Gᵣ| ≈ {g_r_val:.3f} Å⁻¹</label>'
        )
    lines.append("</div>")
    return "\n".join(lines)


def build_interactive_page(fig: go.Figure, context: dict) -> str:
    import plotly.io as pio

    g_values: list[float] = context["g_values"]
    g_group_indices: list[tuple[int, int, int]] = context["g_group_indices"]
    g_two_thetas: list[str] = context["g_two_thetas"]
    g_ring_specs: list[tuple[float, float]] = context["g_ring_specs"]
    g_ring_indices: list[int] = context["g_ring_indices"]
    g_ring_groups: list[tuple[int, int]] = context["g_ring_groups"]
    lattice_visibility: list[bool] = context["lattice_visibility"]
    g_sphere_visibility: list[bool] = context["g_sphere_visibility"]
    g_ring_visibility: list[bool] = context["g_ring_visibility"]
    g_cylinder_visibility: list[bool] = context["g_cylinder_visibility"]
    g_cylinder_specs: list[float] = context["g_cylinder_specs"]
    g_cylinder_groups: list[tuple[int, int, int]] = context["g_cylinder_groups"]

    figure_id = "mono-figure"
    figure_html = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        auto_play=False,
        div_id=figure_id,
        config={"responsive": True},
    )
    selector_controls = _selector_checkbox_html(g_values, g_group_indices, g_two_thetas)
    ring_selector_controls = _ring_selector_checkbox_html(g_ring_specs, g_ring_groups)
    cylinder_selector_controls = _cylinder_selector_checkbox_html(
        g_cylinder_specs, g_cylinder_groups
    )

    return f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Mono simulator</title>
  <style>
    html, body {{ height: 100%; width: 100%; margin: 0; padding: 0; }}
    body {{ font-family: Arial, sans-serif; }}
    #wrapper {{ display: flex; flex-direction: column; width: 100%; height: 100vh; }}
    #note {{ padding: 8px 12px; background: #f2f2f2; width: 100%; box-sizing: border-box; }}
    #controls {{ padding: 8px 12px; width: 100%; box-sizing: border-box; }}
    #figure-container {{ flex: 1 1 auto; width: 100%; min-height: 0; }}
    #figure-container .plotly-graph-div,
    #figure-container .js-plotly-plot {{ height: 100% !important; width: 100% !important; }}
  </style>
</head>
  <body>
    <div id=\"wrapper\">
      <div id=\"note\">Use the pop-out window to toggle 3D powder shells, 2D powder rings, or Gᵣ cylinders, or use the buttons to switch between lattice, 3D Powder, 2D Powder, and Cylinder views.</div>
      <div id=\"controls\">
        <button id=\"open-selector\">Open powder selector window</button>
        <button id=\"download-all\" style=\"margin-left:8px;\">Download all views</button>
        <label style=\"margin-left:12px;\">Ewald opacity:
          <input id=\"ewald-opacity\" type=\"range\" min=\"0\" max=\"1\" step=\"0.05\" value=\"1\">
          <span id=\"ewald-opacity-value\">1.00</span>
        </label>
        <span id=\"selector-status\" style=\"margin-left:8px;color:#444;\"></span>
      </div>
      <div id=\"figure-container\">{figure_html}</div>
    </div>
    <script>
      window.addEventListener('DOMContentLoaded', () => {{
        const figure = document.getElementById('{figure_id}');
        const latticeVis = {json.dumps(lattice_visibility)};
        const gSphereBase = {json.dumps(g_sphere_visibility)};
        const gRingBase = {json.dumps(g_ring_visibility)};
        const gCylinderBase = {json.dumps(g_cylinder_visibility)};
        const sphereSelectorHtml = `{selector_controls}`;
        const ringSelectorHtml = `{ring_selector_controls}`;
        const cylinderSelectorHtml = `{cylinder_selector_controls}`;
        const selectorBtn = document.getElementById('open-selector');
        const downloadBtn = document.getElementById('download-all');
        const selectorStatus = document.getElementById('selector-status');
        const ewaldSlider = document.getElementById('ewald-opacity');
        const ewaldValue = document.getElementById('ewald-opacity-value');
        const meta = figure.layout && figure.layout.meta ? figure.layout.meta : {{}};
        const ewaldIdx = typeof meta.ewald_idx === 'number'
          ? meta.ewald_idx
          : (figure.data || []).findIndex((trace) => trace && trace.name === 'Ewald sphere');
        let selectorMode = '3d';
        let selectorWin = null;

        function applyEwaldOpacity(alpha) {{
          const clamped = Math.min(1, Math.max(0, Number.isFinite(alpha) ? alpha : 1));
          if (ewaldValue) {{
            ewaldValue.textContent = clamped.toFixed(2);
          }}
          if (ewaldSlider && clamped !== parseFloat(ewaldSlider.value)) {{
            ewaldSlider.value = clamped.toString();
          }}
          if (ewaldIdx >= 0) {{
            Plotly.restyle(figure, {{ opacity: clamped }}, [ewaldIdx]);
          }}
          if (Array.isArray(figure.frames)) {{
            figure.frames.forEach((frame) => {{
              if (frame && frame.data && frame.data.length > 0 && frame.data[0]) {{
                frame.data[0].opacity = clamped;
              }}
            }});
          }}
        }}

        if (ewaldSlider) {{
          ewaldSlider.addEventListener('input', (evt) => {{
            const val = parseFloat(evt.target.value);
            applyEwaldOpacity(val);
          }});
          applyEwaldOpacity(parseFloat(ewaldSlider.value));
        }}

        function traceVisibilitySnapshot() {{
          return (figure.data || []).map((trace) => {{
            if (trace.visible === undefined) {{
              return true;
            }}
            return trace.visible;
          }});
        }}

        async function downloadViewSeries() {{
          if (!downloadBtn) {{
            return;
          }}

          downloadBtn.disabled = true;
          selectorStatus.textContent = 'Preparing downloads...';

          const originalVis = traceVisibilitySnapshot();
          const modes = [
            {{ label: 'single_crystal', vis: latticeVis }},
            {{ label: '3d_powder', vis: gSphereBase }},
            {{ label: '2d_powder', vis: gRingBase }},
            {{ label: 'cylinder', vis: gCylinderBase }},
          ];

          try {{
            for (const mode of modes) {{
              const vis = Array.from(mode.vis, (v) => !!v);
              await Plotly.update(figure, {{ visible: vis }});
              const uri = await Plotly.toImage(figure, {{ format: 'png', height: 900, width: 900 }});

              const link = document.createElement('a');
              link.href = uri;
              link.download = `mono_${{mode.label}}.png`;
              document.body.appendChild(link);
              link.click();
              link.remove();
            }}

            selectorStatus.textContent = 'Downloaded Single, 3D, 2D, and Cylinder views.';
          }} catch (err) {{
            selectorStatus.textContent = 'Download failed. Please retry.';
            console.error(err);
          }} finally {{
            await Plotly.update(figure, {{ visible: originalVis }});
            downloadBtn.disabled = false;
          }}
        }}

        function writeSelectorDoc(targetWin, mode = '3d') {{
          const is3D = mode === '3d';
          const isRing = mode === '2d';
          const heading = is3D ? '3D powder shells' : isRing ? '2D powder rings' : 'Cylinders (|Gᵣ|)';
          const intro = is3D
            ? 'Select which |G| shells to display. Default view shows only the smallest |G|.'
            : isRing
              ? 'Select which |Gᵣ| rings to display for the chosen G_z plane.'
              : 'Select which |Gᵣ| cylinders to display along G_z. Rings at lattice G_z slices on each cylinder and the Ewald intersection are shown.';
          const bodyHtml = is3D ? sphereSelectorHtml : isRing ? ringSelectorHtml : cylinderSelectorHtml;
          targetWin.document.open();
          targetWin.document.write(`<!doctype html><html lang="en"><head><meta charset="utf-8"><title>${{heading}}</title>
          <style>body {{ font-family: Arial, sans-serif; padding: 12px; margin: 0; }} .g-option {{ display: block; margin: 6px 0; }}
    </style>
          </head><body><h3>${{heading}}</h3>
          <p>${{intro}}</p>
          <div style="margin-bottom:8px;"><button id="check-all">Show all</button> <button id="check-none">Show none</button></div>
          ${{bodyHtml}}
          </body></html>`);
          targetWin.document.close();
          targetWin.__selectorMode = mode;
        }}

        function hookSelector(targetWin, mode = '3d') {{
          const base = mode === '3d' ? gSphereBase : mode === '2d' ? gRingBase : gCylinderBase;
          const checkboxes = targetWin.document.querySelectorAll('.g-toggle');

          function applySelection() {{
            const vis = Array.from(base);
            checkboxes.forEach((cb) => {{
              const traces = (cb.getAttribute('data-traces') || '').split(',').map(Number).filter((v) => !Number.isNaN(v));
              traces.forEach((idx) => {{ vis[idx] = cb.checked; }});
            }});
            Plotly.update(figure, {{visible: vis}});
          }}

          checkboxes.forEach((cb) => cb.addEventListener('change', applySelection));

          const checkAll = targetWin.document.getElementById('check-all');
          if (checkAll) {{
            checkAll.addEventListener('click', () => {{
              checkboxes.forEach((cb) => {{ cb.checked = true; }});
              applySelection();
            }});
          }}
          const checkNone = targetWin.document.getElementById('check-none');
          if (checkNone) {{
            checkNone.addEventListener('click', () => {{
              checkboxes.forEach((cb) => {{ cb.checked = false; }});
              applySelection();
            }});
          }}

          return applySelection;
        }}

        function openSelectorWindow(focusOnly = false, mode = selectorMode) {{
          selectorMode = mode;
          if (!selectorWin || selectorWin.closed) {{
            selectorWin = window.open('', 'g_selector_window', 'width=340,height=600');
          }}
          if (!selectorWin) {{
            selectorStatus.textContent = 'Pop-up blocked. Click the button to retry after allowing pop-ups.';
            return null;
          }}
          if (!focusOnly || selectorWin.document.body?.childElementCount === 0 || selectorWin.__selectorMode !== mode) {{
            writeSelectorDoc(selectorWin, mode);
          }}
          selectorWin.focus();
          selectorStatus.textContent = mode === '3d'
            ? '3D powder selector opened.'
            : mode === '2d'
              ? '2D powder selector opened.'
              : 'Cylinder selector opened.';
          return hookSelector(selectorWin, mode);
        }}

        let applySelection = openSelectorWindow(true);
        if (!applySelection) {{
          selectorStatus.textContent = 'Pop-up blocked. Please allow pop-ups and click the button.';
        }}

        selectorBtn.addEventListener('click', () => {{
          const applyFn = openSelectorWindow(false, selectorMode);
          if (applyFn) {{
            applySelection = applyFn;
          }}
        }});

        if (downloadBtn) {{
          downloadBtn.addEventListener('click', () => {{
            downloadViewSeries();
          }});
        }}

        figure.on('plotly_buttonclicked', (evt) => {{
          const label = evt.button && evt.button.label;
          if (label === 'Single crystal') {{
            Plotly.update(figure, {{visible: latticeVis}});
          }} else if (label === '3D Powder') {{
            selectorMode = '3d';
            const applyFn = openSelectorWindow(true, '3d');
            if (applyFn) {{
              applySelection = applyFn;
              applySelection();
            }}
          }} else if (label === '2D Powder') {{
            selectorMode = '2d';
            const applyFn = openSelectorWindow(true, '2d');
            if (applyFn) {{
              applySelection = applyFn;
              applySelection();
            }}
          }} else if (label === 'Cylinder') {{
            selectorMode = 'cyl';
            const applyFn = openSelectorWindow(true, 'cyl');
            if (applyFn) {{
              applySelection = applyFn;
              applySelection();
            }}
          }}
        }});
      }});
    </script>
  </body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single Ewald-sphere simulator")
    parser.add_argument("--theta-min", type=float, default=THETA_DEFAULT_MIN,
                        help=f"Minimum θᵢ in degrees (default: {THETA_DEFAULT_MIN})")
    parser.add_argument("--theta-max", type=float, default=THETA_DEFAULT_MAX,
                        help=f"Maximum θᵢ in degrees (default: {THETA_DEFAULT_MAX})")
    parser.add_argument("--frames", type=int, default=N_FRAMES_DEFAULT,
                        help=f"Number of slider steps (default: {N_FRAMES_DEFAULT})")
    return parser.parse_args()


def main(theta_min: float | None = None,
         theta_max: float | None = None,
         frames: int = N_FRAMES_DEFAULT) -> None:
    """Launch the mono simulator."""

    import plotly.io as pio

    pio.renderers.default = "browser"

    th_min = math.radians(theta_min if theta_min is not None else THETA_DEFAULT_MIN)
    th_max = math.radians(theta_max if theta_max is not None else THETA_DEFAULT_MAX)

    fig, context = build_mono_figure(th_min, th_max, frames)
    html = build_interactive_page(fig, context)

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as handle:
        handle.write(html)
        html_path = handle.name

    webbrowser.open(f"file://{html_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args.theta_min, args.theta_max, args.frames)
