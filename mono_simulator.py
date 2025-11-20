#!/usr/bin/env python3
"""Command-line entry for a single Ewald-sphere visualisation.

This script exposes a minimal view that only shows the Ewald sphere and allows
interactive control of the incident angle ``theta_i``.
"""

import argparse
import math

import numpy as np
import plotly.graph_objects as go
from mosaic_sim.geometry import rot_x, sphere


K_MAG_PLOT = 1.5
THETA_BRAGG_002 = math.degrees(math.asin(1.0 / K_MAG_PLOT))
THETA_DEFAULT_MIN = 0.0
THETA_DEFAULT_MAX = THETA_BRAGG_002 + 10.0
N_FRAMES_DEFAULT = 60
CAMERA_EYE = np.array([2.0, 2.0, 1.6])
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
                      n_frames: int = N_FRAMES_DEFAULT) -> go.Figure:
    """Return a Plotly figure showing only the Ewald sphere.

    A slider controls the incident angle ``theta_i`` by rotating the sphere
    about the *qx* axis.
    """

    phi, theta = np.meshgrid(np.linspace(0, math.pi, 100),
                             np.linspace(0, 2 * math.pi, 200))
    Ew_x, Ew_y, Ew_z = sphere(K_MAG_PLOT, phi, theta, (0, K_MAG_PLOT, 0))

    def _intersection_thetas() -> list[float]:
        grid_coords = np.arange(-2, 3)
        gx, gy, gz = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing="ij")
        lattice_points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        thetas: list[float] = []
        for hx, hy, hz in lattice_points:
            if hx == hy == hz == 0:
                continue
            rhs = (hx * hx + hy * hy + hz * hz) / (2.0 * K_MAG_PLOT)
            amplitude = math.hypot(hy, hz)
            if amplitude == 0:
                continue
            ratio = rhs / amplitude
            if abs(ratio) > 1.0:
                continue
            base = math.atan2(hz, hy)
            delta = math.acos(ratio)
            thetas.extend([base + delta, base - delta])
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
            opacity=0.3,
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

    R_MAX = K_MAG_PLOT
    for xyz in [([-R_MAX, R_MAX], [0, 0], [0, 0]),
                ([0, 0], [-R_MAX, 2 * R_MAX], [0, 0]),
                ([0, 0], [0, 0], [-R_MAX, R_MAX])]:
        fig.add_trace(go.Scatter3d(x=xyz[0], y=xyz[1], z=xyz[2],
                                   mode="lines", showlegend=False,
                                   line=dict(color="black",
                                             width=2,
                                             dash="dash")))

    grid_coords = np.arange(-2, 3)
    gx, gy, gz = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing="ij")
    lattice_points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

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
        _, hits = lattice_hits(theta)
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
        _, hits = lattice_hits(theta)
        if hits.size == 0:
            return go.Scatter3d(x=[], y=[], z=[], mode="text", showlegend=False)
        x, y, z = hits[:, 0], hits[:, 1], hits[:, 2]
        labels = [f"({int(hx)}, {int(hy)}, {int(hz)})" for hx, hy, hz in hits]
        distances = np.linalg.norm(hits - CAMERA_EYE, axis=1)
        sizes = (14.0 * distances).tolist()
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

    frames = []
    for i, th in enumerate(theta_all):
        Bx, By, Bz = _ewald_surface(th, Ew_x, Ew_y, Ew_z)
        kx, ky, kz = _k_vector(th)
        frames.append(
            go.Frame(
                name=f"f{i}",
                data=[
                    go.Surface(
                        x=Bx,
                        y=By,
                        z=Bz,
                        opacity=0.3,
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

    fig.update_layout(scene=dict(xaxis=dict(visible=False),
                                 yaxis=dict(visible=False),
                                 zaxis=dict(visible=False),
                                 bgcolor="rgba(0,0,0,0)"),
                      paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, b=0, t=0),
                      sliders=sliders,
                      scene_camera=dict(eye=dict(x=CAMERA_EYE[0],
                                                 y=CAMERA_EYE[1],
                                                 z=CAMERA_EYE[2])))

    return fig


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

    fig = build_mono_figure(th_min, th_max, frames)
    fig.show()


if __name__ == "__main__":
    args = parse_args()
    main(args.theta_min, args.theta_max, args.frames)
