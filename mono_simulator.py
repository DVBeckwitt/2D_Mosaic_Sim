#!/usr/bin/env python3
"""Command-line entry for a single Ewald-sphere visualisation.

This script exposes a minimal view that only shows the Ewald sphere and allows
interactive control of the incident angle ``theta_i``.
"""

import argparse
import math

import numpy as np
import plotly.graph_objects as go

from mosaic_sim.constants import K_MAG
from mosaic_sim.geometry import rot_x, sphere


THETA_DEFAULT_MIN = 5.0
THETA_DEFAULT_MAX = 30.0
N_FRAMES_DEFAULT = 60


def _ewald_surface(theta_i: float, Ew_x: np.ndarray, Ew_y: np.ndarray, Ew_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return rot_x(Ew_x, Ew_y, Ew_z, theta_i)


def _k_vector(theta_i: float) -> tuple[list[float], list[float], list[float]]:
    k_tail = np.array([0.0, K_MAG, 0.0])
    k_head = k_tail * 0.25
    tail_x, tail_y, tail_z = rot_x(np.array([k_tail[0]]), np.array([k_tail[1]]), np.array([k_tail[2]]), theta_i)
    head_x, head_y, head_z = rot_x(np.array([k_head[0]]), np.array([k_head[1]]), np.array([k_head[2]]), theta_i)
    return (
        [tail_x[0], head_x[0]],
        [tail_y[0], head_y[0]],
        [tail_z[0], head_z[0]],
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
    Ew_x, Ew_y, Ew_z = sphere(K_MAG, phi, theta, (0, K_MAG, 0))

    theta_all = np.linspace(theta_min, theta_max, n_frames)

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
            u=[kx[0] - kx[1]],
            v=[ky[0] - ky[1]],
            w=[kz[0] - kz[1]],
            anchor="tail",
            sizemode="absolute",
            sizeref=0.2,
            colorscale=[[0, "black"], [1, "black"]],
            showscale=False,
        )
    )
    cone_idx = len(fig.data) - 1

    R_MAX = K_MAG
    for xyz in [([-R_MAX, R_MAX], [0, 0], [0, 0]),
                ([0, 0], [-R_MAX, 2 * R_MAX], [0, 0]),
                ([0, 0], [0, 0], [-R_MAX, R_MAX])]:
        fig.add_trace(go.Scatter3d(x=xyz[0], y=xyz[1], z=xyz[2],
                                   mode="lines", showlegend=False,
                                   line=dict(color="black",
                                             width=2,
                                             dash="dash")))

    grid_coords = np.arange(-5, 6)
    gx, gy, gz = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing="ij")
    lattice_trace = go.Scatter3d(
        x=gx.ravel(),
        y=gy.ravel(),
        z=gz.ravel(),
        mode="markers",
        marker=dict(size=4, color="red", opacity=0.8),
        name="Integer lattice",
    )
    fig.add_trace(lattice_trace)
    lattice_idx = len(fig.data) - 1

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
                        u=[kx[0] - kx[1]],
                        v=[ky[0] - ky[1]],
                        w=[kz[0] - kz[1]],
                        anchor="tail",
                        sizemode="absolute",
                        sizeref=0.2,
                        colorscale=[[0, "black"], [1, "black"]],
                        showscale=False,
                    ),
                    lattice_trace,
                ],
                traces=[ewald_idx, cone_idx - 1, cone_idx, lattice_idx],
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
                      sliders=sliders)

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
