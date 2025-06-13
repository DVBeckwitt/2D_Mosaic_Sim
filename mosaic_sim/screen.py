# coding: utf-8
"""Far-field detector pattern for the mosaic simulation."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from .constants import a_hex, c_hex, K_MAG, d_hex
from .geometry import intersection_circle
from .intensity import mosaic_intensity

__all__ = ["build_screen_figure", "main"]


def build_screen_figure(
    H: int = 0,
    K: int = 0,
    L: int = 12,
    sigma: float = np.deg2rad(0.8),
    gamma: float = np.deg2rad(5.0),
    eta: float = 0.5,
    det_y: float = 1.0e4,
) -> go.Figure:
    """Return a Plotly figure of the pattern on a distant detector."""

    d_hkl = d_hex(H, K, L, a_hex, c_hex)
    G_MAG = 2 * math.pi / d_hkl

    rx, ry, rz = intersection_circle(G_MAG, K_MAG, K_MAG)
    I_ring = mosaic_intensity(rx, ry, rz, H, K, L, sigma, gamma, eta)

    scale = det_y / ry
    x_det = rx * scale
    z_det = rz * scale

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_det,
            y=z_det,
            mode="markers",
            marker=dict(
                color=I_ring,
                colorscale="Viridis",
                size=6,
                showscale=True,
                colorbar=dict(title="Intensity"),
            ),
        )
    )
    fig.update_xaxes(title="x", scaleanchor="y")
    fig.update_yaxes(title="z")
    fig.update_layout(
        title=f"Flat detector at y={det_y}",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, b=0, t=30),
    )
    return fig


def main() -> None:
    """Launch the far-field detector figure."""

    import plotly.io as pio

    pio.renderers.default = "browser"
    fig = build_screen_figure()
    fig.show()
