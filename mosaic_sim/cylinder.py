"""Static Ewald-sphere figure with a hollow cylinder intersection.

The figure is similar to :mod:`mosaic_sim.animation` but instead of an
animation it exposes a slider allowing manual stepping through rocking
angles.  A hollow cylinder centred on the origin and running along the ``qz``
axis illustrates where the Bragg ring would intersect a rod of scattering.
"""

from __future__ import annotations

import math
import numpy as np
import plotly.graph_objects as go

from .constants import a_hex, c_hex, K_MAG, d_hex
from .geometry import (
    sphere,
    rot_x,
    intersection_circle,
    intersection_cylinder_sphere,
)
from .intensity import mosaic_intensity

__all__ = ["build_cylinder_figure", "main"]


def build_cylinder_figure(H: int = 0, K: int = 0, L: int = 12,
                          sigma: float = np.deg2rad(0.8),
                          gamma: float = np.deg2rad(5.0),
                          eta: float = 0.5) -> go.Figure:
    """Return a Plotly figure showing a Bragg sphere and a hollow cylinder.

    Parameters mirror :func:`mosaic_sim.animation.build_animation` but the
    result is a static figure with a slider.  The radius of the cylinder
    matches the Bragg-ring radius and extends from ``qz=0`` to
    ``qz=5·|qz|``.
    """

    d_hkl = d_hex(H, K, L, a_hex, c_hex)
    G_MAG = 2 * math.pi / d_hkl

    phi, theta = np.meshgrid(np.linspace(0, math.pi, 100),
                             np.linspace(0, 2 * math.pi, 200))
    Ew_x, Ew_y, Ew_z = sphere(K_MAG, phi, theta, (0, K_MAG, 0))
    B0_x, B0_y, B0_z = sphere(G_MAG, phi, theta)

    I_surface = mosaic_intensity(B0_x, B0_y, B0_z, H, K, L,
                                 sigma, gamma, eta)

    br_x, br_y, br_z = intersection_circle(G_MAG, K_MAG, K_MAG)
    gr = math.sqrt(br_x[0] ** 2 + br_z[0] ** 2)

    cyl_line_x, cyl_line_y, cyl_line_z = intersection_cylinder_sphere(
        gr,
        K_MAG,
        K_MAG,
        0.0,
    )

    t_cyl, z_cyl = np.meshgrid(np.linspace(0, 2 * math.pi, 60),
                               np.linspace(0, 5 * abs(gr), 60))
    cyl_x = gr * np.cos(t_cyl)
    cyl_y = gr * np.sin(t_cyl)
    cyl_z = z_cyl

    fig = go.Figure()
    bragg = go.Surface(x=B0_x, y=B0_y, z=B0_z,
                       surfacecolor=I_surface,
                       colorscale=[[0, "rgba(128,128,128,0.25)"],
                                   [1, "rgba(255,0,0,1)"]],
                       showscale=True,
                       colorbar=dict(title="Mosaic<br>Intensity"),
                       name="Bragg sphere")
    fig.add_trace(bragg)
    bragg_idx = len(fig.data) - 1

    fig.add_trace(go.Surface(x=Ew_x, y=Ew_y, z=Ew_z,
                             opacity=0.3, colorscale="Blues", showscale=False,
                             name="Ewald sphere"))

    fig.add_trace(
        go.Surface(
            x=cyl_x,
            y=cyl_y,
            z=cyl_z,
            opacity=0.5,
            showscale=False,
            colorscale=[[0, "rgb(255,191,0)"], [1, "rgb(255,191,0)"]],
            name="Cylinder",
        )
    )
    cyl_idx = len(fig.data) - 1

    fig.add_trace(
        go.Scatter3d(
            x=cyl_line_x,
            y=cyl_line_y,
            z=cyl_line_z,
            mode="lines",
            line=dict(color="green", width=5),
            name="Cylinder/Ewald overlap",
        )
    )
    overlap_idx = len(fig.data) - 1

    k_tail = np.array([0.0, K_MAG, 0.0])
    k_head = k_tail * 0.25
    fig.add_trace(go.Scatter3d(x=[k_tail[0], k_head[0]],
                               y=[k_tail[1], k_head[1]],
                               z=[k_tail[2], k_head[2]],
                               mode="lines", line=dict(color="black", width=5),
                               name="kᵢ"))
    fig.add_trace(go.Cone(x=[k_head[0]], y=[k_head[1]], z=[k_head[2]],
                          u=[-k_head[0]], v=[-k_head[1]], w=[-k_head[2]],
                          anchor="tail", sizemode="absolute", sizeref=0.2,
                          colorscale=[[0, "black"], [1, "black"]],
                          showscale=False))

    R_MAX = max(G_MAG, K_MAG)
    for xyz in [([-R_MAX, R_MAX], [0, 0], [0, 0]),
                ([0, 0], [-R_MAX, 2 * R_MAX], [0, 0]),
                ([0, 0], [0, 0], [-R_MAX, R_MAX])]:
        fig.add_trace(go.Scatter3d(x=xyz[0], y=xyz[1], z=xyz[2],
                                   mode="lines", showlegend=False,
                                   line=dict(color="black",
                                             width=2,
                                             dash="dash")))

    fig.update_layout(scene=dict(xaxis=dict(visible=False),
                                 yaxis=dict(visible=False),
                                 zaxis=dict(visible=False),
                                 bgcolor="rgba(0,0,0,0)"),
                      paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, b=0, t=0))

    THETA_MIN, THETA_MAX = np.deg2rad(5), np.deg2rad(30)
    N_FRAMES = 60
    theta_all = np.linspace(THETA_MIN, THETA_MAX, N_FRAMES)

    frames = []
    for i, th in enumerate(theta_all):
        Bx, By, Bz = rot_x(B0_x, B0_y, B0_z, -th)
        Cx, Cy, Cz = rot_x(cyl_x, cyl_y, cyl_z, -th)

        # Intersection line for the rotated cylinder
        Lx_r, Ly_r, Lz_r = intersection_cylinder_sphere(
            gr,
            K_MAG,
            K_MAG * math.cos(th),
            K_MAG * math.sin(th),
        )
        Lx, Ly, Lz = rot_x(Lx_r, Ly_r, Lz_r, -th)
        frames.append(
            go.Frame(
                name=f"f{i}",
                data=[
                    go.Surface(
                        x=Bx,
                        y=By,
                        z=Bz,
                        surfacecolor=I_surface,
                        colorscale=bragg.colorscale,
                        showscale=False,
                    ),
                    go.Surface(
                        x=Cx,
                        y=Cy,
                        z=Cz,
                        opacity=0.5,
                        showscale=False,
                        colorscale=[[0, "rgb(255,191,0)"], [1, "rgb(255,191,0)"]],
                    ),
                    go.Scatter3d(
                        x=Lx,
                        y=Ly,
                        z=Lz,
                        mode="lines",
                        line=dict(color="green", width=5),
                    ),
                ],
                traces=[bragg_idx, cyl_idx, overlap_idx],
            )
        )
    fig.frames = frames

    steps = [dict(method="animate",
                  args=[[f.name],
                        dict(frame=dict(duration=0, redraw=True),
                             transition=dict(duration=0),
                             mode="immediate")],
                  label=f"{i}")
             for i, f in enumerate(fig.frames)]
    sliders = [dict(steps=steps, currentvalue=dict(prefix="θ index: "),
                    x=0.5, xanchor="center", y=-0.1, len=0.9)]
    fig.update_layout(sliders=sliders)

    return fig


def main() -> None:
    """Launch the static cylinder figure in a browser."""

    import plotly.io as pio
    pio.renderers.default = "browser"
    fig = build_cylinder_figure()
    fig.show()


if __name__ == "__main__":
    main()
