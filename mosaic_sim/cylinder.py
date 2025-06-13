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

__all__ = ["build_cylinder_figure", "main", "dash_main"]


def build_cylinder_figure(H: int = 0, K: int = 0, L: int = 12,
                          sigma: float = np.deg2rad(0.8),
                          gamma: float = np.deg2rad(5.0),
                          eta: float = 0.5) -> go.Figure:
    """Return a Plotly figure showing a Bragg sphere and a hollow cylinder.

    Parameters mirror :func:`mosaic_sim.animation.build_animation` but the
    result is a static figure with a slider.  The radius of the cylinder
    matches the Bragg-ring radius and spans ``qz=-4·|gr|`` to ``qz=4·|gr|``.

"""
    d_hkl = d_hex(H, K, L, a_hex, c_hex)
    G_MAG = 2 * math.pi / d_hkl

    phi, theta = np.meshgrid(np.linspace(0, math.pi, 100),
                             np.linspace(0, 2 * math.pi, 200))
    Ew_x, Ew_y, Ew_z = sphere(K_MAG, phi, theta, (0, K_MAG, 0))
    B0_x, B0_y, B0_z = sphere(G_MAG, phi, theta)

    # ``H``, ``K`` and ``L`` only define the Bragg-sphere radius.  The mosaic
    # intensity is always centred on ``+qz`` so that the reflection orientation
    # does not depend on the Miller indices.
    I_surface = mosaic_intensity(B0_x, B0_y, B0_z, 0, 0, 1,
                                 sigma, gamma, eta)

    ring_x, ring_y, ring_z = intersection_circle(G_MAG, K_MAG, K_MAG)
    gr = math.sqrt(ring_x[0] ** 2 + ring_z[0] ** 2)

    cyl_line_x, cyl_line_y, cyl_line_z = intersection_cylinder_sphere(
        gr,
        K_MAG,
        K_MAG,
        0.0,
    )

    # Intersection of the cylinder with the Bragg sphere (independent of the
    # rocking angle)
    br_line_x, br_line_y, br_line_z = intersection_cylinder_sphere(
        gr,
        G_MAG,
        0.0,
        0.0,
    )

    t_cyl, z_cyl = np.meshgrid(
        np.linspace(0, 2 * math.pi, 60), np.linspace(-4.0 * abs(gr), 4.0 * abs(gr), 60)

    )
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

    fig.add_trace(
        go.Surface(
            x=Ew_x,
            y=Ew_y,
            z=Ew_z,
            opacity=0.3,
            colorscale="Blues",
            showscale=False,
            name="Ewald sphere",
        )
    )
    ewald_idx = len(fig.data) - 1

    fig.add_trace(
        go.Scatter3d(
            x=ring_x,
            y=ring_y,
            z=ring_z,
            mode="lines",
            line=dict(color="red", width=5),
            name="Ewald/Bragg overlap",
        )
    )
    ring_idx = len(fig.data) - 1

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

    fig.add_trace(
        go.Scatter3d(
            x=br_line_x,
            y=br_line_y,
            z=br_line_z,
            mode="lines",
            line=dict(color="orange", width=5),
            name="Cylinder/Bragg overlap",
        )
    )
    br_overlap_idx = len(fig.data) - 1

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

    # Arrow showing the cylinder axis (rotation direction)
    axis_len = 4.0 * abs(gr)
    axis_line = go.Scatter3d(x=[0.0, 0.0],
                             y=[0.0, 0.0],
                             z=[0.0, 0.75 * axis_len],
                             mode="lines",
                             line=dict(color="purple", width=5),
                             name="Rotation axis")
    fig.add_trace(axis_line)
    axis_idx = len(fig.data) - 1
    fig.add_trace(go.Cone(x=[0.0], y=[0.0], z=[0.75 * axis_len],
                          u=[0.0], v=[0.0], w=[0.25 * axis_len],
                          anchor="tail", sizemode="absolute", sizeref=0.2,
                          colorscale=[[0, "purple"], [1, "purple"]],
                          showscale=False))
    cone_idx = len(fig.data) - 1

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
        Bcx, Bcy, Bcz = rot_x(br_line_x, br_line_y, br_line_z, -th)
        ax_line_x, ax_line_y, ax_line_z = rot_x(
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.75 * axis_len]),
            -th,
        )
        ax_tail_x, ax_tail_y, ax_tail_z = rot_x(
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.75 * axis_len]),
            -th,
        )
        ax_vec_x, ax_vec_y, ax_vec_z = rot_x(
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.25 * axis_len]),
            -th,
        )
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
                    go.Scatter3d(
                        x=Bcx,
                        y=Bcy,
                        z=Bcz,
                        mode="lines",
                        line=dict(color="orange", width=5),
                    ),
                    go.Scatter3d(
                        x=ax_line_x,
                        y=ax_line_y,
                        z=ax_line_z,
                        mode="lines",
                        line=dict(color="purple", width=5),
                    ),
                    go.Cone(
                        x=[ax_tail_x[0]],
                        y=[ax_tail_y[0]],
                        z=[ax_tail_z[0]],
                        u=[ax_vec_x[0]],
                        v=[ax_vec_y[0]],
                        w=[ax_vec_z[0]],
                        anchor="tail",
                        sizemode="absolute",
                        sizeref=0.2,
                        colorscale=[[0, "purple"], [1, "purple"]],
                        showscale=False,
                    ),
                ],
                traces=[bragg_idx, cyl_idx, overlap_idx, br_overlap_idx,
                        axis_idx, cone_idx],
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
    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            x=1.05,
            y=1.0,
            xanchor="left",
            yanchor="top",
            direction="down",
            buttons=[
                dict(
                    label="Show Bragg sphere",
                    method="restyle",
                    args=[{"visible": True}, [bragg_idx]],
                ),
                dict(
                    label="Hide Bragg sphere",
                    method="restyle",
                    args=[{"visible": False}, [bragg_idx]],
                ),
            ],
        ),
        dict(
            type="buttons",
            showactive=False,
            x=1.05,
            y=0.9,
            xanchor="left",
            yanchor="top",
            direction="down",
            buttons=[
                dict(
                    label="Show Ewald sphere",
                    method="restyle",
                    args=[{"visible": True}, [ewald_idx]],
                ),
                dict(
                    label="Hide Ewald sphere",
                    method="restyle",
                    args=[{"visible": False}, [ewald_idx]],
                ),
            ],
        ),
        dict(
            type="buttons",
            showactive=False,
            x=1.05,
            y=0.8,
            xanchor="left",
            yanchor="top",
            direction="down",
            buttons=[
                dict(
                    label="Show cylinder",
                    method="restyle",
                    args=[{"visible": True}, [cyl_idx]],
                ),
                dict(
                    label="Hide cylinder",
                    method="restyle",
                    args=[{"visible": False}, [cyl_idx]],
                ),
            ],
        ),
        dict(
            type="buttons",
            showactive=False,
            x=1.05,
            y=0.7,
            xanchor="left",
            yanchor="top",
            direction="down",
            buttons=[
                dict(
                    label="Show cyl/ew intersection",
                    method="restyle",
                    args=[{"visible": True}, [overlap_idx]],
                ),
                dict(
                    label="Hide cyl/ew intersection",
                    method="restyle",
                    args=[{"visible": False}, [overlap_idx]],
                ),
            ],
        ),
        dict(
            type="buttons",
            showactive=False,
            x=1.05,
            y=0.6,
            xanchor="left",
            yanchor="top",
            direction="down",
            buttons=[
                dict(
                    label="Show cyl/br intersection",
                    method="restyle",
                    args=[{"visible": True}, [br_overlap_idx]],
                ),
                dict(
                    label="Hide cyl/br intersection",
                    method="restyle",
                    args=[{"visible": False}, [br_overlap_idx]],
                ),
            ],
        ),
        dict(
            type="buttons",
            showactive=False,
            x=1.05,
            y=0.5,
            xanchor="left",
            yanchor="top",
            direction="down",
            buttons=[
                dict(
                    label="Show ew/br intersection",
                    method="restyle",
                    args=[{"visible": True}, [ring_idx]],
                ),
                dict(
                    label="Hide ew/br intersection",
                    method="restyle",
                    args=[{"visible": False}, [ring_idx]],
                ),
            ],
        ),
    ]

    fig.update_layout(sliders=sliders, updatemenus=updatemenus)

    return fig


def dash_main() -> None:
    """Launch a Dash app allowing live Miller index updates."""

    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output

    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            dcc.Graph(id="cylinder-fig", figure=build_cylinder_figure()),
            html.Div(
                [
                    html.Label("H"),
                    dcc.Input(id="H", type="number", value=0, step=1),
                    html.Label("K"),
                    dcc.Input(id="K", type="number", value=0, step=1),
                    html.Label("L"),
                    dcc.Input(id="L", type="number", value=12, step=1),
                ],
                style={"display": "flex", "gap": "0.5rem"},
            ),
        ]
    )

    @app.callback(
        Output("cylinder-fig", "figure"),
        Input("H", "value"),
        Input("K", "value"),
        Input("L", "value"),
    )
    def update(h, k, l):  # pragma: no cover - UI callback
        h = int(h) if h is not None else 0
        k = int(k) if k is not None else 0
        l = int(l) if l is not None else 0
        return build_cylinder_figure(h, k, l)

    app.run_server(debug=False)


def main() -> None:
    """Launch an interactive cylinder figure that can be updated."""

    import plotly.io as pio

    pio.renderers.default = "browser"

    print("Interactive cylinder simulation")
    print("Enter Miller indices as three integers separated by spaces.")
    print("Press <Enter> for the default (0 0 12) or 'q' to quit.")

    while True:
        try:
            line = input("H K L> ").strip()
        except EOFError:
            break
        if not line:
            h, k, l = 0, 0, 12
        elif line.lower() in {"q", "quit", "exit"}:
            break
        else:
            parts = line.split()
            if len(parts) != 3:
                print("Please enter three integers, e.g. '0 0 12'.")
                continue
            try:
                h, k, l = map(int, parts)
            except ValueError:
                print("Invalid input; please enter integers only.")
                continue

        fig = build_cylinder_figure(h, k, l)
        fig.show()


if __name__ == "__main__":
    main()
