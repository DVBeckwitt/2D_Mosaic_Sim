"""Interactive fibrous Bragg/Ewald/cylinder intersection viewer.

The figure exposes a slider for manual stepping through rocking angles. A
hollow cylinder centred on the origin and running along the ``qz`` axis
illustrates where the Bragg ring would intersect a rod of scattering.
"""

from __future__ import annotations

import argparse
import math
import threading
import webbrowser
from typing import Any
import numpy as np
import plotly.graph_objects as go

from .common import build_error_figure, normalize_peak_params
from .constants import a_hex, c_hex, K_MAG, INTERSECTION_LINE_WIDTH, d_hex
from .detector import extract_scene_camera
from .geometry import (
    sphere,
    rot_x,
    intersection_circle,
    intersection_cylinder_sphere,
)
from .intensity import mosaic_intensity

DEFAULT_H = 0
DEFAULT_K = 0
DEFAULT_L = 12
DEFAULT_SIGMA_DEG = 0.8
DEFAULT_GAMMA_DEG = 5.0
DEFAULT_ETA = 0.5
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8051
DEFAULT_CONTROL_VALUES = (
    DEFAULT_H,
    DEFAULT_K,
    DEFAULT_L,
    DEFAULT_SIGMA_DEG,
    DEFAULT_GAMMA_DEG,
    DEFAULT_ETA,
)
CYLINDER_CAMERA_UIREVISION = "cylinder-camera"

__all__ = [
    "build_cylinder_figure",
    "build_cylinder_app",
    "build_cylinder_error_figure",
    "normalize_cylinder_params",
    "main",
    "dash_main",
]


def _resolve_Gamma(
    Gamma: float | None,
    gamma: float | None,
    *,
    default: float,
) -> float:
    """Resolve the Lorentzian width while supporting legacy ``gamma`` calls."""

    if Gamma is None:
        return default if gamma is None else float(gamma)
    if gamma is not None and not math.isclose(
        float(Gamma),
        float(gamma),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Specify only one Lorentzian width: Γ or gamma")
    return float(Gamma)


def _cylinder_title(
    H: int,
    K: int,
    L: int,
    sigma: float,
    Gamma: float,
    eta: float,
) -> str:
    """Return the shared title string for the fibrous viewer."""

    return (
        f"HKL = ({H}, {K}, {L})"
        f" | Gaussian σ = {math.degrees(sigma):.2f}°"
        f" | Lorentzian Γ = {math.degrees(Gamma):.2f}°"
        f" | Mix η = {eta:.2f}"
    )


def _visibility_toggle_menu(
    label: str,
    trace_idx: int,
    *,
    x: float,
    y: float,
) -> dict[str, object]:
    """Return a styled single-button visibility toggle for one trace."""

    return dict(
        type="buttons",
        active=-1,
        showactive=True,
        x=x,
        y=y,
        xanchor="center",
        yanchor="middle",
        direction="left",
        bgcolor="rgba(255,255,255,0.96)",
        bordercolor="rgba(31,41,55,0.18)",
        borderwidth=1,
        pad=dict(l=8, r=8, t=4, b=4),
        buttons=[
            dict(
                label=label,
                method="restyle",
                args=[{"visible": "legendonly"}, [trace_idx]],
                args2=[{"visible": True}, [trace_idx]],
            )
        ],
    )


def _multi_visibility_toggle_menu(
    label: str,
    trace_indices: list[int],
    *,
    x: float,
    y: float,
) -> dict[str, object]:
    """Return a styled toggle that hides or shows multiple traces at once."""

    return dict(
        type="buttons",
        active=-1,
        showactive=True,
        x=x,
        y=y,
        xanchor="center",
        yanchor="middle",
        direction="left",
        bgcolor="rgba(255,255,255,0.96)",
        bordercolor="rgba(31,41,55,0.18)",
        borderwidth=1,
        pad=dict(l=8, r=8, t=4, b=4),
        buttons=[
            dict(
                label=label,
                method="restyle",
                args=[{"visible": "legendonly"}, trace_indices],
                args2=[{"visible": True}, trace_indices],
            )
        ],
    )


def normalize_cylinder_params(
    H: float | int | None = None,
    K: float | int | None = None,
    L: float | int | None = None,
    sigma_deg: float | None = None,
    Gamma_deg: float | None = None,
    eta: float | None = None,
    *,
    gamma_deg: float | None = None,
    defaults: tuple[int, int, int, float, float, float] = DEFAULT_CONTROL_VALUES,
) -> tuple[int, int, int, float, float, float]:
    """Normalize GUI or CLI parameters into validated fibrous-viewer inputs."""
    return normalize_peak_params(
        H,
        K,
        L,
        sigma_deg,
        Gamma_deg,
        eta,
        gamma_deg=gamma_deg,
        defaults=defaults,
    ).as_tuple()


def build_cylinder_error_figure(message: str) -> go.Figure:
    """Return a compact figure that surfaces invalid GUI inputs."""

    return build_error_figure("Fibrous Bragg/Ewald intersections", message)


def build_cylinder_figure(
    H: int = DEFAULT_H,
    K: int = DEFAULT_K,
    L: int = DEFAULT_L,
    sigma: float = np.deg2rad(DEFAULT_SIGMA_DEG),
    Gamma: float | None = None,
    eta: float = DEFAULT_ETA,
    *,
    gamma: float | None = None,
    camera: dict[str, Any] | None = None,
) -> go.Figure:
    """Return a Plotly figure showing a Bragg sphere and a hollow cylinder.

    The radius of the cylinder matches the Bragg-ring radius and spans
    ``qz=-4·|gr|`` to ``qz=4·|gr|``.

"""
    Gamma = _resolve_Gamma(
        Gamma,
        gamma,
        default=float(np.deg2rad(DEFAULT_GAMMA_DEG)),
    )
    d_hkl = d_hex(H, K, L, a_hex, c_hex)
    G_MAG = 2 * math.pi / d_hkl

    phi, theta = np.meshgrid(np.linspace(0, math.pi, 100),
                             np.linspace(0, 2 * math.pi, 200))
    Ew_x, Ew_y, Ew_z = sphere(K_MAG, phi, theta, (0, K_MAG, 0))
    B0_x, B0_y, B0_z = sphere(G_MAG, phi, theta)

    # ``H``, ``K`` and ``L`` define both the Bragg-sphere radius and the
    # mosaic profile.  Any in-plane component should switch to the belt model
    # rather than the cap profile used for purely out-of-plane reflections.
    I_surface = mosaic_intensity(B0_x, B0_y, B0_z, H, K, L,
                                 sigma, Gamma, eta)

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
            line=dict(color="red", width=INTERSECTION_LINE_WIDTH),
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
            line=dict(color="green", width=INTERSECTION_LINE_WIDTH),
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
            line=dict(color="orange", width=INTERSECTION_LINE_WIDTH),
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

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(0,0,0,0)",
            domain=dict(x=[0.0, 0.78], y=[0.0, 1.0]),
            uirevision=CYLINDER_CAMERA_UIREVISION,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        uirevision=CYLINDER_CAMERA_UIREVISION,
        margin=dict(l=0, r=240, b=90, t=220),
        legend=dict(
            x=0.82,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(31,41,55,0.18)",
            borderwidth=1,
            itemsizing="constant",
        ),
        title=dict(
            text=_cylinder_title(H, K, L, sigma, Gamma, eta),
            x=0.39,
            y=0.95,
            xanchor="center",
        ),
    )

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
                        line=dict(color="green", width=INTERSECTION_LINE_WIDTH),
                    ),
                    go.Scatter3d(
                        x=Bcx,
                        y=Bcy,
                        z=Bcz,
                        mode="lines",
                        line=dict(color="orange", width=INTERSECTION_LINE_WIDTH),
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

    steps = [
        dict(
            method="animate",
            args=[
                [frame.name],
                dict(
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                    mode="immediate",
                ),
            ],
            label=f"{math.degrees(theta_all[i]):.1f}",
        )
        for i, frame in enumerate(fig.frames)
    ]
    sliders = [
        dict(
            active=0,
            steps=steps,
            currentvalue=dict(prefix="θᵢ = ", suffix="°"),
            x=0.39,
            xanchor="center",
            y=-0.1,
            len=0.72,
        )
    ]
    updatemenus = [
        _visibility_toggle_menu("Bragg sphere", bragg_idx, x=0.15, y=1.13),
        _visibility_toggle_menu("Ewald sphere", ewald_idx, x=0.39, y=1.13),
        _visibility_toggle_menu("Cylinder", cyl_idx, x=0.63, y=1.13),
        _visibility_toggle_menu("Bragg/Ewald overlap", ring_idx, x=0.15, y=1.05),
        _visibility_toggle_menu("Cylinder/Ewald overlap", overlap_idx, x=0.39, y=1.05),
        _visibility_toggle_menu("Cylinder/Bragg overlap", br_overlap_idx, x=0.63, y=1.05),
        _multi_visibility_toggle_menu(
            "Disable all",
            [bragg_idx, ewald_idx, cyl_idx, ring_idx, overlap_idx, br_overlap_idx],
            x=0.39,
            y=0.97,
        ),
    ]

    fig.update_layout(sliders=sliders, updatemenus=updatemenus)
    if camera:
        fig.update_layout(scene_camera=camera)

    return fig


def build_cylinder_app(
    initial_H: int = DEFAULT_H,
    initial_K: int = DEFAULT_K,
    initial_L: int = DEFAULT_L,
    initial_sigma: float = np.deg2rad(DEFAULT_SIGMA_DEG),
    initial_Gamma: float | None = None,
    initial_eta: float = DEFAULT_ETA,
    *,
    initial_gamma: float | None = None,
):
    """Return a Dash app for live fibrous-viewer parameter updates."""

    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output, State

    initial_Gamma = _resolve_Gamma(
        initial_Gamma,
        initial_gamma,
        default=float(np.deg2rad(DEFAULT_GAMMA_DEG)),
    )

    control_defaults = (
        int(initial_H),
        int(initial_K),
        int(initial_L),
        math.degrees(initial_sigma),
        math.degrees(initial_Gamma),
        float(initial_eta),
    )

    app = dash.Dash(__name__)
    app.title = "Fibrous Bragg/Ewald Intersections"

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H2("Fibrous Bragg/Ewald intersections", style={"margin": "0"}),
                    html.Div(
                        "Use the in-figure θᵢ slider for rocking angle and the controls below to rebuild HKL and mosaic parameters.",
                        style={"color": "#555"},
                    ),
                ],
                style={"paddingBottom": "0.75rem"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("H"),
                            dcc.Input(
                                id="cylinder-H",
                                type="number",
                                value=control_defaults[0],
                                step=1,
                                debounce=True,
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "0.25rem"},
                    ),
                    html.Div(
                        [
                            html.Label("K"),
                            dcc.Input(
                                id="cylinder-K",
                                type="number",
                                value=control_defaults[1],
                                step=1,
                                debounce=True,
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "0.25rem"},
                    ),
                    html.Div(
                        [
                            html.Label("L"),
                            dcc.Input(
                                id="cylinder-L",
                                type="number",
                                value=control_defaults[2],
                                step=1,
                                debounce=True,
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "0.25rem"},
                    ),
                    html.Div(
                        [
                            html.Label("σ (deg)"),
                            dcc.Input(
                                id="cylinder-sigma",
                                type="number",
                                value=control_defaults[3],
                                step=0.1,
                                debounce=True,
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "0.25rem"},
                    ),
                    html.Div(
                        [
                            html.Label("Γ (deg)"),
                            dcc.Input(
                                id="cylinder-gamma",
                                type="number",
                                value=control_defaults[4],
                                step=0.1,
                                debounce=True,
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "0.25rem"},
                    ),
                    html.Div(
                        [
                            html.Label("η"),
                            dcc.Input(
                                id="cylinder-eta",
                                type="number",
                                value=control_defaults[5],
                                step=0.05,
                                min=0,
                                max=1,
                                debounce=True,
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "0.25rem"},
                    ),
                ],
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "0.75rem",
                    "alignItems": "flex-end",
                    "paddingBottom": "0.75rem",
                },
            ),
            dcc.Graph(
                id="cylinder-fig",
                figure=build_cylinder_figure(
                    H=initial_H,
                    K=initial_K,
                    L=initial_L,
                    sigma=initial_sigma,
                    Gamma=initial_Gamma,
                    eta=initial_eta,
                ),
                style={"height": "82vh"},
                config={"responsive": True},
            ),
        ],
        style={"padding": "1rem"},
    )

    @app.callback(
        Output("cylinder-fig", "figure"),
        Input("cylinder-H", "value"),
        Input("cylinder-K", "value"),
        Input("cylinder-L", "value"),
        Input("cylinder-sigma", "value"),
        Input("cylinder-gamma", "value"),
        Input("cylinder-eta", "value"),
        State("cylinder-fig", "relayoutData"),
    )
    def update_cylinder(h, k, l, sigma_deg, Gamma_deg, eta_value, relayout_data):  # pragma: no cover - UI callback
        try:
            params = normalize_cylinder_params(
                h,
                k,
                l,
                sigma_deg,
                Gamma_deg,
                eta_value,
                defaults=control_defaults,
            )
        except ValueError as exc:
            return build_cylinder_error_figure(str(exc))

        return build_cylinder_figure(*params, camera=extract_scene_camera(relayout_data))

    return app


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the fibrous Bragg/Ewald viewer."""

    parser = argparse.ArgumentParser(
        description=(
            "Fibrous Bragg sphere, Bragg cylinder, and Ewald sphere "
            "intersection viewer with live HKL and mosaic controls"
        )
    )
    parser.add_argument("H", type=int, nargs="?", default=DEFAULT_H,
                        help="Miller index H (default: 0)")
    parser.add_argument("K", type=int, nargs="?", default=DEFAULT_K,
                        help="Miller index K (default: 0)")
    parser.add_argument("L", type=int, nargs="?", default=DEFAULT_L,
                        help="Miller index L (default: 12)")
    parser.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA_DEG,
        help="Gaussian width σ in degrees (default: 0.8)",
    )
    parser.add_argument(
        "--Gamma",
        "--gamma",
        dest="Gamma",
        type=float,
        default=DEFAULT_GAMMA_DEG,
        help="Lorentzian half-width Γ in degrees (default: 5.0)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=DEFAULT_ETA,
        help="Gaussian/Lorentzian mixing factor η, from 0 to 1 (default: 0.5)",
    )
    return parser.parse_args()


def dash_main() -> None:
    """Launch the Dash fibrous viewer using module defaults."""

    app = build_cylinder_app()
    url = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
    threading.Timer(1.0, lambda: webbrowser.open_new(url)).start()
    app.run(debug=False, host=DEFAULT_HOST, port=DEFAULT_PORT)


def main(
    H: int | None = None,
    K: int | None = None,
    L: int | None = None,
    sigma: float | None = None,
    Gamma: float | None = None,
    eta: float | None = None,
    *,
    gamma: float | None = None,
) -> None:
    """Launch the fibrous Dash viewer seeded from CLI or function inputs."""

    if any(value is None for value in (H, K, L, sigma, Gamma, eta)) and gamma is None:
        args = parse_args()
        H, K, L, sigma, Gamma, eta = normalize_cylinder_params(
            args.H,
            args.K,
            args.L,
            args.sigma,
            args.Gamma,
            args.eta,
        )
    else:
        Gamma = _resolve_Gamma(
            Gamma,
            gamma,
            default=float(np.deg2rad(DEFAULT_GAMMA_DEG)),
        )
        H, K, L, sigma, Gamma, eta = normalize_cylinder_params(
            H,
            K,
            L,
            math.degrees(sigma),
            math.degrees(Gamma),
            eta,
        )

    app = build_cylinder_app(
        initial_H=H,
        initial_K=K,
        initial_L=L,
        initial_sigma=sigma,
        initial_Gamma=Gamma,
        initial_eta=eta,
    )
    url = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
    threading.Timer(1.0, lambda: webbrowser.open_new(url)).start()
    app.run(debug=False, host=DEFAULT_HOST, port=DEFAULT_PORT)


if __name__ == "__main__":
    main()
