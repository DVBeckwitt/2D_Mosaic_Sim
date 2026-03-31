"""3-panel detector simulation driven by a ``theta_i`` slider.

The detector view shows how the Bragg sphere intersects the Ewald sphere and
how this maps onto a 2-D detector. The third panel displays the centered
integration profile corresponding to the currently selected rocking angle.
"""
from __future__ import annotations

import argparse
import math
import threading
import webbrowser
from typing import Any
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .common import build_error_figure, normalize_peak_params
from .constants import a_hex, c_hex, K_MAG, INTERSECTION_LINE_WIDTH, d_hex
from .geometry import sphere, rot_x, intersection_circle
from .intensity import mosaic_intensity

DEFAULT_H = 0
DEFAULT_K = 0
DEFAULT_L = 12
DEFAULT_SIGMA_DEG = 0.8
DEFAULT_GAMMA_DEG = 5.0
DEFAULT_ETA = 0.5
DEFAULT_THETA_DEG = 5.0
THETA_MIN_DEG = 5.0
THETA_MAX_DEG = 30.0
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8050
DEFAULT_CONTROL_VALUES = (
    DEFAULT_H,
    DEFAULT_K,
    DEFAULT_L,
    DEFAULT_SIGMA_DEG,
    DEFAULT_GAMMA_DEG,
    DEFAULT_ETA,
)
DETECTOR_CAMERA_UIREVISION = "detector-camera"
K_VECTOR_LINE_WIDTH = 9
K_VECTOR_CONE_SIZE = 0.28
K_VECTOR_LABEL_SCALE = 1.1
K_VECTOR_LABEL_SIZE = 24
THETA_LABEL_SIZE = 24


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


def _detector_title(H: int, K: int, L: int,
                    sigma: float, Gamma: float, eta: float) -> str:
    """Return the shared title string for the detector figure."""

    return (
        f"HKL = ({H}, {K}, {L})"
        f" | Gaussian σ = {math.degrees(sigma):.2f}°"
        f" | Lorentzian Γ = {math.degrees(Gamma):.2f}°"
        f" | Mix η = {eta:.2f}"
    )


def normalize_detector_params(
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
    """Normalize GUI or CLI parameters into validated detector inputs.

    ``sigma_deg`` and ``Gamma_deg`` are accepted in degrees and converted to
    radians for the figure builder.
    """
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


def build_detector_error_figure(message: str) -> go.Figure:
    """Return a compact figure that surfaces invalid GUI inputs."""

    return build_error_figure("Detector mosaic/Ewald view", message)


def _center_ring_profile(phi: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the ring intensity profile as a wrapped angular offset from the peak."""

    peak_index = int(np.argmax(intensity))
    phi_peak = float(phi[peak_index])
    dphi = (phi - phi_peak + math.pi) % (2.0 * math.pi) - math.pi
    order = np.argsort(dphi)
    return dphi[order], intensity[order]


def build_detector_figure(H: int = 0, K: int = 0, L: int = 12,
                          sigma: float = np.deg2rad(0.8),
                          Gamma: float | None = None,
                          eta: float = 0.5,
                          *,
                          gamma: float | None = None,
                          theta_i: float | None = None,
                          camera: dict[str, Any] | None = None) -> go.Figure:
    """Return a Plotly figure with synchronized reciprocal/detector sliders."""
    Gamma = _resolve_Gamma(
        Gamma,
        gamma,
        default=float(np.deg2rad(DEFAULT_GAMMA_DEG)),
    )
    d_hkl = d_hex(H, K, L, a_hex, c_hex)
    G_MAG = 2 * math.pi / d_hkl

    phi, theta = np.meshgrid(np.linspace(0, math.pi, 100),
                             np.linspace(0, 2*math.pi, 200))
    Ew_x, Ew_y, Ew_z = sphere(K_MAG, phi, theta, (0, K_MAG, 0))
    B0_x, B0_y, B0_z = sphere(G_MAG, phi, theta)

    I_surf = mosaic_intensity(B0_x, B0_y, B0_z, H, K, L, sigma, Gamma, eta)

    ring_x_full, ring_y_full, ring_z_full = intersection_circle(G_MAG, K_MAG, K_MAG)
    ring_x = ring_x_full[:-1]
    ring_y = ring_y_full[:-1]
    ring_z = ring_z_full[:-1]
    ring_phi = np.linspace(0.0, 2.0 * math.pi, ring_x.size, endpoint=False)

    theta_min, theta_max, n_frames = np.deg2rad(THETA_MIN_DEG), np.deg2rad(THETA_MAX_DEG), 60
    theta_all = np.linspace(theta_min, theta_max, n_frames)

    r_arc = 0.3 * K_MAG

    def state_for_theta(th: float) -> dict[str, np.ndarray]:
        """Return the synchronized detector state for a rocking angle."""

        Ewx, Ewy, Ewz = rot_x(Ew_x, Ew_y, Ew_z, th)

        rx_full, ry_full, rz_full = rot_x(ring_x_full, ring_y_full, ring_z_full, th)
        rx, ry, rz = rx_full[:-1], ry_full[:-1], rz_full[:-1]
        ring_intensity = mosaic_intensity(rx, ry, rz, H, K, L, sigma, Gamma, eta)
        centered_dphi, centered_intensity = _center_ring_profile(ring_phi, ring_intensity)

        k_head_x, k_head_y, k_head_z = rot_x(
            np.array([0.0]),
            np.array([K_MAG]),
            np.array([0.0]),
            th,
        )
        k_head = np.array([k_head_x[0], k_head_y[0], k_head_z[0]])

        u = np.linspace(0.0, th, 50)
        arc_x = np.zeros_like(u)
        arc_y = r_arc * np.cos(u)
        arc_z = r_arc * np.sin(u)

        return {
            "Ew_x": Ewx,
            "Ew_y": Ewy,
            "Ew_z": Ewz,
            "ring_x": rx_full,
            "ring_y": ry_full,
            "ring_z": rz_full,
            "detector_x": ring_x_full,
            "detector_z": ring_z_full,
            "detector_intensity": np.concatenate([ring_intensity, ring_intensity[:1]]),
            "k_head": k_head,
            "arc_x": arc_x,
            "arc_y": arc_y,
            "arc_z": arc_z,
            "theta_label_y": np.array([r_arc * np.cos(th / 2)]),
            "theta_label_z": np.array([r_arc * np.sin(th / 2)]),
            "dphi": centered_dphi,
            "centered_intensity": centered_intensity,
        }

    theta_states = [state_for_theta(th) for th in theta_all]
    detector_intensity_max = max(
        float(np.max(state["detector_intensity"]))
        for state in theta_states
    )
    centered_intensity_max = max(
        float(np.max(state["centered_intensity"]))
        for state in theta_states
    )
    centered_positive_values = [
        state["centered_intensity"][state["centered_intensity"] > 0.0]
        for state in theta_states
        if np.any(state["centered_intensity"] > 0.0)
    ]
    centered_intensity_min = min(
        float(np.min(values))
        for values in centered_positive_values
    ) if centered_positive_values else max(centered_intensity_max * 1e-6, 1e-12)
    centered_intensity_max = max(centered_intensity_max, centered_intensity_min * 10.0)
    centered_log_range = [
        math.log10(centered_intensity_min),
        math.log10(centered_intensity_max),
    ]

    theta_initial = float(theta_all[0] if theta_i is None else theta_i)
    initial_state = (
        theta_states[0]
        if theta_i is None
        else state_for_theta(theta_initial)
    )

    fig = make_subplots(rows=1, cols=3,
                        specs=[[{"type": "scene"}, {"type": "xy"}, {"type": "xy"}]],
                        column_widths=[0.5, 0.3, 0.2],
                        subplot_titles=("Reciprocal space",
                                        "Detector view",
                                        "Centered integration"))

    fig.add_trace(go.Surface(x=B0_x, y=B0_y, z=B0_z,
                             surfacecolor=I_surf,
                             colorscale=[[0, "rgba(128,128,128,0.25)"],
                                         [1, "rgba(255,0,0,1)"]],
                             showscale=True,
                             colorbar=dict(title="Mosaic<br>Intensity")), 1, 1)
    bragg_idx = len(fig.data) - 1

    fig.add_trace(go.Surface(x=initial_state["Ew_x"], y=initial_state["Ew_y"], z=initial_state["Ew_z"],
                             opacity=0.3, colorscale="Blues", showscale=False), 1, 1)
    ewald_idx = len(fig.data) - 1

    fig.add_trace(go.Scatter3d(x=initial_state["ring_x"], y=initial_state["ring_y"], z=initial_state["ring_z"],
                               mode="lines",
                               line=dict(color="green", width=INTERSECTION_LINE_WIDTH)), 1, 1)
    ring3d_idx = len(fig.data) - 1

    k_head = initial_state["k_head"]
    fig.add_trace(go.Scatter3d(x=[k_head[0], 0.0], y=[k_head[1], 0.0], z=[k_head[2], 0.0],
                               mode="lines", line=dict(color="black", width=K_VECTOR_LINE_WIDTH)), 1, 1)
    kline_idx = len(fig.data) - 1
    fig.add_trace(go.Cone(x=[0.0], y=[0.0], z=[0.0],
                          u=[-k_head[0]], v=[-k_head[1]], w=[-k_head[2]],
                          anchor="tip", sizemode="absolute", sizeref=K_VECTOR_CONE_SIZE,
                          colorscale=[[0, "black"], [1, "black"]], showscale=False), 1, 1)
    kcone_idx = len(fig.data) - 1
    fig.add_trace(go.Scatter3d(x=[k_head[0] * K_VECTOR_LABEL_SCALE], y=[k_head[1] * K_VECTOR_LABEL_SCALE], z=[k_head[2] * K_VECTOR_LABEL_SCALE],
                               mode="text", text=["kᵢ"], textfont=dict(size=K_VECTOR_LABEL_SIZE), showlegend=False), 1, 1)
    ktext_idx = len(fig.data) - 1

    fig.add_trace(go.Scatter3d(x=initial_state["arc_x"], y=initial_state["arc_y"], z=initial_state["arc_z"],
                               mode="lines", line=dict(color="magenta", width=3, dash="dot")), 1, 1)
    arc_idx = len(fig.data) - 1
    fig.add_trace(go.Scatter3d(x=[0], y=initial_state["theta_label_y"], z=initial_state["theta_label_z"],
                               mode="text", text=["θᵢ"], textfont=dict(size=THETA_LABEL_SIZE), showlegend=False), 1, 1)
    thetatext_idx = len(fig.data) - 1

    fig.update_scenes(dict(xaxis=dict(visible=False, showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(visible=False, showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
                           zaxis=dict(visible=False, showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
                           bgcolor="rgba(0,0,0,0)",
                           uirevision=DETECTOR_CAMERA_UIREVISION), row=1, col=1)

    fig.add_trace(go.Scatter(x=initial_state["detector_x"], y=initial_state["detector_z"],
                             mode="markers+lines",
                             marker=dict(size=7, color=initial_state["detector_intensity"], colorscale="Viridis",
                                         cmin=0.0, cmax=detector_intensity_max,
                                         showscale=False, opacity=0.9),
                             line=dict(width=INTERSECTION_LINE_WIDTH, color="grey")), 1, 2)
    ring2d_idx = len(fig.data) - 1
    fig.update_xaxes(visible=False, scaleanchor="y", row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)

    fig.add_trace(
        go.Scatter(
            x=initial_state["dphi"],
            y=initial_state["centered_intensity"],
            mode="lines",
            line=dict(color="crimson", width=2),
            showlegend=False,
        ),
        1,
        3,
    )
    centered_idx = len(fig.data) - 1
    fig.update_xaxes(title="Δφ (rad, peak = 0)", range=[-math.pi, math.pi], row=1, col=3)
    fig.update_yaxes(title="Intensity", type="log", range=centered_log_range, row=1, col=3)

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      showlegend=False,
                      uirevision=DETECTOR_CAMERA_UIREVISION,
                      margin=dict(l=0, r=0, b=0, t=140),
                      title=dict(
                          text=_detector_title(H, K, L, sigma, Gamma, eta),
                          x=0.5,
                          y=0.98,
                          xanchor="center",
                      ))

    if theta_i is None:
        frames = []
        for i, (th, state) in enumerate(zip(theta_all, theta_states, strict=True)):
            ewald_surf = go.Surface(
                x=state["Ew_x"],
                y=state["Ew_y"],
                z=state["Ew_z"],
                opacity=0.3,
                colorscale="Blues",
                showscale=False,
            )
            ring3d = go.Scatter3d(
                x=state["ring_x"],
                y=state["ring_y"],
                z=state["ring_z"],
                mode="lines",
                line=dict(color="green", width=INTERSECTION_LINE_WIDTH),
            )

            ring_up = go.Scatter(x=state["detector_x"], y=state["detector_z"],
                                 mode="markers+lines",
                                 marker=dict(size=7, color=state["detector_intensity"], colorscale="Viridis",
                                             cmin=0.0, cmax=detector_intensity_max,
                                             showscale=False, opacity=0.9),
                                 line=dict(width=INTERSECTION_LINE_WIDTH, color="grey"))

            k_head = state["k_head"]
            k_line = go.Scatter3d(
                x=[k_head[0], 0.0],
                y=[k_head[1], 0.0],
                z=[k_head[2], 0.0],
                mode="lines",
                line=dict(color="black", width=K_VECTOR_LINE_WIDTH),
            )
            k_cone = go.Cone(
                x=[0.0],
                y=[0.0],
                z=[0.0],
                u=[-k_head[0]],
                v=[-k_head[1]],
                w=[-k_head[2]],
                anchor="tip",
                sizemode="absolute",
                sizeref=K_VECTOR_CONE_SIZE,
                colorscale=[[0, "black"], [1, "black"]],
                showscale=False,
            )
            k_text = go.Scatter3d(
                x=[k_head[0] * K_VECTOR_LABEL_SCALE],
                y=[k_head[1] * K_VECTOR_LABEL_SCALE],
                z=[k_head[2] * K_VECTOR_LABEL_SCALE],
                mode="text",
                text=["kᵢ"],
                textfont=dict(size=K_VECTOR_LABEL_SIZE),
                showlegend=False,
            )

            arc_line = go.Scatter3d(x=state["arc_x"], y=state["arc_y"], z=state["arc_z"],
                                    mode="lines", line=dict(color="magenta", width=3, dash="dot"))
            arc_text = go.Scatter3d(x=[0], y=state["theta_label_y"], z=state["theta_label_z"],
                                    mode="text", text=["θᵢ"], textfont=dict(size=THETA_LABEL_SIZE), showlegend=False)
            centered_trace = go.Scatter(
                x=state["dphi"],
                y=state["centered_intensity"],
                mode="lines",
                line=dict(color="crimson", width=2),
                showlegend=False,
            )

            frames.append(
                go.Frame(
                    name=f"theta-{i}",
                    data=[ewald_surf, ring3d, k_line, k_cone, k_text, ring_up, arc_line, arc_text, centered_trace],
                    traces=[ewald_idx, ring3d_idx, kline_idx, kcone_idx, ktext_idx, ring2d_idx, arc_idx, thetatext_idx, centered_idx],
                )
            )

        fig.frames = frames

        slider_steps = [
            dict(
                label=f"{math.degrees(th):.1f}",
                method="animate",
                args=[
                    [f"theta-{i}"],
                    dict(
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0),
                        mode="immediate",
                    ),
                ],
            )
            for i, th in enumerate(theta_all)
        ]

        fig.update_layout(
            updatemenus=[dict(type="buttons", direction="left",
                              x=0.5, y=1.17, xanchor="center",
                              buttons=[
                                  dict(label="▶ Play", method="animate",
                                       args=[None, dict(frame=dict(duration=40, redraw=True),
                                                        transition=dict(duration=0),
                                                        fromcurrent=True, mode="immediate")]),
                                  dict(label="■ Stop", method="animate",
                                       args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                          transition=dict(duration=0),
                                                          mode="immediate")])])],
            sliders=[dict(
                active=0,
                currentvalue=dict(prefix="θᵢ = ", suffix="°"),
                len=0.82,
                x=0.1,
                y=1.05,
                xanchor="left",
                pad=dict(t=10, b=0),
                steps=slider_steps,
            )],
        )
    else:
        fig.frames = []
        fig.update_layout(updatemenus=[], sliders=[])

    if camera:
        fig.update_layout(scene_camera=camera)

    return fig


def build_detector_app(
    initial_H: int = DEFAULT_H,
    initial_K: int = DEFAULT_K,
    initial_L: int = DEFAULT_L,
    initial_sigma: float = np.deg2rad(DEFAULT_SIGMA_DEG),
    initial_Gamma: float | None = None,
    initial_eta: float = DEFAULT_ETA,
    initial_theta_deg: float = DEFAULT_THETA_DEG,
    *,
    initial_gamma: float | None = None,
):
    """Return a Dash app for live detector-parameter updates."""

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
    app.title = "Detector Mosaic/Ewald View"
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H2("Detector mosaic/Ewald view", style={"margin": "0"}),
                    html.Div(
                        "Use the θᵢ slider below to move the Ewald sphere and keep the 3D camera orientation while updating HKL and mosaic parameters.",
                        style={"color": "#555"},
                    ),
                ],
                style={"paddingBottom": "0.75rem"},
            ),
            html.Div(
                [
                    html.Div(
                        [html.Label("H"), dcc.Input(id="detector-H", type="number",
                                                    value=control_defaults[0], step=1, debounce=True)],
                        style={"display": "flex", "flexDirection": "column", "gap": "0.25rem"},
                    ),
                    html.Div(
                        [html.Label("K"), dcc.Input(id="detector-K", type="number",
                                                    value=control_defaults[1], step=1, debounce=True)],
                        style={"display": "flex", "flexDirection": "column", "gap": "0.25rem"},
                    ),
                    html.Div(
                        [html.Label("L"), dcc.Input(id="detector-L", type="number",
                                                    value=control_defaults[2], step=1, debounce=True)],
                        style={"display": "flex", "flexDirection": "column", "gap": "0.25rem"},
                    ),
                    html.Div(
                        [html.Label("σ (deg)"), dcc.Input(id="detector-sigma", type="number",
                                                              value=control_defaults[3], step=0.1, debounce=True)],
                        style={"display": "flex", "flexDirection": "column", "gap": "0.25rem"},
                    ),
                    html.Div(
                        [html.Label("Γ (deg)"), dcc.Input(id="detector-gamma", type="number",
                                                              value=control_defaults[4], step=0.1, debounce=True)],
                        style={"display": "flex", "flexDirection": "column", "gap": "0.25rem"},
                    ),
                    html.Div(
                        [html.Label("η"), dcc.Input(id="detector-eta", type="number",
                                                      value=control_defaults[5], step=0.05,
                                                      min=0, max=1, debounce=True)],
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
            html.Div(
                [
                    html.Label("θᵢ (deg)"),
                    dcc.Slider(
                        id="detector-theta",
                        min=THETA_MIN_DEG,
                        max=THETA_MAX_DEG,
                        step=0.25,
                        value=float(initial_theta_deg),
                        updatemode="drag",
                        marks={
                            THETA_MIN_DEG: f"{THETA_MIN_DEG:.0f}",
                            0.5 * (THETA_MIN_DEG + THETA_MAX_DEG): f"{0.5 * (THETA_MIN_DEG + THETA_MAX_DEG):.1f}",
                            THETA_MAX_DEG: f"{THETA_MAX_DEG:.0f}",
                        },
                    ),
                ],
                style={"paddingBottom": "0.75rem"},
            ),
            dcc.Graph(
                id="detector-fig",
                figure=build_detector_figure(
                    H=initial_H,
                    K=initial_K,
                    L=initial_L,
                    sigma=initial_sigma,
                    Gamma=initial_Gamma,
                    eta=initial_eta,
                    theta_i=math.radians(initial_theta_deg),
                ),
                style={"height": "82vh"},
                config={"responsive": True},
            ),
        ],
        style={"padding": "1rem"},
    )

    @app.callback(
        Output("detector-fig", "figure"),
        Input("detector-H", "value"),
        Input("detector-K", "value"),
        Input("detector-L", "value"),
        Input("detector-sigma", "value"),
        Input("detector-gamma", "value"),
        Input("detector-eta", "value"),
        Input("detector-theta", "value"),
        State("detector-fig", "relayoutData"),
    )
    def update_detector(h, k, l, sigma_deg, Gamma_deg, eta_value, theta_deg, relayout_data):  # pragma: no cover - UI callback
        try:
            params = normalize_detector_params(
                h,
                k,
                l,
                sigma_deg,
                Gamma_deg,
                eta_value,
                defaults=control_defaults,
            )
        except ValueError as exc:
            return build_detector_error_figure(str(exc))

        return build_detector_figure(
            *params,
            theta_i=math.radians(float(theta_deg)),
            camera=extract_scene_camera(relayout_data),
        )

    return app


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the detector/Ewald intersection viewer."""

    parser = argparse.ArgumentParser(
        description=(
            "Detector mosaic/Ewald view with a θᵢ slider and "
            "manuscript-style mosaic-kernel parameters"
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
    """Launch a Dash detector viewer seeded from CLI or function inputs."""

    if any(value is None for value in (H, K, L, sigma, Gamma, eta)) and gamma is None:
        args = parse_args()
        H, K, L, sigma, Gamma, eta = normalize_detector_params(
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
        H, K, L, sigma, Gamma, eta = normalize_detector_params(
            H,
            K,
            L,
            math.degrees(sigma),
            math.degrees(Gamma),
            eta,
        )

    app = build_detector_app(
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
