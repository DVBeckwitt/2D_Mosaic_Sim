"""Interactive cylinder point-cloud probability density figure."""

from __future__ import annotations

import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, callback_context
from dash.dependencies import Input, Output

__all__ = ["build_cylinder_pdf_app", "dash_main"]


# ───────────────── helper functions ──────────────────────────
def I0(z: np.ndarray, sigma_I: float) -> np.ndarray:
    """Axial Gaussian intensity on the original rod."""
    return np.exp(-0.5 * (z / sigma_I) ** 2)


def p_phi(phi: np.ndarray, sigma_phi: float) -> np.ndarray:
    """Gaussian PDF for tilt angle φ."""
    return np.exp(-0.5 * (phi / sigma_phi) ** 2) / (sigma_phi * np.sqrt(2 * np.pi))


def rodrigues(v: np.ndarray, k: np.ndarray, theta: float) -> np.ndarray:
    """Rotate vector ``v`` about unit axis ``k`` by ``theta``."""
    v, k = np.asarray(v), np.asarray(k)
    return (
        v * np.cos(theta)
        + np.cross(k, v) * np.sin(theta)
        + k * np.dot(k, v) * (1 - np.cos(theta))
    )


# ───────────────── Dash app ───────────────────────────────────
def build_cylinder_pdf_app() -> Dash:
    """Return a Dash application for exploring a cylinder PDF."""

    app = Dash(__name__)
    app.layout = html.Div(
        [
            dcc.Graph(id="graph"),
            html.Div(
                [
                    html.Label("Intensity\xa0range"),
                    dcc.RangeSlider(
                        id="range-slider",
                        min=0,
                        max=1,
                        step=0.01,
                        value=[0, 1],
                        marks={0: "0", 1: "1"},
                    ),
                ],
                style={"padding": "10px"},
            ),
            html.Div(
                [
                    html.Label("x₀:"),
                    dcc.Input(id="x0-input", type="number", value=2.0, step=0.1),
                    html.Label("y₀:"),
                    dcc.Input(id="y0-input", type="number", value=2.0, step=0.1),
                    html.Label("σ_I:"),
                    dcc.Input(id="sigmaI-input", type="number", value=10000, step=0.1),
                    html.Label("σ_φ:"),
                    dcc.Input(id="sigmaPhi-input", type="number", value=0.2, step=0.1),
                ],
                style={"padding": "10px", "display": "flex", "gap": "10px", "alignItems": "center"},
            ),
            html.Div(
                [
                    html.Button("Original\xa0Only", id="orig-btn", n_clicks=0),
                    html.Button("Single\xa0Rod", id="single-btn", n_clicks=0),
                    html.Button("Reset\xa0Cloud", id="reset-btn", n_clicks=0),
                ],
                style={"padding": "10px", "display": "flex", "gap": "10px"},
            ),
        ]
    )

    @app.callback(
        Output("graph", "figure"),
        Input("range-slider", "value"),
        Input("x0-input", "value"),
        Input("y0-input", "value"),
        Input("sigmaI-input", "value"),
        Input("sigmaPhi-input", "value"),
        Input("orig-btn", "n_clicks"),
        Input("single-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
    )
    def update_figure(int_range, x0, y0, sigma_I, sigma_phi, n_orig, n_single, n_reset):  # pragma: no cover - UI callback
        # ---- fixed sampling ----
        z_min, z_max = -3.0, 3.0
        n_phi, n_psi, n_z = 60, 60, 30

        # ---- geometry basis ----
        r = np.hypot(x0, y0)
        if r == 0:
            e_r = np.array([1.0, 0.0, 0.0])
        else:
            e_r = np.array([x0, y0, 0.0]) / r
        e_z = np.array([0.0, 0.0, 1.0])
        k_axis = np.cross(e_r, e_z)
        k_axis /= np.linalg.norm(k_axis)

        # ---- deterministic grids ----
        phi_max = 3 * sigma_phi
        phis = np.linspace(-phi_max, phi_max, n_phi)
        psis = np.linspace(0.0, 2 * np.pi, n_psi, endpoint=False)
        zs = np.linspace(z_min, z_max, n_z)
        dphi = phis[1] - phis[0]
        dpsi = psis[1] - psis[0]

        # ---- accumulate point clouds ----
        xs, ys, zs_all, cs = [], [], [], []
        xs0, ys0, zs0, cs0 = [], [], [], []
        xs_s, ys_s, zs_s, cs_s = [], [], [], []

        for phi in phis:
            w_phi = p_phi(phi, sigma_phi) * dphi
            v_phi = rodrigues(e_r * r, k_axis, phi)
            f_phi = rodrigues(e_z, k_axis, phi)
            base_pts = v_phi[None, :] + zs[:, None] * f_phi[None, :]
            Ivals = I0(zs, sigma_I) * w_phi

            for i, psi in enumerate(psis):
                c_ps, s_ps = np.cos(psi), np.sin(psi)
                x_rot = base_pts[:, 0] * c_ps - base_pts[:, 1] * s_ps
                y_rot = base_pts[:, 0] * s_ps + base_pts[:, 1] * c_ps
                w = Ivals * (dpsi / (2 * np.pi))

                xs.extend(x_rot)
                ys.extend(y_rot)
                zs_all.extend(base_pts[:, 2])
                cs.extend(w)
                if i == 0:
                    xs0.extend(x_rot)
                    ys0.extend(y_rot)
                    zs0.extend(base_pts[:, 2])
                    cs0.extend(w)

        for z in zs:
            xs_s.append(x0)
            ys_s.append(y0)
            zs_s.append(z)
            cs_s.append(I0(z, sigma_I))

        xs, ys, zs_all, cs = map(np.array, (xs, ys, zs_all, cs))
        xs0, ys0, zs0, cs0 = map(np.array, (xs0, ys0, zs0, cs0))
        xs_s, ys_s, zs_s, cs_s = map(np.array, (xs_s, ys_s, zs_s, cs_s))

        cs_norm = cs / cs.max()
        cs0_norm = cs0 / cs0.max()
        cs_s_norm = cs_s / cs_s.max()

        trig = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""

        if trig == "single-btn":
            x, y, z, c = xs_s, ys_s, zs_s, cs_s_norm
        elif trig == "orig-btn" and (n_orig % 2) == 1:
            x, y, z, c = xs0, ys0, zs0, cs0_norm
        else:
            lo, hi = int_range
            mask = (cs_norm >= lo) & (cs_norm <= hi)
            x, y, z, c = xs[mask], ys[mask], zs_all[mask], cs_norm[mask]

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(size=3, color=c, colorscale="Inferno", cmin=0, cmax=1),
                )
            ]
        )
        fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        return fig

    return app


def dash_main() -> None:
    """Run the Dash cylinder PDF demo."""
    import webbrowser

    app = build_cylinder_pdf_app()
    webbrowser.open_new("http://127.0.0.1:8050")
    app.run(debug=True, host="127.0.0.1", port=8050)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    dash_main()
