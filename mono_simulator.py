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
                ([0, 0], [-R_MAX, 2 * R_MAX], [0, 0]),
                ([0, 0], [0, 0], [-R_MAX, R_MAX])]:
        fig.add_trace(go.Scatter3d(x=xyz[0], y=xyz[1], z=xyz[2],
                                   mode="lines", showlegend=False,
                                   line=dict(color="black",
                                             width=2,
                                             dash="dash")))
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

    def g_intersection_circle(theta: float, g_val: float, color: str) -> go.Scatter3d:
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
            visible=False,
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

    def _mode_visibility(mode: str, g_mask: list[bool] | None = None) -> list[bool]:
        lattice_related = {lattice_idx, projection_idx, hit_label_idx}
        g_related = g_sphere_indices + g_circle_indices
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
            else:
                vis.append(True)
        return vis

    g_mask_default = [i == 0 for i in range(len(g_sphere_indices))]
    g_mask_default.extend([i == 0 for i in range(len(g_circle_indices))])
    lattice_visibility = _mode_visibility("lattice")
    g_sphere_visibility = _mode_visibility("g_spheres", g_mask_default)

    frames = []
    for i, th in enumerate(theta_all):
        Bx, By, Bz = _ewald_surface(th, Ew_x, Ew_y, Ew_z)
        kx, ky, kz = _k_vector(th)
        circles = [
            g_intersection_circle(th, g_val, palette[j % len(palette)])
            for j, g_val in enumerate(unique_g)
        ]
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
        dict(label="|G| spheres",
             method="update",
             args=[{"visible": g_sphere_visibility}, {}]),
    ]

    fig.update_layout(scene=dict(xaxis=dict(visible=False,
                                            range=[-axis_range, axis_range],
                                            autorange=False),
                                 yaxis=dict(visible=False,
                                            range=[-axis_range, axis_range],
                                            autorange=False),
                                 zaxis=dict(visible=False,
                                            range=[-axis_range, axis_range],
                                            autorange=False),
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
                                g_values=unique_g.tolist()))

    context = dict(
        g_values=unique_g.tolist(),
        g_sphere_indices=g_sphere_indices,
        g_circle_indices=g_circle_indices,
        lattice_visibility=lattice_visibility,
        g_sphere_visibility=g_sphere_visibility,
    )

    return fig, context


def _selector_checkbox_html(g_values: list[float], g_sphere_indices: list[int]) -> str:
    lines = ["<div id=\"g-selector\">"]
    for i, (g_val, trace_idx) in enumerate(zip(g_values, g_sphere_indices, strict=True)):
        checked = "checked" if i == 0 else ""
        lines.append(
            f'<label class="g-option"><input class="g-toggle" type="checkbox" '
            f'data-pos="{i}" data-trace="{trace_idx}" {checked}>|G| = {g_val:.3f} Å⁻¹</label>'
        )
    lines.append("</div>")
    return "\n".join(lines)


def build_interactive_page(fig: go.Figure, context: dict) -> str:
    import plotly.io as pio

    g_values: list[float] = context["g_values"]
    g_sphere_indices: list[int] = context["g_sphere_indices"]
    g_circle_indices: list[int] = context["g_circle_indices"]
    lattice_visibility: list[bool] = context["lattice_visibility"]
    g_sphere_visibility: list[bool] = context["g_sphere_visibility"]

    figure_id = "mono-figure"
    figure_html = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        auto_play=False,
        div_id=figure_id,
        config={"responsive": True},
    )
    selector_controls = _selector_checkbox_html(g_values, g_sphere_indices)

    return f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Mono simulator</title>
  <style>
    html, body {{ height: 100%; width: 100%; margin: 0; padding: 0; }}
    body {{ font-family: Arial, sans-serif; }}
    #wrapper {{ display: flex; flex-direction: column; width: 100%; height: 100%; }}
    #note {{ padding: 8px 12px; background: #f2f2f2; width: 100%; box-sizing: border-box; }}
    #controls {{ padding: 8px 12px; width: 100%; box-sizing: border-box; }}
    #figure-container {{ flex: 1 1 auto; width: 100%; min-height: 0; }}
    #figure-container .plotly-graph-div {{ height: 100% !important; width: 100% !important; }}
  </style>
</head>
  <body>
    <div id=\"wrapper\">
      <div id=\"note\">Use the pop-out window to toggle |G| shells. Default shows the smallest shell only.</div>
      <div id=\"controls\">
        <button id=\"open-selector\">Open |G| selector window</button>
        <span id=\"selector-status\" style=\"margin-left:8px;color:#444;\"></span>
      </div>
      <div id=\"figure-container\">{figure_html}</div>
    </div>
    <script>
      window.addEventListener('DOMContentLoaded', () => {{
        const figure = document.getElementById('{figure_id}');
        const latticeVis = {json.dumps(lattice_visibility)};
        const gSphereBase = {json.dumps(g_sphere_visibility)};
        const gCircleIndices = {json.dumps(g_circle_indices)};
        const selectorBtn = document.getElementById('open-selector');
        const selectorStatus = document.getElementById('selector-status');

        function writeSelectorDoc(targetWin) {{
          targetWin.document.open();
          targetWin.document.write(`<!doctype html><html lang="en"><head><meta charset="utf-8"><title>|G| selector</title>
          <style>body {{ font-family: Arial, sans-serif; padding: 12px; margin: 0; }} .g-option {{ display: block; margin: 6px 0; }}
    </style>
          </head><body><h3>|G| shells</h3>
          <p>Select which |G| spheres to display. Default view shows only the smallest |G|.</p>
          <div style="margin-bottom:8px;"><button id="check-all">Show all</button> <button id="check-none">Show none</button></div>
          {selector_controls}
          </body></html>`);
          targetWin.document.close();
        }}

        function hookSelector(targetWin) {{
          const checkboxes = targetWin.document.querySelectorAll('.g-toggle');

          function applySelection() {{
            const vis = Array.from(gSphereBase);
            checkboxes.forEach((cb) => {{
              const idx = Number(cb.getAttribute('data-trace'));
              const pos = Number(cb.getAttribute('data-pos'));
              vis[idx] = cb.checked;
              const circleIdx = gCircleIndices[pos];
              if (!Number.isNaN(circleIdx)) {{
                vis[circleIdx] = cb.checked;
              }}
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

        function openSelectorWindow(focusOnly = false) {{
          const selectorWin = window.open('', 'g_selector_window', 'width=340,height=600');
          if (!selectorWin) {{
            selectorStatus.textContent = 'Pop-up blocked. Click the button to retry after allowing pop-ups.';
            return null;
          }}
          if (!focusOnly || selectorWin.document.body?.childElementCount === 0) {{
            writeSelectorDoc(selectorWin);
          }}
          selectorWin.focus();
          selectorStatus.textContent = 'Selector window opened.';
          return hookSelector(selectorWin);
        }}

        let applySelection = openSelectorWindow(true);
        if (!applySelection) {{
          selectorStatus.textContent = 'Pop-up blocked. Please allow pop-ups and click the button.';
        }}

        selectorBtn.addEventListener('click', () => {{
          const applyFn = openSelectorWindow();
          if (applyFn) {{
            applySelection = applyFn;
          }}
        }});

        figure.on('plotly_buttonclicked', (evt) => {{
          const label = evt.button && evt.button.label;
          if (label === 'Single crystal') {{
            Plotly.update(figure, {{visible: latticeVis}});
          }} else if (label === '|G| spheres' && applySelection) {{
            applySelection();
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
