"""3-panel rocking-curve detector simulation.

The detector view shows how the Bragg sphere intersects the Ewald sphere and
how this maps onto a 2-D detector.  The third panel displays the integrated
intensity through the diffraction peak as the sample rocks.
"""
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .constants import a_hex, c_hex, K_MAG, d_hex
from .geometry import sphere, rot_x, intersection_circle
from .intensity import mosaic_intensity


def build_detector_figure(H: int = 0, K: int = 0, L: int = 12,
                          sigma: float = np.deg2rad(0.8),
                          gamma: float = np.deg2rad(5.0),
                          eta: float = 0.5) -> go.Figure:
    """Return a Plotly Figure replicating the detector simulation."""
    d_hkl = d_hex(H, K, L, a_hex, c_hex)
    G_MAG = 2 * math.pi / d_hkl

    def gaussian_blur(a: np.ndarray, sigma_blur: float = 5) -> np.ndarray:
        """Apply a simple 1-D Gaussian blur used for smoothing."""

        r = int(3 * sigma_blur)
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_blur) ** 2)
        k /= k.sum()
        return np.convolve(a, k, mode="same")

    phi, theta = np.meshgrid(np.linspace(0, math.pi, 100),
                             np.linspace(0, 2*math.pi, 200))
    Ew_x, Ew_y, Ew_z = sphere(K_MAG, phi, theta, (0, K_MAG, 0))
    B0_x, B0_y, B0_z = sphere(G_MAG, phi, theta)

    I_surf = mosaic_intensity(B0_x, B0_y, B0_z, H, K, L, sigma, gamma, eta)

    ring_x, ring_y, ring_z = intersection_circle(G_MAG, K_MAG, K_MAG)
    hr_x, hr_y, hr_z = ring_x, ring_y, ring_z
    hr_t = np.linspace(0, 2*math.pi, len(hr_x))
    mask_top = hr_z >= 0

    theta_min, theta_max, N_FR = np.deg2rad(5), np.deg2rad(30), 120
    theta_all = np.concatenate([np.linspace(theta_min, theta_max, N_FR//2, endpoint=False),
                                np.linspace(theta_max, theta_min, N_FR//2)])

    sample_ids = np.linspace(0, N_FR-1, 10, dtype=int)
    opacity_vals = np.linspace(0.15, 1.0, 10)
    frame_to_tail = {fid: k for k, fid in enumerate(sample_ids)}

    tails, tail_idx = [], []
    # Pre-compute a handful of intensity traces that will appear sequentially
    # during the animation.  Each trace corresponds to a different rocking
    # angle and fades in as the animation progresses.
    for op, idx in zip(opacity_vals, sample_ids):
        th = theta_all[idx]
        rx, ry, rz = rot_x(hr_x, hr_y, hr_z, th)
        I_blur = gaussian_blur(
            mosaic_intensity(rx, ry, rz, H, K, L, sigma, gamma, eta), 5
        )
        subset = hr_t >= 1.0
        pk = np.argmax(I_blur * subset)
        phi_pk = hr_t[pk]
        dphi = hr_t - phi_pk
        keep = subset & (np.abs(dphi) <= 1.0)
        tails.append(
            go.Scatter(
                x=dphi[keep],
                y=I_blur[keep],
                mode="lines",
                line=dict(color="crimson", width=2),
                opacity=op,
                visible=False,
                showlegend=False,
            )
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

    fig.add_trace(go.Surface(x=Ew_x, y=Ew_y, z=Ew_z,
                             opacity=0.3, colorscale="Blues", showscale=False), 1, 1)

    fig.add_trace(go.Scatter3d(x=ring_x, y=ring_y, z=ring_z,
                               mode="lines", line=dict(color="green", width=5)), 1, 1)

    k_tail, k_head = np.zeros(3), np.array([0, K_MAG, 0])
    fig.add_trace(go.Scatter3d(x=[0, k_head[0]], y=[0, k_head[1]], z=[0, k_head[2]],
                               mode="lines", line=dict(color="black", width=5)), 1, 1)
    fig.add_trace(go.Cone(x=[k_head[0]], y=[k_head[1]], z=[k_head[2]],
                          u=[0], v=[-0.2*K_MAG], w=[0],
                          anchor="tip", sizemode="absolute", sizeref=0.2,
                          colorscale=[[0, "black"], [1, "black"]], showscale=False), 1, 1)
    fig.add_trace(go.Scatter3d(x=[0], y=[K_MAG*1.05], z=[0],
                               mode="text", text=["kᵢ"], showlegend=False), 1, 1)
    ktext_idx = len(fig.data) - 1

    G_dir = np.array([0, math.sin(theta_all[0]), math.cos(theta_all[0])])
    G_vec = G_dir * G_MAG
    fig.add_trace(go.Scatter3d(x=[0, G_vec[0]], y=[0, G_vec[1]], z=[0, G_vec[2]],
                               mode="lines", line=dict(color="red", width=4)), 1, 1)
    Gline_idx = len(fig.data) - 1
    fig.add_trace(go.Cone(x=[G_vec[0]], y=[G_vec[1]], z=[G_vec[2]],
                          u=[G_vec[0]*0.2], v=[G_vec[1]*0.2], w=[G_vec[2]*0.2],
                          anchor="tip", sizemode="absolute", sizeref=0.2,
                          colorscale=[[0, "red"], [1, "red"]], showscale=False), 1, 1)
    Gcone_idx = len(fig.data) - 1
    fig.add_trace(go.Scatter3d(x=[G_vec[0]*1.05], y=[G_vec[1]*1.05], z=[G_vec[2]*1.05],
                               mode="text", text=["G"], showlegend=False), 1, 1)
    Gtext_idx = len(fig.data) - 1

    r_arc = 0.3 * K_MAG
    u = np.linspace(0, theta_all[0], 50)
    arc_y, arc_z = r_arc*np.cos(u), r_arc*np.sin(u)
    fig.add_trace(go.Scatter3d(x=np.zeros_like(arc_y), y=arc_y, z=arc_z,
                               mode="lines", line=dict(color="magenta", width=3, dash="dot")), 1, 1)
    arc_idx = len(fig.data) - 1
    fig.add_trace(go.Scatter3d(x=[0], y=[r_arc*np.cos(theta_all[0]/2)], z=[r_arc*np.sin(theta_all[0]/2)],
                               mode="text", text=["θᵢ"], showlegend=False), 1, 1)
    thetatext_idx = len(fig.data) - 1

    fig.update_scenes(dict(xaxis=dict(title="x"),
                           yaxis=dict(title="y"),
                           zaxis=dict(title="z"),
                           bgcolor="rgba(0,0,0,0)"), row=1, col=1)

    I_blur0 = gaussian_blur(mosaic_intensity(hr_x, hr_y, hr_z, H, K, L, sigma, gamma, eta), 5)[mask_top]
    fig.add_trace(go.Scatter(x=hr_x[mask_top], y=hr_z[mask_top],
                             mode="markers+lines",
                             marker=dict(size=7, color=I_blur0, colorscale="Viridis",
                                         showscale=False, opacity=0.9),
                             line=dict(width=3, color="grey")), 1, 2)
    ring2d_idx = len(fig.data) - 1
    fig.update_xaxes(visible=False, scaleanchor="y", row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)

    for tr in tails:
        fig.add_trace(tr, 1, 3)
        tail_idx.append(len(fig.data) - 1)
    fig.update_xaxes(title="Δφ (rad, peak = 0)", range=[-1, 1], row=1, col=3)
    fig.update_yaxes(title="Intensity", type="log", row=1, col=3)

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, b=0, t=55))

    frames, vis = [], [False]*10
    for f, th in enumerate(theta_all):
        Bx, By, Bz = rot_x(B0_x, B0_y, B0_z, -th)
        surf = go.Surface(x=Bx, y=By, z=Bz,
                          surfacecolor=I_surf,
                          colorscale=fig.data[bragg_idx].colorscale,
                          showscale=False)

        rx, ry, rz = rot_x(hr_x, hr_y, hr_z, th)
        I_blur = gaussian_blur(mosaic_intensity(rx, ry, rz, H, K, L, sigma, gamma, eta), 5)[mask_top]
        ring_up = go.Scatter(x=hr_x[mask_top], y=hr_z[mask_top],
                             mode="markers+lines",
                             marker=dict(size=7, color=I_blur, colorscale="Viridis",
                                         showscale=False, opacity=0.9),
                             line=dict(width=3, color="grey"))

        G_dir = np.array([0, math.sin(th), math.cos(th)])
        G_vec = G_dir * G_MAG
        G_line = go.Scatter3d(x=[0, G_vec[0]], y=[0, G_vec[1]], z=[0, G_vec[2]],
                               mode="lines", line=dict(color="red", width=4))
        G_cone = go.Cone(x=[G_vec[0]], y=[G_vec[1]], z=[G_vec[2]],
                         u=[G_vec[0]*0.2], v=[G_vec[1]*0.2], w=[G_vec[2]*0.2],
                         anchor="tip", sizemode="absolute", sizeref=0.2,
                         colorscale=[[0, "red"], [1, "red"]], showscale=False)
        G_text = go.Scatter3d(x=[G_vec[0]*1.05], y=[G_vec[1]*1.05], z=[G_vec[2]*1.05],
                              mode="text", text=["G"], showlegend=False)

        u = np.linspace(0, th, 50)
        arc_line = go.Scatter3d(x=np.zeros_like(u), y=r_arc*np.cos(u), z=r_arc*np.sin(u),
                                mode="lines", line=dict(color="magenta", width=3, dash="dot"))
        arc_text = go.Scatter3d(x=[0], y=[r_arc*np.cos(th/2)], z=[r_arc*np.sin(th/2)],
                                mode="text", text=["θᵢ"], showlegend=False)

        if f in frame_to_tail:
            vis[frame_to_tail[f]] = True
        tails_update = [go.Scatter(visible=v) for v in vis]

        frames.append(go.Frame(data=[surf, ring_up, G_line, G_cone, G_text, arc_line, arc_text] + tails_update,
                               traces=[bragg_idx, ring2d_idx, Gline_idx, Gcone_idx, Gtext_idx, arc_idx, thetatext_idx] + tail_idx))

    fig.frames = frames

    fig.update_layout(updatemenus=[dict(type="buttons", direction="left",
                                        x=0.5, y=1.18, xanchor="center",
                                        buttons=[
                                            dict(label="▶ Play / Loop", method="animate",
                                                 args=[None, dict(frame=dict(duration=5, redraw=True),
                                                                  transition=dict(duration=0),
                                                                  fromcurrent=True, mode="immediate", loop=True)]),
                                            dict(label="■ Stop", method="animate",
                                                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                                    transition=dict(duration=0),
                                                                    mode="immediate")])])])

    return fig


def main() -> None:
    """Entry point for the ``mosaic-detector`` script."""

    import plotly.io as pio
    pio.renderers.default = "browser"
    fig = build_detector_figure()
    fig.show()


if __name__ == "__main__":
    main()
