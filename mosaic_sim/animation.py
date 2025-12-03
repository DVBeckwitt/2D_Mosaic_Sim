"""Dynamic Bragg-sphere rotation animation.

The functions here generate a Plotly animation that illustrates how the
Bragg sphere rotates relative to the fixed Ewald sphere as the sample is
rocked.  The colour of the Bragg sphere represents the mosaic intensity
distribution calculated via :func:`mosaic_sim.intensity.mosaic_intensity`.
"""
import math
import numpy as np
import plotly.graph_objects as go

from .constants import a_hex, c_hex, K_MAG, d_hex
from .geometry import sphere, rot_x, intersection_circle
from .intensity import mosaic_intensity


def build_animation(H: int = 0, K: int = 0, L: int = 1,
                    sigma: float = np.deg2rad(0.8),
                    gamma: float = np.deg2rad(5.0),
                    eta: float = 0.5) -> go.Figure:
    """Construct the Plotly animation for the given Miller index.

    Parameters
    ----------
    H, K, L:
        Miller indices of the reflection.
    sigma, gamma, eta:
        Parameters of the pseudo-Voigt mosaic spread.

    Returns
    -------
    :class:`plotly.graph_objects.Figure`
        Interactive animation figure.
    """

    # Length of the scattering vector |G| for the chosen reflection
    d_hkl = d_hex(H, K, L, a_hex, c_hex)
    G_MAG = 2 * math.pi / d_hkl

    # Angles covering a unit sphere
    # Keep the surface meshes lightweight so interactive actions (zoom/pan)
    # stay responsive. A coarser grid still renders smoothly while reducing
    # the number of polygons Plotly needs to process by ~4x versus the
    # previous 100x200 sampling.
    phi, theta = np.meshgrid(np.linspace(0, math.pi, 50),
                             np.linspace(0, 2 * math.pi, 100))
    Ew_x, Ew_y, Ew_z = sphere(K_MAG, phi, theta, (0, K_MAG, 0))
    B0_x, B0_y, B0_z = sphere(G_MAG, phi, theta)

    I_surface = mosaic_intensity(B0_x, B0_y, B0_z, H, K, L, sigma, gamma, eta)

    ring_x, ring_y, ring_z = intersection_circle(G_MAG, K_MAG, K_MAG)
    k_tail = np.array([0.0, K_MAG, 0.0])
    k_head = k_tail * 0.25
    cone_vec = -k_head
    R_MAX = max(G_MAG, K_MAG)

    def dyn_state(th: float) -> dict[str, np.ndarray]:
        """Return dynamic state for a rotation angle ``th``."""

        # Rotate the Bragg sphere
        Bx, By, Bz = rot_x(B0_x, B0_y, B0_z, -th)
        # Small arc used to annotate the rocking angle
        r_arc = 0.5
        t_arc = np.linspace(0, th, 50)
        ax = np.zeros_like(t_arc)
        ay = -r_arc * np.cos(t_arc)
        az = -r_arc * np.sin(t_arc)
        theta_lab = (0, -r_arc * math.cos(th / 2), -r_arc * math.sin(th / 2))
        return dict(Bx=Bx, By=By, Bz=Bz, Arc=(ax, ay, az), Theta_lab=theta_lab)

    fig = go.Figure()

    bragg = go.Surface(x=B0_x, y=B0_y, z=B0_z,
                       surfacecolor=I_surface,
                       colorscale=[[0, "rgba(128,128,128,0.25)"],
                                   [1, "rgba(255,0,0,1)"]],
                       showscale=True,
                       hoverinfo="skip",
                       colorbar=dict(title="Mosaic<br>Intensity"),
                       name=f"Bragg sphere {'cap' if H==0 and K==0 else 'band'}")
    fig.add_trace(bragg); bragg_idx = len(fig.data) - 1

    fig.add_trace(go.Surface(x=Ew_x, y=Ew_y, z=Ew_z,
                             opacity=0.3, colorscale="Blues",
                             showscale=False, hoverinfo="skip",
                             name="Ewald sphere"))
    fig.add_trace(go.Scatter3d(x=ring_x, y=ring_y, z=ring_z,
                               mode="lines",
                               line=dict(color="green", width=5),
                               name="2θB ring"))
    fig.add_trace(go.Scatter3d(x=[k_tail[0], k_head[0]], y=[k_tail[1], k_head[1]], z=[k_tail[2], k_head[2]],
                               mode="lines", line=dict(color="black", width=5), name="kᵢ"))
    fig.add_trace(go.Cone(x=[k_head[0]], y=[k_head[1]], z=[k_head[2]],
                          u=[cone_vec[0]], v=[cone_vec[1]], w=[cone_vec[2]],
                          anchor="tail", sizemode="absolute", sizeref=0.2,
                          colorscale=[[0, "black"], [1, "black"]], showscale=False))

    st0 = dyn_state(np.deg2rad(5))
    fig.add_trace(go.Scatter3d(x=st0["Arc"][0], y=st0["Arc"][1], z=st0["Arc"][2],
                               mode="lines",
                               line=dict(color="purple", width=3, dash="dash")))
    fig.add_trace(go.Scatter3d(x=[st0["Theta_lab"][0]], y=[st0["Theta_lab"][1]], z=[st0["Theta_lab"][2]], mode="text", text=["θᵢ"], showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0], y=[K_MAG/2], z=[0], mode="text", text=["kᵢ"], showlegend=False))

    for xyz in [([-R_MAX, R_MAX], [0,0], [0,0]),
                ([0,0], [-R_MAX, 2*R_MAX], [0,0]),
                ([0,0], [0,0], [-R_MAX, R_MAX])]:
        fig.add_trace(go.Scatter3d(x=xyz[0], y=xyz[1], z=xyz[2],
                                   mode="lines", showlegend=False,
                                   line=dict(color="black", width=2, dash="dash")))
    fig.update_layout(scene=dict(xaxis=dict(visible=False),
                                 yaxis=dict(visible=False),
                                 zaxis=dict(visible=False),
                                 bgcolor="rgba(0,0,0,0)"),
                      paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, b=0, t=0))

    THETA_MIN, THETA_MAX = np.deg2rad(5), np.deg2rad(30)
    N_FRAMES = 60
    theta_fwd = np.linspace(THETA_MIN, THETA_MAX, N_FRAMES//2, endpoint=False)
    theta_all = np.concatenate([theta_fwd, theta_fwd[::-1]])

    frames = []
    for i, th in enumerate(theta_all):
        st = dyn_state(th)
        frames.append(
            go.Frame(
                name=f"f{i}",
                data=[
                    go.Surface(x=st["Bx"], y=st["By"], z=st["Bz"],
                               surfacecolor=I_surface,
                               colorscale=bragg.colorscale,
                               hoverinfo="skip", showscale=False),
                    go.Scatter3d(x=st["Arc"][0], y=st["Arc"][1], z=st["Arc"][2],
                                 mode="lines",
                                 line=dict(color="purple", width=3, dash="dash")),
                    go.Scatter3d(x=[st["Theta_lab"][0]], y=[st["Theta_lab"][1]],
                                 z=[st["Theta_lab"][2]], mode="text",
                                 text=["θᵢ"], showlegend=False),
                ],
                traces=[bragg_idx, len(fig.data) - 2, len(fig.data) - 1],
            )
        )
    fig.frames = frames

    fig.update_layout(updatemenus=[dict(type="buttons",
                                        direction="left", x=0.5, y=1.07, xanchor="center",
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
    """Entry point for the ``mosaic-rocking`` script."""

    import plotly.io as pio
    pio.renderers.default = "browser"
    fig = build_animation()
    fig.show()


if __name__ == "__main__":
    main()
