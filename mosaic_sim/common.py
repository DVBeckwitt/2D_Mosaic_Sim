"""Shared parameter normalization and fallback figure helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

PeakDefaults = tuple[int, int, int, float, float, float]


@dataclass(frozen=True)
class PeakFigureParams:
    """Normalized HKL and pseudo-Voigt controls used by multiple viewers."""

    H: int
    K: int
    L: int
    sigma: float
    Gamma: float
    eta: float

    def as_tuple(self) -> tuple[int, int, int, float, float, float]:
        return (self.H, self.K, self.L, self.sigma, self.Gamma, self.eta)


def normalize_peak_params(
    H: float | int | None = None,
    K: float | int | None = None,
    L: float | int | None = None,
    sigma_deg: float | None = None,
    Gamma_deg: float | None = None,
    eta: float | None = None,
    *,
    gamma_deg: float | None = None,
    defaults: PeakDefaults,
) -> PeakFigureParams:
    """Normalize shared HKL/mosaic controls from GUI or CLI inputs."""

    default_H, default_K, default_L, default_sigma, default_Gamma, default_eta = defaults

    H_val = default_H if H is None else int(H)
    K_val = default_K if K is None else int(K)
    L_val = default_L if L is None else int(L)
    sigma_deg_val = default_sigma if sigma_deg is None else float(sigma_deg)
    if Gamma_deg is not None and gamma_deg is not None and not math.isclose(
        float(Gamma_deg),
        float(gamma_deg),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Specify only one Lorentzian width: Γ or gamma")
    Gamma_input = Gamma_deg if Gamma_deg is not None else gamma_deg
    Gamma_deg_val = default_Gamma if Gamma_input is None else float(Gamma_input)
    eta_val = default_eta if eta is None else float(eta)

    if H_val == 0 and K_val == 0 and L_val == 0:
        raise ValueError("At least one of H, K, or L must be non-zero")
    if sigma_deg_val <= 0.0:
        raise ValueError("σ must be greater than 0 degrees")
    if Gamma_deg_val <= 0.0:
        raise ValueError("Γ must be greater than 0 degrees")
    if not 0.0 <= eta_val <= 1.0:
        raise ValueError("η must be between 0 and 1 inclusive")

    return PeakFigureParams(
        H=H_val,
        K=K_val,
        L=L_val,
        sigma=math.radians(sigma_deg_val),
        Gamma=math.radians(Gamma_deg_val),
        eta=eta_val,
    )


def build_error_figure(title: str, message: str) -> go.Figure:
    """Return a compact figure surfacing invalid user inputs."""

    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=18, color="crimson"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        title=title,
        margin=dict(l=40, r=40, b=40, t=80),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def compact_figure_numeric_payload(fig: go.Figure) -> None:
    """Reduce figure payload size by casting numeric trace arrays to float32."""

    keys = ("x", "y", "z", "u", "v", "w", "surfacecolor", "customdata")

    def _cast_trace(trace: object) -> None:
        for key in keys:
            if not hasattr(trace, key):
                continue
            value = getattr(trace, key)
            if value is None:
                continue
            try:
                arr = np.asarray(value)
            except Exception:
                continue
            if arr.dtype.kind not in ("f", "i", "u"):
                continue
            if key == "customdata" and arr.ndim > 0 and arr.dtype.kind != "f":
                arr = arr.astype(np.float64)
            try:
                setattr(trace, key, arr.astype(np.float32))
            except Exception:
                continue
        marker = getattr(trace, "marker", None)
        if marker is not None and hasattr(marker, "color"):
            try:
                color = np.asarray(marker.color)
            except Exception:
                color = None
            if color is not None and color.dtype.kind in ("f", "i", "u"):
                try:
                    marker.color = color.astype(np.float32)
                except Exception:
                    pass

    for trace in fig.data:
        _cast_trace(trace)
    for frame in fig.frames:
        for trace in frame.data:
            _cast_trace(trace)
