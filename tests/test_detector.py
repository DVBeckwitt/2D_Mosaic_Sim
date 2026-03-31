import math

import numpy as np
import pytest
from dash import dcc

from mosaic_sim.constants import K_MAG, a_hex, c_hex, d_hex
from mosaic_sim.detector import (
    DETECTOR_CAMERA_UIREVISION,
    K_VECTOR_LABEL_SIZE,
    THETA_LABEL_SIZE,
    build_detector_app,
    build_detector_figure,
    normalize_detector_params,
)
from mosaic_sim.geometry import intersection_circle, rot_x
from mosaic_sim.intensity import mosaic_intensity


def _expected_detector_ring_profile(
    H: int,
    K: int,
    L: int,
    sigma: float,
    Gamma: float,
    eta: float,
    theta_i: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d_hkl = d_hex(H, K, L, a_hex, c_hex)
    g_mag = 2.0 * math.pi / d_hkl
    ring_x_full, ring_y_full, ring_z_full = intersection_circle(g_mag, K_MAG, K_MAG)
    ring_x = ring_x_full[:-1]
    ring_y = ring_y_full[:-1]
    ring_z = ring_z_full[:-1]
    ring_phi = np.linspace(0.0, 2.0 * math.pi, ring_x.size, endpoint=False)

    rx, ry, rz = rot_x(ring_x, ring_y, ring_z, theta_i)
    ring_intensity = mosaic_intensity(rx, ry, rz, H, K, L, sigma, Gamma, eta)
    detector_intensity = np.concatenate([ring_intensity, ring_intensity[:1]])

    peak_index = int(np.argmax(ring_intensity))
    dphi = (ring_phi - ring_phi[peak_index] + math.pi) % (2.0 * math.pi) - math.pi
    order = np.argsort(dphi)
    return ring_x_full, ring_z_full, detector_intensity, dphi[order], ring_intensity[order]


def test_normalize_detector_params_uses_defaults_and_converts_units():
    params = normalize_detector_params(
        defaults=(1, 2, 3, 0.9, 4.1, 0.25),
    )

    assert params[:3] == (1, 2, 3)
    assert math.isclose(params[3], math.radians(0.9))
    assert math.isclose(params[4], math.radians(4.1))
    assert math.isclose(params[5], 0.25)


def test_normalize_detector_params_rejects_invalid_eta():
    with pytest.raises(ValueError, match="η"):
        normalize_detector_params(0, 0, 12, 0.8, 5.0, 1.2)


def test_build_detector_figure_exposes_parameterized_slider_state():
    fig = build_detector_figure(
        H=0,
        K=0,
        L=12,
        sigma=np.deg2rad(0.8),
        Gamma=np.deg2rad(5.0),
        eta=0.5,
    )

    assert "HKL = (0, 0, 12)" in fig.layout.title.text
    assert "Gaussian σ = 0.80°" in fig.layout.title.text
    assert "Lorentzian Γ = 5.00°" in fig.layout.title.text
    assert "Mix η = 0.50" in fig.layout.title.text
    assert len(fig.frames) == 60
    assert len(fig.layout.sliders) == 1
    assert len(fig.layout.sliders[0].steps) == 60
    assert fig.layout.uirevision == DETECTOR_CAMERA_UIREVISION
    assert fig.layout.scene.uirevision == DETECTOR_CAMERA_UIREVISION
    assert fig.layout.sliders[0].steps[0].args[1]["frame"]["redraw"] is True
    assert fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["redraw"] is True
    assert fig.data[3].line.width > 5
    assert fig.data[3].x[-1] == 0
    assert fig.data[3].y[-1] == 0
    assert fig.data[3].z[-1] == 0
    assert fig.data[4].x[0] == 0
    assert fig.data[4].y[0] == 0
    assert fig.data[4].z[0] == 0
    assert list(fig.data[5].text) == ["kᵢ"]
    assert fig.data[5].textfont.size == K_VECTOR_LABEL_SIZE
    assert list(fig.data[7].text) == ["θᵢ"]
    assert fig.data[7].textfont.size == THETA_LABEL_SIZE
    assert fig.layout.scene.xaxis.visible is False
    assert fig.layout.scene.yaxis.visible is False
    assert fig.layout.scene.zaxis.visible is False
    assert fig.layout.showlegend is False
    assert not any(
        getattr(trace, "text", None) is not None and list(trace.text) == ["G"]
        for trace in fig.data
    )
    assert 1 in fig.frames[0].traces
    assert 0 not in fig.frames[0].traces


def test_build_detector_figure_supports_fixed_theta_without_animation_ui():
    fig = build_detector_figure(
        H=0,
        K=0,
        L=12,
        sigma=np.deg2rad(0.8),
        Gamma=np.deg2rad(5.0),
        eta=0.5,
        theta_i=np.deg2rad(12.0),
    )

    assert len(fig.frames) == 0
    assert len(fig.layout.sliders) == 0
    assert len(fig.layout.updatemenus) == 0
    assert fig.layout.uirevision == DETECTOR_CAMERA_UIREVISION


def test_build_detector_figure_locks_detector_and_centered_scaling_to_global_max():
    fig = build_detector_figure(
        H=0,
        K=0,
        L=12,
        sigma=np.deg2rad(0.8),
        Gamma=np.deg2rad(5.0),
        eta=0.5,
    )

    detector_trace = fig.data[8]
    detector_cmax = float(detector_trace.marker.cmax)
    frame_detector_max = max(
        float(np.max(np.asarray(frame.data[5].marker.color, dtype=float)))
        for frame in fig.frames
    )
    frame_detector_cmax = [
        float(frame.data[5].marker.cmax)
        for frame in fig.frames
    ]
    centered_global_max = max(
        float(np.max(np.asarray(frame.data[8].y, dtype=float)))
        for frame in fig.frames
    )

    assert detector_cmax == pytest.approx(frame_detector_max)
    assert all(value == pytest.approx(detector_cmax) for value in frame_detector_cmax)
    assert 10 ** float(fig.layout.yaxis2.range[1]) == pytest.approx(centered_global_max)


def test_build_detector_figure_plots_raw_ring_intensity_against_wrapped_phi():
    sigma = np.deg2rad(0.2)
    Gamma = np.deg2rad(1.0)
    theta_i = np.deg2rad(20.0)
    fig = build_detector_figure(
        H=0,
        K=0,
        L=12,
        sigma=sigma,
        Gamma=Gamma,
        eta=0.5,
        theta_i=theta_i,
    )

    detector_trace = fig.data[8]
    centered_trace = fig.data[9]
    detector_x, detector_z, detector_intensity, dphi, centered_intensity = _expected_detector_ring_profile(
        0,
        0,
        12,
        sigma,
        Gamma,
        0.5,
        theta_i,
    )

    np.testing.assert_allclose(np.asarray(detector_trace.x, dtype=float), detector_x)
    np.testing.assert_allclose(np.asarray(detector_trace.y, dtype=float), detector_z)
    np.testing.assert_allclose(np.asarray(detector_trace.marker.color, dtype=float), detector_intensity)
    np.testing.assert_allclose(np.asarray(centered_trace.x, dtype=float), dphi)
    np.testing.assert_allclose(np.asarray(centered_trace.y, dtype=float), centered_intensity)
    assert np.asarray(centered_trace.x, dtype=float)[0] < -3.0
    assert np.asarray(centered_trace.x, dtype=float)[-1] > 3.0


def test_build_detector_figure_applies_explicit_camera():
    camera = {
        "eye": {"x": 1.7, "y": 1.1, "z": 0.8},
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
    }

    fig = build_detector_figure(
        H=0,
        K=0,
        L=12,
        sigma=np.deg2rad(0.8),
        Gamma=np.deg2rad(5.0),
        eta=0.5,
        theta_i=np.deg2rad(12.0),
        camera=camera,
    )

    assert fig.layout.scene.camera.eye.x == pytest.approx(1.7)
    assert fig.layout.scene.camera.eye.y == pytest.approx(1.1)
    assert fig.layout.scene.camera.eye.z == pytest.approx(0.8)


def test_build_detector_app_seeds_inputs_and_figure_from_initial_values():
    app = build_detector_app(
        initial_H=1,
        initial_K=2,
        initial_L=7,
        initial_sigma=np.deg2rad(1.1),
        initial_Gamma=np.deg2rad(6.2),
        initial_eta=0.35,
    )

    controls = app.layout.children[1].children
    theta_control = app.layout.children[2]
    graph = app.layout.children[3]

    assert controls[0].children[1].value == 1
    assert controls[1].children[1].value == 2
    assert controls[2].children[1].value == 7
    assert controls[3].children[0].children == "σ (deg)"
    assert controls[4].children[0].children == "Γ (deg)"
    assert controls[5].children[0].children == "η"
    assert math.isclose(controls[3].children[1].value, 1.1)
    assert math.isclose(controls[4].children[1].value, 6.2)
    assert math.isclose(controls[5].children[1].value, 0.35)
    assert math.isclose(theta_control.children[1].value, 5.0)
    assert theta_control.children[1].updatemode == "drag"
    assert isinstance(graph, dcc.Graph)
    assert "HKL = (1, 2, 7)" in graph.figure.layout.title.text
    assert len(graph.figure.layout.sliders) == 0


def test_build_detector_app_callback_preserves_camera_from_relayout_data():
    app = build_detector_app()
    callback = app.callback_map["detector-fig.figure"]["callback"].__wrapped__

    fig = callback(
        0,
        0,
        12,
        0.8,
        5.0,
        0.5,
        12.0,
        {
            "scene.camera.eye.x": 1.5,
            "scene.camera.eye.y": 1.2,
            "scene.camera.eye.z": 0.9,
            "scene.camera.up.x": 0.0,
            "scene.camera.up.y": 0.0,
            "scene.camera.up.z": 1.0,
            "scene.camera.center.x": 0.0,
            "scene.camera.center.y": 0.0,
            "scene.camera.center.z": 0.0,
        },
    )

    assert fig.layout.scene.camera.eye.x == pytest.approx(1.5)
    assert fig.layout.scene.camera.eye.y == pytest.approx(1.2)
    assert fig.layout.scene.camera.eye.z == pytest.approx(0.9)
