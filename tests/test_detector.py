import math

import numpy as np
import pytest
from dash import dcc

import mosaic_sim
from mosaic_sim.constants import INTERSECTION_LINE_WIDTH, K_MAG, a_hex, c_hex, d_hex
import mosaic_sim.detector as detector_module
from mosaic_sim.detector import (
    DETECTOR_CAMERA_UIREVISION,
    K_VECTOR_LABEL_SIZE,
    THETA_LABEL_SIZE,
    build_detector_app,
    build_detector_figure,
    normalize_detector_params,
)
from mosaic_sim.geometry import ewald_bandwidth_k_bounds, ewald_bandwidth_layers, intersection_circle, rot_x
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


def _trace_by_name(fig, name: str):
    return next(trace for trace in fig.data if getattr(trace, "name", "") == name)


def _traces_by_name(fig, name: str):
    return [trace for trace in fig.data if getattr(trace, "name", "") == name]


def _trace_radius_from_center(trace, center: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    center_x, center_y, center_z = np.asarray(center, dtype=float).ravel()
    return np.sqrt(
        (np.asarray(trace.x, dtype=float) - center_x) ** 2
        + (np.asarray(trace.y, dtype=float) - center_y) ** 2
        + (np.asarray(trace.z, dtype=float) - center_z) ** 2
    )


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


def test_package_exports_special_cause_reciprocal_builder():
    assert mosaic_sim.build_special_cause_reciprocal_figure is detector_module.build_special_cause_reciprocal_figure


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
    assert [idx for idx, trace in enumerate(fig.data) if trace.name == "Ewald sphere"] == [1]
    assert [idx for idx, trace in enumerate(fig.data) if trace.name == "Bragg/Ewald overlap"] == [2]
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


def test_build_detector_figure_adds_wavelength_bandwidth_layers_with_matching_opacity():
    theta_i = np.deg2rad(12.0)
    fig = build_detector_figure(
        H=0,
        K=0,
        L=12,
        sigma=np.deg2rad(0.8),
        Gamma=np.deg2rad(5.0),
        eta=0.5,
        theta_i=theta_i,
        wavelength_bandwidth_pct=1.0,
    )

    layers = ewald_bandwidth_layers(K_MAG, 1.0)
    ewald_traces = [trace for trace in fig.data if trace.name == "Ewald sphere"]
    ring_traces = [trace for trace in fig.data if trace.name == "Bragg/Ewald overlap"]

    assert len(ewald_traces) == len(layers)
    assert len(ring_traces) == len(layers)
    assert ewald_traces[len(layers) // 2].opacity > ewald_traces[0].opacity
    assert ewald_traces[len(layers) // 2].opacity > ewald_traces[-1].opacity
    for layer, ewald_trace, ring_trace in zip(layers, ewald_traces, ring_traces, strict=True):
        assert ewald_trace.opacity == pytest.approx(layer.opacity)
        assert ring_trace.opacity == pytest.approx(layer.opacity)

    d_hkl = d_hex(0, 0, 12, a_hex, c_hex)
    g_mag = 2.0 * math.pi / d_hkl
    expected_ring = rot_x(
        *intersection_circle(g_mag, layers[0].k_mag, layers[0].k_mag),
        theta_i,
    )
    np.testing.assert_allclose(np.asarray(ring_traces[0].x, dtype=float), expected_ring[0])
    np.testing.assert_allclose(np.asarray(ring_traces[0].y, dtype=float), expected_ring[1])
    np.testing.assert_allclose(np.asarray(ring_traces[0].z, dtype=float), expected_ring[2])


def test_build_detector_figure_fixed_theta_rotates_bandwidth_surfaces_once(monkeypatch):
    surface_rotations = 0
    original_rot_x = detector_module.rot_x

    def counting_rot_x(x, y, z, ang):
        nonlocal surface_rotations
        if np.asarray(x).ndim == 2:
            surface_rotations += 1
        return original_rot_x(x, y, z, ang)

    monkeypatch.setattr(detector_module, "rot_x", counting_rot_x)

    fig = detector_module.build_detector_figure(
        H=0,
        K=0,
        L=12,
        sigma=np.deg2rad(0.8),
        Gamma=np.deg2rad(5.0),
        eta=0.5,
        theta_i=np.deg2rad(12.0),
        wavelength_bandwidth_pct=1.0,
    )

    ewald_traces = [trace for trace in fig.data if trace.name == "Ewald sphere"]
    assert len(ewald_traces) == len(ewald_bandwidth_layers(K_MAG, 1.0))
    assert surface_rotations == len(ewald_traces)


def test_build_detector_figure_animation_updates_all_bandwidth_layers():
    fig = build_detector_figure(
        H=0,
        K=0,
        L=12,
        sigma=np.deg2rad(0.8),
        Gamma=np.deg2rad(5.0),
        eta=0.5,
        wavelength_bandwidth_pct=1.0,
    )

    ewald_indices = [idx for idx, trace in enumerate(fig.data) if trace.name == "Ewald sphere"]
    ring_indices = [idx for idx, trace in enumerate(fig.data) if trace.name == "Bragg/Ewald overlap"]
    frame_traces = list(fig.frames[0].traces)

    assert len(fig.frames) == 60
    assert set(ewald_indices).issubset(frame_traces)
    assert set(ring_indices).issubset(frame_traces)


def test_build_special_cause_reciprocal_figure_uses_physical_bragg_and_ewald_geometry():
    theta_i = np.deg2rad(12.0)

    fig = detector_module.build_special_cause_reciprocal_figure(
        H=0,
        K=0,
        L=12,
        sigma=np.deg2rad(0.8),
        Gamma=np.deg2rad(5.0),
        eta=0.5,
        theta_i=theta_i,
        wavelength_bandwidth_pct=0.0,
    )

    d_hkl = d_hex(0, 0, 12, a_hex, c_hex)
    g_mag = 2.0 * math.pi / d_hkl
    bragg_trace = _trace_by_name(fig, "Bragg sphere")
    ewald_trace = _trace_by_name(fig, "Ewald sphere")
    overlap_trace = _trace_by_name(fig, "Bragg/Ewald overlap")

    bragg_radius = np.sqrt(
        np.asarray(bragg_trace.x, dtype=float) ** 2
        + np.asarray(bragg_trace.y, dtype=float) ** 2
        + np.asarray(bragg_trace.z, dtype=float) ** 2
    )
    np.testing.assert_allclose(bragg_radius, g_mag, rtol=1e-6)

    ewald_center = rot_x(
        np.array([0.0]),
        np.array([K_MAG]),
        np.array([0.0]),
        theta_i,
    )
    ewald_radius = _trace_radius_from_center(ewald_trace, ewald_center)
    np.testing.assert_allclose(ewald_radius, K_MAG, rtol=1e-6)

    expected_overlap = rot_x(
        *intersection_circle(g_mag, K_MAG, K_MAG),
        theta_i,
    )
    np.testing.assert_allclose(np.asarray(overlap_trace.x, dtype=float), expected_overlap[0])
    np.testing.assert_allclose(np.asarray(overlap_trace.y, dtype=float), expected_overlap[1])
    np.testing.assert_allclose(np.asarray(overlap_trace.z, dtype=float), expected_overlap[2])

    assert len(fig.frames) == 0
    assert len(fig.layout.sliders) == 0
    assert {trace.type for trace in fig.data}.issubset({"surface", "scatter3d", "cone"})
    assert fig.layout.scene.uirevision == DETECTOR_CAMERA_UIREVISION


def test_build_special_cause_reciprocal_figure_defaults_to_requested_peak_and_bandwidth():
    fig = detector_module.build_special_cause_reciprocal_figure()
    layers = ewald_bandwidth_layers(K_MAG, 5.0)

    assert "HKL = (0, 0, 3)" in fig.layout.title.text
    assert "λ bandwidth = 5.00%" in fig.layout.title.text
    assert any(trace.name == "Ewald shell inner" for trace in fig.data)
    assert any(trace.name == "Ewald shell outer" for trace in fig.data)
    assert any(trace.name == "Bragg/Ewald overlap band" for trace in fig.data)
    assert not any(trace.name == "Ewald sphere" for trace in fig.data)
    assert len(_traces_by_name(fig, "Bragg/Ewald overlap")) == len(layers)


def test_build_special_cause_reciprocal_figure_uses_requested_ewald_shell_sample_count():
    fig = detector_module.build_special_cause_reciprocal_figure(
        wavelength_bandwidth_pct=5.0,
        ewald_shell_sample_count=13,
    )

    assert len(_traces_by_name(fig, "Bragg/Ewald overlap")) == 13


@pytest.mark.parametrize("sample_count", [2, 4, 102, 3.5, "many"])
def test_build_special_cause_reciprocal_figure_rejects_invalid_ewald_shell_sample_count(sample_count):
    with pytest.raises(ValueError, match="ewald_shell_sample_count"):
        detector_module.build_special_cause_reciprocal_figure(
            wavelength_bandwidth_pct=5.0,
            ewald_shell_sample_count=sample_count,
        )


def test_build_special_cause_reciprocal_figure_uses_transparent_surfaces_and_opaque_intersections():
    fig = detector_module.build_special_cause_reciprocal_figure()

    transparent_surfaces = [
        _trace_by_name(fig, "Bragg sphere"),
        _trace_by_name(fig, "Ewald shell inner"),
        _trace_by_name(fig, "Ewald shell outer"),
        _trace_by_name(fig, "Bragg/Ewald overlap band"),
    ]
    opaque_intersections = _traces_by_name(fig, "Bragg/Ewald overlap")

    assert all(0.0 < trace.opacity < 1.0 for trace in transparent_surfaces)
    assert {trace.opacity for trace in opaque_intersections} == {1.0}
    assert "rgba" not in str(_trace_by_name(fig, "Bragg sphere").colorscale).lower()


def test_build_special_cause_reciprocal_figure_keeps_geometry_stable_when_mosaic_changes():
    base = detector_module.build_special_cause_reciprocal_figure(
        H=0,
        K=0,
        L=12,
        sigma=np.deg2rad(0.4),
        Gamma=np.deg2rad(1.0),
        eta=0.0,
        theta_i=np.deg2rad(12.0),
        wavelength_bandwidth_pct=0.0,
    )
    changed = detector_module.build_special_cause_reciprocal_figure(
        H=0,
        K=0,
        L=12,
        sigma=np.deg2rad(2.0),
        Gamma=np.deg2rad(10.0),
        eta=1.0,
        theta_i=np.deg2rad(12.0),
        wavelength_bandwidth_pct=0.0,
    )

    base_bragg = _trace_by_name(base, "Bragg sphere")
    changed_bragg = _trace_by_name(changed, "Bragg sphere")
    base_ewald = _trace_by_name(base, "Ewald sphere")
    changed_ewald = _trace_by_name(changed, "Ewald sphere")

    np.testing.assert_allclose(np.asarray(base_bragg.x, dtype=float), np.asarray(changed_bragg.x, dtype=float))
    np.testing.assert_allclose(np.asarray(base_bragg.y, dtype=float), np.asarray(changed_bragg.y, dtype=float))
    np.testing.assert_allclose(np.asarray(base_bragg.z, dtype=float), np.asarray(changed_bragg.z, dtype=float))
    np.testing.assert_allclose(np.asarray(base_ewald.x, dtype=float), np.asarray(changed_ewald.x, dtype=float))
    np.testing.assert_allclose(np.asarray(base_ewald.y, dtype=float), np.asarray(changed_ewald.y, dtype=float))
    np.testing.assert_allclose(np.asarray(base_ewald.z, dtype=float), np.asarray(changed_ewald.z, dtype=float))
    assert not np.allclose(
        np.asarray(base_bragg.surfacecolor, dtype=float),
        np.asarray(changed_bragg.surfacecolor, dtype=float),
    )


def test_build_special_cause_reciprocal_figure_renders_bandwidth_as_ewald_shell():
    theta_i = np.deg2rad(12.0)
    fig = detector_module.build_special_cause_reciprocal_figure(
        H=0,
        K=0,
        L=12,
        sigma=np.deg2rad(0.8),
        Gamma=np.deg2rad(5.0),
        eta=0.5,
        theta_i=theta_i,
        wavelength_bandwidth_pct=5.0,
    )

    assert [trace.name for trace in fig.data].count("Ewald sphere") == 0
    inner_trace = _trace_by_name(fig, "Ewald shell inner")
    outer_trace = _trace_by_name(fig, "Ewald shell outer")
    k_min, k_max = ewald_bandwidth_k_bounds(K_MAG, 5.0)

    fixed_ewald_center = rot_x(
        np.array([0.0]),
        np.array([K_MAG]),
        np.array([0.0]),
        theta_i,
    )
    inner_radius = _trace_radius_from_center(inner_trace, fixed_ewald_center)
    outer_radius = _trace_radius_from_center(outer_trace, fixed_ewald_center)

    np.testing.assert_allclose(inner_radius, k_min, rtol=1e-6)
    np.testing.assert_allclose(outer_radius, k_max, rtol=1e-6)


def test_build_special_cause_reciprocal_figure_renders_bandwidth_overlap_band():
    theta_i = np.deg2rad(12.0)
    fig = detector_module.build_special_cause_reciprocal_figure(
        H=0,
        K=0,
        L=12,
        sigma=np.deg2rad(0.8),
        Gamma=np.deg2rad(5.0),
        eta=0.5,
        theta_i=theta_i,
        wavelength_bandwidth_pct=5.0,
    )

    band_trace = _trace_by_name(fig, "Bragg/Ewald overlap band")
    band_x = np.asarray(band_trace.x, dtype=float)
    band_y = np.asarray(band_trace.y, dtype=float)
    band_z = np.asarray(band_trace.z, dtype=float)

    assert band_trace.type == "surface"
    assert band_x.ndim == 2
    assert band_x.shape[0] > 2
    assert band_x.shape == band_y.shape == band_z.shape

    d_hkl = d_hex(0, 0, 12, a_hex, c_hex)
    g_mag = 2.0 * math.pi / d_hkl
    bragg_radius = np.sqrt(band_x**2 + band_y**2 + band_z**2)
    np.testing.assert_allclose(bragg_radius, g_mag, rtol=1e-6)

    k_min, k_max = ewald_bandwidth_k_bounds(K_MAG, 5.0)
    expected_inner_edge = rot_x(
        *intersection_circle(g_mag, k_min, K_MAG, npts=band_x.shape[1]),
        theta_i,
    )
    expected_outer_edge = rot_x(
        *intersection_circle(g_mag, k_max, K_MAG, npts=band_x.shape[1]),
        theta_i,
    )
    np.testing.assert_allclose(band_x[0], expected_inner_edge[0])
    np.testing.assert_allclose(band_y[0], expected_inner_edge[1])
    np.testing.assert_allclose(band_z[0], expected_inner_edge[2])
    np.testing.assert_allclose(band_x[-1], expected_outer_edge[0])
    np.testing.assert_allclose(band_y[-1], expected_outer_edge[1])
    np.testing.assert_allclose(band_z[-1], expected_outer_edge[2])


def test_build_special_cause_reciprocal_figure_renders_bandwidth_overlap_lines():
    theta_i = np.deg2rad(12.0)
    fig = detector_module.build_special_cause_reciprocal_figure(
        H=0,
        K=0,
        L=12,
        sigma=np.deg2rad(0.8),
        Gamma=np.deg2rad(5.0),
        eta=0.5,
        theta_i=theta_i,
        wavelength_bandwidth_pct=5.0,
    )

    layers = ewald_bandwidth_layers(K_MAG, 5.0)
    overlap_traces = _traces_by_name(fig, "Bragg/Ewald overlap")

    assert len(overlap_traces) == len(layers)
    assert {trace.type for trace in overlap_traces} == {"scatter3d"}
    assert {trace.mode for trace in overlap_traces} == {"lines"}
    assert {trace.opacity for trace in overlap_traces} == {1.0}
    assert {trace.line.color for trace in overlap_traces} == {"rgb(0,128,0)"}
    assert {trace.line.width for trace in overlap_traces} == {INTERSECTION_LINE_WIDTH}
    assert all("rgba" not in trace.line.color.lower() for trace in overlap_traces)

    d_hkl = d_hex(0, 0, 12, a_hex, c_hex)
    g_mag = 2.0 * math.pi / d_hkl
    edge_pairs = (
        (layers[0], overlap_traces[0]),
        (layers[-1], overlap_traces[-1]),
    )
    for layer, trace in edge_pairs:
        expected_overlap = rot_x(
            *intersection_circle(g_mag, layer.k_mag, K_MAG),
            theta_i,
        )
        np.testing.assert_allclose(np.asarray(trace.x, dtype=float), expected_overlap[0])
        np.testing.assert_allclose(np.asarray(trace.y, dtype=float), expected_overlap[1])
        np.testing.assert_allclose(np.asarray(trace.z, dtype=float), expected_overlap[2])


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
    assert controls[6].children[0].children == "λ bandwidth (%)"
    assert controls[6].children[1].value == pytest.approx(0.0)
    assert controls[6].children[1].step == pytest.approx(0.01)
    assert controls[6].children[1].min == pytest.approx(0.0)
    assert controls[6].children[1].max == pytest.approx(100.0)
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
        1.0,
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
