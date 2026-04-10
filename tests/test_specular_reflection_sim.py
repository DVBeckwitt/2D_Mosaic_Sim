import math

import numpy as np
import plotly.graph_objects as go
import pytest
from dash import dcc, html
from dash.exceptions import PreventUpdate

import specular_reflection_sim as specular
from specular_reflection_sim import (
    BeamConfig,
    DetectorConfig,
    DiffractionConfig,
    SampleConfig,
    SPECULAR_CAMERA_UIREVISION,
    build_specular_companion_figure,
    build_detector_frame,
    build_sample_frame,
    build_specular_app,
    build_specular_figure,
    build_specular_outputs,
    configs_from_values,
    project_ray_to_detector,
    trace_specular_simulation,
)


def control_id(name: str, kind: str) -> dict[str, str]:
    return {"type": f"specular-{kind}", "name": name}


def find_component_by_id(component, target_id):
    if getattr(component, "id", None) == target_id:
        return component

    children = getattr(component, "children", None)
    if children is None:
        return None

    if isinstance(children, (list, tuple)):
        for child in children:
            result = find_component_by_id(child, target_id)
            if result is not None:
                return result
        return None

    return find_component_by_id(children, target_id)


def find_label_by_html_for(component, target_html_for):
    if getattr(component, "htmlFor", None) == target_html_for:
        return component

    children = getattr(component, "children", None)
    if children is None:
        return None

    if isinstance(children, (list, tuple)):
        for child in children:
            result = find_label_by_html_for(child, target_html_for)
            if result is not None:
                return result
        return None

    return find_label_by_html_for(children, target_html_for)


def render_component_text(component) -> str:
    if component is None:
        return ""
    if isinstance(component, (str, int, float)):
        return str(component)
    if isinstance(component, (list, tuple)):
        return "".join(render_component_text(child) for child in component)
    if isinstance(component, html.Sub):
        return f"<sub>{render_component_text(component.children)}</sub>"
    return render_component_text(getattr(component, "children", None))


def callback_by_output(app, output_key: str):
    for key, value in app.callback_map.items():
        if key == output_key or output_key in key:
            return value["callback"].__wrapped__
    raise KeyError(output_key)


def test_sample_frame_matches_nominal_incidence_rotation():
    frame = build_sample_frame(SampleConfig(theta_i_deg=12.0))

    theta = math.radians(12.0)
    expected_axis_v = np.array([0.0, math.cos(theta), math.sin(theta)])
    expected_normal = np.array([0.0, -math.sin(theta), math.cos(theta)])

    assert np.allclose(frame.axis_u, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(frame.axis_v, expected_axis_v)
    assert np.allclose(frame.normal, expected_normal)


def test_detector_tilts_keep_direct_beam_at_detector_center():
    detector_config = DetectorConfig(
        distance=250.0,
        beta_deg=11.0,
        gamma_deg=-7.0,
        chi_deg=23.0,
    )
    detector_frame = build_detector_frame(detector_config)

    projection = project_ray_to_detector(
        np.zeros(3, dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        detector_frame,
        detector_config,
    )

    assert projection is not None
    point, uv, pixels = projection
    assert np.allclose(point, detector_frame.origin)
    assert np.allclose(uv, np.array([0.0, 0.0]))
    assert pixels[0] == pytest.approx(detector_config.i0)
    assert pixels[1] == pytest.approx(detector_config.j0)


def test_detector_config_defaults_to_1000_by_1000():
    detector_config = DetectorConfig()

    assert detector_config.width == pytest.approx(1000.0)
    assert detector_config.height == pytest.approx(1000.0)


def test_hkl_diffraction_family_projects_from_sample_hit_point():
    diffraction = DiffractionConfig(H=0, K=0, L=12, sigma_deg=0.8, mosaic_gamma_deg=5.0, eta=0.5)
    result = trace_specular_simulation(
        BeamConfig(
            ray_count=1,
            source_y=-150.0,
            width_x=0.0,
            width_z=0.0,
            divergence_x_deg=0.0,
            divergence_z_deg=0.0,
            z_offset=0.0,
            seed=1,
            display_rays=1,
        ),
        SampleConfig(
            width=50.0,
            height=200.0,
            theta_i_deg=10.0,
            delta_deg=0.0,
            alpha_deg=0.0,
            psi_deg=0.0,
            z_offset=0.0,
        ),
        DetectorConfig(
            distance=200.0,
            width=400.0,
            height=400.0,
            beta_deg=0.0,
            gamma_deg=0.0,
            chi_deg=0.0,
            pixel_u=0.1,
            pixel_v=0.1,
            i0=1000.0,
            j0=1000.0,
        ),
        diffraction,
    )

    assert result.sample_hit_count == 1
    assert 0 < result.diffraction_ray_count < 180
    assert result.exit_dirs.shape == (result.diffraction_ray_count, 3)
    assert result.exit_weights.shape == (result.diffraction_ray_count,)
    assert np.array_equal(np.unique(result.exit_parent_indices), np.array([0]))
    np.testing.assert_allclose(
        np.linalg.norm(result.exit_dirs, axis=1),
        np.ones(result.diffraction_ray_count),
    )
    assert np.all(result.exit_dirs @ result.sample.normal > 0.0)
    assert result.detector_plane_hit_count == result.plane_hit_indices.size
    assert result.detector_hit_count <= result.detector_plane_hit_count
    assert float(np.max(result.exit_weights)) > 0.0


def test_trace_specular_simulation_reports_progress_messages():
    messages: list[str] = []
    diffraction = DiffractionConfig(H=0, K=0, L=12, sigma_deg=0.8, mosaic_gamma_deg=5.0, eta=0.5)

    result = trace_specular_simulation(
        BeamConfig(
            ray_count=1,
            source_y=-150.0,
            width_x=0.0,
            width_z=0.0,
            divergence_x_deg=0.0,
            divergence_z_deg=0.0,
            z_offset=0.0,
            seed=1,
            display_rays=1,
        ),
        SampleConfig(
            width=50.0,
            height=200.0,
            theta_i_deg=10.0,
            delta_deg=0.0,
            alpha_deg=0.0,
            psi_deg=0.0,
            z_offset=0.0,
        ),
        DetectorConfig(
            distance=200.0,
            width=400.0,
            height=400.0,
            beta_deg=0.0,
            gamma_deg=0.0,
            chi_deg=0.0,
            pixel_u=0.1,
            pixel_v=0.1,
            i0=1000.0,
            j0=1000.0,
        ),
        diffraction,
        progress=messages.append,
    )

    assert result.sample_hit_count == 1
    assert messages[0] == "Validating beam, sample, detector, and diffraction inputs"
    assert any("Generating 1 incident beam rays" in message for message in messages)
    assert any("Building sample and detector frames" in message for message in messages)
    assert any("Sample hits on finite sample: 1/1" in message for message in messages)
    assert any("Building HKL diffraction ring for (0, 0, 12)" in message for message in messages)
    assert any("Expanding diffraction families from sample hits: 1/1" in message for message in messages)
    assert any(
        f"Generated {result.diffraction_ray_count} diffraction rays from 1 sample hits" in message
        for message in messages
    )
    assert any("Detector-plane intersections:" in message for message in messages)
    assert messages[-1] == "Trace complete"


def test_trace_specular_simulation_rejects_theta_i_outside_zero_to_ninety():
    with pytest.raises(ValueError, match="sample.theta_i_deg"):
        trace_specular_simulation(sample_config=SampleConfig(theta_i_deg=-0.1))

    with pytest.raises(ValueError, match="sample.theta_i_deg"):
        trace_specular_simulation(sample_config=SampleConfig(theta_i_deg=90.1))

    result = trace_specular_simulation(sample_config=SampleConfig(theta_i_deg=90.0))
    assert result.sample_hit_count >= 0


def test_build_specular_figure_uses_hkl_diffraction_titles_and_summary():
    diffraction = DiffractionConfig(H=0, K=0, L=12)
    result = trace_specular_simulation(diffraction_config=diffraction)
    beam = BeamConfig()
    sample = SampleConfig()
    detector = DetectorConfig()
    fig = build_specular_figure(
        result,
        beam,
        sample,
        detector,
        diffraction,
    )

    assert "HKL = (0, 0, 12)" in fig.layout.title.text
    assert "|G| =" in fig.layout.title.text
    assert "sample " in fig.layout.title.text
    assert "diffraction rays" in fig.layout.title.text
    assert "detector hits" in fig.layout.title.text
    assert "nominal 2θ" not in fig.layout.title.text
    assert len(fig.layout.annotations) == 3
    diffraction_ray_trace = next(trace for trace in fig.data if getattr(trace, "name", "") == "Diffraction rays")
    diffraction_arrow_trace = next(trace for trace in fig.data if getattr(trace, "name", "") == "Diffraction direction")
    assert diffraction_ray_trace.line.color == "rgba(214, 40, 40, 0.16)"
    assert isinstance(diffraction_arrow_trace, go.Cone)
    assert diffraction_arrow_trace.sizemode == "raw"
    assert diffraction_arrow_trace.sizeref == pytest.approx(1.0)
    arrow_norms = np.sqrt(
        np.asarray(diffraction_arrow_trace.u) ** 2
        + np.asarray(diffraction_arrow_trace.v) ** 2
        + np.asarray(diffraction_arrow_trace.w) ** 2
    )
    expected_arrow_tip_length = 0.02 * max(
        sample.width,
        sample.height,
        detector.width,
        detector.height,
        detector.distance,
    )
    assert arrow_norms.min() == pytest.approx(expected_arrow_tip_length)
    assert arrow_norms.max() == pytest.approx(expected_arrow_tip_length)
    assert fig.layout.xaxis.scaleanchor == "y"
    assert fig.layout.xaxis.scaleratio == 1
    assert fig.layout.xaxis.range == (-10.5, 10.5)
    assert fig.layout.yaxis.range == (-42.0, 42.0)
    sample_domain = fig.layout.xaxis.domain[1] - fig.layout.xaxis.domain[0]
    detector_domain = fig.layout.xaxis2.domain[1] - fig.layout.xaxis2.domain[0]
    assert sample_domain > 0.18
    assert sample_domain < detector_domain
    assert fig.layout.xaxis2.scaleanchor == "y2"
    assert fig.layout.xaxis2.scaleratio == 1
    assert fig.layout.xaxis2.range == fig.layout.yaxis2.range


def test_build_specular_companion_figure_removes_redundant_detector_panel():
    fig = build_specular_companion_figure(
        SampleConfig(),
        DiffractionConfig(H=0, K=0, L=12),
    )

    annotation_texts = {annotation.text for annotation in fig.layout.annotations}

    assert annotation_texts == {"Reciprocal space", "Centered integration"}
    assert "Detector view" not in annotation_texts
    assert not any(
        getattr(trace, "xaxis", None) == "x2" and getattr(trace, "yaxis", None) == "y2"
        for trace in fig.data
    )
    ki_tip_arrow = next(
        trace
        for trace in fig.data
        if isinstance(trace, go.Cone) and getattr(trace, "anchor", None) == "tail"
    )
    ki_tip = np.array([ki_tip_arrow.x[0], ki_tip_arrow.y[0], ki_tip_arrow.z[0]], dtype=float)
    ki_arrow = np.array([ki_tip_arrow.u[0], ki_tip_arrow.v[0], ki_tip_arrow.w[0]], dtype=float)
    assert ki_tip_arrow.sizemode == "raw"
    assert ki_tip_arrow.sizeref == pytest.approx(1.0)
    np.testing.assert_allclose(ki_arrow, -0.08 * ki_tip)


def test_build_specular_figure_hides_off_detector_intersections_in_detector_plane():
    beam = BeamConfig(
        ray_count=1,
        source_y=-150.0,
        width_x=0.0,
        width_z=0.0,
        divergence_x_deg=0.0,
        divergence_z_deg=0.0,
        z_offset=0.0,
        seed=1,
        display_rays=1,
    )
    sample = SampleConfig(
        width=50.0,
        height=200.0,
        theta_i_deg=10.0,
        delta_deg=0.0,
        alpha_deg=0.0,
        psi_deg=0.0,
        z_offset=0.0,
    )
    detector = DetectorConfig(
        distance=200.0,
        width=100.0,
        height=100.0,
        beta_deg=0.0,
        gamma_deg=0.0,
        chi_deg=0.0,
        pixel_u=0.1,
        pixel_v=0.1,
        i0=1000.0,
        j0=1000.0,
    )
    diffraction = DiffractionConfig(H=0, K=0, L=12, sigma_deg=0.8, mosaic_gamma_deg=5.0, eta=0.5)
    result = trace_specular_simulation(beam, sample, detector, diffraction)
    fig = build_specular_figure(result, beam, sample, detector, diffraction)

    assert result.detector_plane_hit_count == result.diffraction_ray_count
    assert result.detector_hit_count == 0
    assert not any(getattr(trace, "name", "") == "Off-detector intersections" for trace in fig.data)
    assert not any(getattr(trace, "name", "") == "Detector hits" for trace in fig.data)
    direct_beam_trace = next(trace for trace in fig.data if getattr(trace, "name", "") == "Direct beam center")
    assert direct_beam_trace.x == (0.0,)
    assert direct_beam_trace.y == (0.0,)
    assert fig.layout.xaxis2.range == (-52.5, 52.5)
    assert fig.layout.yaxis2.range == (-52.5, 52.5)


def test_theta_i_changes_live_specular_summary_even_without_active_detector_hits():
    beam = BeamConfig()
    detector = DetectorConfig()
    diffraction = DiffractionConfig(H=0, K=0, L=12, sigma_deg=0.8, mosaic_gamma_deg=5.0, eta=0.5)

    _, summary_low = build_specular_outputs(
        beam,
        SampleConfig(theta_i_deg=10.0),
        detector,
        diffraction,
    )
    _, summary_high = build_specular_outputs(
        beam,
        SampleConfig(theta_i_deg=20.0),
        detector,
        diffraction,
    )

    assert "weighted detector-plane centroid" in summary_low
    assert "weighted detector-plane centroid" in summary_high
    assert summary_low != summary_high


def test_build_specular_figure_compacts_numeric_payload():
    beam = BeamConfig(ray_count=4, display_rays=2)
    sample = SampleConfig()
    detector = DetectorConfig()
    diffraction = DiffractionConfig(H=0, K=0, L=12)
    result = trace_specular_simulation(beam, sample, detector, diffraction)

    fig = build_specular_figure(result, beam, sample, detector, diffraction)

    detector_trace = next(trace for trace in fig.data if getattr(trace, "name", "") == "Detector intersections")
    assert np.asarray(detector_trace.x).dtype == np.float32
    assert np.asarray(detector_trace.marker.color).dtype == np.float32


def test_configs_from_values_falls_back_to_defaults_for_none_gui_values():
    default_beam = BeamConfig(ray_count=40, seed=9, display_rays=12)
    default_sample = SampleConfig(theta_i_deg=14.0, width=25.0)
    default_detector = DetectorConfig(distance=320.0, pixel_u=0.2, pixel_v=0.3)
    default_diffraction = DiffractionConfig(H=1, K=2, L=7, sigma_deg=1.2, mosaic_gamma_deg=4.5, eta=0.3)

    beam, sample, detector, diffraction = configs_from_values(
        rays=None,
        seed=None,
        theta_i=None,
        distance=None,
        H=None,
        K=None,
        L=None,
        sigma_deg=None,
        mosaic_gamma_deg=None,
        eta=None,
        default_beam=default_beam,
        default_sample=default_sample,
        default_detector=default_detector,
        default_diffraction=default_diffraction,
    )

    assert beam.ray_count == 40
    assert beam.seed == 9
    assert beam.display_rays == 12
    assert sample.theta_i_deg == pytest.approx(14.0)
    assert sample.width == pytest.approx(25.0)
    assert detector.distance == pytest.approx(320.0)
    assert detector.pixel_u == pytest.approx(0.2)
    assert detector.pixel_v == pytest.approx(0.3)
    assert diffraction.H == 1
    assert diffraction.K == 2
    assert diffraction.L == 7
    assert diffraction.sigma_deg == pytest.approx(1.2)
    assert diffraction.mosaic_gamma_deg == pytest.approx(4.5)
    assert diffraction.eta == pytest.approx(0.3)


def test_build_specular_app_seeds_inputs_and_figure_from_initial_values():
    initial_sample = SampleConfig(width=30.0, height=70.0, theta_i_deg=8.5, delta_deg=0.4)
    initial_diffraction = DiffractionConfig(H=0, K=0, L=12, sigma_deg=0.8, mosaic_gamma_deg=5.0, eta=0.5)
    app = build_specular_app(
        BeamConfig(ray_count=48, seed=11, display_rays=9, source_y=-125.0),
        initial_sample,
        DetectorConfig(distance=260.0, beta_deg=3.0, pixel_u=0.2, pixel_v=0.25, i0=900.0, j0=875.0),
        initial_diffraction,
    )

    rays_control = find_component_by_id(app.layout, control_id("rays", "slider"))
    rays_input = find_component_by_id(app.layout, control_id("rays", "input"))
    theta_control = find_component_by_id(app.layout, control_id("theta_i", "slider"))
    beta_control = find_component_by_id(app.layout, control_id("beta", "slider"))
    h_control = find_component_by_id(app.layout, control_id("H", "slider"))
    gamma_control = find_component_by_id(app.layout, control_id("mosaic_gamma_deg", "slider"))
    graph = find_component_by_id(app.layout, "specular-fig")
    companion_graph = find_component_by_id(app.layout, "specular-companion-fig")
    summary = find_component_by_id(app.layout, "specular-summary")
    summary_text = render_component_text(summary)

    assert rays_control.value == 48
    assert rays_input.value == 48
    assert theta_control.value == pytest.approx(8.5)
    assert beta_control.value == pytest.approx(3.0)
    assert h_control.value == 0
    assert gamma_control.value == pytest.approx(5.0)
    assert isinstance(graph, dcc.Graph)
    assert isinstance(companion_graph, dcc.Graph)
    assert graph.figure.layout.uirevision == SPECULAR_CAMERA_UIREVISION
    assert graph.figure.layout.scene.uirevision == SPECULAR_CAMERA_UIREVISION
    assert "HKL = (0, 0, 12)" in graph.figure.layout.title.text
    assert "|G| =" in graph.figure.layout.title.text
    assert "diffraction rays" in graph.figure.layout.title.text
    assert "HKL = (0, 0, 12)" in companion_graph.figure.layout.title.text
    assert "(0, 0, 12)" in summary_text
    assert "Diffraction rays" in summary_text


def test_build_specular_app_uses_split_shell_with_sidebar_summary():
    app = build_specular_app()

    sidebar = next(child for child in app.layout.children if getattr(child, "className", None) == "specular-sidebar")
    main = next(child for child in app.layout.children if getattr(child, "className", None) == "specular-main")
    summary = find_component_by_id(app.layout, "specular-summary")
    basic_card = find_component_by_id(app.layout, "specular-basic-card")
    advanced_sections = find_component_by_id(app.layout, "specular-advanced-sections")
    graph = find_component_by_id(app.layout, "specular-fig")
    companion_graph = find_component_by_id(app.layout, "specular-companion-fig")

    assert sidebar.className == "specular-sidebar"
    assert main.className == "specular-main"
    assert summary.id == "specular-summary"
    assert basic_card.id == "specular-basic-card"
    assert advanced_sections.id == "specular-advanced-sections"
    assert isinstance(graph, dcc.Graph)
    assert isinstance(companion_graph, dcc.Graph)
    assert "HKL = (1, 1, 1)" in graph.figure.layout.title.text
    assert "HKL = (1, 1, 1)" in companion_graph.figure.layout.title.text
    assert "Start With the Reflection" in render_component_text(basic_card)
    assert find_component_by_id(basic_card, control_id("theta_i", "slider")) is not None
    assert find_component_by_id(basic_card, control_id("H", "slider")) is not None
    assert find_component_by_id(basic_card, control_id("mosaic_gamma_deg", "slider")) is not None
    for advanced_id in (
        "specular-advanced-sample-geometry",
        "specular-advanced-beam-model",
        "specular-advanced-detector-geometry",
    ):
        section = find_component_by_id(app.layout, advanced_id)
        assert section is not None
        assert section.open is False


def test_build_specular_app_only_keeps_theta_i_live():
    app = build_specular_app()

    theta_control = find_component_by_id(app.layout, control_id("theta_i", "slider"))
    beta_control = find_component_by_id(app.layout, control_id("beta", "slider"))
    sigma_control = find_component_by_id(app.layout, control_id("sigma_deg", "slider"))

    assert theta_control.min == pytest.approx(0.0)
    assert theta_control.max == pytest.approx(90.0)
    assert theta_control.updatemode == "drag"
    assert beta_control.updatemode == "mouseup"
    assert sigma_control.updatemode == "mouseup"


def test_build_specular_app_restores_all_parameter_slider_and_input_controls():
    app = build_specular_app()
    basic_card = find_component_by_id(app.layout, "specular-basic-card")
    sample_advanced = find_component_by_id(app.layout, "specular-advanced-sample-geometry")
    beam_advanced = find_component_by_id(app.layout, "specular-advanced-beam-model")
    detector_advanced = find_component_by_id(app.layout, "specular-advanced-detector-geometry")
    expected_control_names = (
        "rays",
        "seed",
        "display_rays",
        "source_y",
        "beam_width_x",
        "beam_width_z",
        "divergence_x",
        "divergence_z",
        "z_beam",
        "sample_width",
        "sample_height",
        "theta_i",
        "delta",
        "alpha",
        "psi",
        "z_sample",
        "distance",
        "detector_width",
        "detector_height",
        "beta",
        "gamma",
        "chi",
        "pixel_u",
        "pixel_v",
        "H",
        "K",
        "L",
        "sigma_deg",
        "mosaic_gamma_deg",
        "eta",
    )

    for name in expected_control_names:
        assert find_component_by_id(app.layout, control_id(name, "slider")) is not None
        assert find_component_by_id(app.layout, control_id(name, "input")) is not None

    assert find_component_by_id(app.layout, control_id("i0", "slider")) is None
    assert find_component_by_id(app.layout, control_id("j0", "slider")) is None
    assert find_component_by_id(app.layout, control_id("i0", "input")) is None
    assert find_component_by_id(app.layout, control_id("j0", "input")) is None
    for name in specular.SPECULAR_BASIC_CONTROL_NAMES:
        assert find_component_by_id(basic_card, control_id(name, "slider")) is not None
    for name in ("sample_width", "sample_height", "delta", "alpha", "psi", "z_sample"):
        assert find_component_by_id(sample_advanced, control_id(name, "slider")) is not None
    for name in (
        "rays",
        "display_rays",
        "seed",
        "source_y",
        "beam_width_x",
        "beam_width_z",
        "divergence_x",
        "divergence_z",
        "z_beam",
    ):
        assert find_component_by_id(beam_advanced, control_id(name, "slider")) is not None
    for name in ("distance", "detector_width", "detector_height", "beta", "gamma", "chi", "pixel_u", "pixel_v"):
        assert find_component_by_id(detector_advanced, control_id(name, "slider")) is not None


def test_build_specular_app_uses_supplied_initial_outputs(monkeypatch):
    def fail_build_specular_dashboard_outputs(*args, **kwargs):
        raise AssertionError("build_specular_dashboard_outputs should not run when initial outputs are provided")

    monkeypatch.setattr(specular, "build_specular_dashboard_outputs", fail_build_specular_dashboard_outputs)
    supplied_figure = go.Figure()
    supplied_figure.update_layout(title_text="Precomputed figure")
    supplied_companion_figure = go.Figure()
    supplied_companion_figure.update_layout(title_text="Precomputed companion")

    app = specular.build_specular_app(
        BeamConfig(),
        SampleConfig(),
        DetectorConfig(),
        DiffractionConfig(),
        initial_figure=supplied_figure,
        initial_companion_figure=supplied_companion_figure,
        initial_summary="Precomputed summary",
    )

    graph = find_component_by_id(app.layout, "specular-fig")
    companion_graph = find_component_by_id(app.layout, "specular-companion-fig")
    summary = find_component_by_id(app.layout, "specular-summary")

    assert graph.figure.layout.title.text == "Precomputed figure"
    assert companion_graph.figure.layout.title.text == "Precomputed companion"
    assert "Precomputed summary" in render_component_text(summary)


def test_build_specular_app_renders_math_style_control_labels():
    app = build_specular_app()

    beam_width_label = find_label_by_html_for(app.layout, str(control_id("beam_width_x", "slider")))
    divergence_label = find_label_by_html_for(app.layout, str(control_id("divergence_z", "slider")))
    theta_label = find_label_by_html_for(app.layout, str(control_id("theta_i", "slider")))
    h_label = find_label_by_html_for(app.layout, str(control_id("H", "slider")))
    sigma_label = find_label_by_html_for(app.layout, str(control_id("sigma_deg", "slider")))
    gamma_label = find_label_by_html_for(app.layout, str(control_id("mosaic_gamma_deg", "slider")))
    eta_label = find_label_by_html_for(app.layout, str(control_id("eta", "slider")))

    assert beam_width_label is not None
    assert divergence_label is not None
    assert theta_label is not None
    assert h_label is not None
    assert sigma_label is not None
    assert gamma_label is not None
    assert eta_label is not None
    assert render_component_text(beam_width_label.children) == "w<sub>x</sub>"
    assert render_component_text(divergence_label.children) == "Δθ<sub>z</sub> (deg)"
    assert render_component_text(theta_label.children) == "θ<sub>i</sub> (deg)"
    assert render_component_text(h_label.children) == "H"
    assert render_component_text(sigma_label.children) == "σ (deg)"
    assert render_component_text(gamma_label.children) == "Γ (deg)"
    assert render_component_text(eta_label.children) == "η"


def test_build_specular_app_callback_preserves_camera_and_updates_summary():
    app = build_specular_app()
    main_callback = callback_by_output(app, "specular-fig.figure")
    companion_callback = callback_by_output(app, "specular-companion-fig.figure")
    summary_callback = callback_by_output(app, "specular-summary.children")
    graph = find_component_by_id(app.layout, "specular-fig")
    companion_graph = find_component_by_id(app.layout, "specular-companion-fig")
    summary_store = find_component_by_id(app.layout, "specular-summary-text")

    slider_ids = [
        control_id("rays", "slider"),
        control_id("seed", "slider"),
        control_id("display_rays", "slider"),
        control_id("source_y", "slider"),
        control_id("beam_width_x", "slider"),
        control_id("beam_width_z", "slider"),
        control_id("divergence_x", "slider"),
        control_id("divergence_z", "slider"),
        control_id("z_beam", "slider"),
        control_id("sample_width", "slider"),
        control_id("sample_height", "slider"),
        control_id("theta_i", "slider"),
        control_id("delta", "slider"),
        control_id("alpha", "slider"),
        control_id("psi", "slider"),
        control_id("z_sample", "slider"),
        control_id("distance", "slider"),
        control_id("detector_width", "slider"),
        control_id("detector_height", "slider"),
        control_id("beta", "slider"),
        control_id("gamma", "slider"),
        control_id("chi", "slider"),
        control_id("pixel_u", "slider"),
        control_id("pixel_v", "slider"),
        control_id("H", "slider"),
        control_id("K", "slider"),
        control_id("L", "slider"),
        control_id("sigma_deg", "slider"),
        control_id("mosaic_gamma_deg", "slider"),
        control_id("eta", "slider"),
    ]
    slider_values = [
        60,
        7,
        12,
        -150.0,
        0.15,
        0.15,
        0.03,
        0.03,
        0.0,
        20.0,
        80.0,
        12.0,
        0.0,
        0.0,
        0.0,
        0.0,
        200.0,
        180.0,
        180.0,
        0.0,
        0.0,
        0.0,
        0.1,
        0.1,
        0,
        0,
        12,
        0.8,
        5.0,
        0.5,
    ]
    fig = main_callback(
        slider_values,
        slider_ids,
        {
            "scene.camera.eye.x": 1.6,
            "scene.camera.eye.y": 1.2,
            "scene.camera.eye.z": 0.8,
            "scene.camera.up.x": 0.0,
            "scene.camera.up.y": 0.0,
            "scene.camera.up.z": 1.0,
            "scene.camera.center.x": 0.0,
            "scene.camera.center.y": 0.0,
            "scene.camera.center.z": 0.0,
        },
        graph.figure,
    )
    companion_fig = companion_callback(
        slider_values,
        slider_ids,
        {
            "scene.camera.eye.x": 1.25,
            "scene.camera.eye.y": 0.95,
            "scene.camera.eye.z": 0.7,
            "scene.camera.up.x": 0.0,
            "scene.camera.up.y": 0.0,
            "scene.camera.up.z": 1.0,
        },
        companion_graph.figure,
    )
    summary_children, summary_text = summary_callback(slider_values, slider_ids, summary_store.data)
    rendered_summary = render_component_text(summary_children)

    assert fig.layout.scene.camera.eye.x == pytest.approx(1.6)
    assert fig.layout.scene.camera.eye.y == pytest.approx(1.2)
    assert fig.layout.scene.camera.eye.z == pytest.approx(0.8)
    assert companion_fig.layout.scene.camera.eye.x == pytest.approx(1.25)
    assert companion_fig.layout.scene.camera.eye.y == pytest.approx(0.95)
    assert companion_fig.layout.scene.camera.eye.z == pytest.approx(0.7)
    assert "HKL = (0, 0, 12)" in fig.layout.title.text
    assert "diffraction rays" in fig.layout.title.text
    assert "detector hits" in fig.layout.title.text
    assert "HKL = (0, 0, 12)" in companion_fig.layout.title.text
    assert "(0, 0, 12)" in summary_text
    assert "Diffraction rays" in rendered_summary
    assert "Detector hits" in rendered_summary
    assert "Plane hits" in rendered_summary
    assert "nominal 2θ" not in summary_text


def test_specular_companion_callback_skips_main_only_updates():
    app = build_specular_app()
    callback = callback_by_output(app, "specular-companion-fig.figure")
    companion_graph = find_component_by_id(app.layout, "specular-companion-fig")

    slider_ids = [control_id(name, "slider") for name in specular.SPECULAR_CONTROL_NAMES]
    value_map = {
        "rays": 600,
        "seed": 7,
        "display_rays": 80,
        "source_y": -150.0,
        "beam_width_x": 0.15,
        "beam_width_z": 0.15,
        "divergence_x": 0.03,
        "divergence_z": 0.03,
        "z_beam": 0.0,
        "sample_width": 20.0,
        "sample_height": 80.0,
        "theta_i": 10.0,
        "delta": 0.0,
        "alpha": 0.0,
        "psi": 0.0,
        "z_sample": 0.0,
        "distance": 200.0,
        "detector_width": 1000.0,
        "detector_height": 1000.0,
        "beta": 3.5,
        "gamma": 0.0,
        "chi": 0.0,
        "pixel_u": 0.1,
        "pixel_v": 0.1,
        "i0": 1024.0,
        "j0": 1024.0,
        "H": 1,
        "K": 1,
        "L": 1,
        "sigma_deg": 0.8,
        "mosaic_gamma_deg": 5.0,
        "eta": 0.5,
    }
    slider_values = [value_map[control_id["name"]] for control_id in slider_ids]

    with pytest.raises(PreventUpdate):
        callback(slider_values, slider_ids, None, companion_graph.figure)


def test_specular_summary_callback_skips_display_only_updates():
    app = build_specular_app()
    callback = callback_by_output(app, "specular-summary.children")
    summary_store = find_component_by_id(app.layout, "specular-summary-text")

    slider_ids = [control_id(name, "slider") for name in specular.SPECULAR_CONTROL_NAMES]
    value_map = {
        "rays": 600,
        "seed": 7,
        "display_rays": 120,
        "source_y": -150.0,
        "beam_width_x": 0.15,
        "beam_width_z": 0.15,
        "divergence_x": 0.03,
        "divergence_z": 0.03,
        "z_beam": 0.0,
        "sample_width": 20.0,
        "sample_height": 80.0,
        "theta_i": 10.0,
        "delta": 0.0,
        "alpha": 0.0,
        "psi": 0.0,
        "z_sample": 0.0,
        "distance": 200.0,
        "detector_width": 1000.0,
        "detector_height": 1000.0,
        "beta": 0.0,
        "gamma": 0.0,
        "chi": 0.0,
        "pixel_u": 0.1,
        "pixel_v": 0.1,
        "i0": 1024.0,
        "j0": 1024.0,
        "H": 1,
        "K": 1,
        "L": 1,
        "sigma_deg": 0.8,
        "mosaic_gamma_deg": 5.0,
        "eta": 0.5,
    }
    slider_values = [value_map[control_id["name"]] for control_id in slider_ids]

    with pytest.raises(PreventUpdate):
        callback(slider_values, slider_ids, summary_store.data)


def test_trace_specular_simulation_reuses_projection_cache_for_calibration_only_changes():
    specular._cached_projection_trace_context.cache_clear()
    specular._cached_calibrated_projection_context.cache_clear()

    beam = BeamConfig(ray_count=32, display_rays=8)
    sample = SampleConfig()
    diffraction = DiffractionConfig(H=0, K=0, L=12)
    base_detector = DetectorConfig(pixel_u=0.1, pixel_v=0.1, i0=1024.0, j0=1024.0)
    updated_detector = DetectorConfig(pixel_u=0.2, pixel_v=0.3, i0=900.0, j0=875.0)

    trace_specular_simulation(beam, sample, base_detector, diffraction)
    before = specular._cached_projection_trace_context.cache_info()

    trace_specular_simulation(beam, sample, updated_detector, diffraction)
    after = specular._cached_projection_trace_context.cache_info()

    assert after.hits >= before.hits + 1
    assert after.misses == before.misses
