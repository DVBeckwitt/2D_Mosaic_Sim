import math

import numpy as np
import pytest
from dash import dcc, html

from specular_reflection_sim import (
    BeamConfig,
    DetectorConfig,
    SampleConfig,
    SPECULAR_CAMERA_UIREVISION,
    build_specular_app,
    build_detector_frame,
    build_sample_frame,
    configs_from_values,
    nominal_scattering_angle_deg,
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


def test_untilted_detector_places_specular_spot_at_expected_two_theta():
    theta_i_deg = 10.0
    detector_distance = 200.0

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
            theta_i_deg=theta_i_deg,
            delta_deg=0.0,
            alpha_deg=0.0,
            psi_deg=0.0,
            z_offset=0.0,
        ),
        DetectorConfig(
            distance=detector_distance,
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
    )

    assert result.sample_hit_count == 1
    assert result.detector_hit_count == 1
    assert result.detector_uv[0, 0] == pytest.approx(0.0, abs=1e-9)

    expected_v = detector_distance * math.tan(math.radians(2.0 * theta_i_deg))
    assert result.detector_uv[0, 1] == pytest.approx(expected_v, rel=1e-9, abs=1e-9)
    assert nominal_scattering_angle_deg(result.sample) == pytest.approx(2.0 * theta_i_deg)


def test_configs_from_values_falls_back_to_defaults_for_none_gui_values():
    default_beam = BeamConfig(ray_count=40, seed=9, display_rays=12)
    default_sample = SampleConfig(theta_i_deg=14.0, width=25.0)
    default_detector = DetectorConfig(distance=320.0, pixel_u=0.2, pixel_v=0.3)

    beam, sample, detector = configs_from_values(
        rays=None,
        seed=None,
        theta_i=None,
        distance=None,
        default_beam=default_beam,
        default_sample=default_sample,
        default_detector=default_detector,
    )

    assert beam.ray_count == 40
    assert beam.seed == 9
    assert beam.display_rays == 12
    assert sample.theta_i_deg == pytest.approx(14.0)
    assert sample.width == pytest.approx(25.0)
    assert detector.distance == pytest.approx(320.0)
    assert detector.pixel_u == pytest.approx(0.2)
    assert detector.pixel_v == pytest.approx(0.3)


def test_build_specular_app_seeds_inputs_and_figure_from_initial_values():
    initial_sample = SampleConfig(width=30.0, height=70.0, theta_i_deg=8.5, delta_deg=0.4)
    app = build_specular_app(
        BeamConfig(ray_count=48, seed=11, display_rays=9, source_y=-125.0),
        initial_sample,
        DetectorConfig(distance=260.0, beta_deg=3.0, pixel_u=0.2, pixel_v=0.25, i0=900.0, j0=875.0),
    )

    rays_control = find_component_by_id(app.layout, control_id("rays", "slider"))
    rays_input = find_component_by_id(app.layout, control_id("rays", "input"))
    theta_control = find_component_by_id(app.layout, control_id("theta_i", "slider"))
    beta_control = find_component_by_id(app.layout, control_id("beta", "slider"))
    graph = find_component_by_id(app.layout, "specular-fig")
    summary = find_component_by_id(app.layout, "specular-summary")

    assert rays_control.value == 48
    assert rays_input.value == 48
    assert theta_control.value == pytest.approx(8.5)
    assert beta_control.value == pytest.approx(3.0)
    assert isinstance(graph, dcc.Graph)
    assert graph.figure.layout.uirevision == SPECULAR_CAMERA_UIREVISION
    assert graph.figure.layout.scene.uirevision == SPECULAR_CAMERA_UIREVISION
    expected_two_theta = nominal_scattering_angle_deg(build_sample_frame(initial_sample))
    assert f"nominal 2θ = {expected_two_theta:.2f}°" in graph.figure.layout.title.text
    assert f"nominal 2θ from +y: {expected_two_theta:.6f} deg" in summary.children


def test_build_specular_app_renders_math_style_control_labels():
    app = build_specular_app()

    beam_width_label = find_label_by_html_for(app.layout, str(control_id("beam_width_x", "slider")))
    divergence_label = find_label_by_html_for(app.layout, str(control_id("divergence_z", "slider")))
    theta_label = find_label_by_html_for(app.layout, str(control_id("theta_i", "slider")))
    i0_label = find_label_by_html_for(app.layout, str(control_id("i0", "slider")))

    assert beam_width_label is not None
    assert divergence_label is not None
    assert theta_label is not None
    assert i0_label is not None
    assert render_component_text(beam_width_label.children) == "w<sub>x</sub>"
    assert render_component_text(divergence_label.children) == "Δθ<sub>z</sub> (deg)"
    assert render_component_text(theta_label.children) == "θ<sub>i</sub> (deg)"
    assert render_component_text(i0_label.children) == "i<sub>0</sub>"


def test_build_specular_app_callback_preserves_camera_and_updates_summary():
    app = build_specular_app()
    callback = next(
        value["callback"].__wrapped__
        for key, value in app.callback_map.items()
        if "specular-fig.figure" in key and "specular-summary.children" in key
    )

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
        control_id("i0", "slider"),
        control_id("j0", "slider"),
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
        1024.0,
        1024.0,
    ]
    fig, summary = callback(
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
    )

    assert fig.layout.scene.camera.eye.x == pytest.approx(1.6)
    assert fig.layout.scene.camera.eye.y == pytest.approx(1.2)
    assert fig.layout.scene.camera.eye.z == pytest.approx(0.8)
    assert "nominal 2θ = 24.00°" in fig.layout.title.text
    assert "nominal 2θ from +y: 24.000000 deg" in summary
