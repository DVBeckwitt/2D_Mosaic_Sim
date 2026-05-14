from pathlib import Path

from dash import dcc, html
from dash._utils import to_json
from dash.exceptions import PreventUpdate
import numpy as np
import pytest

import mosaic_sim.unified_app as unified_app
from mosaic_sim.unified_app import (
    PNG_EXPORT_CLIENTSIDE_CALLBACK,
    POWDER_QR_CLIENTSIDE_CALLBACK,
    SPECIAL_CAUSE_MATRIX_EXPORT_CLIENTSIDE_CALLBACK,
    SIMULATION_SPECS,
    _updated_mode_state,
    _updated_powder_selection_state,
    _updated_powder_view,
    build_unified_app,
    build_unified_figure,
)


def _specular_section_keys(section) -> list[str]:
    return [control.children[1].children[0].children.id["key"] for control in section.children[1].children]


def _specular_section_input_keys(section) -> list[str]:
    return [control.children[1].children[1].id["key"] for control in section.children[1].children]


def _shell(app):
    return app.layout.children[-1]


def _find_component_by_id(component, target_id):
    if getattr(component, "id", None) == target_id:
        return component

    children = getattr(component, "children", None)
    if children is None:
        return None

    if isinstance(children, (list, tuple)):
        for child in children:
            result = _find_component_by_id(child, target_id)
            if result is not None:
                return result
        return None

    return _find_component_by_id(children, target_id)


def _find_component_by_class(component, class_name: str):
    class_names = str(getattr(component, "className", "") or "").split()
    if class_name in class_names:
        return component

    children = getattr(component, "children", None)
    if children is None:
        return None

    if isinstance(children, (list, tuple)):
        for child in children:
            result = _find_component_by_class(child, class_name)
            if result is not None:
                return result
        return None

    return _find_component_by_class(children, class_name)


def _render_component_text(component) -> str:
    if component is None:
        return ""
    if isinstance(component, (str, int, float)):
        return str(component)
    if isinstance(component, (list, tuple)):
        return "".join(_render_component_text(child) for child in component)
    return _render_component_text(getattr(component, "children", None))


def _callback_by_output(app, output_key: str):
    for key, value in app.callback_map.items():
        if key == output_key or output_key in key:
            return value["callback"].__wrapped__
    raise KeyError(output_key)


def _sidebar(app):
    return _find_component_by_id(app.layout, "simulation-sidebar")


def _main(app):
    return _find_component_by_id(app.layout, "simulation-main")


def _graph(app):
    return _find_component_by_id(app.layout, "simulation-figure")


def _specular_companion_card(app):
    return _find_component_by_id(app.layout, "simulation-specular-companion-card")


def _specular_companion_graph(app):
    return _find_component_by_id(app.layout, "simulation-specular-companion-figure")


def _summary(app):
    return _find_component_by_id(app.layout, "simulation-summary")


def _controls(app):
    return _find_component_by_id(app.layout, "simulation-controls").children


def _control_section_titles(app) -> list[str]:
    return [
        _render_component_text(control_group.children[0])
        for control_group in _controls(app)
        if "simulation-control-section" in str(getattr(control_group, "className", ""))
    ]


def _control_section(app, title: str):
    for control_group in _controls(app):
        if "simulation-control-section" not in str(getattr(control_group, "className", "")):
            continue
        if _render_component_text(control_group.children[0]) == title:
            return control_group
    raise AssertionError(f"Missing control section: {title}")


def _main_figure_callback(app):
    return _callback_by_output(app, "simulation-figure.figure")


def _companion_figure_callback(app):
    return _callback_by_output(app, "simulation-specular-companion-figure.figure")


def test_unified_registry_exposes_all_simulations():
    assert set(SIMULATION_SPECS) == {
        "reciprocal-space",
        "detector-view",
        "special-cause-reciprocal",
        "fibrous-view",
        "specular-view",
    }


def test_build_unified_figure_uses_selected_mode():
    fig = build_unified_figure("detector-view", H=0, K=0, L=12, sigma_deg=0.8, Gamma_deg=5.0, eta=0.5)

    assert "HKL = (0, 0, 12)" in fig.layout.title.text


def test_build_unified_figure_uses_special_cause_reciprocal_mode():
    fig = build_unified_figure(
        "special-cause-reciprocal",
        H=0,
        K=0,
        L=12,
        sigma_deg=0.8,
        Gamma_deg=5.0,
        eta=0.5,
    )

    assert "HKL = (0, 0, 12)" in fig.layout.title.text
    assert any(trace.name == "Bragg sphere" for trace in fig.data)
    assert any(trace.name == "Ewald shell inner" for trace in fig.data)
    assert any(trace.name == "Ewald shell outer" for trace in fig.data)
    assert any(trace.name == "Bragg/Ewald overlap band" for trace in fig.data)
    assert not any(trace.type == "scatter" for trace in fig.data)


def test_build_unified_figure_uses_specular_mode_and_sets_summary_meta():
    fig = build_unified_figure("specular-view")

    assert "HKL = (1, 1, 1)" in fig.layout.title.text
    assert "diffraction rays" in fig.layout.title.text
    assert "nominal 2θ" not in fig.layout.title.text
    assert "HKL = (1, 1, 1)" in fig.layout.meta["simulation_summary"]
    assert "diffraction rays" in fig.layout.meta["simulation_summary"]
    assert "nominal 2θ" not in fig.layout.meta["simulation_summary"]


def test_build_unified_app_seeds_mode_selector_and_initial_figure():
    app = build_unified_app(initial_mode="fibrous-view")

    main = _main(app)
    mode_selector = _find_component_by_id(app.layout, "simulation-mode")
    export_toolbar = main.children[0]
    graph = _graph(app)

    assert isinstance(mode_selector, dcc.RadioItems)
    assert mode_selector.value == "fibrous-view"
    assert export_toolbar.children[0].id == "export-png-button"
    assert export_toolbar.children[0].title == "Download the current visualization as a PNG"
    assert isinstance(graph, dcc.Graph)
    assert "HKL = (0, 0, 12)" in graph.figure.layout.title.text


def test_unified_layout_uses_compact_browser_header_for_mode_switching():
    app = build_unified_app(initial_mode="fibrous-view")

    header = _find_component_by_id(app.layout, "simulation-sidebar-header")
    brand = _find_component_by_class(header, "simulation-brand")
    mode_card = _find_component_by_class(header, "simulation-mode-card")
    mode_helper = _find_component_by_class(header, "simulation-mode-helper")
    mode_selector = _find_component_by_id(header, "simulation-mode")

    assert header is not None
    assert brand is not None
    assert mode_card is not None
    assert "each mode" in _render_component_text(mode_helper)
    assert isinstance(mode_selector, dcc.RadioItems)
    assert mode_selector.inline is True
    assert mode_selector.labelClassName == "simulation-mode-option"
    assert mode_selector.inputClassName == "simulation-mode-input"
    assert [option["value"] for option in mode_selector.options] == list(SIMULATION_SPECS)


def test_unified_graphs_use_dash_responsive_graph_property():
    app = build_unified_app(initial_mode="specular-view")

    assert _graph(app).responsive is True
    assert _specular_companion_graph(app).responsive is True


def test_build_unified_app_seeds_specular_mode_selector_and_initial_figure():
    app = build_unified_app(initial_mode="specular-view")

    mode_selector = _find_component_by_id(app.layout, "simulation-mode")
    graph = _graph(app)
    companion_card = _specular_companion_card(app)
    companion_graph = _specular_companion_graph(app)
    summary = _summary(app)
    basic_card = _find_component_by_id(app.layout, "simulation-specular-basic-card")
    control_sections = _controls(app)

    assert isinstance(mode_selector, dcc.RadioItems)
    assert mode_selector.value == "specular-view"
    assert isinstance(graph, dcc.Graph)
    assert isinstance(companion_graph, dcc.Graph)
    assert "HKL = (1, 1, 1)" in graph.figure.layout.title.text
    assert "diffraction rays" in graph.figure.layout.title.text
    assert "HKL = (1, 1, 1)" in companion_graph.figure.layout.title.text
    assert "(1, 1, 1)" in _render_component_text(summary)
    assert "Diffraction rays" in _render_component_text(summary)
    assert "nominal 2θ" not in _render_component_text(summary)
    assert summary.style.get("display") != "none"
    assert companion_card.style.get("display") != "none"
    assert len(control_sections) == 2
    assert "Start With the Reflection" in _render_component_text(basic_card)
    assert _find_component_by_id(basic_card, {"type": "simulation-hybrid-slider", "key": "H"}) is not None
    assert _find_component_by_id(
        basic_card,
        {"type": "simulation-hybrid-slider", "key": "mosaic_gamma_deg"},
    ) is not None


def test_build_unified_app_seeds_specular_state_payload():
    app = build_unified_app(
        initial_mode="specular-view",
        initial_state={
            "specular-view": {
                "H": 3,
                "K": -1,
                "L": 5,
                "theta_i": 14.5,
                "sigma_deg": 1.2,
                "wavelength_m": 1.3776e-10,
                "lattice_a_m": 4.95e-10,
                "lattice_c_m": 2.68e-9,
            }
        },
    )

    graph = _graph(app)
    companion_graph = _specular_companion_graph(app)
    state_store = app.layout.children[0]

    assert "HKL = (3, -1, 5)" in graph.figure.layout.title.text
    assert "HKL = (3, -1, 5)" in companion_graph.figure.layout.title.text
    assert state_store.data["specular-view"]["wavelength_m"] == pytest.approx(1.3776e-10)
    assert state_store.data["specular-view"]["lattice_a_m"] == pytest.approx(4.95e-10)
    assert state_store.data["specular-view"]["lattice_c_m"] == pytest.approx(2.68e-9)


def test_unified_non_specular_modes_keep_summary_hidden():
    app = build_unified_app(initial_mode="fibrous-view")
    summary = _summary(app)
    companion_card = _specular_companion_card(app)

    assert summary.style.get("display") == "none"
    assert companion_card.style.get("display") == "none"


def test_unified_specular_mode_restores_all_parameter_sections_and_controls():
    app = build_unified_app(initial_mode="specular-view")

    control_sections = _controls(app)
    basic_card = _find_component_by_id(app.layout, "simulation-specular-basic-card")
    sample_advanced = _find_component_by_id(app.layout, "simulation-specular-advanced-sample-geometry")
    beam_advanced = _find_component_by_id(app.layout, "simulation-specular-advanced-beam-model")
    detector_advanced = _find_component_by_id(app.layout, "simulation-specular-advanced-detector-geometry")

    assert len(control_sections) == 2
    for key in ("theta_i", "H", "K", "L", "sigma_deg", "mosaic_gamma_deg", "eta"):
        assert _find_component_by_id(basic_card, {"type": "simulation-hybrid-slider", "key": key}) is not None
        assert _find_component_by_id(basic_card, {"type": "simulation-hybrid-input", "key": key}) is not None
    for key in ("sample_width", "sample_height", "delta", "alpha", "psi", "z_sample"):
        assert _find_component_by_id(sample_advanced, {"type": "simulation-hybrid-slider", "key": key}) is not None
    for key in ("rays", "display_rays", "seed", "source_y", "beam_width_x", "beam_width_z", "divergence_x", "divergence_z", "z_beam"):
        assert _find_component_by_id(beam_advanced, {"type": "simulation-hybrid-slider", "key": key}) is not None
    for key in ("distance", "detector_width", "detector_height", "beta", "gamma", "chi", "pixel_u", "pixel_v"):
        assert _find_component_by_id(detector_advanced, {"type": "simulation-hybrid-slider", "key": key}) is not None


def test_unified_specular_layout_serializes_for_dash():
    app = build_unified_app(initial_mode="specular-view")

    serialized = to_json(app.layout)

    assert '"simulation-mode"' in serialized
    assert '"specular-view"' in serialized
    assert '"simulation-specular-basic-card"' in serialized
    assert '"simulation-specular-companion-figure"' in serialized


def test_unified_detector_theta_slider_updates_continuously():
    app = build_unified_app(initial_mode="detector-view")

    theta_control = _find_component_by_id(app.layout, {"type": "simulation-control", "key": "theta_i_deg"})

    assert isinstance(theta_control, dcc.Slider)
    assert theta_control.updatemode == "drag"


def test_unified_layout_keeps_public_callback_ids_and_state_keys_stable():
    app = build_unified_app(initial_mode="detector-view")

    for component_id in (
        "simulation-mode",
        "simulation-controls",
        "simulation-figure",
        "simulation-state",
    ):
        assert _find_component_by_id(app.layout, component_id) is not None

    state_store = _find_component_by_id(app.layout, "simulation-state")
    assert "detector-view" in state_store.data
    assert "wavelength_bandwidth_pct" in state_store.data["detector-view"]


def test_unified_detector_controls_are_grouped_for_browser_scanning():
    app = build_unified_app(initial_mode="detector-view")

    assert _control_section_titles(app) == ["Incident Angle", "Reflection", "Mosaic Envelope"]
    mosaic_section = _control_section(app, "Mosaic Envelope")

    assert _find_component_by_id(mosaic_section, {"type": "simulation-hybrid-input", "key": "wavelength_bandwidth_pct"}) is not None
    assert "λ bandwidth (%)" in _render_component_text(mosaic_section)


def test_unified_special_cause_reciprocal_controls_are_grouped_for_browser_scanning():
    app = build_unified_app(initial_mode="special-cause-reciprocal")

    mode_selector = _find_component_by_id(app.layout, "simulation-mode")
    graph = _graph(app)
    summary = _summary(app)
    companion_card = _specular_companion_card(app)

    assert mode_selector.value == "special-cause-reciprocal"
    assert "HKL = (0, 0, 3)" in graph.figure.layout.title.text
    assert "λ bandwidth = 5.00%" in graph.figure.layout.title.text
    assert _control_section_titles(app) == [
        "Incident Angle",
        "Reflection",
        "Mosaic Envelope",
        "Ewald Sampling",
    ]
    assert _find_component_by_id(
        _control_section(app, "Reflection"),
        {"type": "simulation-control", "key": "L"},
    ).value == 3
    theta_slider = _find_component_by_id(
        _control_section(app, "Incident Angle"),
        {"type": "simulation-hybrid-slider", "key": "theta_i_deg"},
    )
    theta_input = _find_component_by_id(
        _control_section(app, "Incident Angle"),
        {"type": "simulation-hybrid-input", "key": "theta_i_deg"},
    )
    assert isinstance(theta_slider, dcc.Slider)
    assert isinstance(theta_input, dcc.Input)
    assert theta_slider.value == pytest.approx(5.0)
    assert theta_slider.step == pytest.approx(0.25)
    assert theta_slider.updatemode == "drag"
    assert theta_input.type == "number"
    assert theta_input.value == pytest.approx(5.0)
    assert theta_input.step == "any"
    assert _find_component_by_id(
        _control_section(app, "Mosaic Envelope"),
        {"type": "simulation-hybrid-input", "key": "wavelength_bandwidth_pct"},
    ).value == pytest.approx(5.0)
    center_bragg_toggle = _find_component_by_id(
        _control_section(app, "Incident Angle"),
        {"type": "simulation-control", "key": "center_bragg_only"},
    )
    assert isinstance(center_bragg_toggle, dcc.Checklist)
    assert center_bragg_toggle.value == []
    assert "Hide Ewald + angle helpers" in _render_component_text(_control_section(app, "Incident Angle"))
    sample_input = _find_component_by_id(
        _control_section(app, "Ewald Sampling"),
        {"type": "simulation-control", "key": "ewald_shell_sample_count"},
    )
    assert isinstance(sample_input, dcc.Input)
    assert sample_input.type == "number"
    assert sample_input.value == 99
    assert sample_input.step == 2
    assert sample_input.min == 3
    assert sample_input.max == 101
    assert summary.style.get("display") == "none"
    assert companion_card.style.get("display") == "none"


def test_unified_fibrous_controls_are_grouped_for_browser_scanning():
    app = build_unified_app(initial_mode="fibrous-view")

    assert _control_section_titles(app) == ["Reflection", "Mosaic Envelope"]
    mosaic_section = _control_section(app, "Mosaic Envelope")

    assert _find_component_by_id(mosaic_section, {"type": "simulation-control", "key": "wavelength_bandwidth_pct"}) is not None
    assert "λ bandwidth (%)" in _render_component_text(mosaic_section)


def test_unified_reciprocal_controls_are_grouped_for_browser_scanning():
    app = build_unified_app(initial_mode="reciprocal-space")

    assert _control_section_titles(app)[:2] == ["Sweep", "Render"]
    assert _find_component_by_id(_control_section(app, "Sweep"), {"type": "simulation-control", "key": "theta_i_min_deg"}) is not None
    assert _find_component_by_id(_control_section(app, "Render"), {"type": "simulation-control", "key": "render_profile"}) is not None


def test_unified_detector_mosaic_kernel_controls_use_slider_input_pairs():
    app = build_unified_app(initial_mode="detector-view")

    for key, control in (
        (
            "sigma_deg",
            _find_component_by_id(app.layout, {"type": "simulation-hybrid-slider", "key": "sigma_deg"}),
        ),
        (
            "Gamma_deg",
            _find_component_by_id(app.layout, {"type": "simulation-hybrid-slider", "key": "Gamma_deg"}),
        ),
        ("eta", _find_component_by_id(app.layout, {"type": "simulation-hybrid-slider", "key": "eta"})),
    ):
        slider = control
        number = _find_component_by_id(app.layout, {"type": "simulation-hybrid-input", "key": key})

        assert isinstance(slider, dcc.Slider)
        assert isinstance(number, dcc.Input)
        assert slider.id["type"] == "simulation-hybrid-slider"
        assert slider.id["key"] == key
        assert number.id["type"] == "simulation-hybrid-input"
        assert number.id["key"] == key
        assert number.type == "number"

    bandwidth_slider = _find_component_by_id(
        app.layout,
        {"type": "simulation-hybrid-slider", "key": "wavelength_bandwidth_pct"},
    )
    bandwidth_input = _find_component_by_id(
        app.layout,
        {"type": "simulation-hybrid-input", "key": "wavelength_bandwidth_pct"},
    )

    assert isinstance(bandwidth_slider, dcc.Slider)
    assert isinstance(bandwidth_input, dcc.Input)
    assert bandwidth_slider.id["type"] == "simulation-hybrid-slider"
    assert bandwidth_slider.id["key"] == "wavelength_bandwidth_pct"
    assert bandwidth_slider.min == pytest.approx(0.0)
    assert bandwidth_slider.max == pytest.approx(100.0)
    assert bandwidth_input.id["type"] == "simulation-hybrid-input"
    assert bandwidth_input.id["key"] == "wavelength_bandwidth_pct"
    assert bandwidth_input.step == pytest.approx(0.01)
    assert bandwidth_input.max == pytest.approx(100.0)


def test_unified_fibrous_wavelength_bandwidth_control_uses_number_input():
    app = build_unified_app(initial_mode="fibrous-view")

    bandwidth_input = _find_component_by_id(
        app.layout,
        {"type": "simulation-control", "key": "wavelength_bandwidth_pct"},
    )

    assert isinstance(bandwidth_input, dcc.Input)
    assert bandwidth_input.id["type"] == "simulation-control"
    assert bandwidth_input.id["key"] == "wavelength_bandwidth_pct"
    assert bandwidth_input.type == "number"
    assert bandwidth_input.step == pytest.approx(0.01)
    assert bandwidth_input.min == pytest.approx(0.0)
    assert bandwidth_input.max == pytest.approx(100.0)


def test_unified_wavelength_bandwidth_callbacks_preserve_state_values():
    app = build_unified_app(initial_mode="detector-view")
    state = app.layout.children[0].data

    detector_state = _updated_mode_state(
        "detector-view",
        [],
        [1.25],
        [],
        [{"type": "simulation-hybrid-input", "key": "wavelength_bandwidth_pct"}],
        state,
        {"type": "simulation-hybrid-input", "key": "wavelength_bandwidth_pct"},
    )
    fibrous_state = _updated_mode_state(
        "fibrous-view",
        [0.75],
        [],
        [{"type": "simulation-control", "key": "wavelength_bandwidth_pct"}],
        [],
        state,
        {"type": "simulation-control", "key": "wavelength_bandwidth_pct"},
    )

    assert detector_state["detector-view"]["wavelength_bandwidth_pct"] == pytest.approx(1.25)
    assert fibrous_state["fibrous-view"]["wavelength_bandwidth_pct"] == pytest.approx(0.75)


def test_unified_special_cause_ewald_sample_callback_preserves_state_value():
    app = build_unified_app(initial_mode="special-cause-reciprocal")
    state = app.layout.children[0].data

    updated_state = _updated_mode_state(
        "special-cause-reciprocal",
        [13],
        [],
        [{"type": "simulation-control", "key": "ewald_shell_sample_count"}],
        [],
        state,
        {"type": "simulation-control", "key": "ewald_shell_sample_count"},
    )

    assert updated_state["special-cause-reciprocal"]["ewald_shell_sample_count"] == 13


def test_unified_special_cause_theta_i_callback_preserves_exact_typed_value():
    app = build_unified_app(initial_mode="special-cause-reciprocal")
    state = app.layout.children[0].data

    updated_state = _updated_mode_state(
        "special-cause-reciprocal",
        [],
        [7.13],
        [],
        [{"type": "simulation-hybrid-input", "key": "theta_i_deg"}],
        state,
        {"type": "simulation-hybrid-input", "key": "theta_i_deg"},
    )

    assert updated_state["special-cause-reciprocal"]["theta_i_deg"] == pytest.approx(7.13)


def test_unified_special_cause_center_bragg_callback_preserves_boolean_state():
    app = build_unified_app(initial_mode="special-cause-reciprocal")
    state = app.layout.children[0].data

    updated_state = _updated_mode_state(
        "special-cause-reciprocal",
        [["enabled"]],
        [],
        [{"type": "simulation-control", "key": "center_bragg_only"}],
        [],
        state,
        {"type": "simulation-control", "key": "center_bragg_only"},
    )

    assert updated_state["special-cause-reciprocal"]["center_bragg_only"] is True


def test_special_cause_matrix_figure_builds_fixed_angle_and_peak_grid_with_one_colorbar():
    camera = {
        "eye": {"x": 1.25, "y": 1.1, "z": 0.85},
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
    }

    fig = unified_app._build_special_cause_matrix_figure(
        {
            "sigma_deg": 0.8,
            "Gamma_deg": 5.0,
            "eta": 0.5,
            "wavelength_bandwidth_pct": 5.0,
            "ewald_shell_sample_count": 3,
            "center_bragg_only": True,
        },
        camera=camera,
    )
    layout_json = fig.to_plotly_json()["layout"]
    scene_names = sorted(
        [name for name in layout_json if name.startswith("scene")],
        key=lambda name: int(name[5:] or "1"),
    )
    annotations = layout_json["annotations"]
    annotation_text = " ".join(annotation["text"] for annotation in annotations)
    row_label_annotations = [
        annotation
        for annotation in annotations
        if str(annotation.get("text", "")).startswith("L = ")
    ]
    visible_colorbars = [
        trace
        for trace in fig.data
        if getattr(trace, "type", "") == "surface" and getattr(trace, "showscale", False)
    ]
    trace_names = [getattr(trace, "name", "") for trace in fig.data]
    trace_text = " ".join(
        str(text)
        for trace in fig.data
        for text in (
            getattr(trace, "text", [])
            if isinstance(getattr(trace, "text", []), (list, tuple))
            else [getattr(trace, "text", "")]
        )
    )

    assert scene_names == ["scene", "scene2", "scene3", "scene4", "scene5", "scene6", "scene7", "scene8", "scene9"]
    assert "θᵢ = 5°" in annotation_text
    assert "θᵢ = 10°" in annotation_text
    assert "θᵢ = 15°" in annotation_text
    assert [annotation["text"] for annotation in row_label_annotations] == ["L = 3", "L = 6", "L = 9"]
    assert all(annotation["xref"] == "paper" and annotation["yref"] == "paper" for annotation in row_label_annotations)
    assert all(annotation["x"] < 0 for annotation in row_label_annotations)
    assert all(annotation["textangle"] == -90 for annotation in row_label_annotations)
    assert "003" not in annotation_text
    assert "006" not in annotation_text
    assert "009" not in annotation_text
    assert len(visible_colorbars) == 1
    assert "Bragg sphere" in trace_names
    assert "Bragg/Ewald overlap" in trace_names
    assert "Bragg/Ewald overlap band" not in trace_names
    assert not any(name in {"Ewald shell inner", "Ewald shell outer", "Ewald sphere"} for name in trace_names)
    assert not any(getattr(trace, "type", "") == "cone" for trace in fig.data)
    assert "kᵢ" not in trace_text
    assert "θᵢ" not in trace_text
    for scene_name in scene_names:
        scene_traces = [
            trace
            for trace in fig.data
            if getattr(trace, "scene", "scene") == scene_name
        ]
        scene_trace_names = [getattr(trace, "name", "") for trace in scene_traces]
        assert "Bragg sphere" in scene_trace_names
        assert "Bragg/Ewald overlap" in scene_trace_names
    for scene_name in scene_names:
        assert layout_json[scene_name]["camera"] == camera
    shared_scene_ranges = []
    for scene_name in scene_names:
        scene_layout = layout_json[scene_name]
        assert scene_layout["aspectmode"] == "cube"
        scene_ranges = tuple(tuple(scene_layout[axis]["range"]) for axis in ("xaxis", "yaxis", "zaxis"))
        assert scene_ranges[0] == scene_ranges[1] == scene_ranges[2]
        shared_scene_ranges.append(scene_ranges)
    assert len(set(shared_scene_ranges)) == 1
    largest_visible_coordinate = max(
        float(np.max(np.abs(np.asarray(getattr(trace, coordinate_name), dtype=float))))
        for trace in fig.data
        for coordinate_name in ("x", "y", "z")
        if getattr(trace, coordinate_name, None) is not None
    )
    assert 0.85 <= largest_visible_coordinate / shared_scene_ranges[0][0][1] <= 0.95


def test_special_cause_matrix_figure_keeps_overlap_band_when_helpers_are_visible():
    fig = unified_app._build_special_cause_matrix_figure(
        {
            "sigma_deg": 0.8,
            "Gamma_deg": 5.0,
            "eta": 0.5,
            "wavelength_bandwidth_pct": 5.0,
            "ewald_shell_sample_count": 3,
            "center_bragg_only": False,
        },
    )
    trace_names = [getattr(trace, "name", "") for trace in fig.data]
    layout_json = fig.to_plotly_json()["layout"]

    assert "Bragg/Ewald overlap band" in trace_names
    assert "Ewald shell inner" in trace_names
    assert "Ewald shell outer" in trace_names
    for scene_name in ("scene2", "scene3", "scene5", "scene6", "scene8", "scene9"):
        scene_trace_positions = [
            (index, getattr(trace, "name", ""))
            for index, trace in enumerate(fig.data)
            if getattr(trace, "scene", "scene") == scene_name
        ]
        bragg_position = next(index for index, name in scene_trace_positions if name == "Bragg sphere")
        outline_position = next(index for index, name in scene_trace_positions if name == "Bragg sphere outline")
        assert outline_position > bragg_position
        helper_surface_positions = [
            index
            for index, name in scene_trace_positions
            if name in {"Ewald shell inner", "Ewald shell outer", "Bragg/Ewald overlap band"}
        ]
        assert helper_surface_positions
        assert max(helper_surface_positions) < bragg_position
        helper_surface_traces = [
            trace
            for trace in fig.data
            if getattr(trace, "scene", "scene") == scene_name
            and getattr(trace, "name", "") in {"Ewald shell inner", "Ewald shell outer", "Bragg/Ewald overlap band"}
        ]
        assert helper_surface_traces
        for trace in helper_surface_traces:
            assert trace.type == "scatter3d"
            assert trace.mode == "lines"
            assert len(trace.x) == len(trace.y) == len(trace.z)
            assert any(np.isfinite(np.asarray(trace.x, dtype=float)))
        bragg_trace = next(
            trace
            for trace in fig.data
            if getattr(trace, "scene", "scene") == scene_name
            and getattr(trace, "name", "") == "Bragg sphere"
        )
        assert bragg_trace.lighting.ambient == 1.0
        assert bragg_trace.lighting.diffuse == 0.0
        assert bragg_trace.colorscale[0][1] == "rgb(70,70,70)"
        bragg_outline = next(
            trace
            for trace in fig.data
            if getattr(trace, "scene", "scene") == scene_name
            and getattr(trace, "name", "") == "Bragg sphere outline"
        )
        assert bragg_outline.type == "scatter3d"
        assert bragg_outline.mode == "lines"
        assert any(np.isfinite(np.asarray(bragg_outline.x, dtype=float)))
        ewald_surfaces = [
            trace
            for trace in fig.data
            if getattr(trace, "scene", "scene") == scene_name
            and getattr(trace, "name", "") in {"Ewald shell inner", "Ewald shell outer"}
        ]
        assert ewald_surfaces
    for scene_name in ("scene", "scene4", "scene7"):
        helper_surface_opacities = [
            float(getattr(trace, "opacity", 1.0))
            for trace in fig.data
            if getattr(trace, "scene", "scene") == scene_name
            and getattr(trace, "name", "") in {"Ewald shell inner", "Ewald shell outer", "Bragg/Ewald overlap band"}
        ]
        assert max(helper_surface_opacities) > 0.05
        assert all(
            getattr(trace, "type", None) == "surface"
            for trace in fig.data
            if getattr(trace, "scene", "scene") == scene_name
            and getattr(trace, "name", "") in {"Ewald shell inner", "Ewald shell outer"}
        )
        bragg_trace = next(
            trace
            for trace in fig.data
            if getattr(trace, "scene", "scene") == scene_name
            and getattr(trace, "name", "") == "Bragg sphere"
        )
        assert bragg_trace.lighting.ambient is None
        assert bragg_trace.colorscale[0][1] != "rgb(70,70,70)"
        assert not any(
            getattr(trace, "name", "") == "Bragg sphere outline"
            for trace in fig.data
            if getattr(trace, "scene", "scene") == scene_name
        )
    assert "range" not in layout_json["scene"]["xaxis"]
    assert layout_json["scene"]["aspectmode"] == "data"


def test_unified_specular_sliders_only_keep_theta_i_live():
    app = build_unified_app(initial_mode="specular-view")

    basic_card = _find_component_by_id(app.layout, "simulation-specular-basic-card")
    detector_advanced = _find_component_by_id(app.layout, "simulation-specular-advanced-detector-geometry")

    theta_slider = _find_component_by_id(basic_card, {"type": "simulation-hybrid-slider", "key": "theta_i"})
    beta_slider = _find_component_by_id(
        detector_advanced,
        {"type": "simulation-hybrid-slider", "key": "beta"},
    )
    sigma_slider = _find_component_by_id(
        basic_card,
        {"type": "simulation-hybrid-slider", "key": "sigma_deg"},
    )

    assert theta_slider.updatemode == "drag"
    assert beta_slider.updatemode == "mouseup"
    assert sigma_slider.updatemode == "mouseup"


def test_unified_powder_view_exposes_picker_and_only_relevant_peak_ui():
    app = build_unified_app(initial_mode="reciprocal-space")

    powder_control = _find_component_by_class(app.layout, "simulation-powder-control")
    powder_empty = _find_component_by_class(app.layout, "simulation-powder-empty")
    view_picker = _find_component_by_id(app.layout, "powder-view-mode")

    assert powder_control is not None
    assert powder_empty is not None
    assert isinstance(view_picker, dcc.RadioItems)
    assert view_picker.id == "powder-view-mode"
    assert view_picker.value == "single-crystal"
    assert view_picker.labelClassName == "simulation-powder-option"
    assert view_picker.inputClassName == "simulation-powder-input"
    assert "Peak filters" in _render_component_text(_find_component_by_id(app.layout, "simulation-controls"))
    assert "Single crystal does not use grouped powder peak filters." in _render_component_text(
        _find_component_by_id(app.layout, "simulation-controls")
    )


def test_powder_view_picker_renders_only_active_peak_selector():
    app = build_unified_app(initial_mode="reciprocal-space")
    callback = _callback_by_output(app, "simulation-description.children")
    state = app.layout.children[0].data
    powder_selection_state = app.layout.children[1].data

    _, controls, _, _ = callback(
        "reciprocal-space",
        "2d-powder",
        state,
        powder_selection_state,
    )
    rendered_controls = html.Div(controls)
    peak_selector = _find_component_by_id(
        rendered_controls,
        {"type": "powder-qr-control", "key": "ring_selection"},
    )
    peak_card = _find_component_by_class(rendered_controls, "simulation-peak-selector-card")

    assert isinstance(_find_component_by_id(rendered_controls, "powder-view-mode"), dcc.RadioItems)
    assert peak_card is not None
    assert isinstance(peak_selector, dcc.Checklist)
    assert peak_selector.id["type"] == "powder-qr-control"
    assert peak_selector.id["key"] == "ring_selection"
    assert peak_selector.className == "simulation-peak-selector-list"


def test_unified_styles_define_navigation_and_responsive_workbench_hooks():
    css = (Path(__file__).resolve().parents[1] / "assets" / "specular_ui.css").read_text()
    mobile_css = css.split("@media (max-width: 760px)", maxsplit=1)[1]

    for selector in (
        ".simulation-mode-helper",
        ".simulation-mode-option",
        ".simulation-powder-control",
        ".simulation-peak-selector-card",
        ".simulation-toolbar-button:focus-visible",
        ".simulation-graph-frame--primary",
    ):
        assert selector in css

    assert ".simulation-sidebar--specular {\n    width: auto;" in css
    assert ".simulation-mode-picker {\n    grid-template-columns: minmax(0, 1fr);" in mobile_css
    assert ".simulation-mode-option {\n    box-sizing: border-box;\n    width: 100%;" in mobile_css


def test_unified_app_powder_png_export_restores_series_downloads():
    app = build_unified_app()
    scripts = "\n".join(app._inline_scripts)

    assert 'modeValue === "reciprocal-space"' in scripts
    assert 'mono_single_crystal.png' in scripts
    assert 'reciprocal_3d_powder.png' in scripts
    assert 'reciprocal_2d_powder.png' in scripts
    assert 'mono_cylinder.png' in scripts


def test_unified_app_mosaic_png_export_downloads_panel_files():
    app = build_unified_app(initial_mode="detector-view")
    scripts = "\n".join(app._inline_scripts)

    assert 'modeValue === "detector-view"' in scripts
    assert "mosaic_reciprocal_space.png" in scripts
    assert "mosaic_detector_view.png" in scripts
    assert "mosaic_centered_integration.png" in scripts
    assert "Preparing Mosaic View downloads..." in scripts


def test_unified_app_special_cause_matrix_export_button_is_mode_scoped():
    special_app = build_unified_app(initial_mode="special-cause-reciprocal")
    detector_app = build_unified_app(initial_mode="detector-view")

    special_button = _find_component_by_id(special_app.layout, "export-special-cause-matrix-button")
    detector_button = _find_component_by_id(detector_app.layout, "export-special-cause-matrix-button")
    special_matrix_store = _find_component_by_id(special_app.layout, "special-cause-matrix-export-figure")
    style_callback = _callback_by_output(special_app, "export-special-cause-matrix-button.style")

    assert special_button.children == "Save 3x3 Matrix"
    assert special_button.title == "Download a 3x3 special-cause comparison using the current camera"
    assert isinstance(special_matrix_store, dcc.Store)
    assert special_matrix_store.storage_type == "memory"
    assert special_button.style == {}
    assert detector_button.style.get("display") == "none"
    assert style_callback("special-cause-reciprocal") == {}
    assert style_callback("detector-view") == {"display": "none"}


def test_unified_app_special_cause_matrix_export_callback_uses_current_state_and_camera(monkeypatch):
    app = build_unified_app(initial_mode="special-cause-reciprocal")
    callback = _callback_by_output(app, "special-cause-matrix-export-figure.data")
    state = app.layout.children[0].data
    state["special-cause-reciprocal"] = dict(state["special-cause-reciprocal"])
    state["special-cause-reciprocal"]["ewald_shell_sample_count"] = 3
    state["special-cause-reciprocal"]["center_bragg_only"] = True
    camera = {
        "eye": {"x": 1.35, "y": 1.05, "z": 0.75},
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
    }
    captured = {}

    class FakeFigure:
        def to_plotly_json(self):
            return {
                "data": [{"name": "Bragg sphere"}],
                "layout": {"scene": {"camera": captured["camera"]}},
            }

    def fake_build_special_cause_matrix_figure(values, *, camera=None):
        captured["values"] = values
        captured["camera"] = camera
        return FakeFigure()

    monkeypatch.setattr(
        unified_app,
        "_build_special_cause_matrix_figure",
        fake_build_special_cause_matrix_figure,
    )

    figure_data = callback(
        1,
        "special-cause-reciprocal",
        state,
        {"special-cause-reciprocal": camera},
        None,
    )

    assert captured["values"]["ewald_shell_sample_count"] == 3
    assert captured["values"]["center_bragg_only"] is True
    assert captured["camera"] == camera
    assert figure_data["data"] == [{"name": "Bragg sphere"}]
    assert figure_data["layout"]["scene"]["camera"] == camera
    assert figure_data["export_request"] == 1


def test_unified_app_specular_png_export_hooks_new_mode():
    app = build_unified_app(initial_mode="specular-view")
    scripts = "\n".join(app._inline_scripts)

    assert '"specular-view": "specular_reflection"' in scripts


def test_png_export_callback_targets_rendered_plotly_figure_and_powder_exports():
    assert '.js-plotly-plot' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'Plotly.toImage' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'Plotly.downloadImage' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'Plotly.newPlot' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'sceneCameraBackup' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'key + ".camera"' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'modeValue === "reciprocal-space"' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'modeValue === "detector-view"' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'mono_single_crystal.png' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'reciprocal_3d_powder.png' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'reciprocal_2d_powder.png' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'mono_cylinder.png' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'mosaic_reciprocal_space.png' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'mosaic_detector_view.png' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'mosaic_centered_integration.png' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert '"special-cause-reciprocal": "special_cause_reciprocal"' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert '"specular-view": "specular_reflection"' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'trace?.scene === "scene"' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'trace?.xaxis === "x" && trace?.yaxis === "y"' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'trace?.xaxis === "x2" && trace?.yaxis === "y2"' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'Downloaded Single, 3D, 2D, and Cylinder views.' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'Downloaded reciprocal space, detector view, and centered integration.' in PNG_EXPORT_CLIENTSIDE_CALLBACK


def test_special_cause_matrix_export_clientside_callback_downloads_single_png():
    assert "Plotly.newPlot" in SPECIAL_CAUSE_MATRIX_EXPORT_CLIENTSIDE_CALLBACK
    assert "Plotly.downloadImage" in SPECIAL_CAUSE_MATRIX_EXPORT_CLIENTSIDE_CALLBACK
    assert "special_cause_reciprocal_matrix" in SPECIAL_CAUSE_MATRIX_EXPORT_CLIENTSIDE_CALLBACK
    assert "special-cause-matrix-export-host" in SPECIAL_CAUSE_MATRIX_EXPORT_CLIENTSIDE_CALLBACK
    assert "cleanupExistingMatrixExports" in SPECIAL_CAUSE_MATRIX_EXPORT_CLIENTSIDE_CALLBACK
    assert "document.querySelectorAll" in SPECIAL_CAUSE_MATRIX_EXPORT_CLIENTSIDE_CALLBACK
    assert "purgeAndRemoveHost" in SPECIAL_CAUSE_MATRIX_EXPORT_CLIENTSIDE_CALLBACK
    assert "Plotly.purge(exportHost)" in SPECIAL_CAUSE_MATRIX_EXPORT_CLIENTSIDE_CALLBACK
    assert "document.querySelectorAll(hostSelector).forEach(purgeAndRemoveHost)" in (
        SPECIAL_CAUSE_MATRIX_EXPORT_CLIENTSIDE_CALLBACK
    )
    assert "purgeAndRemoveHost(host)" in SPECIAL_CAUSE_MATRIX_EXPORT_CLIENTSIDE_CALLBACK
    assert SPECIAL_CAUSE_MATRIX_EXPORT_CLIENTSIDE_CALLBACK.index("cleanupExistingMatrixExports();") < (
        SPECIAL_CAUSE_MATRIX_EXPORT_CLIENTSIDE_CALLBACK.index('document.createElement("div")')
    )
    assert "currentMatrixExportRequest" in SPECIAL_CAUSE_MATRIX_EXPORT_CLIENTSIDE_CALLBACK
    assert "requestId !== currentMatrixExportRequest" in SPECIAL_CAUSE_MATRIX_EXPORT_CLIENTSIDE_CALLBACK
    assert "Downloaded special_cause_reciprocal_matrix.png." in SPECIAL_CAUSE_MATRIX_EXPORT_CLIENTSIDE_CALLBACK


def test_powder_qr_clientside_callback_updates_ring_and_cylinder_modes():
    assert 'modeValue !== "reciprocal-space"' in POWDER_QR_CLIENTSIDE_CALLBACK
    assert "powderViewValue" in POWDER_QR_CLIENTSIDE_CALLBACK
    assert "meta.g_group_indices" in POWDER_QR_CLIENTSIDE_CALLBACK
    assert "meta.g_ring_groups" in POWDER_QR_CLIENTSIDE_CALLBACK
    assert "meta.g_cylinder_group_indices" in POWDER_QR_CLIENTSIDE_CALLBACK
    assert 'menus[0].visible = false' in POWDER_QR_CLIENTSIDE_CALLBACK
    assert '"single-crystal"' in POWDER_QR_CLIENTSIDE_CALLBACK
    assert '"3d-powder"' in POWDER_QR_CLIENTSIDE_CALLBACK
    assert '"2d-powder"' in POWDER_QR_CLIENTSIDE_CALLBACK
    assert '"cylinder"' in POWDER_QR_CLIENTSIDE_CALLBACK
    assert "buttons[1]" in POWDER_QR_CLIENTSIDE_CALLBACK
    assert "buttons[2]" in POWDER_QR_CLIENTSIDE_CALLBACK
    assert "buttons[3]" in POWDER_QR_CLIENTSIDE_CALLBACK
    assert "arraysEqual(currentVis, targetVis)" in POWDER_QR_CLIENTSIDE_CALLBACK
    assert "relayoutFigure();" in POWDER_QR_CLIENTSIDE_CALLBACK
    assert '"scene.camera"' in POWDER_QR_CLIENTSIDE_CALLBACK
    assert "powder-selection-state" not in POWDER_QR_CLIENTSIDE_CALLBACK


def test_unified_reciprocal_space_callback_preserves_camera_from_relayout_data():
    app = build_unified_app(initial_mode="reciprocal-space")
    callback = _main_figure_callback(app)
    state = app.layout.children[0].data
    graph = _graph(app)

    fig = callback(
        "reciprocal-space",
        state,
        {},
        {
            "scene.camera": {
                "eye": {"x": 1.6, "y": 1.1, "z": 0.9},
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
                "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            }
        },
        graph.figure,
    )

    assert fig.layout.scene.camera.eye.x == pytest.approx(1.6)
    assert fig.layout.scene.camera.eye.y == pytest.approx(1.1)
    assert fig.layout.scene.camera.eye.z == pytest.approx(0.9)


def test_unified_detector_callback_preserves_camera_from_relayout_data():
    app = build_unified_app(initial_mode="detector-view")
    callback = _main_figure_callback(app)
    state = app.layout.children[0].data
    graph = _graph(app)

    fig = callback(
        "detector-view",
        state,
        {},
        {
            "scene.camera": {
                "eye": {"x": 1.8, "y": 1.0, "z": 0.6},
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
                "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            }
        },
        graph.figure,
    )

    assert fig.layout.scene.camera.eye.x == pytest.approx(1.8)
    assert fig.layout.scene.camera.eye.y == pytest.approx(1.0)
    assert fig.layout.scene.camera.eye.z == pytest.approx(0.6)


def test_unified_fibrous_callback_preserves_camera_from_relayout_data():
    app = build_unified_app(initial_mode="fibrous-view")
    callback = _main_figure_callback(app)
    state = app.layout.children[0].data
    graph = _graph(app)

    fig = callback(
        "fibrous-view",
        state,
        {},
        {
            "scene.camera": {
                "eye": {"x": 1.7, "y": 0.9, "z": 0.7},
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
                "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            }
        },
        graph.figure,
    )

    assert fig.layout.scene.camera.eye.x == pytest.approx(1.7)
    assert fig.layout.scene.camera.eye.y == pytest.approx(0.9)
    assert fig.layout.scene.camera.eye.z == pytest.approx(0.7)


def test_unified_specular_callback_preserves_camera_from_relayout_data():
    app = build_unified_app(initial_mode="specular-view")
    main_callback = _main_figure_callback(app)
    companion_callback = _companion_figure_callback(app)
    state = app.layout.children[0].data
    graph = _graph(app)
    companion_graph = _specular_companion_graph(app)

    fig = main_callback(
        "specular-view",
        state,
        {},
        {
            "scene.camera": {
                "eye": {"x": 1.55, "y": 1.2, "z": 0.75},
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
                "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            }
        },
        graph.figure,
    )
    companion_fig = companion_callback(
        "specular-view",
        state,
        {},
        {
            "scene.camera": {
                "eye": {"x": 1.2, "y": 0.9, "z": 0.7},
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
                "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            }
        },
        companion_graph.figure,
    )

    assert fig.layout.scene.camera.eye.x == pytest.approx(1.55)
    assert fig.layout.scene.camera.eye.y == pytest.approx(1.2)
    assert fig.layout.scene.camera.eye.z == pytest.approx(0.75)
    assert companion_fig.layout.scene.camera.eye.x == pytest.approx(1.2)
    assert companion_fig.layout.scene.camera.eye.y == pytest.approx(0.9)
    assert companion_fig.layout.scene.camera.eye.z == pytest.approx(0.7)
    assert "HKL = (1, 1, 1)" in companion_fig.layout.title.text
    assert "HKL = (1, 1, 1)" in fig.layout.meta["simulation_summary"]
    assert "diffraction rays" in fig.layout.meta["simulation_summary"]
    assert "nominal 2θ" not in fig.layout.meta["simulation_summary"]


def test_unified_specular_companion_callback_skips_main_only_changes():
    app = build_unified_app(initial_mode="specular-view")
    callback = _companion_figure_callback(app)
    state = app.layout.children[0].data
    companion_graph = _specular_companion_graph(app)

    updated_state = dict(state)
    updated_state["specular-view"] = dict(state["specular-view"])
    updated_state["specular-view"]["beta"] = 4.0

    with pytest.raises(PreventUpdate):
        callback(
            "specular-view",
            updated_state,
            {},
            None,
            companion_graph.figure,
        )


def test_unified_summary_callback_no_longer_reads_figure_payload():
    app = build_unified_app(initial_mode="specular-view")
    callback_entry = next(
        value
        for key, value in app.callback_map.items()
        if "simulation-summary.children" in key
    )
    inputs = {(item["id"], item["property"]) for item in callback_entry["inputs"]}

    assert ("simulation-mode", "value") in inputs
    assert ("simulation-state", "data") in inputs
    assert ("simulation-figure", "figure") not in inputs


def test_powder_selection_callback_caches_values():
    app = build_unified_app(initial_mode="reciprocal-space")
    callback = app.callback_map["powder-selection-state.data"]["callback"].__wrapped__

    selection_state = callback(
        [[2], [1, 3], [1]],
        [
            {"type": "powder-qr-control", "key": "sphere_selection"},
            {"type": "powder-qr-control", "key": "ring_selection"},
            {"type": "powder-qr-control", "key": "cylinder_selection"},
        ],
        {},
    )

    assert selection_state["sphere_selection"] == [2]
    assert selection_state["ring_selection"] == [1, 3]
    assert selection_state["cylinder_selection"] == [1]


def test_updated_mode_state_skips_no_op_control_update():
    app = build_unified_app(initial_mode="reciprocal-space")
    state = app.layout.children[0].data
    controls = SIMULATION_SPECS["reciprocal-space"].controls
    control_ids = [
        {"type": "simulation-control", "key": control.key}
        for control in controls
        if control.component == "dropdown" or control.component == "slider" or control.component == "range" or control.component == "number"
    ]
    hybrid_ids: list[dict[str, str]] = []
    values = [state["reciprocal-space"][control_id["key"]] for control_id in control_ids]

    updated_state = _updated_mode_state(
        "reciprocal-space",
        values,
        [],
        control_ids,
        hybrid_ids,
        state,
        {"type": "simulation-control", "key": "theta_i_min_deg"},
    )

    assert updated_state is None


def test_updated_mode_state_returns_new_state_for_real_control_change():
    app = build_unified_app(initial_mode="reciprocal-space")
    state = app.layout.children[0].data
    controls = SIMULATION_SPECS["reciprocal-space"].controls
    control_ids = [
        {"type": "simulation-control", "key": control.key}
        for control in controls
        if control.component == "dropdown" or control.component == "slider" or control.component == "range" or control.component == "number"
    ]
    hybrid_ids: list[dict[str, str]] = []
    values = [state["reciprocal-space"][control_id["key"]] for control_id in control_ids]
    values[0] = 5.0

    updated_state = _updated_mode_state(
        "reciprocal-space",
        values,
        [],
        control_ids,
        hybrid_ids,
        state,
        {"type": "simulation-control", "key": "theta_i_min_deg"},
    )

    assert updated_state is not None
    assert updated_state["reciprocal-space"]["theta_i_min_deg"] == pytest.approx(5.0)


def test_updated_powder_view_skips_no_op_update():
    assert _updated_powder_view("single-crystal", "single-crystal") is None


def test_updated_powder_selection_state_skips_no_op_update():
    app = build_unified_app(initial_mode="reciprocal-space")
    state = app.layout.children[1].data

    updated_state = _updated_powder_selection_state(
        [
            state["sphere_selection"],
            state["ring_selection"],
            state["cylinder_selection"],
        ],
        [
            {"type": "powder-qr-control", "key": "sphere_selection"},
            {"type": "powder-qr-control", "key": "ring_selection"},
            {"type": "powder-qr-control", "key": "cylinder_selection"},
        ],
        state,
    )

    assert updated_state is None


def test_unified_camera_state_callback_caches_per_mode():
    app = build_unified_app(initial_mode="fibrous-view")
    callback = app.callback_map["simulation-camera-state.data"]["callback"].__wrapped__

    camera_state = callback(
        {
            "scene.camera.eye.x": 1.3,
            "scene.camera.eye.y": 1.0,
            "scene.camera.eye.z": 0.8,
            "scene.camera.up.x": 0.0,
            "scene.camera.up.y": 0.0,
            "scene.camera.up.z": 1.0,
        },
        None,
        "fibrous-view",
        {},
    )

    assert camera_state["fibrous-view"]["eye"]["x"] == pytest.approx(1.3)
    assert camera_state["fibrous-view"]["eye"]["y"] == pytest.approx(1.0)
    assert camera_state["fibrous-view"]["eye"]["z"] == pytest.approx(0.8)


def test_unified_camera_state_callback_caches_specular_mode():
    app = build_unified_app(initial_mode="specular-view")
    callback = app.callback_map["simulation-camera-state.data"]["callback"].__wrapped__

    camera_state = callback(
        {
            "scene.camera.eye.x": 1.45,
            "scene.camera.eye.y": 1.05,
            "scene.camera.eye.z": 0.85,
            "scene.camera.up.x": 0.0,
            "scene.camera.up.y": 0.0,
            "scene.camera.up.z": 1.0,
        },
        {
            "scene.camera.eye.x": 1.2,
            "scene.camera.eye.y": 0.95,
            "scene.camera.eye.z": 0.65,
            "scene.camera.up.x": 0.0,
            "scene.camera.up.y": 0.0,
            "scene.camera.up.z": 1.0,
        },
        "specular-view",
        {},
    )

    assert camera_state["specular-view"]["eye"]["x"] == pytest.approx(1.45)
    assert camera_state["specular-view"]["eye"]["y"] == pytest.approx(1.05)
    assert camera_state["specular-view"]["eye"]["z"] == pytest.approx(0.85)
    assert camera_state["specular-view:companion"]["eye"]["x"] == pytest.approx(1.2)
    assert camera_state["specular-view:companion"]["eye"]["y"] == pytest.approx(0.95)
    assert camera_state["specular-view:companion"]["eye"]["z"] == pytest.approx(0.65)


def test_unified_camera_state_callback_skips_no_op_update():
    app = build_unified_app(initial_mode="fibrous-view")
    callback = app.callback_map["simulation-camera-state.data"]["callback"].__wrapped__

    with pytest.raises(PreventUpdate):
        callback(
            {
                "scene.camera.eye.x": 1.3,
                "scene.camera.eye.y": 1.0,
                "scene.camera.eye.z": 0.8,
                "scene.camera.up.x": 0.0,
                "scene.camera.up.y": 0.0,
                "scene.camera.up.z": 1.0,
            },
            None,
            "fibrous-view",
            {
                "fibrous-view": {
                    "eye": {"x": 1.3, "y": 1.0, "z": 0.8},
                    "up": {"x": 0.0, "y": 0.0, "z": 1.0},
                }
            },
        )


def test_unified_main_uses_cli_args_when_called_without_parameters(monkeypatch):
    recorded: dict[str, object] = {}

    class StubApp:
        def run(self, *, debug, host, port):
            recorded["debug"] = debug
            recorded["host"] = host
            recorded["port"] = port

    class StubTimer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("browser timer should not run when --no-browser is set")

    def stub_build_unified_app(initial_mode, *, initial_state=None):
        recorded["initial_mode"] = initial_mode
        recorded["initial_state"] = initial_state
        return StubApp()

    monkeypatch.setattr(
        unified_app,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "mode": "specular-view",
                "host": "127.0.0.1",
                "port": 8067,
                "no_browser": True,
                "state_json": '{"specular-view":{"H":4,"wavelength_m":1.2e-10}}',
            },
        )(),
    )
    monkeypatch.setattr(
        unified_app,
        "build_unified_app",
        stub_build_unified_app,
    )
    monkeypatch.setattr(unified_app.threading, "Timer", StubTimer)

    unified_app.main()

    assert recorded["initial_mode"] == "specular-view"
    assert recorded["initial_state"] == {
        "specular-view": {"H": 4, "wavelength_m": 1.2e-10}
    }
    assert recorded["debug"] is False
    assert recorded["host"] == "127.0.0.1"
    assert recorded["port"] == 8067


def test_unified_main_uses_explicit_parameters_without_parsing_cli(monkeypatch):
    recorded: dict[str, object] = {}

    class StubApp:
        def run(self, *, debug, host, port):
            recorded["debug"] = debug
            recorded["host"] = host
            recorded["port"] = port

    def stub_build_unified_app(initial_mode, *, initial_state=None):
        recorded["initial_mode"] = initial_mode
        recorded["initial_state"] = initial_state
        return StubApp()

    monkeypatch.setattr(
        unified_app,
        "parse_args",
        lambda: (_ for _ in ()).throw(AssertionError("parse_args should not be used")),
    )
    monkeypatch.setattr(
        unified_app,
        "build_unified_app",
        stub_build_unified_app,
    )
    monkeypatch.setattr(
        unified_app.threading,
        "Timer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("browser timer should not run")),
    )

    unified_app.main(
        "specular-view",
        host="0.0.0.0",
        port=9001,
        open_browser=False,
        initial_state={"specular-view": {"H": 2, "lattice_a_m": 5.5e-10}},
    )

    assert recorded["initial_mode"] == "specular-view"
    assert recorded["initial_state"] == {
        "specular-view": {"H": 2, "lattice_a_m": 5.5e-10}
    }
    assert recorded["debug"] is False
    assert recorded["host"] == "0.0.0.0"
    assert recorded["port"] == 9001
