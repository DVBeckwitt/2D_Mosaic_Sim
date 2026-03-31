from dash import dcc
from dash._utils import to_json
from dash.exceptions import PreventUpdate
import pytest

import mosaic_sim.unified_app as unified_app
from mosaic_sim.unified_app import (
    PNG_EXPORT_CLIENTSIDE_CALLBACK,
    POWDER_QR_CLIENTSIDE_CALLBACK,
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


def _sidebar(app):
    return _shell(app).children[0]


def _main(app):
    return _shell(app).children[1]


def _graph(app):
    return _main(app).children[1].children[0]


def _specular_companion_card(app):
    return _main(app).children[2]


def _specular_companion_graph(app):
    return _specular_companion_card(app).children[1].children[0]


def _summary(app):
    return _sidebar(app).children[4]


def _controls(app):
    return _sidebar(app).children[5].children


def _figure_callback(app):
    return next(
        value["callback"].__wrapped__
        for key, value in app.callback_map.items()
        if "simulation-figure.figure" in key
        and "simulation-specular-companion-figure.figure" in key
    )


def test_unified_registry_exposes_all_simulations():
    assert set(SIMULATION_SPECS) == {
        "reciprocal-space",
        "detector-view",
        "fibrous-view",
        "specular-view",
    }


def test_build_unified_figure_uses_selected_mode():
    fig = build_unified_figure("detector-view", H=0, K=0, L=12, sigma_deg=0.8, Gamma_deg=5.0, eta=0.5)

    assert "HKL = (0, 0, 12)" in fig.layout.title.text


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

    sidebar = _sidebar(app)
    main = _main(app)
    mode_dropdown = sidebar.children[2].children[1]
    export_toolbar = main.children[0]
    graph = _graph(app)

    assert mode_dropdown.value == "fibrous-view"
    assert export_toolbar.children[0].id == "export-png-button"
    assert isinstance(graph, dcc.Graph)
    assert "HKL = (0, 0, 12)" in graph.figure.layout.title.text


def test_build_unified_app_seeds_specular_mode_selector_and_initial_figure():
    app = build_unified_app(initial_mode="specular-view")

    sidebar = _sidebar(app)
    mode_dropdown = sidebar.children[2].children[1]
    graph = _graph(app)
    companion_card = _specular_companion_card(app)
    companion_graph = _specular_companion_graph(app)
    summary = _summary(app)
    control_sections = _controls(app)

    assert mode_dropdown.value == "specular-view"
    assert isinstance(graph, dcc.Graph)
    assert isinstance(companion_graph, dcc.Graph)
    assert "HKL = (1, 1, 1)" in graph.figure.layout.title.text
    assert "diffraction rays" in graph.figure.layout.title.text
    assert "HKL = (1, 1, 1)" in companion_graph.figure.layout.title.text
    assert "HKL = (1, 1, 1)" in summary.children
    assert "diffraction rays" in summary.children
    assert "nominal 2θ" not in summary.children
    assert summary.style.get("display") != "none"
    assert companion_card.style.get("display") != "none"
    assert control_sections[0].children[0].children == "Beam"
    assert control_sections[1].children[0].children == "Sample"
    assert control_sections[2].children[0].children == "Detector"
    assert control_sections[3].children[0].children == "Diffraction"
    diffraction_controls = control_sections[3].children[1].children
    assert len(diffraction_controls) == 6
    assert diffraction_controls[0].children[1].children[0].children.id == {
        "type": "simulation-hybrid-slider",
        "key": "H",
    }
    assert diffraction_controls[3].children[1].children[0].children.id == {
        "type": "simulation-hybrid-slider",
        "key": "sigma_deg",
    }
    assert diffraction_controls[4].children[1].children[0].children.id == {
        "type": "simulation-hybrid-slider",
        "key": "mosaic_gamma_deg",
    }
    assert diffraction_controls[5].children[1].children[0].children.id == {
        "type": "simulation-hybrid-slider",
        "key": "eta",
    }


def test_unified_non_specular_modes_keep_summary_hidden():
    app = build_unified_app(initial_mode="fibrous-view")
    summary = _summary(app)
    companion_card = _specular_companion_card(app)

    assert summary.style.get("display") == "none"
    assert companion_card.style.get("display") == "none"


def test_unified_specular_mode_restores_all_parameter_sections_and_controls():
    app = build_unified_app(initial_mode="specular-view")

    control_sections = _controls(app)

    expected_sections = (
        (
            "Beam",
            [
                "rays",
                "seed",
                "display_rays",
                "source_y",
                "beam_width_x",
                "beam_width_z",
                "divergence_x",
                "divergence_z",
                "z_beam",
            ],
        ),
        (
            "Sample",
            [
                "sample_width",
                "sample_height",
                "theta_i",
                "delta",
                "alpha",
                "psi",
                "z_sample",
            ],
        ),
        (
            "Detector",
            [
                "distance",
                "detector_width",
                "detector_height",
                "beta",
                "gamma",
                "chi",
                "pixel_u",
                "pixel_v",
                "i0",
                "j0",
            ],
        ),
        (
            "Diffraction",
            [
                "H",
                "K",
                "L",
                "sigma_deg",
                "mosaic_gamma_deg",
                "eta",
            ],
        ),
    )

    assert len(control_sections) == len(expected_sections)
    for section, (title, keys) in zip(control_sections, expected_sections, strict=True):
        assert section.children[0].children == title
        assert _specular_section_keys(section) == keys
        assert _specular_section_input_keys(section) == keys


def test_unified_specular_layout_serializes_for_dash():
    app = build_unified_app(initial_mode="specular-view")

    serialized = to_json(app.layout)

    assert '"simulation-mode"' in serialized
    assert '"specular-view"' in serialized
    assert '"Diffraction"' in serialized
    assert '"simulation-specular-companion-figure"' in serialized


def test_unified_detector_theta_slider_updates_continuously():
    app = build_unified_app(initial_mode="detector-view")

    controls = _controls(app)
    theta_control = controls[0].children[1]

    assert isinstance(theta_control, dcc.Slider)
    assert theta_control.updatemode == "drag"


def test_unified_detector_mosaic_kernel_controls_use_slider_input_pairs():
    app = build_unified_app(initial_mode="detector-view")

    controls = _controls(app)

    sigma_control = controls[4].children[1]
    Gamma_control = controls[5].children[1]
    eta_control = controls[6].children[1]

    for key, control in (
        ("sigma_deg", sigma_control),
        ("Gamma_deg", Gamma_control),
        ("eta", eta_control),
    ):
        slider = control.children[0].children
        number = control.children[1]

        assert isinstance(slider, dcc.Slider)
        assert isinstance(number, dcc.Input)
        assert slider.id["type"] == "simulation-hybrid-slider"
        assert slider.id["key"] == key
        assert number.id["type"] == "simulation-hybrid-input"
        assert number.id["key"] == key
        assert number.type == "number"


def test_unified_specular_sliders_only_keep_theta_i_live():
    app = build_unified_app(initial_mode="specular-view")

    sections = _controls(app)
    sample_controls = sections[1].children[1].children
    detector_controls = sections[2].children[1].children
    diffraction_controls = sections[3].children[1].children

    theta_slider = sample_controls[2].children[1].children[0].children
    beta_slider = detector_controls[3].children[1].children[0].children
    sigma_slider = diffraction_controls[3].children[1].children[0].children

    assert theta_slider.updatemode == "drag"
    assert beta_slider.updatemode == "mouseup"
    assert sigma_slider.updatemode == "mouseup"


def test_unified_powder_view_exposes_picker_and_only_relevant_peak_ui():
    app = build_unified_app(initial_mode="reciprocal-space")

    controls = _controls(app)
    view_picker = controls[-2].children[1]
    peak_card = controls[-1]

    assert isinstance(view_picker, dcc.RadioItems)
    assert view_picker.id == "powder-view-mode"
    assert view_picker.value == "single-crystal"
    assert peak_card.children[0].children == "Peak filters"
    assert "Single crystal does not use grouped powder peak filters." in peak_card.children[1].children


def test_powder_view_picker_renders_only_active_peak_selector():
    app = build_unified_app(initial_mode="reciprocal-space")
    callback = app.callback_map["..simulation-description.children...simulation-controls.children.."]["callback"].__wrapped__
    state = app.layout.children[0].data
    powder_selection_state = app.layout.children[1].data

    _, controls = callback(
        "reciprocal-space",
        "2d-powder",
        state,
        powder_selection_state,
    )
    peak_selector = controls[-1].children[1]

    assert isinstance(controls[-2].children[1], dcc.RadioItems)
    assert isinstance(peak_selector, dcc.Checklist)
    assert peak_selector.id["type"] == "powder-qr-control"
    assert peak_selector.id["key"] == "ring_selection"


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
    assert '"specular-view": "specular_reflection"' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'trace?.scene === "scene"' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'trace?.xaxis === "x" && trace?.yaxis === "y"' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'trace?.xaxis === "x2" && trace?.yaxis === "y2"' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'Downloaded Single, 3D, 2D, and Cylinder views.' in PNG_EXPORT_CLIENTSIDE_CALLBACK
    assert 'Downloaded reciprocal space, detector view, and centered integration.' in PNG_EXPORT_CLIENTSIDE_CALLBACK


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
    callback = _figure_callback(app)
    state = app.layout.children[0].data

    fig, companion_fig = callback(
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
        None,
    )

    assert fig.layout.scene.camera.eye.x == pytest.approx(1.6)
    assert fig.layout.scene.camera.eye.y == pytest.approx(1.1)
    assert fig.layout.scene.camera.eye.z == pytest.approx(0.9)
    assert len(companion_fig.data) == 0


def test_unified_detector_callback_preserves_camera_from_relayout_data():
    app = build_unified_app(initial_mode="detector-view")
    callback = _figure_callback(app)
    state = app.layout.children[0].data

    fig, companion_fig = callback(
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
        None,
    )

    assert fig.layout.scene.camera.eye.x == pytest.approx(1.8)
    assert fig.layout.scene.camera.eye.y == pytest.approx(1.0)
    assert fig.layout.scene.camera.eye.z == pytest.approx(0.6)
    assert len(companion_fig.data) == 0


def test_unified_fibrous_callback_preserves_camera_from_relayout_data():
    app = build_unified_app(initial_mode="fibrous-view")
    callback = _figure_callback(app)
    state = app.layout.children[0].data

    fig, companion_fig = callback(
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
        None,
    )

    assert fig.layout.scene.camera.eye.x == pytest.approx(1.7)
    assert fig.layout.scene.camera.eye.y == pytest.approx(0.9)
    assert fig.layout.scene.camera.eye.z == pytest.approx(0.7)
    assert len(companion_fig.data) == 0


def test_unified_specular_callback_preserves_camera_from_relayout_data():
    app = build_unified_app(initial_mode="specular-view")
    callback = _figure_callback(app)
    state = app.layout.children[0].data

    fig, companion_fig = callback(
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
        {
            "scene.camera": {
                "eye": {"x": 1.2, "y": 0.9, "z": 0.7},
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
                "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            }
        },
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

    def stub_build_unified_app(initial_mode):
        recorded["initial_mode"] = initial_mode
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

    def stub_build_unified_app(initial_mode):
        recorded["initial_mode"] = initial_mode
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
    )

    assert recorded["initial_mode"] == "specular-view"
    assert recorded["debug"] is False
    assert recorded["host"] == "0.0.0.0"
    assert recorded["port"] == 9001
