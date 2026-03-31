from dash import dcc
import pytest

from mosaic_sim.unified_app import (
    PNG_EXPORT_CLIENTSIDE_CALLBACK,
    POWDER_QR_CLIENTSIDE_CALLBACK,
    SIMULATION_SPECS,
    build_unified_app,
    build_unified_figure,
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

    assert "Specular Beam-Sample-Detector Simulation" in fig.layout.title.text
    assert fig.layout.meta["simulation_summary"].startswith("Specular reflection summary")


def test_build_unified_app_seeds_mode_selector_and_initial_figure():
    app = build_unified_app(initial_mode="fibrous-view")

    shell = app.layout.children[-1]
    sidebar = shell.children[0]
    main = shell.children[1]
    mode_dropdown = sidebar.children[2].children[1]
    export_toolbar = main.children[0]
    graph = main.children[1]

    assert mode_dropdown.value == "fibrous-view"
    assert export_toolbar.children[0].id == "export-png-button"
    assert isinstance(graph, dcc.Graph)
    assert "HKL = (0, 0, 12)" in graph.figure.layout.title.text


def test_build_unified_app_seeds_specular_mode_selector_and_initial_figure():
    app = build_unified_app(initial_mode="specular-view")

    shell = app.layout.children[-1]
    sidebar = shell.children[0]
    main = shell.children[1]
    mode_dropdown = sidebar.children[2].children[1]
    graph = main.children[1]
    summary = main.children[2]
    control_sections = sidebar.children[4].children

    assert mode_dropdown.value == "specular-view"
    assert isinstance(graph, dcc.Graph)
    assert "Specular Beam-Sample-Detector Simulation" in graph.figure.layout.title.text
    assert "Specular reflection summary" in summary.children
    assert summary.style.get("display") != "none"
    assert control_sections[0].children[0].children == "Beam"
    assert control_sections[1].children[0].children == "Sample"
    assert control_sections[2].children[0].children == "Detector"


def test_unified_detector_theta_slider_updates_continuously():
    app = build_unified_app(initial_mode="detector-view")

    shell = app.layout.children[-1]
    sidebar = shell.children[0]
    controls = sidebar.children[4].children
    theta_control = controls[0].children[1]

    assert isinstance(theta_control, dcc.Slider)
    assert theta_control.updatemode == "drag"


def test_unified_detector_mosaic_kernel_controls_use_slider_input_pairs():
    app = build_unified_app(initial_mode="detector-view")

    shell = app.layout.children[-1]
    sidebar = shell.children[0]
    controls = sidebar.children[4].children

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


def test_unified_powder_view_exposes_picker_and_only_relevant_peak_ui():
    app = build_unified_app(initial_mode="reciprocal-space")

    shell = app.layout.children[-1]
    sidebar = shell.children[0]
    controls = sidebar.children[4].children
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
    assert 'Plotly.relayout(graph, {"scene.camera": camera})' in POWDER_QR_CLIENTSIDE_CALLBACK
    assert "powder-selection-state" not in POWDER_QR_CLIENTSIDE_CALLBACK


def test_unified_reciprocal_space_callback_preserves_camera_from_relayout_data():
    app = build_unified_app(initial_mode="reciprocal-space")
    callback = app.callback_map["simulation-figure.figure"]["callback"].__wrapped__
    state = app.layout.children[0].data

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
    )

    assert fig.layout.scene.camera.eye.x == pytest.approx(1.6)
    assert fig.layout.scene.camera.eye.y == pytest.approx(1.1)
    assert fig.layout.scene.camera.eye.z == pytest.approx(0.9)


def test_unified_detector_callback_preserves_camera_from_relayout_data():
    app = build_unified_app(initial_mode="detector-view")
    callback = app.callback_map["simulation-figure.figure"]["callback"].__wrapped__
    state = app.layout.children[0].data

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
    )

    assert fig.layout.scene.camera.eye.x == pytest.approx(1.8)
    assert fig.layout.scene.camera.eye.y == pytest.approx(1.0)
    assert fig.layout.scene.camera.eye.z == pytest.approx(0.6)


def test_unified_fibrous_callback_preserves_camera_from_relayout_data():
    app = build_unified_app(initial_mode="fibrous-view")
    callback = app.callback_map["simulation-figure.figure"]["callback"].__wrapped__
    state = app.layout.children[0].data

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
    )

    assert fig.layout.scene.camera.eye.x == pytest.approx(1.7)
    assert fig.layout.scene.camera.eye.y == pytest.approx(0.9)
    assert fig.layout.scene.camera.eye.z == pytest.approx(0.7)


def test_unified_specular_callback_preserves_camera_from_relayout_data():
    app = build_unified_app(initial_mode="specular-view")
    callback = app.callback_map["simulation-figure.figure"]["callback"].__wrapped__
    state = app.layout.children[0].data

    fig = callback(
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
    )

    assert fig.layout.scene.camera.eye.x == pytest.approx(1.55)
    assert fig.layout.scene.camera.eye.y == pytest.approx(1.2)
    assert fig.layout.scene.camera.eye.z == pytest.approx(0.75)
    assert fig.layout.meta["simulation_summary"].startswith("Specular reflection summary")


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
        "specular-view",
        {},
    )

    assert camera_state["specular-view"]["eye"]["x"] == pytest.approx(1.45)
    assert camera_state["specular-view"]["eye"]["y"] == pytest.approx(1.05)
    assert camera_state["specular-view"]["eye"]["z"] == pytest.approx(0.85)
