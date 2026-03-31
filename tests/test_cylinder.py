import math

import numpy as np
import pytest
from dash import dcc

from mosaic_sim.cylinder import (
    CYLINDER_CAMERA_UIREVISION,
    build_cylinder_app,
    build_cylinder_error_figure,
    build_cylinder_figure,
    normalize_cylinder_params,
)


def test_normalize_cylinder_params_uses_defaults_and_converts_units():
    params = normalize_cylinder_params(
        defaults=(1, 2, 3, 0.9, 4.1, 0.25),
    )

    assert params[:3] == (1, 2, 3)
    assert math.isclose(params[3], math.radians(0.9))
    assert math.isclose(params[4], math.radians(4.1))
    assert math.isclose(params[5], 0.25)


def test_normalize_cylinder_params_rejects_invalid_eta():
    with pytest.raises(ValueError, match="η"):
        normalize_cylinder_params(0, 0, 12, 0.8, 5.0, 1.2)


def test_build_cylinder_figure_exposes_parameterized_slider_state():
    fig = build_cylinder_figure(
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
    assert fig.layout.uirevision == CYLINDER_CAMERA_UIREVISION
    assert fig.layout.scene.uirevision == CYLINDER_CAMERA_UIREVISION


def test_build_cylinder_figure_applies_explicit_camera():
    camera = {
        "eye": {"x": 1.6, "y": 1.0, "z": 0.7},
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
    }

    fig = build_cylinder_figure(
        H=0,
        K=0,
        L=12,
        sigma=np.deg2rad(0.8),
        Gamma=np.deg2rad(5.0),
        eta=0.5,
        camera=camera,
    )

    assert fig.layout.scene.camera.eye.x == pytest.approx(1.6)
    assert fig.layout.scene.camera.eye.y == pytest.approx(1.0)
    assert fig.layout.scene.camera.eye.z == pytest.approx(0.7)


def test_build_cylinder_figure_uses_single_toggle_buttons_without_legend_overlap():
    fig = build_cylinder_figure(
        H=0,
        K=0,
        L=12,
        sigma=np.deg2rad(0.8),
        Gamma=np.deg2rad(5.0),
        eta=0.5,
    )

    expected_labels = [
        "Bragg sphere",
        "Ewald sphere",
        "Cylinder",
        "Bragg/Ewald overlap",
        "Cylinder/Ewald overlap",
        "Cylinder/Bragg overlap",
        "Disable all",
    ]
    menus = fig.layout.updatemenus

    assert len(menus) == len(expected_labels)
    assert [menu.buttons[0].label for menu in menus] == expected_labels
    assert all(len(menu.buttons) == 1 for menu in menus)
    assert all(menu.active == -1 for menu in menus)
    assert all(menu.showactive is True for menu in menus)
    assert all(menu.y > 1.0 for menu in menus[:-1])
    assert 0.9 < menus[-1].y < 1.0
    assert all(menu.buttons[0].args[0]["visible"] == "legendonly" for menu in menus)
    assert all(menu.buttons[0].args2[0]["visible"] is True for menu in menus)
    assert len({(menu.x, menu.y) for menu in menus}) == len(menus)
    assert all(menu.x < fig.layout.legend.x for menu in menus)
    assert fig.layout.margin.r >= 200
    assert fig.layout.legend.x > fig.layout.scene.domain.x[1]
    assert fig.layout.sliders[0].len <= fig.layout.scene.domain.x[1]
    assert list(menus[-1].buttons[0].args[1]) == [0, 1, 3, 2, 4, 5]
    assert list(menus[-1].buttons[0].args2[1]) == [0, 1, 3, 2, 4, 5]


def test_build_cylinder_app_seeds_inputs_and_figure_from_initial_values():
    app = build_cylinder_app(
        initial_H=1,
        initial_K=2,
        initial_L=7,
        initial_sigma=np.deg2rad(1.1),
        initial_Gamma=np.deg2rad(6.2),
        initial_eta=0.35,
    )

    controls = app.layout.children[1].children
    graph = app.layout.children[2]

    assert controls[0].children[1].value == 1
    assert controls[1].children[1].value == 2
    assert controls[2].children[1].value == 7
    assert controls[3].children[0].children == "σ (deg)"
    assert controls[4].children[0].children == "Γ (deg)"
    assert controls[5].children[0].children == "η"
    assert math.isclose(controls[3].children[1].value, 1.1)
    assert math.isclose(controls[4].children[1].value, 6.2)
    assert math.isclose(controls[5].children[1].value, 0.35)
    assert isinstance(graph, dcc.Graph)
    assert "HKL = (1, 2, 7)" in graph.figure.layout.title.text


def test_build_cylinder_app_callback_preserves_camera_from_relayout_data():
    app = build_cylinder_app()
    callback = app.callback_map["cylinder-fig.figure"]["callback"].__wrapped__

    fig = callback(
        0,
        0,
        12,
        0.8,
        5.0,
        0.5,
        {
            "scene.camera.eye.x": 1.4,
            "scene.camera.eye.y": 1.1,
            "scene.camera.eye.z": 0.8,
            "scene.camera.up.x": 0.0,
            "scene.camera.up.y": 0.0,
            "scene.camera.up.z": 1.0,
            "scene.camera.center.x": 0.0,
            "scene.camera.center.y": 0.0,
            "scene.camera.center.z": 0.0,
        },
    )

    assert fig.layout.scene.camera.eye.x == pytest.approx(1.4)
    assert fig.layout.scene.camera.eye.y == pytest.approx(1.1)
    assert fig.layout.scene.camera.eye.z == pytest.approx(0.8)


def test_build_cylinder_error_figure_surfaces_validation_message():
    fig = build_cylinder_error_figure("η must be between 0 and 1 inclusive")

    assert fig.layout.annotations[0].text == "η must be between 0 and 1 inclusive"
