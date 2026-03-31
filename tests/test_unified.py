from dash import dcc

from mosaic_sim.unified import build_unified_app, build_unified_figure


def test_build_unified_figure_uses_detector_mode():
    fig = build_unified_figure(mode="detector-view")

    assert "HKL = (0, 0, 12)" in fig.layout.title.text
    assert len(fig.frames) == 0


def test_build_unified_figure_uses_fibrous_mode():
    fig = build_unified_figure(mode="fibrous-view")

    assert "HKL = (0, 0, 12)" in fig.layout.title.text
    assert len(fig.frames) == 60


def test_build_unified_figure_uses_specular_mode():
    fig = build_unified_figure(mode="specular-view")

    assert "Specular Beam-Sample-Detector Simulation" in fig.layout.title.text
    assert len(fig.frames) == 0


def test_build_unified_app_seeds_initial_mode_and_figure():
    app = build_unified_app(initial_mode="fibrous-view")

    shell = app.layout.children[-1]
    sidebar = shell.children[0]
    graph_panel = shell.children[1]
    mode_selector = sidebar.children[2].children[1]
    controls = sidebar.children[4]
    export_toolbar = graph_panel.children[0]
    graph = graph_panel.children[1]

    assert mode_selector.value == "fibrous-view"
    assert len(controls.children) == 6
    assert export_toolbar.children[0].id == "export-png-button"
    assert isinstance(graph, dcc.Graph)
    assert "HKL = (0, 0, 12)" in graph.figure.layout.title.text


def test_build_unified_app_seeds_specular_mode_and_summary():
    app = build_unified_app(initial_mode="specular-view")

    shell = app.layout.children[-1]
    sidebar = shell.children[0]
    graph_panel = shell.children[1]
    mode_selector = sidebar.children[2].children[1]
    graph = graph_panel.children[1]
    summary = graph_panel.children[2]

    assert mode_selector.value == "specular-view"
    assert isinstance(graph, dcc.Graph)
    assert "Specular Beam-Sample-Detector Simulation" in graph.figure.layout.title.text
    assert "Specular reflection summary" in summary.children
    assert summary.style.get("display") != "none"
