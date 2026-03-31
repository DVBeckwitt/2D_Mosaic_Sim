from dash import dcc

from mosaic_sim.unified import build_unified_app, build_unified_figure


def _shell(app):
    return app.layout.children[-1]


def _sidebar(app):
    return _shell(app).children[0]


def _main(app):
    return _shell(app).children[1]


def _graph(app):
    return _main(app).children[1].children[0]


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

    assert "HKL = (1, 1, 1)" in fig.layout.title.text
    assert "diffraction rays" in fig.layout.title.text
    assert "nominal 2θ" not in fig.layout.title.text
    assert len(fig.frames) == 0


def test_build_unified_app_seeds_initial_mode_and_figure():
    app = build_unified_app(initial_mode="fibrous-view")

    sidebar = _sidebar(app)
    graph_panel = _main(app)
    mode_selector = sidebar.children[2].children[1]
    controls = sidebar.children[5]
    export_toolbar = graph_panel.children[0]
    graph = _graph(app)

    assert mode_selector.value == "fibrous-view"
    assert len(controls.children) == 6
    assert export_toolbar.children[0].id == "export-png-button"
    assert isinstance(graph, dcc.Graph)
    assert "HKL = (0, 0, 12)" in graph.figure.layout.title.text


def test_build_unified_app_seeds_specular_mode_and_summary():
    app = build_unified_app(initial_mode="specular-view")

    sidebar = _sidebar(app)
    mode_selector = sidebar.children[2].children[1]
    graph = _graph(app)
    summary = sidebar.children[4]

    assert mode_selector.value == "specular-view"
    assert isinstance(graph, dcc.Graph)
    assert "HKL = (1, 1, 1)" in graph.figure.layout.title.text
    assert "diffraction rays" in graph.figure.layout.title.text
    assert "HKL = (1, 1, 1)" in summary.children
    assert "diffraction rays" in summary.children
    assert "nominal 2θ" not in summary.children
    assert summary.style.get("display") != "none"
