from dash import dcc

from mosaic_sim.unified import build_unified_app, build_unified_figure


def _shell(app):
    return app.layout.children[-1]


def _sidebar(app):
    return _shell(app).children[0]


def _main(app):
    return _shell(app).children[1]


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


def _render_component_text(component) -> str:
    if component is None:
        return ""
    if isinstance(component, (str, int, float)):
        return str(component)
    if isinstance(component, (list, tuple)):
        return "".join(_render_component_text(child) for child in component)
    return _render_component_text(getattr(component, "children", None))


def _graph(app):
    return _find_component_by_id(_main(app), "simulation-figure")


def test_build_unified_figure_uses_detector_mode():
    fig = build_unified_figure(mode="detector-view")

    assert "HKL = (0, 0, 12)" in fig.layout.title.text
    assert len(fig.frames) == 0


def test_build_unified_figure_uses_detector_mode_with_wavelength_bandwidth():
    fig = build_unified_figure(mode="detector-view", wavelength_bandwidth_pct=1.0)

    assert "λ bandwidth = 1.00%" in fig.layout.title.text


def test_build_unified_figure_accepts_100_percent_wavelength_bandwidth():
    fig = build_unified_figure(mode="special-cause-reciprocal", wavelength_bandwidth_pct=100.0)

    assert "λ bandwidth = 100.00%" in fig.layout.title.text
    assert any(trace.name == "Ewald shell inner" for trace in fig.data)
    assert any(trace.name == "Ewald shell outer" for trace in fig.data)


def test_build_unified_figure_uses_special_cause_reciprocal_mode():
    fig = build_unified_figure(
        mode="special-cause-reciprocal",
        H=0,
        K=0,
        L=12,
        sigma_deg=0.8,
        Gamma_deg=5.0,
        eta=0.5,
        wavelength_bandwidth_pct=1.0,
    )

    assert "HKL = (0, 0, 12)" in fig.layout.title.text
    assert "λ bandwidth = 1.00%" in fig.layout.title.text
    assert len(fig.frames) == 0
    assert any(trace.name == "Bragg sphere" for trace in fig.data)
    assert any(trace.name == "Ewald shell inner" for trace in fig.data)
    assert any(trace.name == "Ewald shell outer" for trace in fig.data)
    assert any(trace.name == "Bragg/Ewald overlap band" for trace in fig.data)


def test_build_unified_figure_passes_special_cause_center_bragg_option():
    fig = build_unified_figure(mode="special-cause-reciprocal", center_bragg_only=True)

    assert any(trace.name == "Bragg sphere" for trace in fig.data)
    assert any(trace.name == "Bragg/Ewald overlap band" for trace in fig.data)
    assert any(trace.name == "Bragg/Ewald overlap" for trace in fig.data)
    assert not any(trace.name == "Ewald sphere" for trace in fig.data)
    assert not any(trace.name == "Ewald shell inner" for trace in fig.data)
    assert not any(trace.name == "Ewald shell outer" for trace in fig.data)
    assert not any(trace.type == "cone" for trace in fig.data)


def test_build_unified_figure_uses_special_cause_ewald_shell_sample_count():
    fig = build_unified_figure(
        mode="special-cause-reciprocal",
        wavelength_bandwidth_pct=5.0,
        ewald_shell_sample_count=13,
    )

    overlap_traces = [trace for trace in fig.data if trace.name == "Bragg/Ewald overlap"]

    assert len(overlap_traces) == 13


def test_build_unified_figure_rejects_invalid_special_cause_ewald_shell_sample_count():
    fig = build_unified_figure(
        mode="special-cause-reciprocal",
        wavelength_bandwidth_pct=5.0,
        ewald_shell_sample_count=4,
    )

    assert "ewald_shell_sample_count" in fig.layout.annotations[0].text


def test_build_unified_figure_uses_special_cause_reciprocal_defaults():
    fig = build_unified_figure(mode="special-cause-reciprocal")

    assert "HKL = (0, 0, 3)" in fig.layout.title.text
    assert "λ bandwidth = 5.00%" in fig.layout.title.text


def test_build_unified_figure_detector_invalid_wavelength_bandwidth_returns_message():
    fig = build_unified_figure(mode="detector-view", wavelength_bandwidth_pct=-1.0)

    assert "wavelength_bandwidth_pct" in fig.layout.annotations[0].text


def test_build_unified_figure_uses_fibrous_mode():
    fig = build_unified_figure(mode="fibrous-view")

    assert "HKL = (0, 0, 12)" in fig.layout.title.text
    assert len(fig.frames) == 60


def test_build_unified_figure_uses_fibrous_mode_with_wavelength_bandwidth():
    fig = build_unified_figure(mode="fibrous-view", wavelength_bandwidth_pct=1.0)

    assert "λ bandwidth = 1.00%" in fig.layout.title.text


def test_build_unified_figure_fibrous_invalid_wavelength_bandwidth_returns_message():
    fig = build_unified_figure(mode="fibrous-view", wavelength_bandwidth_pct=-1.0)

    assert "wavelength_bandwidth_pct" in fig.layout.annotations[0].text


def test_build_unified_figure_uses_specular_mode():
    fig = build_unified_figure(mode="specular-view")

    assert "HKL = (1, 1, 1)" in fig.layout.title.text
    assert "diffraction rays" in fig.layout.title.text
    assert "nominal 2θ" not in fig.layout.title.text
    assert len(fig.frames) == 0


def test_build_unified_app_seeds_initial_mode_and_figure():
    app = build_unified_app(initial_mode="fibrous-view")

    graph_panel = _main(app)
    mode_selector = _find_component_by_id(app.layout, "simulation-mode")
    controls = _find_component_by_id(app.layout, "simulation-controls")
    export_toolbar = graph_panel.children[0]
    graph = _graph(app)

    assert mode_selector.value == "fibrous-view"
    assert _find_component_by_id(controls, {"type": "simulation-control", "key": "wavelength_bandwidth_pct"}) is not None
    assert export_toolbar.children[0].id == "export-png-button"
    assert isinstance(graph, dcc.Graph)
    assert "HKL = (0, 0, 12)" in graph.figure.layout.title.text


def test_build_unified_app_seeds_specular_mode_and_summary():
    app = build_unified_app(initial_mode="specular-view")

    mode_selector = _find_component_by_id(app.layout, "simulation-mode")
    graph = _graph(app)
    summary = _find_component_by_id(app.layout, "simulation-summary")

    assert mode_selector.value == "specular-view"
    assert isinstance(graph, dcc.Graph)
    assert "HKL = (1, 1, 1)" in graph.figure.layout.title.text
    assert "diffraction rays" in graph.figure.layout.title.text
    summary_text = _render_component_text(summary)
    assert "Specular Diffraction" in summary_text
    assert "Diffracted intensity" in summary_text
    assert "nominal 2θ" not in summary_text
    assert summary.style.get("display") != "none"
