from mosaic_sim.mono import (
    MONO_CAMERA_UIREVISION,
    build_interactive_page,
    build_mono_figure,
)


def _assert_flat_surface_lighting(trace) -> None:
    assert trace.lighting.ambient == 1.0
    assert trace.lighting.diffuse == 0.0
    assert trace.lighting.specular == 0.0
    assert trace.lighting.roughness == 1.0
    assert trace.lighting.fresnel == 0.0


def test_build_mono_figure_sets_uirevision_for_camera_persistence():
    fig, _ = build_mono_figure(n_frames=2, render_profile="balanced")

    assert fig.layout.uirevision == MONO_CAMERA_UIREVISION
    assert fig.layout.scene.uirevision == MONO_CAMERA_UIREVISION


def test_build_mono_figure_flattens_ewald_and_g_sphere_lighting():
    fig, _ = build_mono_figure(n_frames=2, render_profile="balanced")

    ewald_trace = next(trace for trace in fig.data if trace.name == "Ewald sphere")
    g_sphere_traces = [
        trace for trace in fig.data if str(getattr(trace, "name", "")).startswith("|G| =")
    ]

    _assert_flat_surface_lighting(ewald_trace)
    _assert_flat_surface_lighting(fig.frames[0].data[0])
    assert g_sphere_traces
    for trace in g_sphere_traces:
        _assert_flat_surface_lighting(trace)


def test_build_interactive_page_exposes_button_disable_toggle():
    fig, context = build_mono_figure(n_frames=2, render_profile="balanced")

    html = build_interactive_page(fig, context)

    assert 'id="toggle-controls"' in html
    assert "Disable buttons" in html
    assert "function setControlsDisabled" in html
    assert "selectorBtn.disabled = controlsDisabled;" in html
    assert "toggleControlsBtn.textContent = controlsDisabled ? 'Enable buttons' : 'Disable buttons';" in html
    assert "function cloneSceneCamera" in html
    assert "function updateFigure(dataUpdate, layoutUpdate = {})" in html
    assert "function relayoutFigure(layoutUpdate = {})" in html
    assert "await relayoutFigure({ updatemenus: menus });" in html
    assert "return { ...layoutUpdate, 'scene.camera': camera };" in html
