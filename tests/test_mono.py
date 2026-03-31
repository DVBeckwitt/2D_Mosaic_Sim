from mosaic_sim.mono import (
    MONO_CAMERA_UIREVISION,
    build_interactive_page,
    build_mono_figure,
)


def test_build_mono_figure_sets_uirevision_for_camera_persistence():
    fig, _ = build_mono_figure(n_frames=2, render_profile="balanced")

    assert fig.layout.uirevision == MONO_CAMERA_UIREVISION
    assert fig.layout.scene.uirevision == MONO_CAMERA_UIREVISION


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
