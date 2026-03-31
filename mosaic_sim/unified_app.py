"""Unified Dash application for the supported simulation views."""

from __future__ import annotations

import argparse
from functools import lru_cache
import math
from pathlib import Path
import threading
import webbrowser
from dataclasses import dataclass
from typing import Any, Callable

import plotly.graph_objects as go
from dash import ALL, MATCH, Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate

from specular_reflection_sim import (
    BeamConfig as SpecularBeamConfig,
    CONTROL_SECTIONS as SPECULAR_CONTROL_SECTIONS,
    DiffractionConfig as SpecularDiffractionConfig,
    DetectorConfig as SpecularDetectorConfig,
    MathLabel as SpecularMathLabel,
    SampleConfig as SpecularSampleConfig,
    build_specular_companion_error_figure,
    build_specular_dashboard_outputs as build_specular_dashboard_views,
    build_specular_error_figure,
    configs_from_values as specular_configs_from_values,
)

from .cylinder import build_cylinder_figure, normalize_cylinder_params
from .detector import (
    DEFAULT_THETA_DEG,
    THETA_MAX_DEG,
    THETA_MIN_DEG,
    build_detector_figure,
    extract_scene_camera,
    normalize_detector_params,
)
from .mono import N_FRAMES_DEFAULT, build_mono_figure

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_MODE",
    "SIMULATION_SPECS",
    "build_unified_figure",
    "build_unified_app",
    "main",
]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8052
DEFAULT_MODE = "reciprocal-space"
DEFAULT_POWDER_VIEW = "single-crystal"
SPECULAR_MODE = "specular-view"
POWDER_SPHERE_SELECTION_KEY = "sphere_selection"
POWDER_RING_SELECTION_KEY = "ring_selection"
POWDER_CYLINDER_SELECTION_KEY = "cylinder_selection"
SCENE_CAMERA_MODES = frozenset(
    {
        DEFAULT_MODE,
        "detector-view",
        "fibrous-view",
        SPECULAR_MODE,
    }
)
PNG_EXPORT_CLIENTSIDE_CALLBACK = """
        function(nClicks, modeValue) {
            if (!nClicks) {
                return "";
            }

            const graphContainer = document.getElementById("simulation-figure");
            const graph = graphContainer?.querySelector?.(".js-plotly-plot") || graphContainer;
            const statusNode = document.getElementById("export-png-status");
            if (!graph || !graph.data || !window.Plotly) {
                return "PNG export failed.";
            }

            const filenames = {
                "reciprocal-space": "powder_views",
                "detector-view": "mosaic_view",
                "fibrous-view": "ewald_cylinder",
                "specular-view": "specular_reflection",
            };
            const exportWidth = 1800;
            const exportHeight = 1800;
            const sliderBackup = graph.layout && graph.layout.sliders
                ? JSON.parse(JSON.stringify(graph.layout.sliders))
                : [];
            const menuBackup = graph.layout && graph.layout.updatemenus
                ? JSON.parse(JSON.stringify(graph.layout.updatemenus))
                : [];
            const sceneCameraBackup = Object.fromEntries(
                Object.entries(graph.layout || {})
                    .filter(([key, value]) => /^scene\\d*$/.test(key) && value && value.camera)
                    .map(([key, value]) => [
                        key + ".camera",
                        JSON.parse(JSON.stringify(value.camera)),
                    ])
            );

            const setStatus = (message) => {
                if (statusNode) {
                    statusNode.textContent = message;
                }
                return message;
            };

            const cloneValue = (value) => JSON.parse(JSON.stringify(value));

            const restoreControls = async () => {
                await Plotly.relayout(graph, {
                    sliders: sliderBackup,
                    updatemenus: menuBackup,
                    ...sceneCameraBackup,
                });
            };

            const hideControls = async () => {
                if (!sliderBackup.length && !menuBackup.length) {
                    if (Object.keys(sceneCameraBackup).length) {
                        await Plotly.relayout(graph, sceneCameraBackup);
                    }
                    return;
                }
                await Plotly.relayout(graph, {
                    sliders: [],
                    updatemenus: [],
                    ...sceneCameraBackup,
                });
            };

            const createHiddenExportDiv = () => {
                const host = document.createElement("div");
                host.style.position = "fixed";
                host.style.left = "-10000px";
                host.style.top = "0";
                host.style.width = exportWidth + "px";
                host.style.height = exportHeight + "px";
                host.style.background = "white";
                host.style.pointerEvents = "none";
                document.body.appendChild(host);
                return host;
            };

            const cleanupHiddenExportDiv = (host) => {
                if (!host) {
                    return;
                }
                try {
                    Plotly.purge(host);
                } catch (purgeErr) {
                    console.error(purgeErr);
                }
                host.remove();
            };

            const buildPanelLayout = (titleText) => ({
                template: graph.layout?.template ? cloneValue(graph.layout.template) : undefined,
                paper_bgcolor: graph.layout?.paper_bgcolor || "white",
                plot_bgcolor: graph.layout?.plot_bgcolor || "white",
                margin: {l: 80, r: 60, b: 80, t: 120},
                title: {
                    text: titleText,
                    x: 0.5,
                    xanchor: "center",
                },
                showlegend: false,
            });

            const downloadStandaloneFigure = async (figureSpec) => {
                const host = createHiddenExportDiv();
                try {
                    await Plotly.newPlot(
                        host,
                        figureSpec.data,
                        figureSpec.layout,
                        {
                            displaylogo: false,
                            displayModeBar: false,
                            responsive: false,
                            staticPlot: true,
                        }
                    );
                    await Plotly.downloadImage(host, {
                        format: "png",
                        filename: figureSpec.filename,
                        width: exportWidth,
                        height: exportHeight,
                    });
                } finally {
                    cleanupHiddenExportDiv(host);
                }
            };

            const downloadSingleView = async (filename) => {
                try {
                    await hideControls();
                    await Plotly.downloadImage(graph, {
                        format: "png",
                        filename: filename,
                        width: exportWidth,
                        height: exportHeight,
                    });
                } catch (err) {
                    console.error(err);
                } finally {
                    try {
                        await restoreControls();
                    } catch (restoreErr) {
                        console.error(restoreErr);
                    }
                }
            };

            const downloadPowderViews = async (modes) => {
                const originalVis = (graph.data || []).map((trace) => {
                    if (trace.visible === undefined) {
                        return true;
                    }
                    return trace.visible;
                });

                setStatus("Preparing downloads...");

                try {
                    await hideControls();
                    for (const mode of modes) {
                        const vis = Array.from(mode.vis, (value) => !!value);
                        await Plotly.update(graph, {visible: vis});
                        const uri = await Plotly.toImage(graph, {
                            format: "png",
                            width: exportWidth,
                            height: exportHeight,
                        });
                        const link = document.createElement("a");
                        link.href = uri;
                        link.download = mode.filename;
                        document.body.appendChild(link);
                        link.click();
                        link.remove();
                    }
                    setStatus("Downloaded Single, 3D, 2D, and Cylinder views.");
                } catch (err) {
                    console.error(err);
                    setStatus("Download failed. Please retry.");
                } finally {
                    try {
                        await Plotly.update(graph, {visible: originalVis});
                    } catch (restoreVisErr) {
                        console.error(restoreVisErr);
                    }
                    try {
                        await restoreControls();
                    } catch (restoreControlsErr) {
                        console.error(restoreControlsErr);
                    }
                }
            };

            const downloadMosaicPanels = async () => {
                const detectorData = graph.data || [];
                const sceneCamera = graph.layout?.scene?.camera
                    ? cloneValue(graph.layout.scene.camera)
                    : null;
                const reciprocalData = detectorData
                    .filter((trace) => trace?.scene === "scene")
                    .map((trace) => cloneValue(trace));
                const detectorViewData = detectorData
                    .filter((trace) => trace?.xaxis === "x" && trace?.yaxis === "y")
                    .map((trace) => {
                        const clonedTrace = cloneValue(trace);
                        delete clonedTrace.xaxis;
                        delete clonedTrace.yaxis;
                        return clonedTrace;
                    });
                const centeredData = detectorData
                    .filter((trace) => trace?.xaxis === "x2" && trace?.yaxis === "y2")
                    .map((trace) => {
                        const clonedTrace = cloneValue(trace);
                        clonedTrace.xaxis = "x";
                        clonedTrace.yaxis = "y";
                        return clonedTrace;
                    });

                if (!reciprocalData.length || !detectorViewData.length || !centeredData.length) {
                    setStatus("Download failed. Please retry.");
                    return;
                }

                const reciprocalLayout = {
                    ...buildPanelLayout("Reciprocal space"),
                    scene: graph.layout?.scene ? cloneValue(graph.layout.scene) : {},
                    uirevision: graph.layout?.uirevision,
                };
                if (sceneCamera) {
                    reciprocalLayout.scene.camera = sceneCamera;
                }

                const detectorLayout = buildPanelLayout("Detector view");
                detectorLayout.xaxis = graph.layout?.xaxis ? cloneValue(graph.layout.xaxis) : {};
                detectorLayout.yaxis = graph.layout?.yaxis ? cloneValue(graph.layout.yaxis) : {};
                delete detectorLayout.xaxis.domain;
                delete detectorLayout.yaxis.domain;

                const centeredLayout = buildPanelLayout("Centered integration");
                centeredLayout.xaxis = graph.layout?.xaxis2 ? cloneValue(graph.layout.xaxis2) : {};
                centeredLayout.yaxis = graph.layout?.yaxis2 ? cloneValue(graph.layout.yaxis2) : {};
                delete centeredLayout.xaxis.domain;
                delete centeredLayout.yaxis.domain;

                const figures = [
                    {
                        filename: "mosaic_reciprocal_space.png",
                        data: reciprocalData,
                        layout: reciprocalLayout,
                    },
                    {
                        filename: "mosaic_detector_view.png",
                        data: detectorViewData,
                        layout: detectorLayout,
                    },
                    {
                        filename: "mosaic_centered_integration.png",
                        data: centeredData,
                        layout: centeredLayout,
                    },
                ];

                setStatus("Preparing Mosaic View downloads...");

                try {
                    for (const figureSpec of figures) {
                        await downloadStandaloneFigure(figureSpec);
                    }
                    setStatus("Downloaded reciprocal space, detector view, and centered integration.");
                } catch (err) {
                    console.error(err);
                    setStatus("Download failed. Please retry.");
                }
            };

            if (modeValue === "reciprocal-space") {
                const buttons = graph.layout && graph.layout.updatemenus && graph.layout.updatemenus.length
                    ? graph.layout.updatemenus[0].buttons || []
                    : [];
                const powderModes = [
                    {label: "Single crystal", filename: "mono_single_crystal.png"},
                    {label: "3D Powder", filename: "reciprocal_3d_powder.png"},
                    {label: "2D Powder", filename: "reciprocal_2d_powder.png"},
                    {label: "Cylinder", filename: "mono_cylinder.png"},
                ].map((mode, index) => ({
                    ...mode,
                    vis: buttons[index] && buttons[index].args && buttons[index].args.length
                        ? buttons[index].args[0].visible
                        : null,
                }));

                if (powderModes.every((mode) => Array.isArray(mode.vis))) {
                    downloadPowderViews(powderModes);
                    return "Saving mono_single_crystal.png, reciprocal_3d_powder.png, reciprocal_2d_powder.png, and mono_cylinder.png...";
                }
            }

            if (modeValue === "detector-view") {
                downloadMosaicPanels();
                return "Saving mosaic_reciprocal_space.png, mosaic_detector_view.png, and mosaic_centered_integration.png...";
            }

            const filename = filenames[modeValue] || "simulation";
            downloadSingleView(filename);
            return "Saving " + filename + ".png...";
        }
        """

POWDER_QR_CLIENTSIDE_CALLBACK = """
        function(modeValue, figureValue, selectionState, powderViewValue) {
            if (modeValue !== "reciprocal-space") {
                return window.dash_clientside.no_update;
            }

            const graphContainer = document.getElementById("simulation-figure");
            const graph = graphContainer?.querySelector?.(".js-plotly-plot") || graphContainer;
            if (!graph || !graph.data || !graph.layout || !window.Plotly) {
                return window.dash_clientside.no_update;
            }

            const menus = graph.layout.updatemenus
                ? JSON.parse(JSON.stringify(graph.layout.updatemenus))
                : [];
            if (!menus.length || !menus[0].buttons || menus[0].buttons.length < 4) {
                return window.dash_clientside.no_update;
            }

            const meta = graph.layout.meta || {};
            const sphereGroups = Array.isArray(meta.g_group_indices) ? meta.g_group_indices : [];
            const ringGroups = Array.isArray(meta.g_ring_groups) ? meta.g_ring_groups : [];
            const cylinderGroups = Array.isArray(meta.g_cylinder_group_indices)
                ? meta.g_cylinder_group_indices
                : [];

            const oldSingleVis = menus[0].buttons[0]?.args?.[0]?.visible;
            const oldSphereVis = menus[0].buttons[1]?.args?.[0]?.visible;
            const oldRingVis = menus[0].buttons[2]?.args?.[0]?.visible;
            const oldCylinderVis = menus[0].buttons[3]?.args?.[0]?.visible;
            if (!Array.isArray(oldSingleVis) || !Array.isArray(oldSphereVis) || !Array.isArray(oldRingVis) || !Array.isArray(oldCylinderVis)) {
                return window.dash_clientside.no_update;
            }

            const cloneCamera = () => {
                const camera = graph.layout?.scene?.camera;
                return camera ? JSON.parse(JSON.stringify(camera)) : null;
            };

            const normalizeSelection = (rawValue, count, defaultValue) => {
                const value = rawValue === null || rawValue === undefined ? defaultValue : rawValue;
                const items = Array.isArray(value) ? value : [value];
                const seen = new Set();
                const normalized = [];
                items.forEach((item) => {
                    const index = Number.parseInt(item, 10);
                    if (Number.isInteger(index) && index >= 0 && index < count && !seen.has(index)) {
                        seen.add(index);
                        normalized.push(index);
                    }
                });
                return normalized;
            };

            const setGroupVisibility = (visibleArray, groups, selected) => {
                const selectedIndices = new Set(selected);
                groups.forEach((group, index) => {
                    const enabled = selectedIndices.has(index);
                    group.forEach((traceIndex) => {
                        visibleArray[traceIndex] = enabled;
                    });
                });
            };

            const arraysEqual = (left, right) =>
                Array.isArray(left) &&
                Array.isArray(right) &&
                left.length === right.length &&
                left.every((value, index) => !!value === !!right[index]);

            const currentVis = (graph.data || []).map((trace) =>
                trace && trace.visible === undefined ? true : !!trace.visible
            );

            const sphereSelection = normalizeSelection(
                selectionState?.sphere_selection,
                sphereGroups.length,
                sphereGroups.length ? [0] : []
            );
            const ringSelection = normalizeSelection(
                selectionState?.ring_selection,
                ringGroups.length,
                ringGroups.map((_, index) => index)
            );
            const cylinderSelection = normalizeSelection(
                selectionState?.cylinder_selection,
                cylinderGroups.length,
                cylinderGroups.length ? [0] : []
            );
            const resolvedPowderView = [
                "single-crystal",
                "3d-powder",
                "2d-powder",
                "cylinder",
            ].includes(powderViewValue)
                ? powderViewValue
                : "single-crystal";

            const nextSphereVis = Array.from(oldSphereVis, (value) => !!value);
            const nextRingVis = Array.from(oldRingVis, (value) => !!value);
            const nextCylinderVis = Array.from(oldCylinderVis, (value) => !!value);
            setGroupVisibility(nextSphereVis, sphereGroups, sphereSelection);
            setGroupVisibility(nextRingVis, ringGroups, ringSelection);
            setGroupVisibility(nextCylinderVis, cylinderGroups, cylinderSelection);

            menus[0].visible = false;
            menus[0].active = {
                "single-crystal": 0,
                "3d-powder": 1,
                "2d-powder": 2,
                "cylinder": 3,
            }[resolvedPowderView] ?? 0;
            menus[0].buttons[1].args[0].visible = nextSphereVis;
            menus[0].buttons[2].args[0].visible = nextRingVis;
            menus[0].buttons[3].args[0].visible = nextCylinderVis;

            const targetVis = {
                "single-crystal": Array.from(oldSingleVis, (value) => !!value),
                "3d-powder": nextSphereVis,
                "2d-powder": nextRingVis,
                "cylinder": nextCylinderVis,
            }[resolvedPowderView];

            const layoutUpdate = { updatemenus: menus };
            const camera = cloneCamera();
            if (camera) {
                layoutUpdate["scene.camera"] = camera;
            }

            const updateFigure = (visibleArray) =>
                Plotly.update(graph, {visible: visibleArray}, layoutUpdate)
                    .catch((error) => console.error(error));

            const relayoutFigure = () =>
                Plotly.relayout(graph, layoutUpdate)
                    .catch((error) => console.error(error));

            if (arraysEqual(currentVis, targetVis)) {
                relayoutFigure();
            } else {
                updateFigure(targetVis);
            }

            return {
                active_view: resolvedPowderView,
                sphere_selection: sphereSelection,
                ring_selection: ringSelection,
                cylinder_selection: cylinderSelection,
                synced_at: Date.now(),
            };
        }
        """


@dataclass(frozen=True)
class ControlSpec:
    key: str
    label: Any
    component: str
    default: Any
    step: float | int | None = None
    min: float | None = None
    max: float | None = None
    options: tuple[tuple[str, Any], ...] = ()
    input_step: float | int | None = None
    updatemode: str | None = None


@dataclass(frozen=True)
class SimulationSpec:
    key: str
    label: str
    description: str
    controls: tuple[ControlSpec, ...]
    build_figure: Callable[[dict[str, Any]], go.Figure]


def _message_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=18, color="crimson"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        title=title,
        margin=dict(l=40, r=40, b=40, t=80),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def _hkl_controls(*, hybrid_mosaic_params: bool = False) -> tuple[ControlSpec, ...]:
    base_controls = (
        ControlSpec("H", "H", "number", 0, step=1),
        ControlSpec("K", "K", "number", 0, step=1),
        ControlSpec("L", "L", "number", 12, step=1),
    )
    if hybrid_mosaic_params:
        return base_controls + (
            ControlSpec(
                "sigma_deg",
                "σ (deg)",
                "slider_input",
                0.8,
                step=0.05,
                min=0.05,
                max=10.0,
                input_step=0.05,
            ),
            ControlSpec(
                "Gamma_deg",
                "Γ (deg)",
                "slider_input",
                5.0,
                step=0.1,
                min=0.1,
                max=30.0,
                input_step=0.1,
            ),
            ControlSpec(
                "eta",
                "η",
                "slider_input",
                0.5,
                step=0.01,
                min=0.0,
                max=1.0,
                input_step=0.01,
            ),
        )
    return base_controls + (
        ControlSpec("sigma_deg", "σ (deg)", "number", 0.8, step=0.1, min=0.0),
        ControlSpec("Gamma_deg", "Γ (deg)", "number", 5.0, step=0.1, min=0.0),
        ControlSpec("eta", "η", "number", 0.5, step=0.05, min=0.0, max=1.0),
    )


@lru_cache(maxsize=1)
def _specular_control_sections() -> tuple[tuple[str, tuple[ControlSpec, ...]], ...]:
    default_by_group = {
        "beam": SpecularBeamConfig(),
        "sample": SpecularSampleConfig(),
        "detector": SpecularDetectorConfig(),
        "diffraction": SpecularDiffractionConfig(),
    }
    sections: list[tuple[str, tuple[ControlSpec, ...]]] = []
    for title, controls in SPECULAR_CONTROL_SECTIONS:
        unified_controls = tuple(
            ControlSpec(
                key=control.name,
                label=control.label,
                component="slider_input",
                default=getattr(default_by_group[control.config_group], control.attr_name),
                step=control.step,
                min=control.min_value,
                max=control.max_value,
                input_step=control.step,
                updatemode=control.updatemode,
            )
            for control in controls
        )
        sections.append((title, unified_controls))
    return tuple(sections)


def _specular_controls() -> tuple[ControlSpec, ...]:
    return tuple(
        control
        for _, section_controls in _specular_control_sections()
        for control in section_controls
    )


def _mode_state_from_spec(spec: SimulationSpec) -> dict[str, Any]:
    return {control.key: control.default for control in spec.controls}


def _default_state() -> dict[str, dict[str, Any]]:
    return {key: _mode_state_from_spec(spec) for key, spec in SIMULATION_SPECS.items()}


def _resolve_mode(mode: str | None) -> str:
    if mode in SIMULATION_SPECS:
        return str(mode)
    return DEFAULT_MODE


def _merged_mode_state(mode: str, state: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = SIMULATION_SPECS[mode]
    merged = _mode_state_from_spec(spec)
    if state:
        merged.update({key: value for key, value in state.items() if key in merged})
    return merged


def _updated_mode_state(
    mode_value: Any,
    values: list[Any],
    hybrid_values: list[Any],
    control_ids: list[dict[str, Any]],
    hybrid_ids: list[dict[str, Any]],
    state_value: dict[str, Any] | None,
    triggered_id: Any,
) -> dict[str, dict[str, Any]] | None:
    mode_key = _resolve_mode(mode_value)
    current_state = dict(state_value or {})
    previous_mode_state = _merged_mode_state(mode_key, current_state.get(mode_key))
    mode_state = dict(previous_mode_state)

    if isinstance(triggered_id, dict):
        pairs: list[tuple[dict[str, Any], Any]]
        if triggered_id.get("type") == "simulation-control":
            pairs = list(zip(control_ids, values, strict=True))
        elif triggered_id.get("type") == "simulation-hybrid-input":
            pairs = list(zip(hybrid_ids, hybrid_values, strict=True))
        else:
            pairs = []
        for control_id, value in pairs:
            if control_id["key"] == triggered_id.get("key"):
                mode_state[control_id["key"]] = value
                break
    else:
        for control_id, value in zip(control_ids, values, strict=True):
            mode_state[control_id["key"]] = value
        for control_id, value in zip(hybrid_ids, hybrid_values, strict=True):
            mode_state[control_id["key"]] = value

    if mode_state == previous_mode_state:
        return None

    next_state = dict(current_state)
    next_state[mode_key] = mode_state
    if next_state == current_state:
        return None
    return next_state


def _normalize_hkl_mosaic_values(values: dict[str, Any]) -> tuple[int, int, int, float, float, float]:
    return normalize_detector_params(
        values.get("H"),
        values.get("K"),
        values.get("L"),
        values.get("sigma_deg"),
        values.get("Gamma_deg", values.get("gamma_deg")),
        values.get("eta"),
    )


def _build_reciprocal_space_figure(values: dict[str, Any]) -> go.Figure:
    theta_min_deg = float(values.get("theta_i_min_deg", values.get("theta_min_deg", 0.0)))
    theta_max_deg = float(values.get("theta_i_max_deg", values.get("theta_max_deg", 90.0)))
    frames = int(values.get("frames", N_FRAMES_DEFAULT))
    render_profile = str(values.get("render_profile", "balanced"))

    if frames < 2:
        return _message_figure("Powder Views", "frames must be at least 2")
    if theta_max_deg <= theta_min_deg:
        return _message_figure(
            "Powder Views",
            "θᵢ max must be greater than θᵢ min",
        )

    try:
        fig, _ = build_mono_figure(
            math.radians(theta_min_deg),
            math.radians(theta_max_deg),
            frames,
            render_profile=render_profile,
        )
    except ValueError as exc:
        return _message_figure("Powder Views", str(exc))
    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].visible = False
    return fig


def _build_detector_adapter(values: dict[str, Any]) -> go.Figure:
    try:
        params = _normalize_hkl_mosaic_values(values)
    except ValueError as exc:
        return _message_figure("Mosaic View", str(exc))
    theta_deg = float(values.get("theta_i_deg", values.get("theta_deg", DEFAULT_THETA_DEG)))
    return build_detector_figure(*params, theta_i=math.radians(theta_deg))


def _build_fibrous_adapter(values: dict[str, Any]) -> go.Figure:
    try:
        params = normalize_cylinder_params(
            values.get("H"),
            values.get("K"),
            values.get("L"),
            values.get("sigma_deg"),
            values.get("Gamma_deg", values.get("gamma_deg")),
            values.get("eta"),
        )
    except ValueError as exc:
        return _message_figure("Ewald Cylinder", str(exc))
    return build_cylinder_figure(*params)


def _build_specular_dashboard_adapter(
    values: dict[str, Any],
    *,
    camera: dict[str, Any] | None = None,
    companion_camera: dict[str, Any] | None = None,
) -> tuple[go.Figure, go.Figure, str]:
    try:
        beam_config, sample_config, detector_config, diffraction_config = specular_configs_from_values(
            rays=values.get("rays"),
            seed=values.get("seed"),
            display_rays=values.get("display_rays"),
            source_y=values.get("source_y"),
            beam_width_x=values.get("beam_width_x"),
            beam_width_z=values.get("beam_width_z"),
            divergence_x=values.get("divergence_x"),
            divergence_z=values.get("divergence_z"),
            z_beam=values.get("z_beam"),
            sample_width=values.get("sample_width"),
            sample_height=values.get("sample_height"),
            theta_i=values.get("theta_i"),
            delta=values.get("delta"),
            alpha=values.get("alpha"),
            psi=values.get("psi"),
            z_sample=values.get("z_sample"),
            distance=values.get("distance"),
            detector_width=values.get("detector_width"),
            detector_height=values.get("detector_height"),
            beta=values.get("beta"),
            gamma=values.get("gamma"),
            chi=values.get("chi"),
            pixel_u=values.get("pixel_u"),
            pixel_v=values.get("pixel_v"),
            i0=values.get("i0"),
            j0=values.get("j0"),
            h_index=values.get("H"),
            k_index=values.get("K"),
            l_index=values.get("L"),
            sigma_deg=values.get("sigma_deg"),
            mosaic_gamma_deg=values.get("mosaic_gamma_deg"),
            eta=values.get("eta"),
            default_beam=SpecularBeamConfig(),
            default_sample=SpecularSampleConfig(),
            default_detector=SpecularDetectorConfig(),
            default_diffraction=SpecularDiffractionConfig(),
        )
    except ValueError as exc:
        summary = f"Error: {exc}"
        figure = build_specular_error_figure(str(exc))
        figure.update_layout(meta={"simulation_summary": summary})
        companion_figure = build_specular_companion_error_figure(str(exc))
        return figure, companion_figure, summary

    figure, companion_figure, summary = build_specular_dashboard_views(
        beam_config,
        sample_config,
        detector_config,
        diffraction_config,
        camera=camera,
        companion_camera=companion_camera,
    )
    meta = dict(figure.layout.meta) if isinstance(figure.layout.meta, dict) else {}
    meta["simulation_summary"] = summary
    figure.update_layout(meta=meta)
    return figure, companion_figure, summary


def _build_specular_adapter(
    values: dict[str, Any],
    *,
    camera: dict[str, Any] | None = None,
) -> tuple[go.Figure, str]:
    figure, _, summary = _build_specular_dashboard_adapter(values, camera=camera)
    return figure, summary


def _build_simulation_outputs(
    mode: str,
    values: dict[str, Any],
    *,
    camera: dict[str, Any] | None = None,
) -> tuple[go.Figure, str]:
    if mode == SPECULAR_MODE:
        return _build_specular_adapter(values, camera=camera)

    spec = SIMULATION_SPECS[mode]
    figure = spec.build_figure(values)
    if camera and mode in SCENE_CAMERA_MODES:
        figure.update_layout(scene_camera=camera)
    return figure, ""


def _extract_summary_from_figure_value(figure_value: Any) -> str:
    if isinstance(figure_value, go.Figure):
        meta = figure_value.layout.meta
    elif isinstance(figure_value, dict):
        meta = (figure_value.get("layout") or {}).get("meta")
    else:
        meta = None

    if not isinstance(meta, dict):
        return ""

    summary = meta.get("simulation_summary")
    return summary if isinstance(summary, str) else ""


def _summary_style(mode: str, summary: str = "") -> dict[str, Any]:
    style = {
        "whiteSpace": "pre-wrap",
        "backgroundColor": "#f7fafc",
        "border": "1px solid #e2e8f0",
        "borderRadius": "10px",
        "padding": "0.85rem",
        "fontFamily": "Consolas, Monaco, monospace",
        "fontSize": "0.92rem",
    }
    if mode != SPECULAR_MODE or not summary:
        style["display"] = "none"
    return style


def _specular_companion_style(mode: str) -> dict[str, Any]:
    style = {"display": "grid", "gap": "0.75rem", "minHeight": "0"}
    if mode != SPECULAR_MODE:
        style["display"] = "none"
    return style


def _specular_companion_camera_key(mode: str) -> str:
    return f"{mode}:companion"


def _shell_class_names(mode: str) -> tuple[str, str]:
    shell_class = "simulation-shell"
    sidebar_class = "simulation-sidebar"
    if mode == SPECULAR_MODE:
        shell_class += " simulation-shell--specular"
        sidebar_class += " simulation-sidebar--specular"
    return shell_class, sidebar_class


def _build_specular_control_sections(values: dict[str, Any]) -> list[html.Div]:
    sections: list[html.Div] = []
    for title, controls in _specular_control_sections():
        sections.append(
            html.Div(
                [
                    html.H3(title, style={"margin": "0", "fontSize": "1rem"}, className="specular-section-title"),
                    html.Div(
                        [_build_control(control, values[control.key]) for control in controls],
                        style={"display": "grid", "gap": "0.9rem"},
                        className="specular-section-grid",
                    ),
                ],
                className="specular-section-card",
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "0.85rem",
                    "padding": "0.9rem",
                    "backgroundColor": "#fbfcfd",
                    "border": "1px solid #d8dee9",
                    "borderRadius": "10px",
                },
            )
        )
    return sections


def _format_ring_selector_label(g_r_value: float, g_z_value: float) -> str:
    if math.isclose(g_r_value, 0.0, abs_tol=1e-9):
        return f"00L peak, Gz ~= {g_z_value:.3f} A^-1"
    return f"Qr ~= {g_r_value:.3f} A^-1, Gz ~= {g_z_value:.3f} A^-1"


def _format_sphere_selector_label(g_value: float, two_theta_value: str) -> str:
    return f"|G| ~= {g_value:.3f} A^-1, 2θ ~= {two_theta_value}"


def _format_cylinder_selector_label(g_r_value: float) -> str:
    return f"Qr ~= {g_r_value:.3f} A^-1"


def _resolved_powder_view(value: Any) -> str:
    if value in {"single-crystal", "3d-powder", "2d-powder", "cylinder"}:
        return str(value)
    return DEFAULT_POWDER_VIEW


@lru_cache(maxsize=1)
def _powder_selector_catalog() -> dict[str, Any]:
    _, context = build_mono_figure(n_frames=2, render_profile="balanced")
    sphere_options = tuple(
        (_format_sphere_selector_label(g_value, two_theta_value), index)
        for index, (g_value, two_theta_value) in enumerate(
            zip(context["g_values"], context["g_two_thetas"], strict=True)
        )
    )
    ring_options = tuple(
        (_format_ring_selector_label(g_r_value, g_z_value), index)
        for index, (g_r_value, g_z_value) in enumerate(context["g_ring_specs"])
    )
    cylinder_options = tuple(
        (_format_cylinder_selector_label(g_r_value), index)
        for index, g_r_value in enumerate(context["g_cylinder_specs"])
    )
    return {
        "sphere_options": sphere_options,
        "ring_options": ring_options,
        "cylinder_options": cylinder_options,
        "sphere_default": (0,) if sphere_options else (),
        "ring_default": tuple(index for index, _ in enumerate(ring_options)),
        "cylinder_default": (0,) if cylinder_options else (),
    }


def _normalize_powder_selection(
    value: Any,
    option_count: int,
    default: tuple[int, ...],
) -> list[int]:
    if value is None:
        candidates = list(default)
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        candidates = [value]

    normalized: list[int] = []
    for raw_value in candidates:
        try:
            index = int(raw_value)
        except (TypeError, ValueError):
            continue
        if index < 0 or index >= option_count or index in normalized:
            continue
        normalized.append(index)
    return normalized


def _default_powder_selection_state() -> dict[str, list[int]]:
    catalog = _powder_selector_catalog()
    return {
        POWDER_SPHERE_SELECTION_KEY: list(catalog["sphere_default"]),
        POWDER_RING_SELECTION_KEY: list(catalog["ring_default"]),
        POWDER_CYLINDER_SELECTION_KEY: list(catalog["cylinder_default"]),
    }


def _resolved_powder_selection_state(state: dict[str, Any] | None = None) -> dict[str, list[int]]:
    catalog = _powder_selector_catalog()
    source = dict(state or {})
    return {
        POWDER_SPHERE_SELECTION_KEY: _normalize_powder_selection(
            source.get(POWDER_SPHERE_SELECTION_KEY),
            len(catalog["sphere_options"]),
            catalog["sphere_default"],
        ),
        POWDER_RING_SELECTION_KEY: _normalize_powder_selection(
            source.get(POWDER_RING_SELECTION_KEY),
            len(catalog["ring_options"]),
            catalog["ring_default"],
        ),
        POWDER_CYLINDER_SELECTION_KEY: _normalize_powder_selection(
            source.get(POWDER_CYLINDER_SELECTION_KEY),
            len(catalog["cylinder_options"]),
            catalog["cylinder_default"],
        ),
    }


def _updated_powder_view(value: Any, current_value: Any) -> str | None:
    resolved_current = _resolved_powder_view(current_value)
    resolved_value = _resolved_powder_view(value if value is not None else current_value)
    if resolved_value == resolved_current:
        return None
    return resolved_value


def _updated_powder_selection_state(
    values: list[Any],
    control_ids: list[dict[str, Any]],
    state_value: dict[str, Any] | None,
) -> dict[str, list[int]] | None:
    current_state = _resolved_powder_selection_state(state_value)
    selection_state = dict(current_state)
    for control_id, value in zip(control_ids, values, strict=True):
        selection_state[control_id["key"]] = value
    resolved_state = _resolved_powder_selection_state(selection_state)
    if resolved_state == current_state:
        return None
    return resolved_state


def _build_powder_view_control(active_view: Any = None) -> html.Div:
    resolved_view = _resolved_powder_view(active_view)
    return html.Div(
        [
            html.Label("Powder view", style={"fontWeight": "600"}),
            dcc.RadioItems(
                id="powder-view-mode",
                options=[
                    {"label": "Single crystal", "value": "single-crystal"},
                    {"label": "3D powder", "value": "3d-powder"},
                    {"label": "2D powder", "value": "2d-powder"},
                    {"label": "Cylinder", "value": "cylinder"},
                ],
                value=resolved_view,
                labelStyle={
                    "display": "block",
                    "padding": "0.4rem 0.55rem",
                    "border": "1px solid #dbe4ee",
                    "borderRadius": "0.5rem",
                    "backgroundColor": "#ffffff",
                    "marginBottom": "0.35rem",
                },
                inputStyle={"marginRight": "0.5rem"},
            ),
        ],
        style={"display": "flex", "flexDirection": "column", "gap": "0.35rem"},
    )


def _build_peak_selector_card(
    key: str,
    label: str,
    options: tuple[tuple[str, Any], ...],
    value: list[int],
) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(label, style={"fontWeight": "600"}),
                    html.Div(
                        f"{len(value)} of {len(options)} peaks visible",
                        style={"fontSize": "0.85rem", "color": "#64748b"},
                    ),
                ],
                style={"display": "flex", "flexDirection": "column", "gap": "0.15rem"},
            ),
            dcc.Checklist(
                id={"type": "powder-qr-control", "key": key},
                options=[
                    {"label": option_label, "value": option_value}
                    for option_label, option_value in options
                ],
                value=value,
                labelStyle={
                    "display": "block",
                    "marginBottom": "0.4rem",
                    "padding": "0.35rem 0.4rem",
                    "borderRadius": "0.45rem",
                },
                inputStyle={"marginRight": "0.5rem"},
                style={"maxHeight": "16rem", "overflowY": "auto"},
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "0.6rem",
            "padding": "0.8rem",
            "border": "1px solid #dbe4ee",
            "borderRadius": "0.75rem",
            "backgroundColor": "#ffffff",
        },
    )


def _build_powder_selector_controls(
    selection_state: dict[str, Any] | None = None,
    powder_view: Any = None,
) -> list[html.Div]:
    catalog = _powder_selector_catalog()
    selection = _resolved_powder_selection_state(selection_state)
    resolved_view = _resolved_powder_view(powder_view)

    controls: list[html.Div] = [_build_powder_view_control(resolved_view)]
    selector_cards = {
        "3d-powder": _build_peak_selector_card(
            POWDER_SPHERE_SELECTION_KEY,
            "Visible |G| shells",
            catalog["sphere_options"],
            selection[POWDER_SPHERE_SELECTION_KEY],
        ),
        "2d-powder": _build_peak_selector_card(
            POWDER_RING_SELECTION_KEY,
            "Visible 2D powder peaks",
            catalog["ring_options"],
            selection[POWDER_RING_SELECTION_KEY],
        ),
        "cylinder": _build_peak_selector_card(
            POWDER_CYLINDER_SELECTION_KEY,
            "Visible cylinder peaks",
            catalog["cylinder_options"],
            selection[POWDER_CYLINDER_SELECTION_KEY],
        ),
    }

    active_selector = selector_cards.get(resolved_view)
    if active_selector is not None:
        controls.append(active_selector)
    else:
        controls.append(
            html.Div(
                [
                    html.Div("Peak filters", style={"fontWeight": "600"}),
                    html.Div(
                        "Single crystal does not use grouped powder peak filters.",
                        style={"color": "#64748b", "lineHeight": "1.5"},
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "0.45rem",
                    "padding": "0.8rem",
                    "border": "1px solid #dbe4ee",
                    "borderRadius": "0.75rem",
                    "backgroundColor": "#ffffff",
                },
            )
        )
    return controls


SIMULATION_SPECS: dict[str, SimulationSpec] = {
    "reciprocal-space": SimulationSpec(
        key="reciprocal-space",
        label="Powder Views",
        description=(
            "Single-crystal, 3D powder, 2D powder, and cylinder reciprocal-space "
            "views in one Plotly figure. Use the sidebar powder view picker and "
            "peak filters, plus the in-figure θᵢ slider."
        ),
        controls=(
            ControlSpec("theta_i_min_deg", "θᵢ min (deg)", "number", 0.0, step=1.0),
            ControlSpec("theta_i_max_deg", "θᵢ max (deg)", "number", 90.0, step=1.0),
            ControlSpec("frames", "frames", "number", N_FRAMES_DEFAULT, step=1, min=2.0),
            ControlSpec(
                "render_profile",
                "render profile",
                "dropdown",
                "balanced",
                options=(("Balanced", "balanced"), ("Full", "full")),
            ),
        ),
        build_figure=_build_reciprocal_space_figure,
    ),
    "detector-view": SimulationSpec(
        key="detector-view",
        label="Mosaic View",
        description=(
            "Three-panel detector simulation linking reciprocal-space geometry, detector "
            "intensity, and centered integration. Use the θᵢ slider in the sidebar to "
            "move the Ewald sphere without resetting the camera."
        ),
        controls=(
            ControlSpec(
                "theta_i_deg",
                "θᵢ (deg)",
                "slider",
                DEFAULT_THETA_DEG,
                step=0.25,
                min=THETA_MIN_DEG,
                max=THETA_MAX_DEG,
            ),
            *_hkl_controls(hybrid_mosaic_params=True),
        ),
        build_figure=_build_detector_adapter,
    ),
    "fibrous-view": SimulationSpec(
        key="fibrous-view",
        label="Ewald Cylinder",
        description=(
            "Fibrous Bragg/Ewald/cylinder overlap view with the same HKL and mosaic controls."
        ),
        controls=_hkl_controls(),
        build_figure=_build_fibrous_adapter,
    ),
    SPECULAR_MODE: SimulationSpec(
        key=SPECULAR_MODE,
        label="Specular Diffraction",
        description=(
            "Beam -> sample -> detector diffraction geometry with live beam, sample, "
            "detector, and HKL/mosaic controls. The summary panel below the figure "
            "tracks HKL, |G|, and diffraction hit counts."
        ),
        controls=_specular_controls(),
        build_figure=lambda values: _build_specular_adapter(values)[0],
    ),
}


def _build_control(control: ControlSpec, value: Any) -> html.Div:
    control_id = {"type": "simulation-control", "key": control.key}
    label_content: Any
    if isinstance(control.label, SpecularMathLabel):
        label_content = html.Span(
            [
                control.label.symbol,
                html.Sub(control.label.subscript) if control.label.subscript is not None else None,
                control.label.suffix,
            ]
        )
    else:
        label_content = control.label

    if control.component == "dropdown":
        component = dcc.Dropdown(
            id=control_id,
            options=[{"label": label, "value": option_value} for label, option_value in control.options],
            value=value,
            clearable=False,
        )
    elif control.component == "slider":
        midpoint = 0.5 * (float(control.min) + float(control.max))
        component = dcc.Slider(
            id=control_id,
            min=control.min,
            max=control.max,
            step=control.step,
            value=value,
            updatemode="drag" if control.key == "theta_i_deg" else "mouseup",
            marks={
                float(control.min): f"{control.min:g}",
                midpoint: f"{midpoint:g}",
                float(control.max): f"{control.max:g}",
            },
        )
    elif control.component == "slider_input":
        slider_id = {"type": "simulation-hybrid-slider", "key": control.key}
        input_id = {"type": "simulation-hybrid-input", "key": control.key}
        marks = {
            float(control.min): f"{control.min:g}",
            float(control.max): f"{control.max:g}",
        }
        component = html.Div(
            [
                html.Div(
                    dcc.Slider(
                        id=slider_id,
                        min=control.min,
                        max=control.max,
                        step=control.step,
                        value=value,
                        updatemode=control.updatemode or "mouseup",
                        marks=marks,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    style={"flex": "1 1 auto"},
                ),
                dcc.Input(
                    id=input_id,
                    type="number",
                    value=value,
                    step=control.input_step if control.input_step is not None else control.step,
                    min=control.min,
                    max=control.max,
                    debounce=True,
                    style={"width": "88px", "minWidth": "88px"},
                ),
            ],
            className="simulation-control-pair",
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "0.75rem",
            },
        )
    elif control.component == "range":
        component = dcc.RangeSlider(
            id=control_id,
            min=control.min,
            max=control.max,
            step=control.step,
            value=list(value),
            allowCross=False,
            marks={
                float(control.min): f"{control.min:g}",
                0.5 * (float(control.min) + float(control.max)): f"{0.5 * (float(control.min) + float(control.max)):g}",
                float(control.max): f"{control.max:g}",
            },
        )
    else:
        component = dcc.Input(
            id=control_id,
            type="number",
            value=value,
            step=control.step,
            min=control.min,
            max=control.max,
            debounce=True,
            style={"width": "100%"},
        )

    return html.Div(
        [
            html.Label(label_content, style={"fontWeight": "600"}),
            component,
        ],
        className="simulation-control",
        style={"display": "flex", "flexDirection": "column", "gap": "0.35rem"},
    )


def _build_controls_for_mode(
    mode: str,
    values: dict[str, Any],
    powder_selection_state: dict[str, Any] | None = None,
    powder_view: Any = None,
) -> list[html.Div]:
    if mode == SPECULAR_MODE:
        return _build_specular_control_sections(values)

    spec = SIMULATION_SPECS[mode]
    controls = [_build_control(control, values[control.key]) for control in spec.controls]
    if mode == "reciprocal-space":
        controls.extend(_build_powder_selector_controls(powder_selection_state, powder_view))
    return controls


def build_unified_figure(mode: str = DEFAULT_MODE, **values: Any) -> go.Figure:
    """Return the figure for the selected simulation mode."""

    mode_key = _resolve_mode(mode)
    merged_values = _merged_mode_state(mode_key, values)
    return _build_simulation_outputs(mode_key, merged_values)[0]


def build_unified_app(initial_mode: str = DEFAULT_MODE) -> Dash:
    """Return a single Dash app that can switch between the supported views."""

    mode = _resolve_mode(initial_mode)
    state = _default_state()
    powder_selection_state = _default_powder_selection_state()
    powder_view_state = DEFAULT_POWDER_VIEW
    initial_values = _merged_mode_state(mode, state.get(mode))
    initial_spec = SIMULATION_SPECS[mode]
    if mode == SPECULAR_MODE:
        initial_figure, initial_companion_figure, initial_summary = _build_specular_dashboard_adapter(
            initial_values
        )
    else:
        initial_figure, initial_summary = _build_simulation_outputs(mode, initial_values)
        initial_companion_figure = go.Figure()
    shell_class_name, sidebar_class_name = _shell_class_names(mode)

    assets_folder = Path(__file__).resolve().parent.parent / "assets"
    app = Dash(__name__, suppress_callback_exceptions=True, assets_folder=str(assets_folder))
    app.title = "Unified Mosaic Simulator"
    app.layout = html.Div(
        [
            dcc.Store(id="simulation-state", data=state),
            dcc.Store(id="powder-selection-state", data=powder_selection_state),
            dcc.Store(id="powder-view-state", data=powder_view_state),
            dcc.Store(id="simulation-camera-state", data={}),
            dcc.Store(id="powder-selector-sync", data={}),
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("Unified Mosaic Simulator", style={"margin": "0"}),
                            html.Div(
                                "One GUI for Powder Views, Mosaic View, Ewald Cylinder, and Specular Diffraction, backed by the same package-level builders.",
                                style={"color": "#4b5563", "lineHeight": "1.5"},
                            ),
                            html.Div(
                                [
                                    html.Label("simulation", style={"fontWeight": "600"}),
                                    dcc.Dropdown(
                                        id="simulation-mode",
                                        options=[
                                            {"label": spec.label, "value": spec.key}
                                            for spec in SIMULATION_SPECS.values()
                                        ],
                                        value=mode,
                                        clearable=False,
                                    ),
                                ],
                                style={"display": "flex", "flexDirection": "column", "gap": "0.35rem"},
                            ),
                            html.Div(
                                initial_spec.description,
                                id="simulation-description",
                                style={"color": "#374151", "lineHeight": "1.5"},
                            ),
                            html.Pre(
                                id="simulation-summary",
                                children=initial_summary,
                                className="specular-summary-card",
                                style=_summary_style(mode, initial_summary),
                            ),
                            html.Div(
                                _build_controls_for_mode(
                                    mode,
                                    initial_values,
                                    powder_selection_state,
                                    powder_view_state,
                                ),
                                id="simulation-controls",
                                className="simulation-controls",
                                style={"display": "grid", "gap": "0.9rem"},
                            ),
                        ],
                        id="simulation-sidebar",
                        className=sidebar_class_name,
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "1rem",
                            "padding": "1.25rem",
                            "backgroundColor": "#f8fafc",
                            "borderRight": "1px solid #dbe4ee",
                            "boxSizing": "border-box",
                            "maxHeight": "100vh",
                            "overflowY": "auto",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Button(
                                        "Save PNG",
                                        id="export-png-button",
                                        n_clicks=0,
                                        style={
                                            "padding": "0.6rem 0.9rem",
                                            "border": "1px solid #cbd5e1",
                                            "backgroundColor": "#f8fafc",
                                            "borderRadius": "0.5rem",
                                            "cursor": "pointer",
                                            "fontWeight": "600",
                                        },
                                    ),
                                    html.Div(
                                        id="export-png-status",
                                        style={"color": "#475569", "fontSize": "0.95rem"},
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "justifyContent": "space-between",
                                    "alignItems": "center",
                                    "paddingBottom": "0.75rem",
                                    "gap": "1rem",
                                },
                            ),
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="simulation-figure",
                                        figure=initial_figure,
                                        style={"height": "100%", "minHeight": "0"},
                                        config={"responsive": True, "displaylogo": False},
                                    ),
                                ],
                                className="simulation-graph-frame simulation-graph-frame--primary",
                            ),
                            html.Section(
                                [
                                    html.Div(
                                        [
                                            html.H3(
                                                "Reciprocal Space and Integrated Response",
                                                style={"margin": "0"},
                                            ),
                                            html.Div(
                                                "The synchronized mosaic reciprocal-space and centered-integration panels for the current specular HKL and θᵢ.",
                                                style={"color": "#475569", "lineHeight": "1.5"},
                                            ),
                                        ],
                                        className="simulation-visual-header",
                                        style={"display": "flex", "flexDirection": "column", "gap": "0.35rem"},
                                    ),
                                    html.Div(
                                        [
                                            dcc.Graph(
                                                id="simulation-specular-companion-figure",
                                                figure=initial_companion_figure,
                                                style={"height": "100%", "minHeight": "0"},
                                                config={"responsive": True, "displaylogo": False},
                                            ),
                                        ],
                                        className="simulation-graph-frame simulation-graph-frame--secondary",
                                    ),
                                ],
                                id="simulation-specular-companion-card",
                                className="simulation-visual-card",
                                style=_specular_companion_style(mode),
                            ),
                        ],
                        id="simulation-main",
                        className="simulation-main",
                        style={"flex": "1 1 auto", "padding": "1rem", "minWidth": "0"},
                    ),
                ],
                id="simulation-shell",
                className=shell_class_name,
            ),
        ]
    )

    @app.callback(
        Output("simulation-state", "data"),
        Input({"type": "simulation-control", "key": ALL}, "value"),
        Input({"type": "simulation-hybrid-input", "key": ALL}, "value"),
        State("simulation-mode", "value"),
        State({"type": "simulation-control", "key": ALL}, "id"),
        State({"type": "simulation-hybrid-input", "key": ALL}, "id"),
        State("simulation-state", "data"),
        prevent_initial_call=True,
    )
    def cache_control_values(
        values,
        hybrid_values,
        mode_value,
        control_ids,
        hybrid_ids,
        state_value,
    ):  # pragma: no cover - UI callback
        updated_state = _updated_mode_state(
            mode_value,
            values,
            hybrid_values,
            control_ids,
            hybrid_ids,
            state_value,
            ctx.triggered_id,
        )
        if updated_state is None:
            raise PreventUpdate
        return updated_state

    @app.callback(
        Output({"type": "simulation-hybrid-slider", "key": MATCH}, "value"),
        Output({"type": "simulation-hybrid-input", "key": MATCH}, "value"),
        Input({"type": "simulation-hybrid-slider", "key": MATCH}, "value"),
        Input({"type": "simulation-hybrid-input", "key": MATCH}, "value"),
        prevent_initial_call=True,
    )
    def sync_hybrid_control(slider_value, input_value):  # pragma: no cover - UI callback
        triggered_id = ctx.triggered_id
        if not isinstance(triggered_id, dict):
            raise PreventUpdate

        if triggered_id.get("type") == "simulation-hybrid-input":
            value = input_value
            if value is None:
                value = slider_value
        else:
            value = slider_value

        if value is None:
            raise PreventUpdate

        return value, value

    @app.callback(
        Output("powder-view-state", "data"),
        Input("powder-view-mode", "value"),
        State("powder-view-state", "data"),
        prevent_initial_call=True,
    )
    def cache_powder_view_value(value, current_value):  # pragma: no cover - UI callback
        updated_view = _updated_powder_view(value, current_value)
        if updated_view is None:
            raise PreventUpdate
        return updated_view

    @app.callback(
        Output("powder-selection-state", "data"),
        Input({"type": "powder-qr-control", "key": ALL}, "value"),
        State({"type": "powder-qr-control", "key": ALL}, "id"),
        State("powder-selection-state", "data"),
        prevent_initial_call=True,
    )
    def cache_powder_selection_values(values, control_ids, state_value):  # pragma: no cover - UI callback
        updated_state = _updated_powder_selection_state(values, control_ids, state_value)
        if updated_state is None:
            raise PreventUpdate
        return updated_state

    @app.callback(
        Output("simulation-description", "children"),
        Output("simulation-controls", "children"),
        Input("simulation-mode", "value"),
        Input("powder-view-state", "data"),
        State("simulation-state", "data"),
        State("powder-selection-state", "data"),
        prevent_initial_call=True,
    )
    def render_mode_controls(
        mode_value,
        powder_view_value,
        state_value,
        powder_state_value,
    ):  # pragma: no cover - UI callback
        mode_key = _resolve_mode(mode_value)
        spec = SIMULATION_SPECS[mode_key]
        values = _merged_mode_state(mode_key, (state_value or {}).get(mode_key))
        return spec.description, _build_controls_for_mode(
            mode_key,
            values,
            powder_state_value,
            powder_view_value,
        )

    @app.callback(
        Output("simulation-shell", "className"),
        Output("simulation-sidebar", "className"),
        Input("simulation-mode", "value"),
        prevent_initial_call=True,
    )
    def render_shell_classes(mode_value):  # pragma: no cover - UI callback
        return _shell_class_names(_resolve_mode(mode_value))

    @app.callback(
        Output("simulation-specular-companion-card", "style"),
        Input("simulation-mode", "value"),
        prevent_initial_call=True,
    )
    def render_specular_companion_card(mode_value):  # pragma: no cover - UI callback
        return _specular_companion_style(_resolve_mode(mode_value))

    @app.callback(
        Output("simulation-camera-state", "data"),
        Input("simulation-figure", "relayoutData"),
        Input("simulation-specular-companion-figure", "relayoutData"),
        State("simulation-mode", "value"),
        State("simulation-camera-state", "data"),
        prevent_initial_call=True,
    )
    def cache_camera_state(
        relayout_data,
        companion_relayout_data,
        mode_value,
        camera_state_value,
    ):  # pragma: no cover - UI callback
        mode_key = _resolve_mode(mode_value)
        camera_state = dict(camera_state_value or {})
        updated_state = dict(camera_state)
        changed = False

        if mode_key in SCENE_CAMERA_MODES:
            camera = extract_scene_camera(relayout_data)
            if camera and camera_state.get(mode_key) != camera:
                updated_state[mode_key] = camera
                changed = True

        if mode_key == SPECULAR_MODE:
            companion_key = _specular_companion_camera_key(mode_key)
            companion_camera = extract_scene_camera(companion_relayout_data)
            if companion_camera and camera_state.get(companion_key) != companion_camera:
                updated_state[companion_key] = companion_camera
                changed = True

        if not changed:
            raise PreventUpdate
        return updated_state

    @app.callback(
        Output("simulation-figure", "figure"),
        Output("simulation-specular-companion-figure", "figure"),
        Input("simulation-mode", "value"),
        Input("simulation-state", "data"),
        State("simulation-camera-state", "data"),
        State("simulation-figure", "relayoutData"),
        State("simulation-specular-companion-figure", "relayoutData"),
        prevent_initial_call=True,
    )
    def render_figure(
        mode_value,
        state_value,
        camera_state_value,
        relayout_data,
        companion_relayout_data,
    ):  # pragma: no cover - UI callback
        mode_key = _resolve_mode(mode_value)
        values = _merged_mode_state(mode_key, (state_value or {}).get(mode_key))
        camera_state = dict(camera_state_value or {})
        camera = None
        if mode_key in SCENE_CAMERA_MODES:
            camera = camera_state.get(mode_key) or extract_scene_camera(relayout_data)

        if mode_key == SPECULAR_MODE:
            companion_camera = (
                camera_state.get(_specular_companion_camera_key(mode_key))
                or extract_scene_camera(companion_relayout_data)
            )
            figure, companion_figure, _ = _build_specular_dashboard_adapter(
                values,
                camera=camera,
                companion_camera=companion_camera,
            )
            return figure, companion_figure

        return _build_simulation_outputs(mode_key, values, camera=camera)[0], go.Figure()

    @app.callback(
        Output("simulation-summary", "children"),
        Output("simulation-summary", "style"),
        Input("simulation-mode", "value"),
        Input("simulation-figure", "figure"),
        prevent_initial_call=True,
    )
    def render_summary(mode_value, figure_value):  # pragma: no cover - UI callback
        mode_key = _resolve_mode(mode_value)
        summary = _extract_summary_from_figure_value(figure_value) if mode_key == SPECULAR_MODE else ""
        return summary, _summary_style(mode_key, summary)

    app.clientside_callback(
        PNG_EXPORT_CLIENTSIDE_CALLBACK,
        Output("export-png-status", "children"),
        Input("export-png-button", "n_clicks"),
        State("simulation-mode", "value"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        POWDER_QR_CLIENTSIDE_CALLBACK,
        Output("powder-selector-sync", "data"),
        Input("simulation-mode", "value"),
        Input("simulation-figure", "figure"),
        Input("powder-selection-state", "data"),
        Input("powder-view-state", "data"),
        prevent_initial_call=True,
    )

    return app


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the unified simulator."""

    parser = argparse.ArgumentParser(
        description="Unified GUI for the supported simulation views"
    )
    parser.add_argument(
        "--mode",
        choices=tuple(SIMULATION_SPECS),
        default=DEFAULT_MODE,
        help="initial simulation mode",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Dash host (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Dash port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="start the server without opening a browser tab",
    )
    return parser.parse_args()


def main(
    mode: str | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    open_browser: bool | None = None,
) -> None:
    """Launch the unified simulator app."""

    if mode is None and host is None and port is None and open_browser is None:
        args = parse_args()
        resolved_mode = _resolve_mode(args.mode)
        resolved_host = str(args.host)
        resolved_port = int(args.port)
        resolved_open_browser = not args.no_browser
    else:
        resolved_mode = _resolve_mode(mode)
        resolved_host = DEFAULT_HOST if host is None else str(host)
        resolved_port = DEFAULT_PORT if port is None else int(port)
        resolved_open_browser = True if open_browser is None else bool(open_browser)

    app = build_unified_app(initial_mode=resolved_mode)
    url = f"http://{resolved_host}:{resolved_port}"
    if resolved_open_browser:
        threading.Timer(1.0, lambda: webbrowser.open_new(url)).start()
    app.run(debug=False, host=resolved_host, port=resolved_port)


if __name__ == "__main__":
    args = parse_args()
    main(
        args.mode,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )
