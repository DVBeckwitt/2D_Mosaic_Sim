# ADR-001: Special Cause Reciprocal Mode and Ewald Bandwidth Sampling

## Status
Accepted

## Date
2026-05-10

## Context
The unified Dash app needed a one-panel special-cause view that matches the reciprocal-space geometry from the mosaic detector view. The view must use the physical Cu-K alpha Ewald sphere radius, the Bi2Se3 Bragg sphere radius from `d_hex(H,K,L)`, and the Bragg/Ewald intersection while preserving the existing mosaic parameters `sigma`, `Gamma`, and `eta`.

The detector and fibrous views also need an optional way to visualize finite Ewald wavelength spread without changing the default monochromatic behavior.

## Decision
Add `special-cause-reciprocal` as a unified-app mode backed by `build_special_cause_reciprocal_figure`.

Use one shared `wavelength_bandwidth_pct` parameter for detector and fibrous Ewald layer stacks. The default remains `0.0`, which returns the original single-Ewald-sphere behavior. Non-zero values render an odd stack of sampled Ewald layers with the central wavelength at the highest opacity.

## Alternatives Considered

### Reuse the Full Mosaic Detector Figure
- Pros: Minimal code.
- Cons: Still renders detector and centered-integration panels, which is not the requested one-panel reciprocal-space view.
- Rejected because the new mode needs a focused reciprocal-space surface.

### Add a Separate Special-Cause Parameter Set
- Pros: Isolates the new mode.
- Cons: Duplicates mosaic controls and risks diverging from the detector view.
- Rejected because the mode is defined as the reciprocal-space view of the mosaic detector geometry.

### Ignore Wavelength Spread
- Pros: Smaller implementation.
- Cons: Makes finite Ewald sphere thickness unavailable in related views.
- Rejected because the layer-stack model is additive and keeps the default monochromatic behavior.

## Consequences
- `Special Cause Reciprocal` shares HKL, incidence angle, `sigma`, `Gamma`, `eta`, and wavelength bandwidth controls with `Mosaic View`.
- The public Python API adds `build_special_cause_reciprocal_figure`.
- `wavelength_bandwidth_pct=0.0` is the migration path for existing users; no existing script arguments need to change.
- Larger non-zero bandwidth values increase Plotly payload size because they add Ewald surfaces and overlap traces.

## Verification
- `python -m pytest -q`
- Browser smoke test of `special-cause-reciprocal`, `detector-view`, and switching back to `special-cause-reciprocal`.
