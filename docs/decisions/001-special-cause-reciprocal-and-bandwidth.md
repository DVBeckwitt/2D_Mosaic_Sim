# ADR-001: Special Cause Reciprocal Mode and Ewald Bandwidth Rendering

## Status
Accepted

## Date
2026-05-10

## Context
The unified Dash app needed a one-panel special-cause view that matches the reciprocal-space geometry from the mosaic detector view. The view must use the physical Cu-K alpha Ewald sphere radius, the Bi2Se3 Bragg sphere radius from `d_hex(H,K,L)`, and the Bragg/Ewald intersection while preserving the existing mosaic parameters `sigma`, `Gamma`, and `eta`.

The detector and fibrous views also need an optional way to visualize finite Ewald wavelength spread without changing the default monochromatic behavior.

## Decision
Add `special-cause-reciprocal` as a unified-app mode backed by `build_special_cause_reciprocal_figure`.

Use one shared `wavelength_bandwidth_pct` parameter for detector, special-cause reciprocal, and fibrous Ewald spread. The base detector and fibrous views keep the sampled Ewald layer stack. `special-cause-reciprocal` renders non-zero bandwidth as a continuous hollow Ewald shell bounded by the minimum and maximum `k = 2π / λ` radii, with a Bragg/Ewald overlap surface spanning the valid intersections across that interval and one green Bragg/Ewald overlap line for each sampled Ewald bandwidth layer.

The default remains `0.0` for existing detector/fibrous behavior. The special-cause reciprocal mode and its public figure builder default to `theta_i_deg=5.0`, `(H,K,L)=(0,0,3)`, and `wavelength_bandwidth_pct=5.0`.

Render the special-cause reciprocal Bragg, shell, overlap-band, and sampled overlap-line geometry with uniform opaque traces. Use an RGB Bragg colorscale rather than RGBA colorscale stops so Plotly opacity is controlled in one place and the shell/band surfaces remain readable.

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

### Keep Layer-Specific Transparency in Special-Cause Reciprocal
- Pros: Matches the sampled detector layer-stack visual treatment.
- Cons: The special-cause view is a focused geometry view, not the sampled detector stack; semi-transparent layers make the shell and overlap band harder to inspect.
- Rejected because the one-panel special-cause view should prioritize readable physical geometry.

## Consequences
- `Special Cause Reciprocal` shares HKL, incidence angle, `sigma`, `Gamma`, `eta`, and wavelength bandwidth controls with `Mosaic View`.
- `Special Cause Reciprocal` defaults to `theta_i_deg=5.0`, the `(0,0,3)` peak, and `wavelength_bandwidth_pct=5.0` in both the unified GUI and direct `build_special_cause_reciprocal_figure()` calls.
- The public Python API adds `build_special_cause_reciprocal_figure`.
- `wavelength_bandwidth_pct=0.0` is the migration path for monochromatic rendering. Direct callers can pass `L=12, wavelength_bandwidth_pct=0.0` to reproduce the previous helper default.
- Larger non-zero bandwidth values increase Plotly payload size because they add Ewald shell surfaces, the continuous overlap band, and sampled overlap-line traces.
- The special-cause reciprocal traces are intentionally opaque; future translucent rendering should be treated as a new visual design choice, not a compatibility requirement.
- The sampled green overlap lines are part of the intended non-zero bandwidth visual contract; they make the finite Ewald thickness inspectable instead of relying on the continuous band alone.

## Verification
- `python -m pytest -q`
- Browser smoke test of `special-cause-reciprocal`, `detector-view`, and switching back to `special-cause-reciprocal`.
- `python -m pytest`
- `python -m compileall mosaic_sim specular_reflection_sim.py`
- `git diff --check`
- Local Dash/Chromium runtime check across every unified GUI mode, every powder submode, and 390px mobile layout. The check found no page errors, no HTTP `>=400` responses, and no horizontal mobile overflow.
- `python -m pytest tests/test_detector.py -q`
- `python -m pip check`
- `python -m build`
- Local Dash/Chromium runtime check of `special-cause-reciprocal` at `λ bandwidth = 5.00%`: seven green `Bragg/Ewald overlap` line traces, one overlap band, and inner/outer shell traces rendered with no failed requests or HTTP `>=400` responses.
