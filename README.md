# 2D Mosaic Simulation

Interactive tools for visualizing how reciprocal-space features intersect the Ewald sphere and map onto detector-space observables for layered or mosaic materials.

## Current PNG Gallery (`docs/images`)

All current PNG assets in `docs/images` are shown below so the README top matches what is in the repo right now.

### `docs/images/mono_cylinder` (captured from `single_crystal_powder_cylinder_viewer.py`)

| Single Crystal | 3D Powder |
| --- | --- |
| ![Single crystal reciprocal-space view](docs/images/mono_cylinder/mono_single_crystal.png) | ![3D powder reciprocal-space view](docs/images/mono_cylinder/mono_3d_powder.png) |
| 2D Powder | Cylinder |
| ![2D powder reciprocal-space view](docs/images/mono_cylinder/mono_2d_powder.png) | ![Cylinder reciprocal-space view](docs/images/mono_cylinder/mono_cylinder.png) |

### Reciprocal-space mode explanations

1. `Single Crystal`: Discrete reciprocal-lattice points. You only get scattering when a point lies on the Ewald sphere.
2. `3D Powder`: Lattice points are azimuthally averaged into full spherical shells (`|G|` constant), producing full rings when intersected.
3. `2D Powder`: In-plane averaging with out-of-plane order gives reciprocal rings (`|G_r|` at fixed `G_z`), producing ring-arc behavior.
4. `Cylinder`: Reciprocal features extend continuously along `qz`, modeling rod-like scattering from layered/fibrous order.

### `docs/images/placeholders` (PNG placeholders to replace with real screenshots)

These are intentionally temporary images. Keep the same filenames when you replace them with real captures.

### Detector Mosaic/Ewald View (`detector_mosaic_ewald_view.py`)

![Detector feature placeholder](docs/images/placeholders/detector_placeholder.png)

- File: `docs/images/placeholders/detector_placeholder.png`
- Purpose: 3-panel browser UI linking reciprocal-space geometry to detector intensity and centered integration, with live `theta_i`, HKL, `σ`, `Γ`, and `η` controls in the GUI.
- What to capture: Full figure with all three panels visible, including colorbar, integration subplot, and the `theta_i` control.
- Run: `python detector_mosaic_ewald_view.py 0 0 12 --sigma 0.8 --gamma 5 --eta 0.5 --wavelength-bandwidth-pct 0.5`
  The CLI values seed the initial browser state; you can then change HKL and the mosaic parameters directly in the GUI.

### Fibrous Bragg/Ewald Intersections (`fibrous_bragg_ewald_intersections.py`)

![Fibrous feature placeholder](docs/images/placeholders/fibrous_placeholder.png)

- File: `docs/images/placeholders/fibrous_placeholder.png`
- Purpose: Browser-based Bragg/Ewald/cylinder intersection view with live HKL / `σ` / `Γ` / `η` controls below the plot.
- What to capture: A frame showing all three overlap curves clearly (`Ewald/Bragg`, `Cylinder/Ewald`, `Cylinder/Bragg`).
- Run: `python fibrous_bragg_ewald_intersections.py 0 0 12 --sigma 0.8 --gamma 5 --eta 0.5 --wavelength-bandwidth-pct 0.5`
  The CLI values seed the initial browser state; you can then change HKL and the mosaic parameters directly in the GUI.

## Installation

The project targets Python `3.11+`.

```bash
pip install -e .
```

## Project Status

Status date: 2026-05-21

- Documentation status: matrix export design is recorded in `docs/decisions/001-special-cause-matrix-export.md`; launch scope, verification, migration, monitoring, and rollback are recorded in `docs/launch/2026-05-13-special-cause-matrix-export.md`.
- Feature: `Special Cause Reciprocal` is implemented in the unified GUI. It renders the Cu-K alpha Ewald shell, Bi2Se3 Bragg sphere, continuous Bragg/Ewald overlap band, and sampled intensity-colored Bragg/Ewald overlap lines as a one-panel reciprocal-space view using the same HKL, `theta_i`, `σ`, `Γ`, and `η` controls as `Mosaic View`. Its defaults are `theta_i = 5°`, `(H,K,L) = (0,0,3)`, `λ bandwidth (%) = 5.0`, `Ewald samples (odd) = 99`, and `Hide Ewald + angle helpers` off.
- Feature: finite Ewald wavelength spread is available through `λ bandwidth (%)` for mosaic detector, special-cause reciprocal, and fibrous views. Special-cause reciprocal uses the bandwidth as concentric Ewald radii around the fixed central Cu-Kα Ewald center, plus one pink-to-purple Bragg/Ewald overlap line per sampled bandwidth layer; each sampled line point is colored from the same Bragg-sphere mosaic intensity field at that reciprocal-space position. Mosaic detector and fibrous views keep the sampled layer-stack rendering. The default `0.0` keeps the previous monochromatic behavior in mosaic detector and fibrous views.
- Feature: unified GUI navigation has been refreshed. The mode picker now uses styled full-width choices on small screens, the powder peak filters use shared CSS classes instead of inline style dictionaries, and inactive powder peak selector cards are no longer constructed.
- Feature: special-cause reciprocal rendering now keeps the Bragg sphere mosaic map prominent while leaving auxiliary Ewald shell surfaces and the continuous Bragg/Ewald overlap band transparent. The sampled Bragg/Ewald intersection lines remain fully opaque, and the Bragg sphere keeps an RGB mosaic colorscale with a visible colorbar.
- Feature: `Hide Ewald + angle helpers` is available only in `Special Cause Reciprocal`. When enabled, it hides the Ewald shell/sphere surfaces, `k_i`, `theta_i`, and the theta arc while keeping the Bragg sphere mosaic, continuous Bragg/Ewald overlap band, and sampled intensity-colored Bragg/Ewald intersections visible.
- Feature: `Special Cause Reciprocal` can export a single 3x3 matrix PNG from the reference `L = 9` matrix view and current non-matrix settings, including the `Hide Ewald + angle helpers` state. The matrix columns use `theta_i = 5°, 10°, 15°`; rows use `003`, `006`, and `009` with left-side `L = 3`, `L = 6`, and `L = 9` labels; all cells share one mosaic-intensity color legend. The active save path renders each sphere as a transparent sprite, crops the sprite to visible content, scales all rows from the shared `L = 9` Bragg reference extent, and composites one final 2D canvas with labels and colorbar added once. The exported matrix intentionally ignores the live single-view camera, uses the default/reference matrix orientation, and avoids independently auto-fitting smaller rows. When helpers are hidden, the exported matrix omits the continuous overlap-band helper surface and keeps the sampled Bragg/Ewald overlap lines. When helpers are visible, the `10°` and `15°` matrix columns render the Ewald shell and broad overlap helper as wireframes and add a high-contrast Bragg outline so the full Bragg sphere remains visible in the final PNG; the `5°` column keeps the existing filled helper rendering.
- Browser/runtime status: prior local Dash/Chromium verification confirmed `Special Cause Reciprocal` at `λ bandwidth = 5.00%` renders live `Bragg/Ewald overlap` line traces, one overlap band, and the inner/outer shell traces with no failed requests and no HTTP `>=400` responses. Matrix export verification covers both hidden-helper exports and visible-helper exports; the visible-helper run confirmed the `10°` and `15°` panels download with wireframe Ewald helpers, a full Bragg outline, no page errors, and zero leftover off-screen export hosts. The sprite-composite path was also verified through headless Edge CDP: the browser captured one `1800 x 1800` PNG blob, reported the matrix-download success status, left zero export hosts, produced zero page error events, filled each `L = 9` cell above the regression threshold, and kept row extents increasing from `L = 3` to `L = 9`. The fixed-center bandwidth, shell sample-count, pink-to-purple line intensity coloring, centered Bragg view, and unified GUI control/state path are covered by figure-level regression tests. Current known limits are that the private subplot matrix helper still exists only as a test/debug support surface, raw/cropped sprite debug image dumps are not exposed in the normal UI, and browser-side rendering still has to create and crop nine Plotly sprite renders before composing the final canvas.
- Bug/error status: fixed the special-cause rendering regression where the Bragg sphere mosaic became hard to see because it shared the same transparent opacity as the Ewald shell and overlap-band helper surfaces. The intensity-colored overlap-line update also guards the `100%` bandwidth edge case where some sampled shell layers produce empty intersection traces. Matrix export now respects `Hide Ewald + angle helpers`, drops the broad overlap-band helper surface from hide-enabled matrix panels, uses sprite cropping plus one shared `L = 9` Bragg reference scale so the biggest Bragg sphere fills its final cell without making smaller rows autorange independently, no longer inherits an over-zoomed live camera, keeps the `10°` and `15°` Bragg spheres legible when Ewald helpers are visible, removes any prior off-screen export host before rendering the next export, and guards stale async export status updates. Browser-managed duplicate download names such as `special_cause_reciprocal_matrix (1).png` are outside the app's control.
- Migration status: no deprecation or migration is required for users. No existing mode, CLI argument, console entry point, public Python function signature, public Dash component ID, or trace name was removed. The private matrix figure helper no longer accepts a camera override because matrix export uses the reference view by design. The rejected subplot-first export path is no longer the active save path; its remaining private helper/test coverage is internal and can be removed later if no debug fallback is needed. Existing detector/fibrous scripts keep their previous behavior unless `--wavelength-bandwidth-pct` is supplied. Direct callers of `build_special_cause_reciprocal_figure()` still use the special-case `(0,0,3)` / `5%` / `99 samples` default; pass `L=12, wavelength_bandwidth_pct=0.0` for the old direct-call geometry view.
- CI status: `.github/workflows/ci.yml` runs editable install, `pip check`, compile, tests, and package build on pull requests, pushes to `main`, and manual dispatch across Python 3.11 and 3.13.
- Shipping status: local `pip check`, compile, full test suite, package build, code review, and diff whitespace checks pass for this update; rollback is a normal git revert of the matrix export feature commit. See `docs/launch/2026-05-13-special-cause-matrix-export.md`.

## Usage

Run scripts from the project root:

```bash
python mosaic_simulator.py                                          # unified GUI for the supported simulations
python mosaic_simulator.py --mode specular-view                     # open the unified GUI directly in Specular Diffraction mode
python mosaic_simulator.py --mode special-cause-reciprocal          # one-panel Cu-Kα/Bi2Se3 reciprocal-space overlap view, defaulting to theta_i=5°, (0,0,3), and 5% bandwidth
python single_crystal_powder_cylinder_viewer.py                # single-crystal / 3D powder / 2D powder / cylinder views
python single_crystal_powder_cylinder_viewer.py --full-quality # higher quality render profile
python detector_mosaic_ewald_view.py 0 0 12 --sigma 0.8 --gamma 5 --eta 0.5  # launches the detector Dash UI with live HKL/mosaic controls
python detector_mosaic_ewald_view.py 0 0 12 --sigma 0.8 --gamma 5 --eta 0.5 --wavelength-bandwidth-pct 0.5  # adds a finite Ewald wavelength layer stack
python fibrous_bragg_ewald_intersections.py 0 0 12 --sigma 0.8 --gamma 5 --eta 0.5  # fibrous Bragg/Ewald/cylinder GUI with live HKL/mosaic controls
python fibrous_bragg_ewald_intersections.py 0 0 12 --sigma 0.8 --gamma 5 --eta 0.5 --wavelength-bandwidth-pct 0.5  # adds a finite Ewald wavelength layer stack
python specular_reflection_sim.py                                  # standalone specular diffraction GUI with lab geometry plus reciprocal-space / integrated-response companion panels
```

`λ bandwidth (%)` is the full `Δλ / λ` percentage, not FWHM. The GUI controls allow values from `0.0` through `100.0`; the default `0.0` keeps the monochromatic Ewald sphere behavior.

The unified GUI includes these switchable modes in one Dash app:

- `Powder Views`
- `Mosaic View`
- `Special Cause Reciprocal`
- `Ewald Cylinder`
- `Specular Diffraction`

`Special Cause Reciprocal` is the reciprocal-space panel of the mosaic detector geometry by itself. It uses the Cu-Kα Ewald sphere size, the Bi2Se3 `d_hex(H,K,L)` Bragg-sphere size, and the Bragg/Ewald intersection with the same HKL, `theta_i`, `σ`, `Γ`, `η`, and `λ bandwidth (%)` controls as `Mosaic View`. With non-zero bandwidth, it keeps the Ewald center fixed at the central Cu-Kα vector, expands/contracts the Ewald radius into concentric shell bounds, shows the continuous overlap band, and draws one pink-to-purple intensity-colored overlap line for each sampled Ewald layer. The special-cause `theta_i` control includes both a slider and a numeric input so exact angles can be typed directly. Use `Ewald samples (odd)` to set the sampled shell-line count from `3` to `101`; the default is `99`. Enable `Hide Ewald + angle helpers` in the incident-angle section to hide the Ewald shell/sphere surfaces and incident-angle helpers while leaving the Bragg mosaic surface, overlap band, and intensity-colored overlap lines visible. Use `Save 3x3 Matrix` to export one PNG from the reference `L = 9` matrix view and current non-matrix settings; the export varies columns by `theta_i = 5°, 10°, 15°` and rows by `003`, `006`, and `009`, with left-side `L = 3`, `L = 6`, and `L = 9` row labels and a single mosaic-intensity color legend. The export renders transparent sphere sprites, crops to their visible pixels, scales from the shared `L = 9` Bragg reference extent, and composites one final canvas, so the largest sphere fills its cell while the smaller rows remain proportionally smaller. If `Hide Ewald + angle helpers` is enabled, the exported matrix omits those helpers plus the broad overlap-band helper surface in all nine cells while keeping the sampled overlap lines. If helpers are visible, the `10°` and `15°` exported cells use wireframe Ewald/overlap helpers plus a Bragg wire outline to avoid the exported Bragg sphere collapsing visually to only its lower hemisphere. This mode opens at `theta_i = 5°` on the `(0,0,3)` peak with `5%` bandwidth by default.

### Specular Diffraction notes

- Default specular HKL is `(1, 1, 1)`.
- Default specular detector size is `1000 x 1000`.
- The specular UI shows a lab-geometry figure together with a companion reciprocal-space and centered-integration figure.
- In the lab geometry, outgoing diffraction rays are rendered with transparent red traces and outward arrowheads.

Installed console entry points:

- `mosaic-simulator`
- `mosaic-single-crystal-powder-cylinder-viewer`
- `mosaic-detector-mosaic-ewald-view`
- `mosaic-fibrous-bragg-ewald-intersections`
