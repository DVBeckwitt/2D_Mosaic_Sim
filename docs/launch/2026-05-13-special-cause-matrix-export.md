# Special Cause Matrix Export Launch Note

## Status
Ready for local use as of 2026-05-20. Updated on 2026-05-21 so the active save path is content-driven sprite composition instead of nine 3D subplots. Merged into local `main` with the already-shipped mosaic UI bandwidth work present, cleaned up duplicate helper/test code without changing behavior, then fixed the sprite-composite matrix sizing so each cell scales and centers from the Bragg-only footprint.

## Scope
- Adds `Save 3x3 Matrix` to `Special Cause Reciprocal` only.
- Exports one PNG using the reference `L = 9` matrix view and current non-matrix settings.
- Does not inherit the live single-view camera; matrix exports use the default/reference matrix orientation.
- Matrix columns are `theta_i = 5°`, `10°`, and `15°`.
- Matrix rows are `003`, `006`, and `009`.
- Matrix row labels appear on the left side as `L = 3`, `L = 6`, and `L = 9`.
- Keeps one mosaic-intensity color legend for the whole figure.
- Renders each matrix cell as a transparent visual sphere sprite, keeps Bragg-only anchor figures in the export spec for regression coverage, crops the visual sprite to visible pixels, measures the centered non-cyan Bragg footprint inside that crop, scales and centers from that Bragg footprint, clips helper geometry to the cell, and composites the final PNG on one 2D canvas.
- Keeps title, row labels, column labels, and colorbar out of the sprite renders so they cannot shrink the sphere content.
- Respects the current `Hide Ewald + angle helpers` state in exported panels, omitting Ewald helper geometry and the broad overlap-band helper surface in all cells while keeping sampled Bragg/Ewald overlap lines visible. All sprites are re-centered on their Bragg sphere, and final placement defaults to local per-cell Bragg filling rather than shared `L` proportionality.
- Keeps helper-visible `theta_i = 10°` and `15°` panels legible by rendering Ewald shell and broad overlap helpers as wireframes and adding a Bragg sphere outline in those columns.
- Repeated exports remove any previous off-screen matrix export host before rendering the next one.

## Non-Goals
- No batch export API.
- No configurable matrix dimensions.
- No server-side file writing.
- No change to existing single-view PNG export behavior.
- No attempt to keep tuning subplot spacing, figure size, scene distance, or per-axis zoom as the default export strategy.

## Verification
- 2026-05-20 local release gate: `python -m pip check`, `python -m compileall -q mosaic_sim specular_reflection_sim.py`, `python -m pytest`, `python -m build`, and `git diff --check`.
- 2026-05-21 merge gate: `python -m pytest tests/test_unified_app.py -k "special_cause_matrix_export_spec or special_cause_matrix_export_callback or special_cause_matrix_export_clientside_callback" -q`, `python -m pytest tests/test_unified_app.py -k "special_cause_matrix" -q`, `python -m pip check`, `python -m compileall -q mosaic_sim specular_reflection_sim.py`, `python -m pytest`, `python -m build`, and `git diff --check`.
- 2026-05-21 cleanup gate: `python -m pytest tests/test_unified_app.py -k "special_cause_matrix" -q`, `python -m compileall -q mosaic_sim specular_reflection_sim.py`, `python -m pytest`, `python -m pip check`, and `git diff --check`.
- 2026-05-21 Bragg-bbox sizing gate: `python -m pytest tests/test_unified_app.py -k "special_cause_matrix_export_spec or special_cause_matrix_export_clientside_callback" -q`, `py -3.11 -m pytest tests/test_unified_app.py -k "special_cause_matrix" -q`, `python -m compileall -q mosaic_sim specular_reflection_sim.py`, headless Edge/CDP export smoke, exported-PNG footprint analysis, and `git diff --check`.
- Full `python -m pytest -q` on the local Python 3.13 environment currently aborts in unrelated Plotly-heavy cylinder/fibrous figure construction with a Windows access violation after many tests have already passed. `py -3.11 -m pytest -q --assert=plain` also aborts in an unrelated Plotly cylinder test with the latest resolved Plotly package. Focused matrix export regressions pass on both Python 3.13 and Python 3.11.
- `python -m pytest -q`
- `python -m compileall -q mosaic_sim specular_reflection_sim.py`
- `python -m pip check`
- `git diff --check`
- Sprite-composite regression checks: `python -m pytest tests/test_unified_app.py -k "special_cause_matrix_export" -q`
- Special-cause matrix regression checks: `python -m pytest tests/test_unified_app.py -k "special_cause_matrix" -q`
- Browser callback parse check: extracted `SPECIAL_CAUSE_MATRIX_EXPORT_CLIENTSIDE_CALLBACK` and evaluated it with Node; the callback parsed to a function.
- Sprite-composite browser smoke via headless Edge CDP at `http://127.0.0.1:8057`: clicking `Save 3x3 Matrix` captured one `image/png` export blob, decoded to `1800 x 1800`, reported `Downloaded special_cause_reciprocal_matrix.png.`, left zero off-screen export hosts, and produced zero page error events.
- Post-merge sprite-composite browser smoke via headless Edge CDP at `http://127.0.0.1:8059`: clicking `Save 3x3 Matrix` downloaded `special_cause_reciprocal_matrix.png`, reported `Downloaded special_cause_reciprocal_matrix.png.`, left zero off-screen export hosts, and produced no export failure. CDP also reported browser-side canceled download requests and a Plotly canvas readback warning, which are monitored noise rather than app failure.
- Runtime pixel measurement of the generated PNG found `L = 9` content extents of `411`, `407`, and `409` px across cells whose limiting dimension is about `486.7` px; `L = 3`, `L = 6`, and `L = 9` extents increased in every column.
- Local Dash health check returned HTTP 200 at `http://127.0.0.1:8050`.
- Browser export verification with `Hide Ewald + angle helpers` enabled downloaded `special_cause_reciprocal_matrix.png`, left zero off-screen export hosts, produced no console or page errors, confirmed the broad pale overlap-band helper surface and Ewald helpers were absent, and confirmed the largest sphere fills its panel while smaller rows stay proportionally smaller.
- Browser export verification with `Hide Ewald + angle helpers` disabled downloaded `special_cause_reciprocal_matrix.png`, left zero off-screen export hosts, produced no console or page errors, and confirmed the `10°`/`15°` columns show wireframe Ewald helpers plus a full Bragg sphere outline instead of only the lower Bragg hemisphere.
- For the Bragg-bbox sizing fix, headless Edge/CDP reached the Dash page, found the `Save 3x3 Matrix` button, downloaded `special_cause_reciprocal_matrix.png` in 171 seconds, reported `Downloaded special_cause_reciprocal_matrix.png.`, and produced an `1800 x 1800` PNG. Exported-PNG analysis measured Bragg footprint extents of at least `0.805 * min(cellWidth, cellHeight)` in every cell.

## CI/CD
`.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual dispatch. It installs the package in editable mode with dev dependencies, checks installed dependencies, compiles Python sources, runs pytest, and builds the package on Python 3.11 and 3.13.

## Deprecation and Migration
No migration is required. The change is additive for users: no existing mode, CLI argument, console script, public Python function signature, Dash component ID, or trace name was removed. The private matrix figure helper no longer accepts a camera override because the matrix export intentionally uses the reference view. The export spec now includes private `bragg_anchor_figure`, `bragg_cell_fill_fraction`, and `preserve_relative_l_scale` fields for browser composition and regression coverage. The old subplot matrix builder is no longer the active save path, but it remains private code/test support. The 2026-05-21 cleanup did not change public or private call signatures. Existing users can ignore the new button and keep the previous workflow.

## Known Limits
The app cleans up its off-screen Plotly export host before each matrix render. It cannot delete files already saved by the browser, so duplicate filenames in `Downloads` may still receive browser-managed suffixes such as `(1)`.

The active save path is the sprite-composite exporter. The old subplot matrix builder remains private code/test support only; it is not called by the normal `Save 3x3 Matrix` action. Remove it in a later cleanup if no debug fallback is needed.

The browser callback can draw debug cell, full-sprite, and scaled Bragg-bbox boxes when `debug_layout` is supplied, and it stores placement metadata on `window.__specialCauseMatrixDebug`. The normal app does not expose raw sprite or cropped sprite image dumps. Add that as an explicit diagnostic feature before relying on it for support workflows.

The browser now renders and crops nine full visual Plotly 3D sprite figures and reuses one hidden Plotly host with `Plotly.react` across sprite renders. Bragg footprint measurement is image-based to avoid a second WebGL export per cell. Keep monitoring export latency and browser memory use if the matrix dimensions, render size, or trace density increase.

Code review on 2026-05-21 found no avoidable new files, no new dependencies, and no obvious security issue. The Bragg-bbox sizing fix modifies existing source/tests/docs only, derives anchor figures from already-built sprite JSON without deep-copying discarded helper traces, and avoids duplicate browser-side WebGL exports by measuring the centered Bragg footprint from each visual sprite image. Remaining optional cleanup is to remove the private subplot matrix helper if the debug/test fallback is no longer useful, and to replace substring-only JavaScript callback assertions with an automated parse or browser smoke in CI if browser export regressions become common.

## Rollback
Revert the matrix export fix commit if the button or export behavior regresses:

```bash
git revert <matrix-export-fix-commit>
```

After revert, run the CI gate locally:

```bash
python -m pytest -q
python -m compileall -q mosaic_sim specular_reflection_sim.py
python -m pip check
git diff --check
```

## Monitoring
For local GUI usage, monitor the browser console, Dash server output, whether repeated clicks download the latest matrix without overlapping export hosts, whether sprite cropping leaves tight content bounds, whether each Bragg sphere fills most of its own final cell without clipping, whether helper traces stay clipped inside their cells, whether `Hide Ewald + angle helpers` hides helpers and the broad overlap-band helper surface in all exported matrix cells, whether helper-visible `10°` and `15°` matrix cells continue to show the full Bragg sphere outline with Ewald helpers present, whether the export stays on the centered reference matrix camera instead of the live single-view camera, and whether `L = 3`, `L = 6`, and `L = 9` remain left-side row labels. The main risk is browser-side 3D export performance for rendering and cropping dense visual sprite figures before final canvas composition.
