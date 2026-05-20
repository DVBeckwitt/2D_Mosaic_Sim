# Special Cause Matrix Export Launch Note

## Status
Ready for local use as of 2026-05-20.

## Scope
- Adds `Save 3x3 Matrix` to `Special Cause Reciprocal` only.
- Exports one PNG using the reference `L = 9` matrix view and current non-matrix settings.
- Does not inherit the live single-view camera; matrix exports use the default/reference matrix orientation.
- Matrix columns are `theta_i = 5°`, `10°`, and `15°`.
- Matrix rows are `003`, `006`, and `009`.
- Matrix row labels appear on the left side as `L = 3`, `L = 6`, and `L = 9`.
- Keeps one mosaic-intensity color legend for the whole figure.
- Respects the current `Hide Ewald + angle helpers` state in exported panels, omitting the broad overlap-band helper surface while keeping sampled Bragg/Ewald overlap lines visible. The top-left reference panel still includes Ewald sphere/shell geometry even when helpers are hidden, and all panels use one shared `L = 9` Bragg reference scale so the largest Bragg sphere fills its panel.
- Keeps helper-visible `theta_i = 10°` and `15°` panels legible by rendering Ewald shell and broad overlap helpers as wireframes and adding a Bragg sphere outline in those columns.
- Repeated exports remove any previous off-screen matrix export host before rendering the next one.

## Non-Goals
- No batch export API.
- No configurable matrix dimensions.
- No server-side file writing.
- No change to existing single-view PNG export behavior.

## Verification
- 2026-05-20 local release gate: `python -m pip check`, `python -m compileall -q mosaic_sim specular_reflection_sim.py`, `python -m pytest`, `python -m build`, and `git diff --check`.
- `python -m pytest -q`
- `python -m compileall -q mosaic_sim specular_reflection_sim.py`
- `python -m pip check`
- `git diff --check`
- Local Dash health check returned HTTP 200 at `http://127.0.0.1:8050`.
- Browser export verification with `Hide Ewald + angle helpers` enabled downloaded `special_cause_reciprocal_matrix.png`, left zero off-screen export hosts, produced no console or page errors, confirmed the broad pale overlap-band helper surface was absent, and confirmed the largest sphere fills its panel while smaller rows stay proportionally smaller.
- Browser export verification with `Hide Ewald + angle helpers` disabled downloaded `special_cause_reciprocal_matrix.png`, left zero off-screen export hosts, produced no console or page errors, and confirmed the `10°`/`15°` columns show wireframe Ewald helpers plus a full Bragg sphere outline instead of only the lower Bragg hemisphere.

## CI/CD
`.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual dispatch. It installs the package in editable mode with dev dependencies, checks installed dependencies, compiles Python sources, runs pytest, and builds the package on Python 3.11 and 3.13.

## Deprecation and Migration
No migration is required. The change is additive for users: no existing mode, CLI argument, console script, public Python function signature, Dash component ID, or trace name was removed. The private matrix figure helper no longer accepts a camera override because the matrix export intentionally uses the reference view. Existing users can ignore the new button and keep the previous workflow.

## Known Limits
The app cleans up its off-screen Plotly export host before each matrix render. It cannot delete files already saved by the browser, so duplicate filenames in `Downloads` may still receive browser-managed suffixes such as `(1)`.

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
For local GUI usage, monitor the browser console, Dash server output, whether repeated clicks download the latest matrix without overlapping export hosts, whether `Hide Ewald + angle helpers` still hides helpers and the broad overlap-band helper surface in exported matrices except for the top-left Ewald sphere/shell reference, whether the `L = 9` Bragg sphere fills its panel without clipping while smaller rows remain proportional, whether helper-visible `10°` and `15°` matrix cells continue to show the full Bragg sphere outline with Ewald helpers present, whether the export stays on the reference matrix orientation instead of the live single-view camera, and whether `L = 3`, `L = 6`, and `L = 9` remain left-side row labels. The main risk is browser-side 3D export performance for the nine-panel figure.
