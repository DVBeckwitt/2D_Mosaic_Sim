# Special Cause Matrix Export Launch Note

## Status
Ready for local use as of 2026-05-13.

## Scope
- Adds `Save 3x3 Matrix` to `Special Cause Reciprocal` only.
- Exports one PNG using current camera and current non-matrix settings.
- Matrix columns are `theta_i = 5°`, `10°`, and `15°`.
- Matrix rows are `003`, `006`, and `009`.
- Matrix row labels appear on the left side as `L = 3`, `L = 6`, and `L = 9`.
- Keeps one mosaic-intensity color legend for the whole figure.
- Respects the current `Hide Ewald + angle helpers` state in all nine exported panels.
- Repeated exports remove any previous off-screen matrix export host before rendering the next one.

## Non-Goals
- No batch export API.
- No configurable matrix dimensions.
- No server-side file writing.
- No change to existing single-view PNG export behavior.

## Verification
- `python -m pytest -q`
- `python -m compileall -q mosaic_sim specular_reflection_sim.py`
- `python -m pip check`
- `git diff --check`
- Local Dash health check returned HTTP 200 at `http://127.0.0.1:8050`.
- Browser export verification with `Hide Ewald + angle helpers` enabled downloaded `special_cause_reciprocal_matrix.png`, left zero off-screen export hosts, and produced no console or page errors.

## CI/CD
`.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual dispatch. It installs the package in editable mode with dev dependencies, checks installed dependencies, compiles Python sources, runs pytest, and builds the package on Python 3.11 and 3.13.

## Deprecation and Migration
No migration is required. The change is additive: no existing mode, CLI argument, console script, public Python function signature, or trace name was removed. Existing users can ignore the new button and keep the previous workflow.

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
For local GUI usage, monitor the browser console, Dash server output, whether repeated clicks download the latest matrix without overlapping export hosts, whether `Hide Ewald + angle helpers` still hides helpers in exported matrices, and whether `L = 3`, `L = 6`, and `L = 9` remain left-side row labels. The main risk is browser-side 3D export performance for the nine-panel figure.
