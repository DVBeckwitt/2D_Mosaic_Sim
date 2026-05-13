# ADR-001: Special Cause Matrix Export Uses Browser Plotly Rendering

## Status
Accepted

## Date
2026-05-13

## Context
`Special Cause Reciprocal` needs a one-click export that captures the current camera and non-matrix settings as one 3x3 figure. Columns vary `theta_i` across `5°`, `10°`, and `15°`; rows vary the reciprocal peak across `003`, `006`, and `009`; all panels share one mosaic-intensity color legend.

The app already exports PNGs from rendered Plotly.js figures in the browser. A server-side Plotly/Kaleido PNG export was tested as an alternative, but a reduced nine-scene smoke case timed out locally, making it too slow and brittle for this interactive workflow.

## Decision
Build the 3x3 comparison as a Plotly subplot figure on the Dash server, pass that figure spec through an in-memory `dcc.Store`, then render it off-screen in the browser with Plotly.js and download it with `Plotly.downloadImage`.

The visible interface remains a mode-scoped `Save 3x3 Matrix` button. No public Python function signature, console entry point, CLI argument, or existing Dash component ID was removed.

## Alternatives Considered

### Server-side Kaleido export
- Pros: Keeps large figure serialization off the client callback path.
- Cons: Timed out locally for the nine-scene 3D matrix, adds export latency, and requires relying on a heavier static-image path for an interactive browser app.
- Rejected: The export must complete predictably from the GUI.

### Nine separate PNG downloads
- Pros: Reuses existing single-figure export logic.
- Cons: Does not satisfy the requirement for one matrix figure with one color legend.
- Rejected: User needs one formatted figure.

### Static precomputed matrix
- Pros: Fastest download.
- Cons: Would ignore current camera and settings.
- Rejected: User needs to choose the perspective and other settings before saving.

## Consequences
- Export quality follows the browser Plotly renderer already used by the app.
- The generated figure spec is transient and stored in memory only.
- A click nonce is included so repeated exports with unchanged settings still trigger downloads.
- Each matrix export removes any previous off-screen matrix export host before rendering the next one, then purges and removes its own host after completion.
- Browser-managed duplicate filenames remain outside the app boundary; the app does not delete files from the user's download directory.
- Rollback is a normal git revert of the feature commit; no migration or data cleanup is needed.
