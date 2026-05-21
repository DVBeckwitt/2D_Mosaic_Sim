# ADR-001: Special Cause Matrix Export Uses Browser Sprite Composition

## Status
Accepted

## Date
2026-05-13

## Context
`Special Cause Reciprocal` needs a one-click export that captures current non-matrix settings as a 3x3 cell-image bundle using a reference `L = 9` matrix view. Columns vary `theta_i` across `5°`, `10°`, and `15°`; rows vary the reciprocal peak across `003`, `006`, and `009`; all cells share one mosaic-intensity color legend.

The app already exports PNGs from rendered Plotly.js figures in the browser. A server-side Plotly/Kaleido PNG export was tested as an alternative, but a reduced nine-scene smoke case timed out locally, making it too slow and brittle for this interactive workflow.

A previous browser export path rendered one nine-subplot Plotly figure. That kept labels and the colorbar simple, but the saved PNG was dominated by subplot whitespace, scene padding, and 3D camera margins. The matrix export needs to size the visible sphere content, not the 3D axes around it.

## Decision
Build the 3x3 comparison as reusable cropped sphere sprites. The Dash server returns an in-memory export spec containing nine label-free, transparent Plotly visual sprite figures, nine matching Bragg-only anchor figures, and matrix metadata. The browser renders each visual sprite off-screen with Plotly.js, renders the matching anchor figure for measurement, crops the visual sprite to visible content, measures the Bragg footprint from the anchor image, embeds the cell metadata in the cropped PNG, writes a JSON sidecar for each cell, and downloads one ZIP bundle containing all nine cell PNGs plus a manifest. The separate Python compositor reads that bundle and draws the final 2D matrix canvas with the title, row labels, column labels, and a single colorbar.

The visible interface remains a mode-scoped `Save 3x3 Cells` button. It now saves `special_cause_reciprocal_matrix_cells.zip` instead of the final matrix PNG. The new `mosaic-special-cause-matrix` console entry point builds the final PNG offline from the bundle and exposes manual sizing controls. The active export spec does not expose a camera override because the matrix export is defined by the reference view rather than the live single-view camera. The old subplot matrix builder can remain as a private helper/test surface, but it is no longer the normal save path.

## Alternatives Considered

### Server-side Kaleido export
- Pros: Keeps large figure serialization off the client callback path.
- Cons: Timed out locally for the nine-scene 3D matrix, adds export latency, and requires relying on a heavier static-image path for an interactive browser app.
- Rejected: The export must complete predictably from the GUI.

### Nine separate PNG downloads
- Pros: Reuses existing single-figure export logic.
- Cons: Does not satisfy the requirement for one offline-built matrix figure with one color legend.
- Rejected: User needs one formatted figure.

### One nine-subplot browser export
- Pros: Keeps title, labels, scenes, and colorbar in one Plotly figure.
- Cons: Sizes subplot axes instead of visible sphere content, producing tiny spheres floating in large empty cells.
- Rejected: This is the failure mode the export must avoid.

### Static precomputed matrix
- Pros: Fastest download.
- Cons: Would ignore current settings.
- Rejected: User needs the current non-matrix settings reflected before saving.

## Consequences
- Export quality follows the browser Plotly renderer already used by the app for each visual sprite, with final layout handled by the separate Python compositor.
- The generated sprite export spec is transient and stored in memory only.
- A click nonce is included so repeated exports with unchanged settings still trigger downloads.
- The exported matrix uses the current `Hide Ewald + angle helpers` state, while still fixing matrix columns and rows to the requested `theta_i` and `00L` comparison values. In hide-helper matrix exports, Ewald helper geometry and the broad continuous overlap-band helper surface are omitted in all cells so the sampled Bragg/Ewald overlap lines remain readable. Each sprite is re-centered on its Bragg sphere, and each final cell placement is scaled from the cell's own Bragg-only anchor bbox by default so every Bragg sphere fills most of its available cell region. Absolute inter-row `L` proportionality is retained only as an internal opt-in export-spec flag. All matrix sprites use a fixed centered reference camera rather than the live single-view camera. In helper-visible matrix exports, the `10°` and `15°` columns avoid stacked transparent helper surfaces by converting Ewald shell and broad overlap helper surfaces to wireframe traces, then add a high-contrast Bragg outline so Plotly's 3D export keeps the full Bragg sphere visible.
- Single-view camera state remains available for normal plot interactions and single-figure PNG export, but matrix export does not read or persist it.
- Each matrix export removes any previous off-screen matrix export host before rendering the next one, then purges and removes its own host after completion.
- The final matrix colorbar, title, and row/column labels are drawn once by the offline compositor and never participate in sprite cropping or sprite scaling.
- The old subplot matrix builder is retained only as private test/debug support and is not the active save path.
- The private Bragg-anchor export-spec field is retained for matrix composition only; it carries the same camera, coordinate transform, axis range, and hidden-axis layout as the visual sprite while excluding helper traces, labels, annotations, axes, and colorbars from the measurement image.
- Full diagnostic image dumps for raw sprites remain a future support feature; the downloaded bundle exposes cropped cell PNGs, JSON sidecars, embedded PNG metadata, and the offline compositor can draw debug boxes for cell, pasted-image, and scaled Bragg-bbox geometry.
- Browser-managed duplicate filenames remain outside the app boundary; the app does not delete files from the user's download directory.
- Rollback is a normal git revert of the feature commit; no migration or data cleanup is needed.
