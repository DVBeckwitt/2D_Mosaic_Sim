# mosaic_sim

Utilities for visualising 2‑D oriented powder X‑ray diffraction conditions.
The code produces interactive Plotly figures showing how the Bragg sphere
rotates relative to the Ewald sphere and how this maps onto a detector.

Three command line scripts are provided:

- `simulate_detector.py` – static 3‑panel detector simulation
- `simulate_mosaic.py`  – dynamic Bragg‑sphere animation
- `simulate_cylinder.py` – static Ewald sphere with cylinder slider prompting
  for the Miller indices to update the figure and optionally showing the
  cylinder/Bragg intersection line

Both scripts rely on the shared `mosaic_sim` package which exposes physical
constants, geometry helpers and intensity kernels.

## Installation

The project targets Python 3.11 or newer.  After cloning the repository install
the dependencies with `pip`:

```bash
pip install -e .
```

## Usage

Run the examples directly:

```bash
python simulate_detector.py    # Open the static detector figure
python simulate_mosaic.py      # Launch the interactive animation
python simulate_cylinder.py    # Cylinder intersection with slider
python simulate_cylinder_live.py  # Update H, K, L interactively
python simulate_screen.py      # Pattern on a distant flat detector
```

When installed as a package the scripts are available as console entry points
`mosaic-detector`, `mosaic-rocking`, `mosaic-cylinder`, `mosaic-cylinder-live` and `mosaic-screen`.
