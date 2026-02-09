# 2D Mosaic Simulation

This repository collects small utilities for visualising diffraction from two‑dimensionally oriented powders.
Each tool shows how reciprocal‑lattice rods from a layered material intersect a three‑dimensional Bragg sphere,
rotate relative to the Ewald construction and map onto an area detector.  The visualisations build intuition for
the hybrid ring–cap patterns produced by thin films and other samples with a common out‑of‑plane axis but random
in‑plane orientation.

Interactive figures are implemented with Plotly and expose physical constants, geometry helpers and intensity
kernels through the shared `mosaic_sim` package.

## Command‑line scripts

The following scripts demonstrate different aspects of the geometry.  Run them with `python <script>` from the
project root or via the console entry points installed with the package.

- `simulate_detector.py` – static three‑panel view of reciprocal space, Bragg sphere and detector
- `simulate_mosaic.py` – animated Bragg sphere showing rotation and mosaic spread
- `fibrous_simulator.py` – Ewald sphere with a cylindrical reciprocal‑space rod that updates with Miller indices
- `mono_simulator.py` – minimal Ewald-sphere view with adjustable incident angle
- `simulate_cylinder_pdf.py` – interactive probability‑density plot for a tilted rod
- `simulate_screen.py` – diffraction pattern on a distant flat detector

## Installation

The project targets Python 3.11 or newer.  After cloning the repository install the dependencies with `pip`:

```bash
pip install -e .
```

## Usage

Run the examples directly from the source tree:

```bash
python simulate_detector.py         # Open the static detector figure
python simulate_mosaic.py           # Launch the interactive animation
python fibrous_simulator.py         # Cylinder intersection with slider
python mono_simulator.py            # Single Ewald sphere (balanced fast default)
python mono_simulator.py --full-quality  # Full-resolution mono view
python simulate_cylinder_pdf.py     # Explore orientation PDF interactively
python simulate_screen.py           # Pattern on a distant flat detector
```

When installed as a package the scripts are also available as console entry points
`mosaic-detector`, `mosaic-rocking`, `mosaic-cylinder`, `mosaic-cylinder-pdf` and `mosaic-screen`.
