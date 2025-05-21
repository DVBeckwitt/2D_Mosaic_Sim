# mosaic_sim

This package provides utilities for simulating rocking curves and mosaic
intensity distributions for Bi₂Se₃ crystals.  Two command line scripts are
included:

- `simulate_detector.py` – 3‑panel rocking‑curve figure
- `simulate_mosaic.py`  – dynamic Bragg‑sphere animation

Both rely on the shared `mosaic_sim` package which exposes physical constants,
geometry helpers and intensity kernels.
