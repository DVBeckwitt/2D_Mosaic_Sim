#!/usr/bin/env python3
"""Command-line entry for the detector simulation.

This thin wrapper allows the example to be executed directly or installed as
the ``mosaic-detector`` console script.
"""
from mosaic_sim.detector import main

if __name__ == "__main__":
    main()
