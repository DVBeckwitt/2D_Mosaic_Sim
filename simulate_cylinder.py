#!/usr/bin/env python3
"""Command-line entry for the static cylinder figure.

This provides an executable wrapper similar to the existing examples and can
be installed as a console script.
"""
import argparse
import math

from mosaic_sim.cylinder import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static cylinder figure")
    parser.add_argument("H", type=int, nargs="?", default=0,
                        help="Miller index H (default: 0)")
    parser.add_argument("K", type=int, nargs="?", default=0,
                        help="Miller index K (default: 0)")
    parser.add_argument("L", type=int, nargs="?", default=12,
                        help="Miller index L (default: 12)")
    parser.add_argument("--sigma", type=float, default=0.8,
                        help="Mosaic spread σ in degrees (default: 0.8)")
    parser.add_argument("--gamma", type=float, default=5.0,
                        help="In-plane mosaic spread γ in degrees (default: 5.0)")
    parser.add_argument("--eta", type=float, default=0.5,
                        help="Lorentzian/Gaussian mixing factor η (default: 0.5)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.H, args.K, args.L,
         math.radians(args.sigma),
         math.radians(args.gamma),
         args.eta)
