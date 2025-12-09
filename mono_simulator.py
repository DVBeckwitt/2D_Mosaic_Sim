#!/usr/bin/env python3
"""Quick helper to evaluate PbI₂ Bragg angles with Cu-Kα radiation."""

import argparse
import math

from mosaic_sim.constants import (
    CU_K_ALPHA_WAVELENGTH,
    PBI2_2H_A_HEX,
    PBI2_2H_C_HEX,
    d_hex,
)


def bragg_two_theta(h: int, k: int, l: int) -> tuple[float, float]:
    """Return (d, 2θ°) for the given (h k l) in 2H PbI₂."""
    d_spacing = d_hex(h, k, l, a=PBI2_2H_A_HEX, c=PBI2_2H_C_HEX)
    theta = math.asin(CU_K_ALPHA_WAVELENGTH / (2 * d_spacing))
    return d_spacing, math.degrees(2 * theta)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the d-spacing and 2θ position of a PbI₂ (h k l) reflection "
            "for Cu-Kα radiation."
        )
    )
    parser.add_argument("h", type=int, nargs="?", default=0, help="H Miller index")
    parser.add_argument("k", type=int, nargs="?", default=0, help="K Miller index")
    parser.add_argument("l", type=int, nargs="?", default=2, help="L Miller index")
    args = parser.parse_args()

    d_spacing, two_theta = bragg_two_theta(args.h, args.k, args.l)
    print(
        f"(h k l)=({args.h} {args.k} {args.l}) in 2H PbI₂: "
        f"d = {d_spacing * 1e10:.4f} Å, 2θ = {two_theta:.2f}° "
        f"(λ = {CU_K_ALPHA_WAVELENGTH * 1e10:.4f} Å)"
    )


if __name__ == "__main__":
    main()
