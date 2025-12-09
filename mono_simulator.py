#!/usr/bin/env python3
"""Quick helper to evaluate PbI₂ Bragg angles with Cu-Kα radiation."""

import argparse
import math
from typing import Iterable

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


def _bragg_curve(two_theta_deg: float, angles: Iterable[float], fwhm: float = 0.3) -> list[float]:
    """Return normalized Gaussian intensities centered at ``two_theta_deg``."""

    sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
    return [math.exp(-0.5 * ((x - two_theta_deg) / sigma) ** 2) for x in angles]


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
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip showing a plot (always printed to stdout).",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=5.0,
        help="Half-width of 2θ span around the peak to display (degrees).",
    )
    args = parser.parse_args()

    d_spacing, two_theta = bragg_two_theta(args.h, args.k, args.l)
    print(
        f"(h k l)=({args.h} {args.k} {args.l}) in 2H PbI₂: "
        f"d = {d_spacing * 1e10:.4f} Å, 2θ = {two_theta:.2f}° "
        f"(λ = {CU_K_ALPHA_WAVELENGTH * 1e10:.4f} Å)"
    )

    if args.no_plot:
        return

    import matplotlib.pyplot as plt

    two_theta_values = [two_theta + delta for delta in _linspace(-args.window, args.window, 500)]
    intensity = _bragg_curve(two_theta, two_theta_values)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(two_theta_values, intensity, color="tab:blue")
    ax.axvline(two_theta, color="tab:red", linestyle="--", linewidth=1, label=f"2θ = {two_theta:.2f}°")
    ax.set_xlabel("2θ (degrees)")
    ax.set_ylabel("Relative intensity (a.u.)")
    ax.set_title(f"PbI₂ (h k l)=({args.h} {args.k} {args.l}) using Cu-Kα")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)

    plt.tight_layout()
    plt.show()


def _linspace(start: float, stop: float, num: int) -> list[float]:
    """Simple linspace to avoid numpy dependency."""

    if num < 2:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


if __name__ == "__main__":
    main()
