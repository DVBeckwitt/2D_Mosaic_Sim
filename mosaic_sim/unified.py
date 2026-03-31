"""Compatibility wrapper for the unified mosaic simulator."""

from .unified_app import (
    DEFAULT_HOST,
    DEFAULT_MODE,
    DEFAULT_PORT,
    SIMULATION_SPECS,
    build_unified_app,
    build_unified_figure,
    main,
    parse_args,
)

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_MODE",
    "DEFAULT_PORT",
    "SIMULATION_SPECS",
    "build_unified_app",
    "build_unified_figure",
    "main",
    "parse_args",
]
