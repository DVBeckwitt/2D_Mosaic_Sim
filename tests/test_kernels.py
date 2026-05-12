from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from mosaic_sim.constants import K_MAG
from mosaic_sim.intensity import cap_intensity, belt_intensity, mosaic_intensity
from mosaic_sim.geometry import (
    ewald_bandwidth_k_bounds,
    ewald_bandwidth_layers,
    normalize_wavelength_bandwidth_pct,
    sphere,
)


def test_cap_normalisation():
    phi, theta = np.meshgrid(np.linspace(0, np.pi, 20), np.linspace(0, 2*np.pi, 40))
    Qx, Qy, Qz = sphere(1.0, phi, theta)
    I = cap_intensity(Qx, Qy, Qz, np.deg2rad(0.8), np.deg2rad(5), 0.5)
    assert np.isclose(I.max(), 1.0)

def test_belt_handles_1d():
    # 1-D ring of points should not raise and should be normalised
    t = np.linspace(0, 2*np.pi, 200)
    Qx = np.cos(t)
    Qy = np.sin(t)
    Qz = np.zeros_like(t)
    I = belt_intensity(Qx, Qy, Qz, 0.0, 0.0, 1.0, np.deg2rad(0.8), np.deg2rad(5), 0.5)

    assert np.isclose(I.max(), 1.0)


def test_mosaic_intensity_switches_profile_for_hk():
    phi, theta = np.meshgrid(np.linspace(0, np.pi, 10), np.linspace(0, 2*np.pi, 20))
    Qx, Qy, Qz = sphere(1.0, phi, theta)

    I_expected = belt_intensity(Qx, Qy, Qz, 1, 1, 0, np.deg2rad(0.8), np.deg2rad(5), 0.5)
    I_actual = mosaic_intensity(Qx, Qy, Qz, 1, 1, 0, np.deg2rad(0.8), np.deg2rad(5), 0.5)

    assert np.allclose(I_expected, I_actual)


def test_normalize_wavelength_bandwidth_pct_uses_default_and_accepts_valid_values():
    assert normalize_wavelength_bandwidth_pct(None, default=0.25) == pytest.approx(0.25)
    assert normalize_wavelength_bandwidth_pct(1) == pytest.approx(1.0)
    assert normalize_wavelength_bandwidth_pct(100.0) == pytest.approx(100.0)


@pytest.mark.parametrize("value", [-0.01, 200.0, float("nan"), float("inf")])
def test_normalize_wavelength_bandwidth_pct_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="wavelength_bandwidth_pct"):
        normalize_wavelength_bandwidth_pct(value)


def test_geometry_module_does_not_import_plotly_coupled_common_helpers():
    geometry_path = Path(__file__).resolve().parents[1] / "mosaic_sim" / "geometry.py"
    code = """
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import types

geometry_path = Path(sys.argv[1])
package = types.ModuleType("mosaic_sim")
package.__path__ = [str(geometry_path.parent)]
package.__package__ = "mosaic_sim"
package.__spec__ = importlib.machinery.ModuleSpec("mosaic_sim", loader=None, is_package=True)
sys.modules["mosaic_sim"] = package

spec = importlib.util.spec_from_file_location("mosaic_sim.geometry", geometry_path)
module = importlib.util.module_from_spec(spec)
sys.modules["mosaic_sim.geometry"] = module
spec.loader.exec_module(module)

print(any(name == "plotly" or name.startswith("plotly.") for name in sys.modules))
"""

    output = subprocess.check_output(
        [sys.executable, "-c", code, str(geometry_path)],
        text=True,
    ).strip()

    assert output == "False"


def test_ewald_bandwidth_layers_zero_bandwidth_returns_central_layer():
    layers = ewald_bandwidth_layers(K_MAG, 0.0)

    assert len(layers) == 1
    assert layers[0].relative_wavelength_offset == pytest.approx(0.0)
    assert layers[0].k_mag == pytest.approx(K_MAG)
    assert layers[0].opacity == pytest.approx(0.30)


def test_ewald_bandwidth_layers_sample_odd_stack_with_central_max_opacity():
    layers = ewald_bandwidth_layers(K_MAG, 1.0, layer_count=6)
    offsets = [layer.relative_wavelength_offset for layer in layers]
    opacities = [layer.opacity for layer in layers]
    center = len(layers) // 2

    assert len(layers) == 7
    assert offsets[0] == pytest.approx(-0.005)
    assert offsets[-1] == pytest.approx(0.005)
    assert offsets[center] == pytest.approx(0.0)
    assert layers[center].k_mag == pytest.approx(K_MAG)
    assert opacities[center] == pytest.approx(max(opacities))
    assert opacities[0] < opacities[1] < opacities[center]
    assert opacities[-1] < opacities[-2] < opacities[center]
    for layer in layers:
        assert layer.k_mag == pytest.approx(K_MAG / (1.0 + layer.relative_wavelength_offset))


def test_ewald_bandwidth_k_bounds_zero_bandwidth_returns_central_radius():
    k_min, k_max = ewald_bandwidth_k_bounds(K_MAG, 0.0)

    assert k_min == pytest.approx(K_MAG)
    assert k_max == pytest.approx(K_MAG)


def test_ewald_bandwidth_k_bounds_returns_ordered_shell_radii():
    k_min, k_max = ewald_bandwidth_k_bounds(K_MAG, 5.0)

    assert k_min == pytest.approx(K_MAG / 1.025)
    assert k_max == pytest.approx(K_MAG / 0.975)
    assert k_min < K_MAG < k_max


def test_ewald_bandwidth_k_bounds_accepts_100_percent_bandwidth():
    k_min, k_max = ewald_bandwidth_k_bounds(K_MAG, 100.0)

    assert k_min == pytest.approx(K_MAG / 1.5)
    assert k_max == pytest.approx(K_MAG / 0.5)
    assert k_min < K_MAG < k_max
