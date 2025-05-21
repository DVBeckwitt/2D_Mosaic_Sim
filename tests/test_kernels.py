import numpy as np
from mosaic_sim.intensity import cap_intensity, belt_intensity
from mosaic_sim.geometry import sphere


def test_cap_normalisation():
    phi, theta = np.meshgrid(np.linspace(0, np.pi, 20), np.linspace(0, 2*np.pi, 40))
    Qx, Qy, Qz = sphere(1.0, phi, theta)
    I = cap_intensity(Qx, Qy, Qz, np.deg2rad(0.8))
    assert np.isclose(I.max(), 1.0)


def test_belt_handles_1d():
    t = np.linspace(0, 2*np.pi, 200)
    x = np.cos(t)
    y = np.zeros_like(t)
    z = np.sin(t)
    I = belt_intensity(x, y, z, 0.0, 0.0, 1.0, np.deg2rad(0.8), np.deg2rad(5), 0.5)
    assert np.isclose(I.max(), 1.0)
