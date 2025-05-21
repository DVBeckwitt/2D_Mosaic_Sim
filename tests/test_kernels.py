import numpy as np
from mosaic_sim.intensity import cap_intensity, belt_intensity
from mosaic_sim.geometry import sphere


def test_cap_normalisation():
    phi, theta = np.meshgrid(np.linspace(0, np.pi, 20), np.linspace(0, 2*np.pi, 40))
    Qx, Qy, Qz = sphere(1.0, phi, theta)
    I = cap_intensity(Qx, Qy, Qz, np.deg2rad(0.8))
    assert np.isclose(I.max(), 1.0)
