import numpy as np
from mosaic_sim.intensity import cap_intensity, belt_intensity
from mosaic_sim.geometry import sphere, intersection_circle



def test_cap_normalisation():
    phi, theta = np.meshgrid(np.linspace(0, np.pi, 20), np.linspace(0, 2*np.pi, 40))
    Qx, Qy, Qz = sphere(1.0, phi, theta)
    I = cap_intensity(Qx, Qy, Qz, np.deg2rad(0.8))
    assert np.isclose(I.max(), 1.0)

def test_belt_normalisation_1d():
    x, y, z = intersection_circle(1.0, 1.0, 1.0)
    I = belt_intensity(x, y, z, 0.0, 0.0, 1.0, np.deg2rad(0.8), np.deg2rad(5), 0.5)
    assert np.isclose(I.max(), 1.0)

