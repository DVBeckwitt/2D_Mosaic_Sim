import numpy as np
from mosaic_sim.intensity import cap_intensity, belt_intensity, mosaic_intensity
from mosaic_sim.geometry import sphere


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
