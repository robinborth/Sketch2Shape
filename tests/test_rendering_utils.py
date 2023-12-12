import numpy as np

from lib.rendering.utils import elev_azim_to_unit_shere


def test_elev_azim_to_unit_shere_azim_0():
    point = elev_azim_to_unit_shere(elev=0, azim=0)
    assert np.allclose(point, np.array([0, 0, 1]))


def test_elev_azim_to_unit_shere_azim_90():
    point = elev_azim_to_unit_shere(elev=0, azim=90)
    assert np.allclose(point, np.array([1, 0, 0]))


def test_elev_azim_to_unit_shere_azim_180():
    point = elev_azim_to_unit_shere(elev=0, azim=180)
    assert np.allclose(point, np.array([0, 0, -1]))


def test_elev_azim_to_unit_shere_azim_270():
    point = elev_azim_to_unit_shere(elev=0, azim=270)
    assert np.allclose(point, np.array([-1, 0, 0]))


def test_elev_azim_to_unit_shere_azim_360():
    point = elev_azim_to_unit_shere(elev=0, azim=360)
    assert np.allclose(point, np.array([0, 0, 1]))


def test_elev_azim_to_unit_shere_elev_0():
    point = elev_azim_to_unit_shere(elev=0, azim=0)
    assert np.allclose(point, np.array([0, 0, 1]))


def test_elev_azim_to_unit_shere_elev_90():
    point = elev_azim_to_unit_shere(elev=90, azim=0)
    assert np.allclose(point, np.array([0, 1, 0]))


def test_elev_azim_to_unit_shere_elev_180():
    point = elev_azim_to_unit_shere(elev=180, azim=0)
    assert np.allclose(point, np.array([0, 0, -1]))


def test_elev_azim_to_unit_shere_elev_270():
    point = elev_azim_to_unit_shere(elev=270, azim=0)
    assert np.allclose(point, np.array([0, -1, 0]))


def test_elev_azim_to_unit_shere_elev_360():
    point = elev_azim_to_unit_shere(elev=360, azim=0)
    assert np.allclose(point, np.array([0, 0, 1]))
