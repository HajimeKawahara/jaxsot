"""test for map."""

import pytest
import numpy as np
from jaxsot.io.earth import binarymap, multibandmap
from jaxsot.core.map import rotating_map

def test_rotating_map_binary():
    mmap = binarymap(nside=16, show=False)
    Porb = 40.0
    N = 1024
    obst = np.linspace(0.0, Porb, N)
    rotM = rotating_map(mmap, obst, rotthetamax = np.pi / 2.0)
    assert np.sum(rotM) == pytest.approx(1017666.1972822)

def test_rotating_map_multiband():
    _, A, _ = multibandmap(show=False)
    Porb = 40.0
    N = 1024
    obst = np.linspace(0.0, Porb, N)
    rotA = rotating_map(A, obst, rotthetamax = np.pi / 2.0)
    assert np.sum(rotA) == pytest.approx(12582912.00000001)

if __name__ == '__main__':
    test_rotating_map_binary()
    test_rotating_map_multiband()
