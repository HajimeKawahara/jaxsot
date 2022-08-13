"""test for lc."""

import pytest
import numpy as np
import healpy as hp
from jaxsot.core.weight import comp_weight, comp_omega
from jaxsot.core.lc import gen_lightcurve, gen_dynamic_lightcurve
from jaxsot.io.earth import binarymap, multibandmap


def test_lc():
    mmap = binarymap(nside=16, show=False)
    nside = 16
    inc = 0.0
    Thetaeq = np.pi
    zeta = np.pi/3.0
    Pspin = 23.9344699/24.0
    wspin = 2*np.pi/Pspin
    Porb = 40.0
    worb = 2.*np.pi/Porb
    N = 1024
    obst = np.linspace(0.0, Porb, N)
    Thetav = worb*obst
    Phiv = np.mod(wspin*obst, 2*np.pi)
    omega = comp_omega(nside)
    WI, WV = comp_weight(nside, zeta, inc, Thetaeq, Thetav, Phiv, omega)
    W = WI*WV
    lc = gen_lightcurve(W, mmap, 0.0)
    assert np.abs(np.sum(lc)-63856.86) < 0.1


def test_dy_lc():
    np.random.seed(1)
    mmap, Ainit, Xinit = multibandmap(show=False)
    npix = mmap.shape[0]
    nside = hp.npix2nside(npix)
    inc = 0.0
    Thetaeq = np.pi
    zeta = np.pi/3.0
    Pspin = 23.9344699/24.0
    wspin = 2 * np.pi / Pspin
    Porb = 40.0
    worb = 2. * np.pi / Porb
    N = 1024
    obst = np.linspace(0.0, Porb, N)
    Thetav = worb * obst
    Phiv = np.mod(wspin * obst, 2 * np.pi)
    omega = comp_omega(nside)

    WI, WV = comp_weight(nside, zeta, inc, Thetaeq, Thetav, Phiv, omega)
    W = np.array(WI * WV)
    lc = gen_dynamic_lightcurve(W, Ainit, Xinit, obst, 0.1)
    assert np.sum(lc) == pytest.approx(1019453.1112199578)

if __name__ == '__main__':
    test_lc()
    test_dy_lc()
