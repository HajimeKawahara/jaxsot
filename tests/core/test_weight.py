"""test for weight."""

import pytest
import numpy as np
import jax.numpy as jnp
from jaxsot.core.weight import comp_weight, comp_omega


def test_comp_weight():
    nside = 16
    inc = 0.0
    Thetaeq = np.pi
    zeta = np.pi/3.0
    Pspin = 23.9344699/24.0
    wspin = 2*np.pi/Pspin
    Porb = 40.0
    worb = 2.*np.pi/Porb
    N = 1024
    obst = jnp.linspace(0.0, Porb, N)
    Thetav = worb*obst
    Phiv = jnp.mod(wspin*obst, 2*np.pi)
    omega = comp_omega(nside)
    WI, WV = comp_weight(nside, zeta, inc, Thetaeq, Thetav, Phiv, omega)
    assert np.abs(np.sum(WI*WV)-166871) < 1.0


if __name__ == '__main__':
    test_comp_weight()
