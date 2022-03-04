import numpy as np
import jax.numpy as jnp
import jaxopt
import healpy as hp
from jaxsot.mockdata.testdata import static_singleband
from jaxsot.core.neighbor import calc_neighbor_weightmatrix
import pytest

def test_l1tsv():
    np.random.seed(1)
    nside=8
    npix=hp.nside2npix(nside)
    lc,W, mmap_input=static_singleband(nside)
    wtsv, _ = calc_neighbor_weightmatrix(nside)    
    wtsv, _ = calc_neighbor_weightmatrix(nside)
    
    def objective(params,lamtsv):
        residuals=lc - jnp.dot(W,params)
        regtsv = lamtsv * jnp.dot(params, jnp.dot(wtsv,params))
        return jnp.mean(residuals ** 2) + regtsv

    pg = jaxopt.ProximalGradient(fun=objective, prox=jaxopt.prox.prox_non_negative_lasso, maxiter=5000)
    res = pg.run(init_params=np.random.normal(0.0,1.0,npix),  hyperparams_prox=0.01, lamtsv=0.01)
    params, state = res
    refs=304.90924
    assert np.abs(np.sum(params)-refs)<1.e-16

if __name__=="__main__":
    test_l1tsv()
