import numpy as np
import jax.numpy as jnp
import jaxopt
import healpy as hp
import matplotlib.pyplot as plt
from jaxsot.mockdata.testdata import static_singleband

np.random.seed(1)
nside=8
npix=hp.nside2npix(nside)
lc,W,mmap_input=static_singleband(nside)

def objective(params,lam):
    residuals=lc - jnp.dot(W,params)
    return jnp.mean(residuals ** 2) + lam*lam * jnp.dot(params,params)

gd = jaxopt.GradientDescent(fun=objective, maxiter=500)
res = gd.run(init_params=np.random.normal(0.0,1.0,npix), lam=0.1)
params, state = res
hp.mollview(params, flip="geo")
plt.show()

