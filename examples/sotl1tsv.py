import numpy as np
import jax.numpy as jnp
from jaxsot.core.weight import comp_weight, comp_omega
from jaxsot.core.lc import gen_lightcurve
from jaxsot.core.neighbor import calc_neighbor_weightmatrix
from jaxsot.io.earth import binarymap
import jaxopt
import healpy as hp
import matplotlib.pyplot as plt

nside=8
mmap=binarymap(nside=nside,show=False)
inc=0.0
Thetaeq=np.pi
zeta=np.pi/3.0
Pspin=23.9344699/24.0
wspin=2*np.pi/Pspin
Porb=40.0
worb=2.*np.pi/Porb
N=1024
obst=np.linspace(0.0,Porb,N)
Thetav=worb*obst
Phiv=np.mod(wspin*obst,2*np.pi)
omega=comp_omega(nside)
WI,WV=comp_weight(nside,zeta,inc,Thetaeq,Thetav,Phiv,omega)
W=jnp.array(WI*WV)
lc=gen_lightcurve(W,mmap,0.1)
wtsv, _ = calc_neighbor_weightmatrix(nside)

def objective(params,lamtsv):
    residuals=lc - jnp.dot(W,params)
    regtsv = lamtsv * jnp.dot(params, jnp.dot(wtsv,params))
    return jnp.mean(residuals ** 2) + regtsv

#pg = jaxopt.ProximalGradient(fun=objective, prox=jaxopt.prox.prox_lasso, maxiter=5000)
pg = jaxopt.ProximalGradient(fun=objective, prox=jaxopt.prox.prox_non_negative_lasso, maxiter=5000)
res = pg.run(init_params=np.random.normal(0.0,1.0,len(mmap)),  hyperparams_prox=0.01, lamtsv=0.01)
params, state = res
hp.mollview(params)
plt.show()

