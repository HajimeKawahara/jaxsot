import numpy as np
import jax.numpy as jnp
from jaxsot.core.weight import comp_weight, comp_omega
from jaxsot.core.lc import gen_lightcurve
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

def objective(params,lam):
    residuals=lc - jnp.dot(W,params)
    return jnp.mean(residuals ** 2) + lam*lam * jnp.dot(params,params)

gd = jaxopt.GradientDescent(fun=objective, maxiter=500)
res = gd.run(init_params=np.random.normal(0.0,1.0,len(mmap)), lam=0.1)
params, state = res
hp.mollview(params)
plt.show()

