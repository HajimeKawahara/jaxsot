"""testdata module makes fiducial testdata sets from a mock Earth.

"""

import numpy as np
import jax.numpy as jnp
from jaxsot.core.weight import comp_weight, comp_omega
from jaxsot.core.lc import gen_lightcurve
from jaxsot.mockdata.earth import binarymap

def fiducial_geometry():
    """provide a fiducial geometry set

    Returns:
       zeta,inc,Thetaeq,Thetav,Phiv,omega

    """
    N=1024    
    inc=0.0
    Thetaeq=np.pi
    zeta=np.pi/3.0
    Pspin=23.9344699/24.0
    wspin=2*np.pi/Pspin
    Porb=40.0
    worb=2.*np.pi/Porb
    obst=np.linspace(0.0,Porb,N)
    Thetav=worb*obst
    Phiv=np.mod(wspin*obst,2*np.pi)
    return zeta,inc,Thetaeq,Thetav,Phiv


def static_singleband(nside):
    """generate test data for a single band observation of a static geography

    Args:
        nside: nside of Healpix

    Returns: 
        test light curve, weight matrix, input static map
 
    """
    mmap=binarymap(nside=nside,show=False)
    zeta,inc,Thetaeq,Thetav,Phiv=fiducial_geometry()
    omega=comp_omega(nside)
    WI,WV=comp_weight(nside,zeta,inc,Thetaeq,Thetav,Phiv,omega)
    W=jnp.array(WI*WV)
    lc=gen_lightcurve(W,mmap,0.1)
    return lc, W, mmap

def dynamic_singleband(nside):
    """generate test data for a single band observation of a rotating geography

    Args:
        nside: nside of Healpix

    Returns: 
        test light curve, weight matrix, input dynamic map
 
    """
    mmap=binarymap(nside=nside,show=False)
    zeta,inc,Thetaeq,Thetav,Phiv=fiducial_geometry()
    WI,WV=comp_weight(nside,zeta,inc,Thetaeq,Thetav,Phiv,omega)
    W=jnp.array(WI*WV)
    M=gendymap_rotation(mmap,obst,rotthetamax=np.pi/2.0)
    lc=gen_dynamic_lightcurve(W,M,0.1)

    return lc, W, M
