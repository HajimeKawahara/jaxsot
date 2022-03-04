import jax.numpy as jnp
from jax import jit

@jit
def mean_stsotgp(W,KS,lc,Pid):
    """
    Args:
       W: weight
       KS: spatial kernel (positive semidefinite)
       lc: lightcurve       
       Pid: data precision matrix (inverse of the data covariance)

    Returns:
       mean static map, Kw

    """
    Kw=(W@KS@W.T)
    Nn=np.shape(Kw)[0]
    IKw=jnp.eye(Nn)+Pid@Kw        
    Xlc=jnp.linalg.solve(IKw,Pid@lc,assume_a="pos")
    return KS@W.T@Xlc, Kw


@jit
def covariance_sotgp(Kw,Sigmad):
    """P of the posterior given theta and g

    Args:
       Kw: weighted kernel (positive semidefinite)
       Sigmad: the data covariance (positive semidefinite)

    Returns:
       P = (Sigmad + Kw)^-1

    """
    P=jnp.linalg.inv(Sigmad+Kw)
    return P
