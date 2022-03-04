import jax.numpy as jnp
from jax import jit

@jit
def Kw_dysot(W,KS,KT,alpha):
    """compute weighted kernel

    Args:
       W: weight
       KS: spatial kernel (positive semidefinite)
       KT: temporal kernel (positive semidefinite)
       alpha: norm

    Return:
       weighted kernel

    """
    return alpha*KT*(W@KS@W.T)

@jit
def mean_dysotgp(W,KS,KT,alpha,lc,Pid):
    """mean value of a dynamic SOT using gaussian process

    Args:
       W: weight
       KS: spatial kernel (positive semidefinite)
       KT: temporal kernel (positive semidefinite)
       alpha: norm
       lc: lightcurve       
       Pid: data precision matrix (inverse of the data covariance)

    Returns:
       mean dynamic map, Kw

    """
    Ni,Nj=jnp.shape(W)
    Kw=Kw_dysot(W,KS,KT,alpha)
    IKw=jnp.eye(Ni)+Pid@Kw
    Xlc=jnp.linalg.solve(IKw,Pid@lc)
    Aast=alpha*KT@(W.T*Xlc).T@KS
    return Aast, Kw

@jit
def covariance_dysotgp(Kw,Sigmad):
    """P of the posterior given theta and g

    Args:
       Kw: weighted kernel (positive semidefinite)
       Sigmad: the data covariance (positive semidefinite)

    Returns:
       P = (Sigmad + Kw)^-1

    """
    P=jnp.linalg.inv(Sigmad+Kw)
    return P
