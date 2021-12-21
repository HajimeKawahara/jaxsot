import numpy as np


def gen_lightcurve(W,mmap,sigma_relative):
    """light curve generator

    """
    lc=np.dot(W,mmap)
    sigma=np.mean(lc)*sigma_relative
    noise=np.random.normal(0.0,sigma,len(lc))
    lc=lc+noise
    return lc


if __name__=="__main__":
    from jaxsot.core.weight import comp_weight, comp_omega
    from jaxsot.io.earth import binarymap

    mmap=binarymap(nside=16,show=False)

    nside=16
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
    W=WI*WV
    lc=gen_lightcurve(W,mmap,0.0)
    print(np.sum(lc))
    import matplotlib.pyplot as plt
    plt.plot(obst,lc)
    plt.show()
