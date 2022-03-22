from email.base64mime import body_decode
import numpy as np
import jax.numpy as jnp
from jaxsot.core.weight import comp_weight, comp_omega
from jaxsot.core.lc import gen_lightcurve
from jaxsot.io.earth import binarymap
from jaxsot.io.earth import multibandmap
import jaxopt
import healpy as hp
import matplotlib.pyplot as plt

np.random.seed(34)

mmap,Ainit,Xinit=multibandmap(show=False)
npix=mmap.shape[0]
nside=hp.npix2nside(npix)

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
W=np.array(WI*WV)
lc=gen_lightcurve(W,mmap,0.1)

nside=16
npix=hp.nside2npix(nside)
omega=comp_omega(nside)
WI,WV=comp_weight(nside,zeta,inc,Thetaeq,Thetav,Phiv,omega)
W=jnp.array(WI*WV)
Nk=3

A0=np.random.rand(npix,Nk)
X0=np.random.rand(Nk,np.shape(lc)[1])
print(A0.shape,X0.shape)

#Ni=np.shape(lcall)[0]
Nl=np.shape(lc)[1]
#Nk=np.shape(A0)[1]
Nj=np.shape(A0)[0]
A=np.copy(A0)
X=np.copy(X0)
Y=np.copy(lc)

lamA=10**(2)
lamX=10**(2)
maxiter=10

for i in range(0,maxiter):
    print('i=',i)
    for k in range(0,Nk):

        # xk
        AX=np.dot(np.delete(A,obj=k,axis=1),np.delete(X,obj=k,axis=0))
        Delta=Y-np.dot(W,AX)
        ak=A[:,k]
        Wa=np.dot(W,ak)
        W_x=np.dot(Wa,Wa)*np.eye(Nl)
        bx=-np.dot(np.dot(Delta.T,W),ak)
        Xminus = np.delete(X,obj=k,axis=0)
        XXTinverse=np.linalg.inv(np.dot(Xminus,Xminus.T))
        K=np.eye(Nl) - np.dot(np.dot(Xminus.T,XXTinverse),Xminus)
        K=K*np.linalg.det(np.dot(Xminus,Xminus.T))*lamX
        
        QP_obj_xk = lambda params: 0.5*jnp.dot(params, jnp.dot((W_x+K),params)) + jnp.dot(bx,params)
        pg = jaxopt.ProjectedGradient(fun=QP_obj_xk, projection=jaxopt.projection.projection_non_negative,maxiter=100)
        res = pg.run(init_params=jnp.array(X[k,:]))

        # ak
        xk=X[k,:]
        W_a=(np.dot(xk,xk))*(np.dot(W.T,W))
        b=-np.dot(np.dot(W.T,Delta),xk)
        T_a=lamA*np.eye(Nj)

        QP_obj_ak = lambda params: 0.5*jnp.dot(params, jnp.dot((W_a+T_a),params)) + jnp.dot(b,params)
        pg = jaxopt.ProjectedGradient(fun=QP_obj_ak, projection=jaxopt.projection.projection_non_negative,maxiter=100)
        res = pg.run(init_params=jnp.array(A[:,k]))
        A[:,k]=res.params

        print('k=',k)

fig=plt.figure(figsize=(10,6))
ax=fig.add_subplot(211)
ax.plot(X.T)

for k in range(Nk):
    hp.mollview(A[:,k],flip="geo",sub=(2,3,k+4))
plt.show()

