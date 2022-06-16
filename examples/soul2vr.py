import numpy as np
import jax.numpy as jnp
from jaxsot.core.weight import comp_weight, comp_omega
from jaxsot.core.lc import gen_lightcurve
from jaxsot.core.block_coordinate_descent import opt_map_l2, opt_ref_vr
from jaxsot.io.earth import binarymap
from jaxsot.io.earth import multibandmap
import healpy as hp
import matplotlib.pyplot as plt
import tqdm

np.random.seed(34)

mmap, Ainit, Xinit = multibandmap(show=False)
npix = mmap.shape[0]
nside = hp.npix2nside(npix)

inc = 0.0
Thetaeq = np.pi
zeta = np.pi / 3.0
Pspin = 23.9344699 / 24.0
wspin = 2 * np.pi / Pspin
Porb = 40.0
worb = 2. * np.pi / Porb
N = 1024
obst = np.linspace(0.0, Porb, N)

Thetav = worb * obst
Phiv = np.mod(wspin * obst, 2 * np.pi)
omega = comp_omega(nside)

WI, WV = comp_weight(nside, zeta, inc, Thetaeq, Thetav, Phiv, omega)
W = np.array(WI * WV)
lc = gen_lightcurve(W, mmap, 0.1)

nside = 16
npix = hp.nside2npix(nside)
omega = comp_omega(nside)
WI, WV = comp_weight(nside, zeta, inc, Thetaeq, Thetav, Phiv, omega)
W = jnp.array(WI * WV)
Nk = 3

A0 = np.random.rand(npix, Nk)
X0 = np.random.rand(Nk, np.shape(lc)[1])
print(A0.shape, X0.shape)

#Ni=np.shape(lcall)[0]
Nl = np.shape(lc)[1]
#Nk=np.shape(A0)[1]
Nj = np.shape(A0)[0]
A = np.copy(A0)
X = np.copy(X0)
Y = np.copy(lc)

lamA = 10**(-1)
lamX = 10**(2)
maxiter = 100

for i in tqdm.tqdm(range(0, maxiter)):
    for k in range(0, Nk):
        X[k, :] = opt_ref_vr(k, Y, W, A, X, lamX)
        A[:, k] = opt_map_l2(k, Y, W, A, X, lamA)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(221)
ax.plot(X.T)

for k in range(Nk):
    hp.mollview(A[:, k], flip="geo", sub=(2, 3, k + 4))
plt.show()
