import numpy as np
import jax.numpy as jnp
from jaxsot.core.block_coordinate_descent import opt_ref_vr, opt_map_l1tsv
from jaxsot.core.testdata import static_multiband
import healpy as hp
import matplotlib.pyplot as plt
import tqdm

np.random.seed(34)
lc, W = static_multiband()
npix=W.shape[1]
Nk = 3

A0 = np.random.rand(npix, Nk)
X0 = np.random.rand(Nk, np.shape(lc)[1])
A = np.copy(A0)
X = np.copy(X0)
Y = np.copy(lc)

laml1 = 10 ** (-3.5)
lamtsv = 10 ** (-4)
lamX = 10 ** (2)
maxiter = 100

for i in tqdm.tqdm(range(0, maxiter)):
    for k in range(0, Nk):
        X[k, :] = opt_ref_vr(k, Y, W, A, X, lamX)
        A[:, k] = opt_map_l1tsv(k, Y, W, A, X, laml1, lamtsv)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(221)
ax.plot(X.T)

for k in range(Nk):
    hp.mollview(A[:, k], flip="geo", sub=(2, 3, k + 4))
plt.show()
