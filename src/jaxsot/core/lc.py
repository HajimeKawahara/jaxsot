import numpy as np
from jaxsot.core.map import rotating_map


def gen_lightcurve(W, mmap, sigma_relative):
   """light curve generator

   Args:
      W: geometric kernel
      mmap: map vector (binary) or matrix (multiband)
      sigma_relative: standard deviation of noise

   Returns:
      light curve

   """
   lc = np.dot(W, mmap)
   sigma = np.mean(lc) * sigma_relative
   noise = np.random.normal(0.0, sigma, np.shape(lc))
   lc = lc + noise
   return lc


def gen_dynamic_lightcurve(W, A, X, obst, sigma_relative):
    """light curve generator
    Args:
        W: geometric kernel; Ni (time) x Nj (pixel)
        A: multiband map; Nj (pixel) x Nk (components)
        X: multiband reflectivity; Nk (components) x Nl (bands)
        obst: observation times
        sigma_relative: standard deviation of noise
    Returns:
        dynamic light curve
    """
    rotA = rotating_map(A, obst, rotthetamax = np.pi / 2.0)
    lc = np.dot(np.sum(W[:, :, np.newaxis] * rotA, axis = 1), X)
    sigma = np.mean(lc) * sigma_relative
    noise = np.random.normal(0.0, sigma, np.shape(lc))
    lc = lc + noise
    return lc

if __name__ == "__main__":
   from jaxsot.core.weight import comp_weight, comp_omega
   from jaxsot.io.earth import binarymap

   mmap = binarymap(nside=16, show=False)

   nside = 16
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
   W = WI * WV
   lc = gen_lightcurve(W, mmap, 0.0)
   print(np.sum(lc))
   import matplotlib.pyplot as plt
   plt.plot(obst, lc)
   plt.show()
