# JAXSOT

In the near future, direct imaging missions will search for Earth-like planets around nearby stars. One of the problems is how to characterize the planet surface. To address this question, we are developing a surface map and components reconstruction method using a one-dimensional light curve of a direct-imaged planet. The orbital motion and spin rotation of a planet conveys information about the spherical surface to the time-series of the light curve. In the future, this theoretical work will be tested in the era of space direct imaging of exoplanets.

JAXSOT is a successor of [SOT](https://github.com/HajimeKawahara/sot) package, using a autodiff/XLA package JAX.
Therefore, JAXSOT is compatible with 
- an optimization package using jaxopt,
- an HMC-NUTS using PPLs such as NumPyro.
The aim of JAXSOT is to unify various methods related to the spin-orbit mapping into one python/JAX framework. It makes easy to develop new algorithms by connecting with techniques developed so far.


