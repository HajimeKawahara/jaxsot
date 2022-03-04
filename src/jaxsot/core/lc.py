import numpy as np


def gen_lightcurve(W, mmap, sigma_relative):
    """light curve generator using a static map.

    Args:
       W: weight matrix
       mmap: static healpix map
       sigma_relative: relative noise level

    Returns:
       lightcurve
    """
    lc = np.dot(W, mmap)
    sigma = np.mean(lc)*sigma_relative
    noise = np.random.normal(0.0, sigma, len(lc))
    lc = lc+noise
    return lc


def gen_dynamic_lightcurve(W, M, sigma_relative):
    """light curve generator using a dynamic map.

    Args:
       W: weight matrix
       M: dynamic map
       sigma_relative: relative noise level

    Returns:
       lightcurve
    """

    lc = np.sum(W*M, axis=1)
    sigma = np.mean(lc)*sigma_relative
    noise = np.random.normal(0.0, sigma, len(lc))
    lc = lc+noise
    return lc
