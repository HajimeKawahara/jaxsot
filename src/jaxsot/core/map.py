import numpy as np
from jaxsot.io.reflectivity import set_meanalbedo

def gen_multibandmap(cmap, refsurfaces, sky, vals, bands, onsky=False):
    """multiband map generator

    Args:
       cmap: classmap
       refsurfaces: list of target component reflectivities
       sky: sky reflectivity 
       vals: list of target component values
       bands: list of [waves, wavee]
       onsky: if True, consider sky reflectivity

    Returns:
       multiband map in healpix, multi-component map in healpix, multi-component reflectivity

    """
    nbands = np.shape(bands)[0]
    ncomp = len(refsurfaces)
    if len(refsurfaces) != len(vals):
        print("inconsisitent val and refsurces. CHECK IT.")
        
    #map
    Ain=np.zeros((len(cmap),ncomp))
    for i in range(0, ncomp):
        mask = (cmap == vals[i])
        Ain[mask, i] = 1.0

    # spectra
    Xin = []
    for ibands in range(0, nbands):
        waves = bands[ibands][0]
        wavee = bands[ibands][1]
        malbedo_band = set_meanalbedo(
            waves, wavee, refsurfaces, sky, onsky)
        Xin.append(malbedo_band)

    Xin = np.array(Xin).T
    mmap=np.dot(Ain,Xin)
    return mmap, Ain, Xin


if __name__=="__main__":
    print("-")