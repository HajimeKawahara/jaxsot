import numpy as np
import healpy as hp
import tqdm
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
        assert ValueError("inconsisitent val and refsurces.")

    #map
    Ain = np.zeros((len(cmap), ncomp))
    for i in range(0, ncomp):
        mask = (cmap == vals[i])
        Ain[mask, i] = 1.0

    # spectra
    Xin = []
    for ibands in range(0, nbands):
        waves = bands[ibands][0]
        wavee = bands[ibands][1]
        malbedo_band = set_meanalbedo(waves, wavee, refsurfaces, sky, onsky)
        Xin.append(malbedo_band)

    Xin = np.array(Xin).T
    mmap = np.dot(Ain, Xin)
    return mmap, Ain, Xin


def rotate_map(hmap, rot_phi, rot_theta):
    """rotating map generator

    Args:
        hmap: map vector
        rot_phi: rotation angle around the axis parallel to the spin axis
        rot_theta: rotation angle around the axis perpendicular to the spin axis

    Returns:
        rotated map vector

    """

    nside = hp.npix2nside(len(hmap))
    t,p = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    r = hp.Rotator(deg=False, rot=[rot_phi,rot_theta])
    trot, prot = r(t,p)
    rmap = hp.get_interp_val(hmap, trot, prot)
    return rmap

def rotating_map(mmap,obst,rotphimax=0.0,rotthetamax=3*np.pi/4):
    """rotating map generator

    Args:
        mmap: static map vector (binary) or matrix (multiband)
        obst: observation times
        rotphimax: maximum of the rotation angle around the axis parallel to the spin axis
        rotthetamax: maximum of the rotation angle around the axis perpendicular to the spin axis

    Returns:
        rotated map matrix (binary) or tensor (multiband)

    """

    Nt=len(obst)
    rotphi=np.linspace(0.0,rotphimax,Nt)
    rottheta=np.linspace(0.0,rotthetamax,Nt)
    ndim=np.shape(np.shape(mmap))[0]

    if ndim==1:
        M=[]
        for i in range(0,Nt):
            M.append(rotate_map(mmap, rotphi[i], rottheta[i]))
        M=np.array(M)

    elif ndim==2:
        Nj,Nl=np.shape(mmap)

        M=[]
        for i in tqdm.tqdm(range(0,Nt)):
            MM=[]
            for l in (range(0,Nl)):
                MM.append(rotate_map(mmap[:,l], rotphi[i], rottheta[i]))
            M.append(np.array(MM).T)
        M=np.array(M)
        
    return M
