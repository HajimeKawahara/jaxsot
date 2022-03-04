"""Generating a rotating earth dynamic map."""
import numpy as np
import healpy as hp
import tqdm


def rotate_map(hmap, rot_theta, rot_phi):
    """rotate a healpix map.

    Args:
       hmap: input healpix map
       rot_theta: rotation angle theta
       rot_phi: rotation angle phi

    Returns:
       rotated healpix map
    """
    nside = hp.npix2nside(len(hmap))
    t, p = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    r = hp.Rotator(deg=False, rot=[rot_phi, rot_theta])
    trot, prot = r(t, p)
    rmap = hp.get_interp_val(hmap, trot, prot)
    return rmap


def gendymap_rotation(mmap, obst, rotphimax=0.0, rotthetamax=3*np.pi/4):
    """generate a dynamic map by rotating continents.

    Args:
       mmap: input healpix map
       obst: observation time series
       rotphimax: maximum rotation angle of phi
       rotthetamax: maximum rotation angle of theta

    Returns:
       dynamic map
    """

    Nt = len(obst)
    rotphi = np.linspace(0.0, rotphimax, Nt)
    rottheta = np.linspace(0.0, rotthetamax, Nt)
    ndim = np.shape(np.shape(mmap))[0]
    if ndim == 1:
        M = []
        for i in range(0, Nt):
            M.append(rotate_map(mmap, rottheta[i], rotphi[i]))
        M = np.array(M)
    elif ndim == 2:
        Nj, Nl = np.shape(mmap)

        M = []
        for i in tqdm.tqdm(range(0, Nt)):
            MM = []
            for l in (range(0, Nl)):
                MM.append(rotate_map(mmap[:, l], rottheta[i], rotphi[i]))
            M.append(np.array(MM).T)
        M = np.array(M)

    return M
