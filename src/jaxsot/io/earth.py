import healpy as hp
import numpy as np
import pkg_resources
from jaxsot.io.reflectivity import load_refdata

default_band=bands=[[0.4,0.45],[0.45,0.5],[0.5,0.55],[0.55,0.6],[0.6,0.65],[0.65,0.7],[0.7,0.75],[0.75,0.8],[0.8,0.85],[0.85,0.9]]

def binarymap(nside=16,show=False):
    """Load a binary map of Earth

    Args:
       nside: nside in Healpix
       show: if True, display a map 

    returns:
       binary map in healpix   
     

    """
    # test map
    filename="data/mockalbedo"+str(nside)+".fits"
    fitsfile=(pkg_resources.resource_filename('jaxsot', filename))
    mmap=(hp.read_map(fitsfile))
    mask=(mmap>0.0)
    mmap[mask]=1.0
    mmap=np.asarray(mmap)
    if show:
        import matplotlib.pyplot as plt
        hp.mollview(mmap, title="Cloud-subtracted Earth",flip="geo",cmap=plt.cm.bone,min=0,max=1)
        hp.graticule(color="white")
        plt.show()

    return mmap

def load_classification_map(nclass=3):
    """Load a multiband map of Earth

    Args:
       nclass: number of the classes (3 or 4)
    
    Returns:
       dataclass

    """
    filename="data/cmap"+str(nclass)+"class.npz"
    npzfile=(pkg_resources.resource_filename('jaxsot', filename))
    dataclass=np.load(npzfile)
    return dataclass

def multibandmap(band=default_band,show=False):
    """Load a multiband map of Earth

    Args:
       nside: nside in Healpix

    returns:
       multiband map in healpix   
     

    """
    # test map
    nclass=3
    dataclass=load_classification_map(nclass)
    cmap=dataclass["arr_0"]
    npix=len(cmap)
    nclass=(len(np.unique(cmap)))
    nside=hp.npix2nside(npix)
    vals=dataclass["arr_1"]
    valexp=dataclass["arr_2"]
    cloud, cloud_ice, snow_fine, snow_granular, snow_med, soil, veg, ice, water, clear_sky\
    =load_refdata()

    
    refsurfaces=[water,soil,veg]
    #malbedo=io_surface_type.set_meanalbedo(0.8,0.9,refsurfaces,clear_sky)
    
    #mmap,Ainit,Xinit=toymap.make_multiband_map(cmap,refsurfaces,clear_sky,vals,bands)
    #ave_band=np.mean(np.array(bands),axis=1)
    #io_surface_type.plot_albedo(veg,soil,cloud,snow_med,water,clear_sky,ave_band,Xinit,valexp)
    

    
    return 


if __name__=="__main__":
    mmap=binarymap(nside=16,show=True)
    print(len(mmap[mmap==1.0]))
