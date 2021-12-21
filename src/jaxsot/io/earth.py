import healpy as hp
import numpy as np
import pkg_resources

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
    fitsfile=(pkg_resources.resource_filename('jaxsot', filename))
    dataclass=np.load(fitsfile)
    return dataclass

def multibandmap(show=False):
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

    return 


if __name__=="__main__":
    mmap=binarymap(nside=16,show=True)
    print(len(mmap[mmap==1.0]))
