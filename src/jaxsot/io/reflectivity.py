import numpy as np
import os
import pkg_resources


def get_path(filename):
    path="data/refdata/"+filename
    path=(pkg_resources.resource_filename('jaxsot', path))
    return path

def load_refdata():
    """load reflectivity data
    
    Returns:
        reflectivity of cloud, cloud_ice, snow_fine, snow_granular, snow_med, soil, veg, ice, water, clear_sky

    """
    # from astrobio/ipynb/reflectivity.ipynb
    cloud = np.loadtxt(get_path("clouds.txt"))
    cloud_ice = np.loadtxt(get_path("clouds_ice.txt"))
    snow_fine = np.loadtxt(get_path("fine_snow.txt"))
    snow_granular = np.loadtxt(get_path("granular_snow.txt"))
    snow_med = np.loadtxt(get_path("medium_snow.txt"))
    soil = np.loadtxt(get_path("soil.txt"))
    veg = np.loadtxt(get_path("veg_deciduous.txt"))
    ice = np.loadtxt(get_path("ice.txt"))
    water = np.loadtxt(get_path("ocean_McLinden.csv"))
    cs=np.load(get_path("clear_sky.npz"))    
    clear_sky = cs["arr_0"].T
    clear_sky[:, 0] = clear_sky[:, 0]/1000.0  # nm->micron
    water[:, 0] = water[:, 0]/1000.0

    snow_fine[:, 1] = snow_fine[:, 1]/100.0
    snow_granular[:, 1] = snow_granular[:, 1]/100.0
    snow_med[:, 1] = snow_med[:, 1]/100.0
    soil[:, 1] = soil[:, 1]/100.0
    veg[:, 1] = veg[:, 1]/100.0

    return cloud, cloud_ice, snow_fine, snow_granular, snow_med, soil, veg, ice, water, clear_sky

def get_meanalbedo(ref, waves, wavee):
    mask = (ref[:, 0] >= waves)*(ref[:, 0] <= wavee)
    return np.mean(ref[mask, 1])

def set_meanalbedo(waves, wavee, refsurfaces, sky, onsky=False):
    ma = []
    if onsky:
        atm = get_meanalbedo(sky, waves, wavee)
        for i in range(0, len(refsurfaces)):
            ma.append(get_meanalbedo(refsurfaces[i], waves, wavee)+atm)
    else:
        for i in range(0, len(refsurfaces)):
            ma.append(get_meanalbedo(refsurfaces[i], waves, wavee))
        
    return np.array(ma)

def plot_albedo(veg,soil,cloud,snow_med,water,clear_sky,ave_band,malbedo,valexp):
    import matplotlib.pyplot as plt
    fig= plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(veg[:,0],veg[:,1],c="black",lw=2,label="vegitation (deciduous)")
    ax.plot(soil[:,0],soil[:,1],c="gray",lw=1,label="soil")
    ax.plot(cloud[:,0],cloud[:,1],c="black",ls="dashed",label="cloud (water)")
    ax.plot(snow_med[:,0],snow_med[:,1],c="gray",ls="dashed",label="snow (medium granular)")
    ax.plot(water[:,0],water[:,1],c="gray",ls="-.",label="water")
    ax.plot(clear_sky[:,0],clear_sky[:,1],c="gray",ls="dotted",label="clear sky")
    for i in range(0,len(valexp)):
        ax.plot(ave_band,malbedo[i,:],"+",label=valexp[i])
    plt.xlim(0.4,1.5)
    plt.legend(bbox_to_anchor=(1.1, 0.3))
    plt.show()

if __name__=="__main__":
    load_refdata()
