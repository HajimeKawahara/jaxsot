import numpy as np
import os
import pkg_resources


def get_path(filename):
    path = 'data/refdata/'+filename
    path = (pkg_resources.resource_filename('jaxsot', path))
    return path


def load_refdata():
    """load reflectivity data.

    Returns:
        reflectivity of cloud, cloud_ice, snow_fine, snow_granular, snow_med, soil, veg, ice, water, clear_sky
    """
    # from astrobio/ipynb/reflectivity.ipynb
    cloud = np.loadtxt(get_path('clouds.txt'))
    cloud_ice = np.loadtxt(get_path('clouds_ice.txt'))
    snow_fine = np.loadtxt(get_path('fine_snow.txt'))
    snow_granular = np.loadtxt(get_path('granular_snow.txt'))
    snow_med = np.loadtxt(get_path('medium_snow.txt'))
    soil = np.loadtxt(get_path('soil.txt'))
    veg = np.loadtxt(get_path('veg_deciduous.txt'))
    ice = np.loadtxt(get_path('ice.txt'))
    water = np.loadtxt(get_path('ocean_McLinden.csv'))
    cs = np.load(get_path('clear_sky.npz'))
    clear_sky = cs['arr_0'].T
    clear_sky[:, 0] = clear_sky[:, 0]/1000.0  # nm->micron
    water[:, 0] = water[:, 0]/1000.0

    snow_fine[:, 1] = snow_fine[:, 1]/100.0
    snow_granular[:, 1] = snow_granular[:, 1]/100.0
    snow_med[:, 1] = snow_med[:, 1]/100.0
    soil[:, 1] = soil[:, 1]/100.0
    veg[:, 1] = veg[:, 1]/100.0

    return cloud, cloud_ice, snow_fine, snow_granular, snow_med, soil, veg, ice, water, clear_sky


if __name__ == '__main__':
    load_refdata()
