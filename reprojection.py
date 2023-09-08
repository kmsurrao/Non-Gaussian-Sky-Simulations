import numpy as np
from pixell import enmap, reproject, enplot, utils


def healpix2CAR(inp, healpix_map):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    healpix_map: numpy array, either [I,Q,U] healpix maps if pol, or just one intensity healpix map if not pol
    
    RETURNS
    -------
    car_map: ndarray, either [I,Q,U] CAR maps if pol, or just one intensity CAR map if not pol
    '''
    res = np.pi/inp.ellmax*(180/np.pi)*60.
    shape, wcs = enmap.fullsky_geometry(res=res * utils.arcmin, proj='car')
    ncomp = 3 if inp.pol else 1
    car_map = reproject.enmap_from_healpix(healpix_map, shape, wcs, ncomp=ncomp, unit=1, lmax=inp.ellmax, rot="gal,equ")
    return car_map


def eshow(x,**kwargs): 
    '''
    Function to visualize CAR map
    '''
    enplot.show(enplot.plot(x,**kwargs))