import numpy as np
import healpy as hp
from pixell import enmap, reproject, enplot, utils


def eshow(x,**kwargs): 
    '''
    Function to visualize CAR map
    '''
    enplot.show(enplot.plot(x,**kwargs))


def apply_beam(map_, beamfile, pol):
    '''
    ARGUMENTS
    ---------
    map_: numpy array, either [I,Q,U] healpix maps if pol, or just one intensity healpix map if not pol
    beamfile: str, path of file containing ACT beam information
    pol: Bool, whether [I,Q,U] or just I map is provided
    
    RETURNS
    -------
    frequencies: numpy array of frequencies (GHz) in the passband
    bandpass_weights: numpy array of bandpass weights (with nu^2 divided out)
    '''
    data = np.loadtxt(beamfile)
    Bl = data[:,1]
    Bl /= Bl[0]
    beam_convolved_map = hp.smoothing(map_, beam_window=Bl, pol=pol)
    return beam_convolved_map


def healpix2CAR(healpix_map, ellmax, pol):
    '''
    ARGUMENTS
    ---------
    healpix_map: numpy array, either [I,Q,U] healpix maps if pol, or just one intensity healpix map if not pol
    ellmax: int, maximum ell for which to include information in CAR map
    pol: Bool, whether [I,Q,U] or just I map is provided
    
    RETURNS
    -------
    car_map: numpy array, either [I,Q,U] CAR maps if pol, or just one intensity CAR map if not pol
    '''
    res = np.pi/ellmax*(180/np.pi)*60.
    shape, wcs = enmap.fullsky_geometry(res=res * utils.arcmin, proj='car')
    ncomp = 3 if pol else 1
    car_map = reproject.enmap_from_healpix(healpix_map, shape, wcs, ncomp=ncomp, unit=1, lmax=ellmax, rot="gal,equ")
    return car_map