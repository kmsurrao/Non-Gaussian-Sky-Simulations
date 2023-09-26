import numpy as np
from pixell import enmap, reproject, enplot, utils
import multiprocessing as mp


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
    car_map = reproject.healpix2map(healpix_map, shape, wcs, lmax=inp.ellmax, rot="gal,equ")
    return car_map


def eshow(x,**kwargs): 
    '''
    Function to visualize CAR map
    '''
    enplot.show(enplot.plot(x,**kwargs))


def get_all_CAR_maps(inp, beam_convolved_maps):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input specifications
    beam_convolved_maps: ndarray of shape (Nfreqs, Nsplits, Npix) if not pol,
        or shape (Nfreqs, Nsplits, 3 for I/Q/U, Npix) if pol
        contains beam-convolved healpix maps for each frequency and split
    
    RETURNS
    -------
    car_maps: list of maps in CAR format

    '''
    car_maps = []
    freqs = [220, 150, 90]

    pool = mp.Pool(min(inp.nsims, 12))
    results = pool.starmap(healpix2CAR, [(inp, beam_convolved_maps[i//4, i%4]) for i in range(12)])
    pool.close()
    
    idx = 0
    for freq in range(3):
        for split in range(4):
            car_maps.append(results[idx])
            idx += 1
            enmap.write_map(f'sim_{freqs[freq]}GHz_split{split}', results[idx])
    
    return car_maps