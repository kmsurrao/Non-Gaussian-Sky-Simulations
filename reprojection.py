import numpy as np
from utils import get_CAR_shape_and_wcs
from pixell import enmap, reproject, enplot
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
    shape, wcs = get_CAR_shape_and_wcs(inp)
    rot = None if inp.noise_dir else "gal,equ"
    car_map = reproject.healpix2map(healpix_map, shape, wcs, lmax=inp.ellmax, rot=rot)
    return car_map


def eshow(x,**kwargs): 
    '''
    Function to visualize CAR map
    '''
    enplot.show(enplot.plot(x,**kwargs))


def get_all_CAR_maps(inp, beam_convolved_maps, parallel=False):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input specifications
    beam_convolved_maps: ndarray of shape (Nfreqs, Nsplits, Npix) if not pol,
        or shape (Nfreqs, Nsplits, 3 for I/Q/U, Npix) if pol
        contains beam-convolved healpix maps for each frequency and split
    parallel: Bool, whether to run reprojection in parallel (can cause memory issues)
    
    RETURNS
    -------
    car_maps: list of maps in CAR format

    '''
    car_maps = []
    freqs = [220, 150, 90]

    shape, wcs = get_CAR_shape_and_wcs(inp)

    if parallel:
        pool = mp.Pool(12)
        results = pool.starmap(healpix2CAR, [(inp, beam_convolved_maps[i//4, i%4]) for i in range(12)])
        pool.close()
    
    idx = 0
    for freq in range(3):
        for split in range(4):
            if parallel:
                map_to_write = results[idx]
            else:
                map_to_write = healpix2CAR(inp, beam_convolved_maps[freq, split])
            map_to_write = enmap.ndmap(map_to_write, wcs) 
            car_maps.append(map_to_write)
            enmap.write_map(f'{inp.output_dir}/sim_{freqs[freq]}GHz_split{split}', map_to_write)
            idx += 1
    
    return car_maps