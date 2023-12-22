from utils import get_CAR_shape_and_wcs
from pixell import enmap, reproject, enplot
from tqdm import tqdm



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
    car_map = reproject.healpix2map(healpix_map, shape, wcs, lmax=inp.ellmax, rot=None)
    return car_map


def eshow(x,**kwargs): 
    '''
    Function to visualize CAR map
    '''
    enplot.show(enplot.plot(x,**kwargs))


def get_all_CAR_maps(inp, rotated_maps):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input specifications
    rotated_maps: ndarray of shape (3 for PA4 PA5 PA6, 2 for freqs in array in ascending order, Nsplits, Npix) if not pol,
        or shape (3 for PA4 PA5 PA6, 2 for freqs in array in ascending order, Nsplits, 3 for I/Q/U, Npix) if pol
        contains beam-convolved healpix maps for each frequency and split (already rotated to equatorial coordinates)
    
    RETURNS
    -------
    car_maps: list of maps in CAR format

    '''
    print('Reprojecting to CAR...', flush=True)
    car_maps = []
    shape, wcs = get_CAR_shape_and_wcs(inp)
    pa_freq_dict = {4:[150,220], 5:[90,150], 6:[90,150]}
    for p, pa in enumerate(tqdm([4, 5, 6], desc='PA')):
        for i, freq in enumerate(tqdm(pa_freq_dict[pa], desc='Freq')):
            for split in tqdm(range(4), desc='Split'):
                map_to_write = healpix2CAR(inp, rotated_maps[p, i, split])
                map_to_write = enmap.ndmap(map_to_write, wcs) 
                car_maps.append(map_to_write)
                enmap.write_map(f'{inp.output_dir}/sim_pa{pa}_{freq}GHz_split{split}_noiseless', map_to_write)
        
    return car_maps