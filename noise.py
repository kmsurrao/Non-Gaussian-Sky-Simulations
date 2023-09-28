import numpy as np
from pixell import enmap, utils
import multiprocessing as mp

def add_noise(inp, freq, split):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input specifications
    freq: int, frequency in GHz (either 220, 150, or 90)
    split: int, split number (either 0, 1, 2, or 3)
    
    RETURNS
    -------
    map_to_write: CAR map with noise added
    '''

    res = np.pi/inp.ellmax*(180/np.pi)*60.
    shape, wcs = enmap.fullsky_geometry(res=res * utils.arcmin, proj='car')

    CAR_map = enmap.read_map(f'{inp.output_dir}/sim_{freq}GHz_split{split}')
    if freq==220:
        noise_file = f'act_dr6v4_tile_cmbmask_pa4_f150_pa4_f220_lmax10800_4way_set{split}_noise_sim_map9061021.fits'
        noise_map = 10**(-6)*enmap.read_map(noise_file)[1,0]
    elif freq==150:
        noise_file = f'act_dr6v4_tile_cmbmask_pa5_f090_pa5_f150_lmax10800_4way_set{split}_noise_sim_map9061021.fits'
        noise_map = 10**(-6)*enmap.read_map(noise_file)[1,0]
    elif freq==90:
        noise_file = f'act_dr6v4_tile_cmbmask_ivfwhm2_pa6_f090_pa6_f150_lmax10800_4way_set{split}_noise_sim_map9061021.fits'
        noise_map = 10**(-6)*enmap.read_map(noise_file)[0,0]
    
    if not inp.pol:
        noise_map = noise_map[0]

    CAR_map += noise_map
    map_to_write = enmap.ndmap(CAR_map, wcs)

    enmap.write_map(f'{inp.output_dir}/sim_{freq}GHz_split{split}', map_to_write)
    return map_to_write



def save_all_noise_added_maps(inp):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input specifications
    
    RETURNS
    -------
    car_maps: list of maps in CAR format

    '''
    freqs = [220, 150, 90]
    freqs_list = []
    splits_list = []
    for freq in freqs:
        for split in range(4):
            freqs_list.append(freq)
            splits_list.append(split)

    pool = mp.Pool(12)
    car_maps = pool.starmap(add_noise, [(inp, freqs_list[i], splits_list[i]) for i in range(12)])
    pool.close()
    
    return car_maps