import numpy as np
from utils import get_CAR_shape_and_wcs
from pixell import enmap
import multiprocessing as mp

def add_noise(inp, freq, split, pa):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input specifications
    freq: int, frequency in GHz (either 220, 150, or 90)
    split: int, split number (either 0, 1, 2, or 3)
    pa: int, PA (either 4, 5, or 6)
    
    RETURNS
    -------
    map_to_write: CAR map with noise added
    '''

    shape, wcs = get_CAR_shape_and_wcs(inp)

    CAR_map = enmap.read_map(f'{inp.output_dir}/sim_pa{pa}_{freq}GHz_split{split}_noiseless')
    if pa == 4:
        if inp.noise_type == 'tiled':
            noise_file = f'{inp.noise_dir}/act_dr6v4_tile_cmbmask_pa4_f150_pa4_f220_lmax10800_4way_set{split}_noise_sim_map9061021.fits'
        elif inp.noise_type == 'stitched':
            noise_file = f'{inp.noise_dir}/act_dr6v4_fdw_cmbmask_4000_prof_cosine_5000_tile_cmbmask_pa4_f150_pa4_f220_lmax10800_4way_set{split}_noise_sim_map40294_103094.fits'
    elif pa == 5:
        if inp.noise_type == 'tiled':
            noise_file = f'{inp.noise_dir}/act_dr6v4_tile_cmbmask_pa5_f090_pa5_f150_lmax10800_4way_set{split}_noise_sim_map9061021.fits'
        elif inp.noise_type == 'stitched':
            noise_file = f'{inp.noise_dir}/act_dr6v4_fdw_cmbmask_4000_prof_cosine_5000_tile_cmbmask_pa5_f090_pa5_f150_lmax10800_4way_set{split}_noise_sim_map40294_103094.fits'
    elif pa == 6:
        if inp.noise_type == 'tiled':
            noise_file = f'{inp.noise_dir}/act_dr6v4_tile_cmbmask_ivfwhm2_pa6_f090_pa6_f150_lmax10800_4way_set{split}_noise_sim_map9061021.fits'
        elif inp.noise_type == 'stitched':
            noise_file = f'{inp.noise_dir}/act_dr6v4_fdw_cmbmask_ivfwhm2_4000_prof_cosine_5000_tile_cmbmask_ivfwhm2_pa6_f090_pa6_f150_lmax10800_4way_set{split}_noise_sim_map40294_103094.fits'
    full_noise_maps = 10**(-6)*enmap.read_map(noise_file)
    if freq == 90:
        noise_map = full_noise_maps[0,0]
    elif freq == 150:
        if pa == 4:
            noise_map = full_noise_maps[0,0]
        else:
            noise_map = full_noise_maps[1,0]
    elif freq == 220:
        noise_map = full_noise_maps[1,0]
    
    if not inp.pol:
        noise_map = noise_map[0]

    CAR_map += noise_map
    map_to_write = enmap.ndmap(CAR_map, wcs)

    enmap.write_map(f'{inp.output_dir}/sim_pa{pa}_{freq}GHz_split{split}_{inp.noise_type}noise', map_to_write)
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
    freqs_list = []
    splits_list = []
    pa_list = []
    pa_freq_dict = {4:[150,220], 5:[90,150], 6:[90,150]}
    for p, pa in enumerate([4, 5, 6]):
        for i, freq in enumerate(pa_freq_dict[pa]):
            for split in range(4):
                freqs_list.append(freq)
                splits_list.append(split)
                pa_list.append(pa)

    pool = mp.Pool(12)
    car_maps = pool.starmap(add_noise, [(inp, freqs_list[i], splits_list[i], pa_list[i]) for i in range(len(freqs_list))])
    pool.close()
    
    return car_maps