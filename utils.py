import numpy as np
import matplotlib.pyplot as plt
from pixell import enmap, utils

def plot_histogram(inp, cmap, freq, comp, string='before'):
    '''
    Plots histogram of pixel values and saves them in inp.plot_dir

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    cmap: 1D numpy array if not inp.pol, otherwise 3D numpy array for T,Q,U containing
        maps in healpix format
    freq: int, frequency of input map in GHz
    comp: str, name of component shown in the map
    string: str, 'before' or 'after' based on whether plotting histogram 
                    before or after applying flux cut

    RETURNS
    -------
    None
    '''
    types = ['I', 'Q', 'U']
    for t in range(3):
        plt.clf()
        if inp.pol:
            plt.hist(cmap[t], density=True, alpha=0.5)
        else:
            plt.hist(cmap, density=True, alpha=0.5)
        plt.yscale('log')
        plt.savefig(f'{inp.plot_dir}/pixel_hist_{freq}ghz_{types[t]}_{comp}_{string}.png')
        plt.close()
        if not inp.pol:
            break
    return


def get_CAR_shape_and_wcs(inp):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications

    RETURNS
    -------
    shape: 2D array containing shape of CAR map (same as noise maps if adding noise)
    wcs: wcs pixell object (same as noise maps if adding noise)

    '''
    if not inp.noise_dir:
        res = np.pi/inp.ellmax*(180/np.pi)*60.
        shape, wcs = enmap.fullsky_geometry(res=res * utils.arcmin, proj='car')
    else:
        if inp.noise_type == 'tiled':
            noise_file = f'{inp.noise_dir}/act_dr6v4_tile_cmbmask_pa4_f150_pa4_f220_lmax10800_4way_set0_noise_sim_map9061021.fits'
        elif inp.noise_type == 'stitched':
            noise_file = f'{inp.noise_dir}/act_dr6v4_fdw_cmbmask_4000_prof_cosine_5000_tile_cmbmask_pa4_f150_pa4_f220_lmax10800_4way_set0_noise_sim_map40294_103094.fits'
        noise_map = 10**(-6)*enmap.read_map(noise_file)[1,0]
        if not inp.pol:
            noise_map = noise_map[0]
        shape, wcs = noise_map.shape, noise_map.wcs
    return shape, wcs
