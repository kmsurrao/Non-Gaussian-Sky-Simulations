import os
import subprocess
import pickle
import argparse
import numpy as np
import healpy as hp
from input import Info
from bandpass_integration import get_all_bandpassed_maps, get_all_bandpass_freqs_and_weights
from beams import get_all_beam_convolved_maps, get_all_beam_convolved_alm
from reprojection import get_all_CAR_maps
from make_plots import plot_outputs
from galactic_mask import compute_coupling_matrices, plot_and_save_mask_deconvolved_spectra
from rotation import get_all_rotated_maps
from noise import save_all_noise_added_maps

def main():
    '''
    RETURNS
    -------
    car_maps: list of CAR maps at frequencies 220, 150, and 90 GHz for each split. 
        If pol, each map contains [I,Q,U], otherwise just I.
    '''

    # main input file containing most specifications 
    parser = argparse.ArgumentParser(description="Non-gaussian full sky simulations.")
    parser.add_argument("--config", default="example_yaml_files/stampede.yaml")
    args = parser.parse_args()
    input_file = args.config

    # read in the input file and set up relevant info object
    inp = Info(input_file)

    # set up output directory
    if not os.path.isdir(inp.output_dir):
        env = os.environ.copy()
        subprocess.call(f'mkdir {inp.output_dir}', shell=True, env=env)

    # set up output directory for plots
    if inp.plot_dir and not os.path.isdir(inp.plot_dir):
        env = os.environ.copy()
        subprocess.call(f'mkdir {inp.plot_dir}', shell=True, env=env)

    # get bandpass frequencies and weights
    all_bandpass_freqs, all_bandpass_weights = get_all_bandpass_freqs_and_weights(inp)
    pickle.dump(all_bandpass_freqs, open(f'{inp.output_dir}/all_bandpass_freqs.p', 'wb'))
    pickle.dump(all_bandpass_weights, open(f'{inp.output_dir}/all_bandpass_weights.p', 'wb'))
    print('Got passbands', flush=True)

    # create NaMaster workspace object if computing mask-deconvolved spectra
    if 'component_mask_deconvolution' in inp.checks or 'freq_map_mask_deconvolution' in inp.checks:
        compute_coupling_matrices(inp)
        print('Computed coupling matrix for mask deconvolution', flush=True)

    # get galactic and extragalactic component maps
    all_central_freqs = [220, 150, 90]
    all_maps = get_all_bandpassed_maps(inp, all_bandpass_freqs, all_central_freqs, all_bandpass_weights=all_bandpass_weights, parallel=False)
    pickle.dump(all_maps, open(f'{inp.output_dir}/maps_before_beam.p', 'wb'), protocol=4)
    print('Got maps of galactic and extragalactic components at 220, 150, and 90 GHz', flush=True)
    
    # get beam-convolved healpix map alm at each frequency
    beam_convolved_alm = get_all_beam_convolved_alm(inp, all_maps)
    pickle.dump(beam_convolved_alm, open(f'{inp.output_dir}/beam_convolved_alm.p', 'wb'), protocol=4)
    print('\nGot beam-convolved alm', flush=True)

    # save mask-deconvolved spectra using 70% fsky galactic mask (do not plot yet)
    if 'freq_map_mask_deconvolution' in inp.checks:
        if not inp.pol:
            beam_convolved_maps = np.zeros((3, 2, 4, 12*inp.nside**2), dtype=np.float32)
        else:
            beam_convolved_maps = np.zeros((3, 2, 4, 3, 12*inp.nside**2), dtype=np.float32)
        for p in range(3):
            for i in range(2):
                for split in range(4):
                    beam_convolved_maps[p,i,split] = hp.alm2map(beam_convolved_alm[p,i,split], inp.nside)
        plot_and_save_mask_deconvolved_spectra(inp, beam_convolved_maps, save_only=True)

    # rotate beam-convolved healpix maps from galactic to equatorial coordinates
    rotated_maps = get_all_rotated_maps(inp, beam_convolved_maps)
    pickle.dump(rotated_maps, open(f'{inp.output_dir}/rotated_healpix_maps.p', 'wb'), protocol=4)
    print('\nGot rotated maps', flush=True)

    # convert each frequency map from healpix to CAR
    car_maps = get_all_CAR_maps(inp, rotated_maps)
    print('\nGot CAR maps', flush=True)

    # add noise if noise_dir provided
    if inp.noise_dir is not None:
        car_maps = save_all_noise_added_maps(inp)
        print('\nAdded noise to maps', flush=True)

    # make plots
    if len(inp.plots_to_make) > 0:
        plot_outputs(inp)

    return car_maps


if __name__=='__main__':
    main()
    
