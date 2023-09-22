import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import pickle
import argparse
from pixell import enmap
import pymaster as nmt
import healpy as hp
from input import Info
from bandpass_integration import get_bandpass_freqs_and_weights, get_galactic_comp_maps, get_extragalactic_comp_maps
from beams import get_all_beam_convolved_maps
from reprojection import get_all_CAR_maps
from make_plots import plot_outputs
from galactic_mask import plot_and_save_mask_deconvolved_spectra

def main():
    '''
    RETURNS
    -------
    car_maps: list of CAR maps at frequencies 220, 150, and 90 GHz. If pol, each 
            map contains [I,Q,U], otherwise just I
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
    
    #Set up output directory for plots
    if inp.plot_dir and not os.path.isdir(inp.plot_dir):
        env = os.environ.copy()
        subprocess.call(f'mkdir {inp.plot_dir}', shell=True, env=env)

    # get bandpass frequencies and weights
    all_bandpass_freqs = []
    all_bandpass_weights = []
    for central_freq in [220, 150, 90]:
        bandpass_freqs, bandpass_weights = get_bandpass_freqs_and_weights(central_freq, inp.passband_file)
        all_bandpass_freqs.append(bandpass_freqs)
        all_bandpass_weights.append(bandpass_weights)
    pickle.dump(all_bandpass_freqs, open(f'{inp.output_dir}/all_bandpass_freqs.p', 'wb'))
    pickle.dump(all_bandpass_weights, open(f'{inp.utput_dir}/all_bandpass_weights.p', 'wb'))
    print('Got passbands', flush=True)
    

    # create NaMaster workspace object if computing mask-deconvolved spectra
    if 'component_mask_deconvolution' in inp.checks or 'freq_map_mask_deconvolution' in inp.checks:
        inp.wsp = nmt.NmtWorkspace()
        inp.b = nmt.NmtBin.from_lmax_linear(inp.ellmax, inp.ells_per_bin, is_Dell=True)
        mask = hp.read_map(inp.mask_file, field=(3)) #70% fsky
        mask = hp.ud_grade(mask, inp.nside)
        f_tmp = nmt.NmtField(mask, [np.ones_like(mask)])
        inp.wsp.compute_coupling_matrix(f_tmp, f_tmp, inp.b)
        if inp.pol:
            inp.wsp2 = nmt.NmtWorkspace() #temperature x pol
            inp.wsp3 = nmt.NmtWorkspace() #pol x pol
            f_tmp_pol = nmt.NmtField(mask, [np.ones_like(mask), np.ones_like(mask)])
            inp.wsp2.compute_coupling_matrix(f_tmp, f_tmp_pol, inp.b)
            inp.wsp3.compute_coupling_matrix(f_tmp_pol, f_tmp_pol, inp.b)
        print('Computed coupling matrix for mask deconvolution', flush=True)


    #get galactic and extragalactic component maps
    #not enough memory to save individual component maps, so save one for all galactic components and one for all extragalactic components
    if not inp.pol: #index as all_maps[freq, gal or extragal, pixel], freqs in decreasing order
        all_maps = np.zeros((3, 2, 12*inp.nside**2), dtype=np.float32)
    else: #index as all_maps[freq, gal or extragal, I/Q/U, pixel], freqs in decreasing order
        all_maps = np.zeros((3, 2, 3, 12*inp.nside**2), dtype=np.float32)
    for i, freq in enumerate([220, 150, 90]):
        gal_comps = get_galactic_comp_maps(inp, all_bandpass_freqs[i], central_freq=freq,
                                        bandpass_weights=all_bandpass_weights[i])
        extragal_comps = get_extragalactic_comp_maps(inp, freq, plot_hist=(True if inp.plot_dir else False))
        all_maps[i,0] = gal_comps
        all_maps[i,1] = extragal_comps
        plt.close('all')
    pickle.dump(all_maps, open(f'{inp.output_dir}/gal_and_extragal_before_beam.p', 'wb'), protocol=4)
    print('Got maps of galactic and extragalactic components at 220, 150, and 90 GHz', flush=True)
    

    # get beam-convolved healpix maps at each frequency
    beam_convolved_maps = get_all_beam_convolved_maps(inp, all_maps)
    pickle.dump(beam_convolved_maps, open(f'{inp.output_dir}/beam_convolved_maps.p', 'wb'), protocol=4)
    print('Got beam-convolved maps', flush=True)
    
    # save mask-deconvolved spectra using 70% fsky galactic mask (do not plot yet)
    if 'freq_map_mask_deconvolution' in inp.checks:
        plot_and_save_mask_deconvolved_spectra(inp, beam_convolved_maps, save_only=True)


    # convert each frequency map from healpix to CAR
    car_maps = get_all_CAR_maps(inp, beam_convolved_maps)
    print('Got CAR maps', flush=True)

    # make plots
    if len(inp.plots_to_make) > 0:
        plot_outputs(inp)


    return car_maps


if __name__=='__main__':
    main()
    
