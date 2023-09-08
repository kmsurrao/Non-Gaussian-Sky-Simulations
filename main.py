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
from beams import apply_beam 
from reprojection import healpix2CAR
from make_plots import plot_outputs
from galactic_mask import plot_and_save_mask_deconvolved_spectra

def main(inp):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input specifications
    
    RETURNS
    -------
    car_maps: list of CAR maps at frequencies 220, 150, and 90 GHz. If pol, each 
            map contains [I,Q,U], otherwise just I
    '''


    # set up output directory
    if not os.path.isdir(inp.output_dir):
        env = os.environ.copy()
        subprocess.call(f'mkdir {inp.output_dir}', shell=True, env=env)


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
    if inp.mask_file and inp.ells_per_bin:
        inp.wsp = nmt.NmtWorkspace()
        inp.b = nmt.NmtBin.from_lmax_linear(inp.ellmax, inp.ells_per_bin, is_Dell=True)
        mask = hp.read_map(inp.mask_file, field=(3)) #70% fsky
        mask = hp.ud_grade(mask, inp.nside)
        f_tmp = nmt.NmtField(mask, [np.ones_like(mask)])
        inp.wsp.compute_coupling_matrix(f_tmp, f_tmp, inp.b)
        print('Computed coupling matrix for mask deconvolution', flush=True)


    #get galactic and extragalactic component maps
    num_extragal_comps = 5
    if inp.ksz_reionization_file is not None:
        num_extragal_comps += 1
    if not inp.pol: #index as all_maps[freq, comp, pixel], freqs in decreasing order
        all_maps = np.zeros((3, len(inp.galactic_components)+num_extragal_comps, 12*inp.nside**2))
    if inp.pol: #index as all_maps[freq, comp, I/Q/U, pixel], freqs in decreasing order
        all_maps = np.zeros((3, len(inp.galactic_components)+num_extragal_comps, 3, 12*inp.nside**2))
    for i, freq in enumerate([220, 150, 90]):
        all_maps[i,:len(inp.galactic_components)] = get_galactic_comp_maps(inp, all_bandpass_freqs[i], central_freq=freq, 
                                        bandpass_weights=all_bandpass_weights[i], plot_hist=(True if inp.plot_dir else False))
        all_maps[i,len(inp.galactic_components):] = get_extragalactic_comp_maps(inp, freq)
    pickle.dump(all_maps, open(f'{inp.output_dir}/gal_and_extragal_before_beam.p', 'wb'))
    print('Got maps of galactic and extragalactic components at 220, 150, and 90 GHz', flush=True)
    

    # get beam-convolved healpix maps at each frequency
    freq_maps = np.sum(all_maps, axis=1) #shape Nfreqs, 1 if not pol or 3 if pol, Npix
    if not inp.pol: #index as all_maps[freq, pixel], freqs in decreasing order
        beam_convolved_maps = np.zeros((3, 12*inp.nside**2))
    if inp.pol: #index as all_maps[freq, I/Q/U, pixel], freqs in decreasing order
        beam_convolved_maps = np.zeros((3, 3, 12*inp.nside**2))
    for freq in range(3):
        if freq==0:
            beamfile = f'{inp.beam_dir}/coadd_pa4_f220_night_beam_tform_jitter_cmb.txt'
        elif freq==1:
            beamfile = f'{inp.beam_dir}/coadd_pa5_f150_night_beam_tform_jitter_cmb.txt'
        elif freq==2:
            beamfile = f'{inp.beam_dir}/coadd_pa6_f090_night_beam_tform_jitter_cmb.txt'
        beam_convolved_maps[freq] = apply_beam(freq_maps[freq], beamfile, inp.pol)
    pickle.dump(beam_convolved_maps, open(f'{inp.output_dir}/beam_convolved_maps.p', 'wb'))
    print('Got beam-convolved maps', flush=True)
    
    # save mask-deconvolved spectra using 70% fsky galactic mask (do not plot yet)
    if inp.mask_file and inp.ells_per_bin:
        plot_and_save_mask_deconvolved_spectra(inp, beam_convolved_maps, save_only=True)


    # convert each frequency map from healpix to CAR
    car_maps = []
    freqs = [220, 150, 90]
    for freq in range(3):
        car_map = healpix2CAR(inp, beam_convolved_maps[freq])
        car_maps.append(car_map)
        enmap.write_map(f'sim_{freqs[freq]}GHz', car_map)
    print('Got CAR maps', flush=True)

    return car_maps


if __name__=='__main__':

    # main input file containing most specifications 
    parser = argparse.ArgumentParser(description="Non-gaussian full sky simulations.")
    parser.add_argument("--config", default="example_yaml_files/stampede.yaml")
    args = parser.parse_args()
    input_file = args.config

    # read in the input file and set up relevant info object
    inp = Info(input_file)

    main(inp)
    
    if inp.mask_file and inp.ells_per_bin:
        beam_convolved_maps = pickle.load(open(f'{inp.output_dir}/beam_convolved_maps.p', 'rb'))
        plot_and_save_mask_deconvolved_spectra(inp, save_only=True)

    
    if len(inp.plots_to_make) > 0:
        plot_outputs(inp)
    
