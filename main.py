import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import pickle
from pixell import enmap
from bandpass_integration import get_bandpass_freqs_and_weights, get_galactic_comp_map, get_extragalactic_comp_map
from beams import apply_beam, healpix2CAR

def main(nside, ellmax, galactic_components, passband_file, agora_sims_dir, beam_dir, pol=False, 
        ksz_reionization_file=None, save_intermediate=False, verbose=False):
    '''
    ARGUMENTS
    ---------
    nside: int, resolution parameter for maps
    ellmax: int, maximum ell for which to compute power spectra
    galactic_components: list of preset strings for pysm components
    passband_file: str, path to h5 file containing passband information
    agora_sims_dir: str, directory containing agora extragalactic sims with ACT passbands
    beam_dir: str, directory containing beam files
    pol: Bool, False if only computing intensity maps, True if computing E-mode maps
    ksz_reionization_file: str, file containing Cl for reionization kSZ
    save_intermediate: Bool, whether to save maps at intermediate steps
    

    RETURNS
    -------
    car_maps: list of CAR maps at frequencies 220, 150, and 90 GHz. If pol, each 
            map contains [I,Q,U], otherwise just I
    '''

    #set up output directory if save_intermediate is True
    if save_intermediate:
        output_dir = 'outputs'
        if not os.path.isdir(output_dir):
            env = os.environ.copy()
            subprocess.call(f'mkdir {output_dir}', shell=True, env=env)


    #get bandpass frequencies and weights
    all_bandpass_freqs = []
    all_bandpass_weights = []
    for central_freq in [220, 150, 90]:
        bandpass_freqs, bandpass_weights = get_bandpass_freqs_and_weights(central_freq, passband_file, plot=True)
        all_bandpass_freqs.append(bandpass_freqs)
        all_bandpass_weights.append(bandpass_weights)
    if save_intermediate:
        pickle.dump(all_bandpass_freqs, open(f'{output_dir}/all_bandpass_freqs.p', 'wb'))
        pickle.dump(all_bandpass_weights, open(f'{output_dir}/all_bandpass_weights.p', 'wb'))
    if verbose:
        print('Got passbands', flush=True)


    #get galactic and extragalactic component maps
    if not pol: #index as all_maps[freq, gal or extragal, pixel], freqs in decreasing order
        all_maps = np.zeros((3, 2, 12*nside**2))
    if pol: #index as all_maps[freq, gal or extragal, I/Q/U, pixel], freqs in decreasing order
        all_maps = np.zeros((3, 2, 3, 12*nside**2))
    for i, freq in enumerate([220, 150, 90]):
        all_maps[i,0] = get_galactic_comp_map(galactic_components, nside, bandpass_freqs, central_freq=None, bandpass_weights=None, plot=False, pol=pol)
        all_maps[i,1] = get_extragalactic_comp_map(freq, nside, ellmax, agora_sims_dir, ksz_reionization_file=ksz_reionization_file, plot=False, pol=pol)
    if save_intermediate:
        pickle.dump(all_maps, open(f'{output_dir}/gal_and_extragal_before_beam.p', 'wb'))
    if verbose:
        print('Got maps of galactic and extragalactic components at 220, 150, and 90 GHz', flush=True)
    

    #get beam-convolved healpix maps at each frequency
    freq_maps = np.sum(all_maps, axis=1) #shape Nfreqs, 1 if not pol or 3 if pol, Npix
    if not pol: #index as all_maps[freq, gal or extragal, pixel], freqs in decreasing order
        beam_convolved_maps = np.zeros((3, 12*nside**2))
    if pol: #index as all_maps[freq, gal or extragal, I/Q/U, pixel], freqs in decreasing order
        beam_convolved_maps = np.zeros((3, 3, 12*nside**2))
    for freq in range(3):
        if freq==0:
            beamfile = f'{beam_dir}/coadd_pa4_f220_night_beam_tform_jitter_cmb.txt'
        elif freq==1:
            beamfile = f'{beam_dir}/coadd_pa5_f150_night_beam_tform_jitter_cmb.txt'
        elif freq==2:
            beamfile = f'{beam_dir}/coadd_pa6_f090_night_beam_tform_jitter_cmb.txt'
        beam_convolved_maps[freq] = apply_beam(freq_maps[freq], beamfile, pol)
    if save_intermediate:
        pickle.dump(beam_convolved_maps, open(f'{output_dir}/beam_convolved_maps.p', 'wb'))
    if verbose:
        print('Got beam-convolved maps', flush=True)


    #convert each frequency map from healpix to CAR
    car_maps = []
    for freq in range(3):
        car_map = healpix2CAR(beam_convolved_maps[freq], ellmax, pol)
        car_maps.append(car_map)
        if freq==0:
            enmap.write_map('sim_220GHz', car_map)
        elif freq==1:
            enmap.write_map('sim_150GHz', car_map)
        else:
            enmap.write_map('sim_90GHz', car_map)
    if verbose:
        print('Got CAR maps', flush=True)

    return car_maps


if __name__=='__main__':

    ##### DEFINITIONS AND FILE PATHS, MODIFY HERE #####
    nside = 32 #nside at which to create maps, ideally 8192
    ellmax = 50 #maximum ell for which to compute power spectra, ideally 10000
    galactic_components = ['d1', 's1', 'a1', 'f1'] #pysm predefined galactic component strings
    pol = False #whether or not to compute E-mode maps
    passband_file = "passbands_20220316/AdvACT_Passbands.h5" #file containing ACT passband information
    agora_sims_dir = 'agora' #directory containing agora extragalactic sims
    ksz_reionization_file = 'agora/FBN_kSZ_PS_patchy.txt' #file with columns ell, D_ell (uK^2) of patchy kSZ, set to None if no such file
    beam_dir = 'beams'

    main(nside, ellmax, galactic_components, passband_file, agora_sims_dir, beam_dir, pol=pol, 
        ksz_reionization_file=ksz_reionization_file, save_intermediate=True, verbose=True)