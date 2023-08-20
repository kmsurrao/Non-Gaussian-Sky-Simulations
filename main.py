import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import pickle
from pixell import enmap
from bandpass_integration import get_bandpass_freqs_and_weights, get_galactic_comp_map, get_extragalactic_comp_map
from beams import apply_beam, healpix2CAR
from make_plots import plot_outputs
from galactic_mask import get_mask_deconvolved_spectrum, plot_and_save_mask_deconvolved_spectra

def main(nside, ellmax, galactic_components, passband_file, agora_sims_dir, beam_dir, pol=False, 
        ksz_reionization_file=None, save_intermediate=False, verbose=False, output_dir='outputs'):
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
    output_dir: str, directory in which to put output files if save_intermediate is True
    

    RETURNS
    -------
    car_maps: list of CAR maps at frequencies 220, 150, and 90 GHz. If pol, each 
            map contains [I,Q,U], otherwise just I
    '''

    #set up output directory if save_intermediate is True
    if save_intermediate and not os.path.isdir(output_dir):
        env = os.environ.copy()
        subprocess.call(f'mkdir {output_dir}', shell=True, env=env)


    #get bandpass frequencies and weights
    all_bandpass_freqs = []
    all_bandpass_weights = []
    for central_freq in [220, 150, 90]:
        bandpass_freqs, bandpass_weights = get_bandpass_freqs_and_weights(central_freq, passband_file)
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
        output_dir_here = output_dir if save_intermediate else None
        all_maps[i,0] = get_galactic_comp_map(galactic_components, nside, all_bandpass_freqs[i], central_freq=None, bandpass_weights=all_bandpass_weights[i], pol=pol, ellmax=ellmax, output_dir=output_dir_here)
        all_maps[i,1] = get_extragalactic_comp_map(freq, nside, ellmax, agora_sims_dir, ksz_reionization_file=ksz_reionization_file, pol=pol, output_dir=output_dir_here)
    if save_intermediate:
        pickle.dump(all_maps, open(f'{output_dir}/gal_and_extragal_before_beam.p', 'wb'))
    if verbose:
        print('Got maps of galactic and extragalactic components at 220, 150, and 90 GHz', flush=True)
    

    #get beam-convolved healpix maps at each frequency
    freq_maps = np.sum(all_maps, axis=1) #shape Nfreqs, 1 if not pol or 3 if pol, Npix
    if not pol: #index as all_maps[freq, pixel], freqs in decreasing order
        beam_convolved_maps = np.zeros((3, 12*nside**2))
    if pol: #index as all_maps[freq, I/Q/U, pixel], freqs in decreasing order
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


    # #convert each frequency map from healpix to CAR
    # car_maps = []
    # freqs = [220, 150, 90]
    # for freq in range(3):
    #     car_map = healpix2CAR(beam_convolved_maps[freq], ellmax, pol)
    #     car_maps.append(car_map)
    #     enmap.write_map(f'sim_{freqs[freq]}GHz', car_map)
    # if verbose:
    #     print('Got CAR maps', flush=True)

    # return car_maps


if __name__=='__main__':

    ##### DEFINITIONS AND FILE PATHS, MODIFY HERE #####
    base_dir = '/scratch/09334/ksurrao/ACT_sims' #'/scratch/09334/ksurrao/ACT_sims' for Stampede, '.' for terremoto
    nside = 8192 #nside at which to create maps, ideally 8192
    ellmax = 10000 #maximum ell for which to compute power spectra, ideally 10000
    galactic_components = ['d10', 's5', 'a1', 'f1'] #pysm predefined galactic component strings
    pol = False #whether or not to compute E-mode maps
    passband_file = f'{base_dir}/passbands_20220316/AdvACT_Passbands.h5' #file containing ACT passband information
    agora_sims_dir = f'{base_dir}/agora' #directory containing agora extragalactic sims, /global/cfs/cdirs/act/data/agora_sims on NERSC
    ksz_reionization_file = f'{agora_sims_dir}/FBN_kSZ_PS_patchy.txt' #file with columns ell, D_ell (uK^2) of patchy kSZ, set to None if no such file
    beam_dir = f'{base_dir}/beams', #/global/cfs/cdirs/act/data/adriaand/beams/20230130_beams on NERSC
    output_dir = f'{base_dir}/outputs_nside{nside}' #directory in which to put outputs (can be full path)
    plot = True #whether to produce plots
    plot_dir = f'{base_dir}/plots_nside{nside}' #only needs to be defined if plot==True
    plots_to_make = ['passband', 'gal_and_extragal_comps', 'freq_maps_no_beam', 'beam_convolved_maps', 'all_comp_spectra'] #only needs to be defined if plot==True

    main(nside, ellmax, galactic_components, passband_file, agora_sims_dir, beam_dir, pol=pol, 
        ksz_reionization_file=ksz_reionization_file, save_intermediate=True, verbose=True, output_dir=output_dir)
    
    if plot:
        plot_outputs(output_dir, plot_dir, ellmax, pol, which=plots_to_make)
    
    # #plot and save mask-deconvolved spectra using 70% fsky galactic mask
    # ells_per_bin = 50
    # beam_convolved_maps = pickle.load(open(f'{output_dir}/beam_convolved_maps.p', 'rb'))
    # mask_file = f'{base_dir}/HFI_Mask_GalPlane-apo5_2048_R2.00.fits'
    # plot_and_save_mask_deconvolved_spectra(nside, output_dir, plot_dir, beam_convolved_maps, mask_file, ellmax, ells_per_bin, pol=pol)