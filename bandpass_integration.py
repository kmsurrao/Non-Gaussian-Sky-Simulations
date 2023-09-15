import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pysm3
import pysm3.units as u
import h5py
from scipy.interpolate import interp1d
import argparse
import pickle
from galactic_mask import compute_master
from input import Info
from apply_flux_cut import initial_masking


def get_bandpass_freqs_and_weights(central_freq, passband_file):
    '''
    ARGUMENTS
    ---------
    central_freq: int or float, central frequency of the passband
    passband_file: str, path to h5 file containing passband information
    
    RETURNS
    -------
    frequencies: numpy array of frequencies (GHz) in the passband
    bandpass_weights: numpy array of bandpass weights (with nu^2 divided out)
    '''

    with h5py.File(passband_file, "r") as f:
        assert central_freq in {220, 150, 90}, f"No bandpass information for frequency {central_freq} GHz"
        if central_freq == 220:
            band = f[list(f.keys())[1]] #PA4_220
        elif central_freq == 150:
            band = f[list(f.keys())[3]] #PA5_150
        elif central_freq == 90:
            band = f[list(f.keys())[4]] #PA6_90

        frequencies_key = list(band.keys())[0]
        mean_band_key = list(band.keys())[4]
        frequencies = band[frequencies_key][()]
        mean_band = band[mean_band_key][()]
        bandpass_weights = mean_band/frequencies**2
            
    return frequencies, bandpass_weights



def get_galactic_comp_maps(inp, bandpass_freqs, central_freq=None, bandpass_weights=None):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input specifications
    bandpass_freqs: numpy array of frequencies in the passband (in GHz)
    central_freq: float, central banpass frequency (in GHz)
    bandpass_weights: numpy array of weights for each frequency in the passband 
                     (length should be the same as bandpass_freqs)
    
    RETURNS
    -------
    all_maps: ndarray containing bandpass integrated maps of each galactic component in healpix format
        if pol: all_maps has shape (num_galactic_components, 3 for IQU, Npix)
        if not pol: all_maps has shape (num_galactic_components, Npix)
    '''

    all_maps = []
    if central_freq is None: 
        central_freq = np.mean(bandpass_freqs)

    if 'component_power_spectra' in inp.checks:
        if not inp.pol:
            power_spectra = np.zeros((len(inp.galactic_components), inp.ellmax+1), dtype=np.float32)
        else:
            power_spectra = np.zeros((len(inp.galactic_components), 6, inp.ellmax+1), dtype=np.float32) #6 for TT, EE, BB, TE, EB, TE

    if 'component_mask_deconvolution' in inp.checks:
        mask = hp.read_map(inp.mask_file, field=(3)) #70% fsky
        mask = hp.ud_grade(mask, inp.nside)

    if 'component_power_spectra' in inp.checks or 'component_mask_deconvolution' in inp.checks:
        
        for c, comp in enumerate(inp.galactic_components):

            print(f'Generating pysm bandpass-integrated map for {comp}, {central_freq} GHz', flush=True)
            sky = pysm3.Sky(nside=inp.nside, preset_strings=[comp])
            map_ = sky.get_emission(bandpass_freqs*u.GHz, bandpass_weights)
            map_ = map_.to(u.K_CMB, equivalencies=u.cmb_equivalencies(central_freq*u.GHz))
            all_maps.append(map_)
            map_to_use = map_[0] if not inp.pol else map_

            if 'component_power_spectra' in inp.checks:
                power_spectra[c] = hp.anafast(map_to_use, lmax=inp.ellmax, pol=inp.pol)

            if 'component_mask_deconvolution' in inp.checks:
                ell_eff, Cl = compute_master(inp, mask, map_to_use)
                if c==0: #first component
                    pickle.dump(ell_eff, open(f'{inp.output_dir}/ell_eff.p', 'wb'))
                    Nbins = len(ell_eff)
                    if not inp.pol:
                        deconvolved_spectra = np.zeros((len(inp.galactic_components), Nbins), dtype=np.float32)
                    else:
                        deconvolved_spectra = np.zeros((len(inp.galactic_components), 6, Nbins), dtype=np.float32) #6 for TT, EE, BB, TE, EB, TE
                deconvolved_spectra[c] = Cl

        if 'component_power_spectra' in inp.checks:
            pickle.dump(power_spectra, open(f'{inp.output_dir}/gal_comp_spectra_{central_freq}.p', 'wb'))
        if 'component_mask_deconvolution' in inp.checks:
            pickle.dump(deconvolved_spectra, open(f'{inp.output_dir}/gal_comp_mask_deconvolved_spectra_{central_freq}.p', 'wb'))
    
        all_maps = np.array(all_maps)
        return all_maps if inp.pol else all_maps[:,0]
    
    print(f'Generating pysm bandpass-integrated map for all galactic components, {central_freq} GHz', flush=True)
    sky = pysm3.Sky(nside=inp.nside, preset_strings=inp.galactic_components)
    map_ = sky.get_emission(bandpass_freqs*u.GHz, bandpass_weights)
    map_ = map_.to(u.K_CMB, equivalencies=u.cmb_equivalencies(central_freq*u.GHz))
    all_maps.append(map_)
    return all_maps if inp.pol else all_maps[:,0]



def get_extragalactic_comp_maps(inp, freq, plot_hist=True):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input specifications
    freq: int, central banpass frequency (in GHz)
    plot_hist: Bool, whether to plot histogram values before and after inpainting
        for CIB and radio components
    
    RETURNS
    -------
    all_maps: ndarray containing bandpass integrated maps of each galactic component in healpix format
        if pol: all_maps has shape (num_galactic_components, 3 for IQU, Npix)
        if not pol: all_maps has shape (num_galactic_components, Npix)
    '''

    comps = ['lcmbNG', 'lkszNGbahamas80', 'ltszNGbahamas80', 'lcibNG', 'lradNG']
    all_maps = []

    if 'component_power_spectra' in inp.checks:
        if not inp.pol:
            power_spectra = np.zeros((len(comps)+1, inp.ellmax+1), dtype=np.float32) #CMB, kSZ, tSZ, CIB, radio, reionization kSZ
        else:
            power_spectra = np.zeros((len(comps)+1, 6, inp.ellmax+1), dtype=np.float32) #6 for TT, EE, BB, TE, EB, TE
    
    for c, comp in enumerate(comps):
        print(f'Processing agora map for {comp}, {freq} GHz', flush=True)

        if not inp.pol:
            cmap = 10**(-6)*hp.read_map(f'{inp.agora_sims_dir}/agora_act_{freq}ghz_{comp}_uk.fits')
        else:
            cmap = 10**(-6)*hp.read_map(f'{inp.agora_sims_dir}/agora_act_{freq}ghz_{comp}_uk.fits', field=[0,1,2])
        cmap = hp.ud_grade(cmap, inp.nside)
        
        if comp == 'lcibNG' or comp=='lradNG':
            if not inp.pol:
                map150 = 10**(-6)*hp.read_map(f'{inp.agora_sims_dir}/agora_act_150ghz_{comp}_uk.fits')
            else:
                map150 = 10**(-6)*hp.read_map(f'{inp.agora_sims_dir}/agora_act_150ghz_{comp}_uk.fits', field=[0,1,2])
            map150 = hp.ud_grade(map150, inp.nside)
            if plot_hist:
                plot_hist(inp, cmap, freq, comp)
            cmap = initial_masking(inp, cmap, map150)
            if plot_hist:
                plot_hist(inp, cmap, freq, comp)

        all_maps.append(cmap)
        if 'component_power_spectra' in inp.checks:
            power_spectra[c] = hp.anafast(cmap, lmax=inp.ellmax, pol=inp.pol)

    if inp.ksz_reionization_file is not None:
        ells_ksz_patchy, ksz_patchy = np.transpose(np.loadtxt(inp.ksz_reionization_file))
        f = interp1d(ells_ksz_patchy, ksz_patchy, fill_value="extrapolate", kind="cubic")
        ells = np.arange(inp.ellmax+1)
        ksz_patchy = f(ells)
        ksz_patchy = 10**(-12)*ksz_patchy/((ells)*(ells+1))*(2*np.pi)
        ksz_patchy[0] = 0
        ksz_patchy_realization = hp.synfast(ksz_patchy, inp.nside)
        if not inp.pol:
            if 'component_power_spectra' in inp.checks:
                power_spectra[-1] = ksz_patchy
            all_maps.append(ksz_patchy_realization)
        else:
            if 'component_power_spectra' in inp.checks:
                power_spectra[-1, 0] = ksz_patchy
            all_maps.append([ksz_patchy_realization, np.zeros_like(ksz_patchy_realization), np.zeros_like(ksz_patchy_realization)])

    if 'component_power_spectra' in inp.checks:
        pickle.dump(power_spectra, open(f'{inp.output_dir}/extragal_comp_spectra_{freq}.p', 'wb'))
    
    all_maps = np.array(all_maps)
    return all_maps