import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pysm3
import pysm3.units as u
import h5py
from scipy.interpolate import interp1d
import pickle


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
        
        if central_freq == 220:
            band = f[list(f.keys())[1]] #PA4_220
        elif central_freq == 150:
            band = f[list(f.keys())[3]] #PA5_150
        elif central_freq == 90:
            band = f[list(f.keys())[4]] #PA6_90
        else:
            print(f"No information for frequency {central_freq} GHz")
            return

        frequencies_key = list(band.keys())[0]
        mean_band_key = list(band.keys())[4]
        frequencies = band[frequencies_key][()]
        mean_band = band[mean_band_key][()]
        bandpass_weights = mean_band/frequencies**2
            
    return frequencies, bandpass_weights


def get_galactic_comp_map(components, nside, bandpass_freqs, central_freq=None, bandpass_weights=None, pol=False, ellmax=None, output_dir=None):
    '''
    ARGUMENTS
    ---------
    components: list of preset strings for pysm components
    nside: int, resolution parameter for maps
    bandpass_freqs: numpy array of frequencies in the passband (in GHz)
    central_freq: float, central banpass frequency (in GHz)
    bandpass_weights: numpy array of weights for each frequency in the passband 
                     (length should be the same as bandpass_freqs)
    plot: Bool, whether to make healpy plot of temperature map
    pol: Bool, returns [I,Q,U] if true, otherwise just I
    ellmax: int, maximum ell for which to compute power spectra of each component
    output_dir: str, directory in which to put power spectrum pickle file, 
                power spectrum not saved if output_dir is None
    
    RETURNS
    -------
    if pol: returns bandpass integrated [I,Q,U] in healpix format
    if not pol: returns bandpass integrated temperature (intensity) map only in healpix format
    '''

    total_map = None
    if output_dir:
        if ellmax is None:
            ellmax = 3*nside-1
        if not pol:
            power_spectra = np.zeros((len(components), ellmax+1))
        else:
            power_spectra = np.zeros((len(components), 6, ellmax+1))
        for c, comp in enumerate(components):
            sky = pysm3.Sky(nside=nside, preset_strings=[comp])
            map_ = sky.get_emission(bandpass_freqs*u.GHz, bandpass_weights)
            if central_freq is None: central_freq = np.mean(bandpass_freqs)
            map_ = map_.to(u.K_CMB, equivalencies=u.cmb_equivalencies(central_freq*u.GHz))
            if total_map is None:
                total_map = map_
            else:
                total_map += map_
            if not pol:
                power_spectra[c] = hp.anafast(map_[0], lmax=ellmax)
            else:
                power_spectra[c] = hp.anafast(map_, lmax=ellmax, pol=True)
        pickle.dump(power_spectra, open(f'{output_dir}/gal_comp_spectra_{central_freq}.p', 'wb'))

    else:
        sky = pysm3.Sky(nside=nside, preset_strings=components)
        map_ = sky.get_emission(bandpass_freqs*u.GHz, bandpass_weights)
        if central_freq is None: central_freq = np.mean(bandpass_freqs)
        total_map = map_.to(u.K_CMB, equivalencies=u.cmb_equivalencies(central_freq*u.GHz))

    return total_map if pol else total_map[0]

def get_extragalactic_comp_map(freq, nside, ellmax, agora_sims_dir, ksz_reionization_file=None, pol=False, output_dir=None):
    '''
    ARGUMENTS
    ---------
    freq: int, central banpass frequency (in GHz)
    nside: int, resolution parameter for maps
    ellmax: int, maximum ell for which to compute power spectra
    agora_sims_dir: str, directory containing agora extragalactic sims with ACT passbands
    ksz_reionization_file: str, file containing Cl for reionization kSZ
    pol: Bool, returns [I,Q,U] if true, otherwise just I
    output_dir: str, directory in which to put power spectrum pickle file, 
            power spectrum not saved if output_dir is None
    
    RETURNS
    -------
    if pol: returns bandpass integrated numpy array of [I,Q,U] in healpix format, units of K
    if not pol: returns bandpass integrated temperature (intensity) map only in healpix format, units of K
    '''

    filename = f'{agora_sims_dir}/agora_act_{freq}ghz_lcmbNG_lcibNG_ltszNGbahamas80_lkszNGbahamas80_lradNG_uk.fits'
    if pol:
        I,Q,U = 10**(-6)*hp.read_map(filename, field=[0,1,2])
        I = hp.ud_grade(I, nside)
        Q = hp.ud_grade(Q, nside)
        U = hp.ud_grade(U, nside)
    else:
        I = 10**(-6)*hp.read_map(filename)
        I = hp.ud_grade(I, nside)

    if ksz_reionization_file is not None:
        ells_ksz_patchy, ksz_patchy = np.transpose(np.loadtxt(ksz_reionization_file))
        ksz_patchy *= 10**(-12) #convert from uK^2 to K^2
        f = interp1d(ells_ksz_patchy, ksz_patchy, fill_value="extrapolate", kind='cubic')
        ells = np.arange(ellmax+1)
        ksz_patchy = f(ells)
        ksz_patchy = ksz_patchy/((ells)*(ells+1))*(2*np.pi)
        ksz_patchy[0] = 0
        ksz_patchy_realization = hp.synfast(ksz_patchy, nside)
        I += ksz_patchy_realization

    if output_dir:
        freqs = [220, 150, 90]
        comps = ['lcmbNG', 'lkszNGbahamas80', 'ltszNGbahamas80', 'lcibNG', 'lradNG']
        for freq in freqs:
            if not pol:
                power_spectra = np.zeros((len(comps)+1, ellmax+1)) #CMB, kSZ, tSZ, CIB, radio, reionization kSZ
            else:
                power_spectra = np.zeros((len(comps)+1, 6, ellmax+1))
            for c, comp in enumerate(comps):
                if not pol:
                    cmap = hp.read_map(f'{agora_sims_dir}/agora_act_{freq}ghz_{comp}_uk.fits')
                else:
                    cmap = hp.read_map(f'{agora_sims_dir}/agora_act_{freq}ghz_{comp}_uk.fits', field=[0,1,2])
                power_spectra[c] = hp.anafast(cmap, lmax=ellmax, pol=pol)
            if not pol:
                power_spectra[-1] = ksz_patchy
            else:
                power_spectra[-1, 0] = ksz_patchy
            pickle.dump(power_spectra, open(f'{output_dir}/extragal_comp_spectra_{freq}.p', 'wb'))

    return np.array([I,Q,U]) if pol else I