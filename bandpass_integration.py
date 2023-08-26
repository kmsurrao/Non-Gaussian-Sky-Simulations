import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pysm3
import pysm3.units as u
import h5py
from scipy.interpolate import interp1d
import pickle
from galactic_mask import get_mask_deconvolved_spectrum


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


def get_galactic_comp_map(components, nside, bandpass_freqs, central_freq=None, bandpass_weights=None, 
    pol=False, ellmax=None, output_dir=None, mask_file=None, ells_per_bin=None):
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
    mask_file: str, file galactic mask, set to None if not computing
        mask-deconvolved spectra
    ells_per_bin: int, number of ells per bin for NaMaster mask deconvolution, set to None
        if not computing mask-deconvolved spectra
    
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
        if mask_file and ells_per_bin:
            mask = hp.read_map(mask_file, field=(3)) #70% fsky
            mask = hp.ud_grade(mask, nside)
        for c, comp in enumerate(components):
            sky = pysm3.Sky(nside=nside, preset_strings=[comp])
            map_ = sky.get_emission(bandpass_freqs*u.GHz, bandpass_weights)
            if central_freq is None: central_freq = np.mean(bandpass_freqs)
            map_ = map_.to(u.K_CMB, equivalencies=u.cmb_equivalencies(central_freq*u.GHz))
            if total_map is None:
                total_map = map_
            else:
                total_map += map_
            map_to_use = map_[0] if not pol else map_
            power_spectra[c] = hp.anafast(map_to_use, lmax=ellmax, pol=pol)
            if mask_file and ells_per_bin:
                ell_eff, Cl = get_mask_deconvolved_spectrum(nside, mask, ells_per_bin, map_to_use, ellmax=ellmax, pol=pol)
                if c==0: #first component
                    pickle.dump(ell_eff, open(f'{output_dir}/ell_eff.p', 'wb'))
                    Nbins = len(ell_eff)
                    if not pol:
                        deconvolved_spectra = np.zeros((len(components), Nbins))
                    else:
                        deconvolved_spectra = np.zeros((len(components), 6, Nbins))
                deconvolved_spectra[c] = Cl
        pickle.dump(power_spectra, open(f'{output_dir}/gal_comp_spectra_{central_freq}.p', 'wb'))
        if mask_file and ells_per_bin:
            pickle.dump(deconvolved_spectra, open(f'{output_dir}/gal_comp_mask_deconvolved_spectra_{central_freq}.p', 'wb'))

    else:
        sky = pysm3.Sky(nside=nside, preset_strings=components)
        map_ = sky.get_emission(bandpass_freqs*u.GHz, bandpass_weights)
        if central_freq is None: central_freq = np.mean(bandpass_freqs)
        total_map = map_.to(u.K_CMB, equivalencies=u.cmb_equivalencies(central_freq*u.GHz))

    return total_map if pol else total_map[0]



def get_extragalactic_comp_map(freq, nside, ellmax, agora_sims_dir, ksz_reionization_file=None, pol=False, 
    output_dir=None, mask_file=None, ells_per_bin=None):
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
    mask_file: str, file galactic mask, set to None if not computing
        mask-deconvolved spectra
    ells_per_bin: int, number of ells per bin for NaMaster mask deconvolution, set to None
        if not computing mask-deconvolved spectra
    
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
        f = interp1d(ells_ksz_patchy, ksz_patchy, fill_value="extrapolate", kind='cubic')
        ells = np.arange(ellmax+1)
        ksz_patchy = f(ells)
        ksz_patchy = ksz_patchy/((ells)*(ells+1))*(2*np.pi)
        ksz_patchy[0] = 0
        ksz_patchy_realization = hp.synfast(ksz_patchy, nside)
        I += ksz_patchy_realization

    if output_dir:
        comps = ['lcmbNG', 'lkszNGbahamas80', 'ltszNGbahamas80', 'lcibNG', 'lradNG']
        if not pol:
            power_spectra = np.zeros((len(comps)+1, ellmax+1)) #CMB, kSZ, tSZ, CIB, radio, reionization kSZ
        else:
            power_spectra = np.zeros((len(comps)+1, 6, ellmax+1))
        if mask_file and ells_per_bin:
            mask = hp.read_map(mask_file, field=(3)) #70% fsky
            mask = hp.ud_grade(mask, nside)
        for c, comp in enumerate(comps):
            if not pol:
                cmap = 10**(-6)*hp.read_map(f'{agora_sims_dir}/agora_act_{freq}ghz_{comp}_uk.fits')
            else:
                cmap = 10**(-6)*hp.read_map(f'{agora_sims_dir}/agora_act_{freq}ghz_{comp}_uk.fits', field=[0,1,2])
            power_spectra[c] = hp.anafast(cmap, lmax=ellmax, pol=pol)
            if mask_file and ells_per_bin:
                ell_eff, Cl = get_mask_deconvolved_spectrum(nside, mask, ells_per_bin, cmap, ellmax=ellmax, pol=pol)
                if c==0: #first component
                    pickle.dump(ell_eff, open(f'{output_dir}/ell_eff.p', 'wb'))
                    Nbins = len(ell_eff)
                    if not pol:
                        deconvolved_spectra = np.zeros((len(comps)+1, Nbins))
                    else:
                        deconvolved_spectra = np.zeros((len(comps)+1, 6, Nbins))
                deconvolved_spectra[c] = Cl
        if not pol:
            power_spectra[-1] = ksz_patchy
        else:
            power_spectra[-1, 0] = ksz_patchy
        if mask_file and ells_per_bin:
            ell_eff, Cl = get_mask_deconvolved_spectrum(nside, mask, ells_per_bin, ksz_patchy_realization, ellmax=ellmax, pol=False)
            if not pol:
                deconvolved_spectra[-1] = Cl
            else:
                deconvolved_spectra[-1, 0] = Cl
            pickle.dump(deconvolved_spectra, open(f'{output_dir}/extragal_comp_mask_deconvolved_spectra_{freq}.p', 'wb'))
        pickle.dump(power_spectra, open(f'{output_dir}/extragal_comp_spectra_{freq}.p', 'wb'))
       

    return np.array([I,Q,U]) if pol else I


if __name__=='__main__':
    nside = 8192
    ellmax = 10000
    ells_per_bin = 50 #number of ells per bin for NaMaster mask deconvolution
    pol = False
    galactic_components = ['d10', 's5', 'a1', 'f1'] #pysm predefined galactic component strings
    base_dir = '/scratch/09334/ksurrao/ACT_sims' #'/scratch/09334/ksurrao/ACT_sims' for Stampede, '.' for terremoto
    agora_sims_dir = f'{base_dir}/agora' #directory containing agora extragalactic sims, /global/cfs/cdirs/act/data/agora_sims on NERSC
    ksz_reionization_file = f'{agora_sims_dir}/FBN_kSZ_PS_patchy.txt' #file with columns ell, D_ell (uK^2) of patchy kSZ, set to None if no such file
    mask_file = f'{base_dir}/HFI_Mask_GalPlane-apo5_2048_R2.00.fits' #file containing Planck galactic masks
    output_dir = f'{base_dir}/outputs_nside{nside}' #directory in which to put outputs (can be full path)
    
    all_bandpass_freqs = pickle.load(open(f'{output_dir}/all_bandpass_freqs.p', 'rb'))
    all_bandpass_weights = pickle.load(open(f'{output_dir}/all_bandpass_weights.p', 'rb'))
    

    if not pol: #index as all_maps[freq, gal or extragal, pixel], freqs in decreasing order
        all_maps = np.zeros((3, 2, 12*nside**2))
    if pol: #index as all_maps[freq, gal or extragal, I/Q/U, pixel], freqs in decreasing order
        all_maps = np.zeros((3, 2, 3, 12*nside**2))
    for i, freq in enumerate([220, 150, 90]):
        print(f'On frequency {freq}', flush=True)
        all_maps[i,0] = get_galactic_comp_map(galactic_components, nside, all_bandpass_freqs[i], 
                    central_freq=freq, bandpass_weights=all_bandpass_weights[i], pol=pol, 
                    ellmax=ellmax, output_dir=output_dir, mask_file=mask_file, ells_per_bin=ells_per_bin)
        all_maps[i,1] = get_extragalactic_comp_map(freq, nside, ellmax, agora_sims_dir, 
                    ksz_reionization_file=ksz_reionization_file, pol=pol, output_dir=output_dir,
                    mask_file=mask_file, ells_per_bin=ells_per_bin)
    pickle.dump(all_maps, open(f'{output_dir}/gal_and_extragal_before_beam.p', 'wb'))
    print('Got maps of galactic and extragalactic components at 220, 150, and 90 GHz', flush=True)