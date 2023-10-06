import numpy as np
import healpy as hp
import pysm3
import pysm3.units as u
import h5py
from scipy.interpolate import interp1d
import pickle
import matplotlib.pyplot as plt
import multiprocessing as mp
from galactic_mask import compute_master
from apply_flux_cut import initial_masking
from utils import plot_histogram


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


def get_all_bandpass_freqs_and_weights(inp):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    
    RETURNS
    -------
    all_bandpass_freqs: (Nfreqs, Nfreqs_in_passband) ndarray of frequencies in the 
        passband (in GHz)
    all_bandpass_weights: (Nfreqs, Nfreqs_in_passband) ndarray of weights for each 
        frequency in the passband (with nu^2 divided out)
    '''
    all_bandpass_freqs = []
    all_bandpass_weights = []
    for central_freq in [220, 150, 90]:
        bandpass_freqs, bandpass_weights = get_bandpass_freqs_and_weights(central_freq, inp.passband_file)
        all_bandpass_freqs.append(bandpass_freqs)
        all_bandpass_weights.append(bandpass_weights)
    return all_bandpass_freqs, all_bandpass_weights



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
    gal_comps: ndarray containing bandpass integrated map of all galactic components in healpix format
        if pol: gal_comps has shape (3 for IQU, Npix)
        if not pol: gal_comps has shape (Npix, )
    '''

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
            if c==0:
                gal_comps = map_
            else:
                gal_comps += map_
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
    
        gal_comps = np.array(gal_comps, dtype=np.float32)
        return gal_comps if inp.pol else gal_comps[0]
    
    print(f'Generating pysm bandpass-integrated map for all galactic components, {central_freq} GHz', flush=True)
    sky = pysm3.Sky(nside=inp.nside, preset_strings=inp.galactic_components)
    map_ = sky.get_emission(bandpass_freqs*u.GHz, bandpass_weights)
    map_ = map_.to(u.K_CMB, equivalencies=u.cmb_equivalencies(central_freq*u.GHz))
    gal_comps = np.array(map_, dtype=np.float32)
    return gal_comps if inp.pol else gal_comps[0]



def get_extragalactic_comp_maps(inp, freq, plot_hist=True):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input specifications
    freq: int, central bandpass frequency (in GHz)
    plot_hist: Bool, whether to plot histogram values before and after inpainting
        for CIB and radio components
    
    RETURNS
    -------
    extragal_comps: ndarray containing bandpass integrated map of all extragalactic components in healpix format
        if pol: extragal_comps has shape (3 for IQU, Npix)
        if not pol: extragal_comps has shape (Npix,)
    '''

    comps = ['lcmbNG', 'lkszNGbahamas80', 'ltszNGbahamas80', 'lcibNG', 'lradNG']

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
                plot_histogram(inp, cmap, freq, comp, string='before')
            cmap = initial_masking(inp, cmap, map150)
            if plot_hist:
                plot_histogram(inp, cmap, freq, comp, string='after')

        if c==0:
            extragal_comps = cmap
        else:
            extragal_comps += cmap
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
            extragal_comps += ksz_patchy_realization
        else:
            if 'component_power_spectra' in inp.checks:
                power_spectra[-1, 0] = ksz_patchy
            extragal_comps[0] += ksz_patchy_realization

    if 'component_power_spectra' in inp.checks:
        pickle.dump(power_spectra, open(f'{inp.output_dir}/extragal_comp_spectra_{freq}.p', 'wb'))
    
    extragal_comps = np.array(extragal_comps, dtype=np.float32)
    return extragal_comps



def combined_map_before_beam(inp, bandpass_freqs, central_freq=None, bandpass_weights=None, plot_hist=True):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input specifications
    bandpass_freqs: numpy array of frequencies in the passband (in GHz)
    central_freq: float, central banpass frequency (in GHz)
    bandpass_weights: numpy array of weights for each frequency in the passband 
                     (length should be the same as bandpass_freqs)
    plot_hist: Bool, whether to plot histogram values before and after inpainting
        for CIB and radio components
    
    RETURNS
    -------
    None

    SAVES
    -----
    combined_map: ndarray containing bandpass integrated map of all galactic and
        extragalactic components in healpix format
        if pol: combined_map has shape (3 for IQU, Npix)
        if not pol: combined_map has shape (Npix, )
    '''
    gal_comps = get_galactic_comp_maps(inp, bandpass_freqs, central_freq=central_freq, bandpass_weights=bandpass_weights)
    extragal_comps = get_extragalactic_comp_maps(inp, central_freq, plot_hist=True)
    combined_map = gal_comps + extragal_comps
    combined_map = np.array(combined_map, dtype=np.float32)
    pickle.dump(combined_map, open(f'{inp.output_dir}/combined_map_{central_freq}.p', 'wb'), protocol=4)
    return None



def get_all_bandpassed_maps(inp, all_bandpass_freqs, all_central_freqs=None, all_bandpass_weights=None, parallel=True):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input specifications
    all_bandpass_freqs: (Nfreqs, Nfreqs_in_passband) ndarray of frequencies in the 
        passband (in GHz)
    all_central_freqs: (Nfreqs, ) array-like of floats, containing central banpass 
        frequencies (in GHz)
    all_bandpass_weights: (Nfreqs, Nfreqs_in_passband) ndarray of weights for each 
        frequency in the passband 
    parallel: Bool, whether to run in parallel (may cause memory issues if nside too high)
    
    RETURNS
    -------
    all_maps: ndarray containing bandpass integrated maps of 
        all galactic and extragalactic combined maps in healpix format
        if pol: all_maps has shape (Nfreqs, 3 for IQU, Npix)
        if not pol: all_maps has shape (Nfreqs, Npix)
    '''
    plot_hist = True if inp.plot_dir else False
    if parallel:
        pool = mp.Pool(3)
        tmp = pool.starmap(combined_map_before_beam, [(inp, all_bandpass_freqs[i], all_central_freqs[i], all_bandpass_weights[i], plot_hist) for i in range(3)])
        pool.close()
        plt.close('all')
    else:
        for i in range(3):
            combined_map_before_beam(inp, all_bandpass_freqs[i], all_central_freqs[i], all_bandpass_weights[i], plot_hist)
    if inp.pol:
        all_maps = np.zeros((3, 3, 12*inp.nside**2), dtype=np.float32)
    else:
        all_maps = np.zeros((3, 12*inp.nside**2), dtype=np.float32)
    for i, freq in enumerate(all_central_freqs):
        all_maps[i] = pickle.load(open(f'{inp.output_dir}/combined_map_{freq}.p', 'rb'))
    return all_maps