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
    if inp.ellmax is None:
        ellmax = 3*inp.nside-1
    else:
        ellmax = inp.ellmax
    if not inp.pol:
        power_spectra = np.zeros((len(inp.galactic_components), ellmax+1), dtype=np.float32)
    else:
        power_spectra = np.zeros((len(inp.galactic_components), 6, ellmax+1), dtype=np.float32) #6 for TT, EE, BB, TE, EB, TE

    if inp.mask_file and inp.ells_per_bin:
        mask = hp.read_map(inp.mask_file, field=(3)) #70% fsky
        mask = hp.ud_grade(mask, inp.nside)

    for c, comp in enumerate(inp.galactic_components):
        if central_freq is None: 
            central_freq = np.mean(bandpass_freqs)
        print(f'Generating pysm bandpass-integrated map for {comp}, {central_freq} GHz', flush=True)
        sky = pysm3.Sky(nside=inp.nside, preset_strings=[comp])
        map_ = sky.get_emission(bandpass_freqs*u.GHz, bandpass_weights)
        map_ = map_.to(u.K_CMB, equivalencies=u.cmb_equivalencies(central_freq*u.GHz))
        all_maps.append(map_)
        map_to_use = map_[0] if not inp.pol else map_
        power_spectra[c] = hp.anafast(map_to_use, lmax=ellmax, pol=inp.pol)
        if inp.mask_file and inp.ells_per_bin:
            ell_eff, Cl = compute_master(inp, mask, map_to_use)
            if c==0: #first component
                pickle.dump(ell_eff, open(f'{inp.output_dir}/ell_eff.p', 'wb'))
                Nbins = len(ell_eff)
                if not inp.pol:
                    deconvolved_spectra = np.zeros((len(inp.galactic_components), Nbins), dtype=np.float32)
                else:
                    deconvolved_spectra = np.zeros((len(inp.galactic_components), 6, Nbins), dtype=np.float32) #6 for TT, EE, BB, TE, EB, TE
            deconvolved_spectra[c] = Cl

    pickle.dump(power_spectra, open(f'{inp.output_dir}/gal_comp_spectra_{central_freq}.p', 'wb'))
    if inp.mask_file and inp.ells_per_bin:
        pickle.dump(deconvolved_spectra, open(f'{inp.output_dir}/gal_comp_mask_deconvolved_spectra_{central_freq}.p', 'wb'))
    
    all_maps = np.array(all_maps)
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
                types = ['I', 'Q', 'U']
                if inp.pol:
                    plt.clf()
                    plt.hist(cmap)
                    plt.savefig(f'{inp.plot_dir}/pixel_hist_{freq}ghz_I_{comp}_before.png')
                else:
                    for t in range(3):
                        plt.clf()
                        plt.hist(cmap)
                        plt.savefig(f'{inp.plot_dir}/pixel_hist_{freq}ghz_{types[t]}_{comp}_before.png')

            cmap = initial_masking(inp, cmap, map150)

            if plot_hist:
                types = ['I', 'Q', 'U']
                if inp.pol:
                    plt.clf()
                    plt.hist(cmap)
                    plt.savefig(f'{inp.plot_dir}/pixel_hist_{freq}ghz_I_{comp}_before.png')
                else:
                    for t in range(3):
                        plt.clf()
                        plt.hist(cmap)
                        plt.savefig(f'{inp.plot_dir}/pixel_hist_{freq}ghz_{types[t]}_{comp}_before.png')


        all_maps.append(cmap)
        power_spectra[c] = hp.anafast(cmap, lmax=inp.ellmax, pol=inp.pol)

    if inp.ksz_reionization_file is not None:
        ells_ksz_patchy, ksz_patchy = np.transpose(np.loadtxt(inp.ksz_reionization_file))
        f = interp1d(ells_ksz_patchy, ksz_patchy, fill_value="extrapolate", kind='cubic')
        ells = np.arange(inp.ellmax+1)
        ksz_patchy = f(ells)
        ksz_patchy = 10**(-6)*ksz_patchy/((ells)*(ells+1))*(2*np.pi)
        ksz_patchy[0] = 0
        ksz_patchy_realization = hp.synfast(ksz_patchy, inp.nside)
        if not inp.pol:
            power_spectra[-1] = ksz_patchy
            all_maps.append(ksz_patchy_realization)
        else:
            power_spectra[-1, 0] = ksz_patchy
            all_maps.append([ksz_patchy_realization, np.zeros_like(ksz_patchy_realization), np.zeros_like(ksz_patchy_realization)])

    pickle.dump(power_spectra, open(f'{inp.output_dir}/extragal_comp_spectra_{freq}.p', 'wb'))
    
    all_maps = np.array(all_maps)
    return all_maps


if __name__=='__main__':

    # main input file containing most specifications 
    parser = argparse.ArgumentParser(description="Non-gaussian full sky simulations.")
    parser.add_argument("--config", default="example_yaml_files/stampede.yaml")
    args = parser.parse_args()
    input_file = args.config

    # read in the input file and set up relevant info object
    inp = Info(input_file)
    
    all_bandpass_freqs = pickle.load(open(f'{inp.output_dir}/all_bandpass_freqs.p', 'rb'))
    all_bandpass_weights = pickle.load(open(f'{inp.output_dir}/all_bandpass_weights.p', 'rb'))
    

    if not inp.pol: #index as all_maps[freq, gal or extragal, pixel], freqs in decreasing order
        all_maps = np.zeros((3, 2, 12*inp.nside**2), dtype=np.float32)
    if inp.pol: #index as all_maps[freq, gal or extragal, I/Q/U, pixel], freqs in decreasing order
        all_maps = np.zeros((3, 2, 3, 12*inp.nside**2), dtype=np.float32)
    for i, freq in enumerate([220, 150, 90]):
        print(f'On frequency {freq}', flush=True)
        all_maps[i,0] = get_galactic_comp_maps(inp, all_bandpass_freqs[i], 
                central_freq=freq, bandpass_weights=all_bandpass_weights[i])
        all_maps[i,1] = get_extragalactic_comp_maps(inp, freq)
    pickle.dump(all_maps, open(f'{inp.output_dir}/gal_and_extragal_before_beam.p', 'wb'))
    print('Got maps of galactic and extragalactic components at 220, 150, and 90 GHz', flush=True)