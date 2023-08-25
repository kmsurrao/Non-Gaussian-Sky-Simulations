import numpy as np
import healpy as hp
import pickle
import pymaster as nmt
import matplotlib.pyplot as plt

def get_mask_deconvolved_spectrum(nside, mask, ells_per_bin, map1, map2=None, ellmax=None, pol=False):
    '''
    ARGUMENTS
    ---------
    nside: int, resolution parameter for maps
    mask: 1D numpy array in healpix format containing mask
    ells_per_bin: int, number of ells to put in a bin
    map1: 1D (if not pol) or 3D (if pol) numpy array in healpix format containing map
    map2: 1D (if not pol) or 3D (if pol) numpy array in healpix format containing map2 (if different from map1)
    ellmax: int, maximum ell for which to compute results (if not given, goes to 3*nside)
    pol: Bool, True if passing in [I,Q,U], False if just I
    
    RETURNS
    -------
    ell_arr: array of effective ell values for bins
    cl_12: array containing binned mask-deconvolved power spectrum
        1D if not pol, 3D if pol (TT, EE, BB)
    ''' 

    if ellmax:
        b = nmt.NmtBin.from_lmax_linear(ellmax, ells_per_bin, is_Dell=True)
    else:
        b = nmt.NmtBin.from_nside_linear(nside, ells_per_bin, is_Dell=True)
    
    f_1 = nmt.NmtField(mask, [map1])
    if map2:
        f_2 = nmt.NmtField(mask, [map2])
    else:
        f_2 = f_1
    dl_12 = nmt.compute_full_master(f_1, f_2, b)
    ell_arr = b.get_effective_ells()
    to_dl = ell_arr*(ell_arr+1)/2/np.pi

    if not pol:
        return ell_arr, dl_12/to_dl
    
    else:
        f_1_pol = nmt.NmtField(mask, map1[1:])
        if map2:
            f_2_pol = nmt.NmtField(mask, map2[1:])
        else:
            f_2_pol = f_1_pol
        dl_12_pol = nmt.compute_full_master(f_1_pol, f_2_pol, b)
        return ell_arr, np.array([dl_12/to_dl, dl_12_pol[0]/to_dl, dl_12_pol[3]/to_dl])
    

def plot_and_save_mask_deconvolved_spectra(nside, output_dir, plot_dir, maps, mask_file, ellmax, ells_per_bin, pol=False, save_only=False, plot_only=False):
    '''
    ARGUMENTS
    ---------
    nside: int, resolution parameter for maps
    output_dir: str, directory in which output files were put
    plot_dir: str, directory in which to save plots
    maps: ndarray containing beam-convolved maps
        index as maps[freq][I,Q,U][pixel] if pol
        index as maps[freq][pixel] if not pol
    mask_file: str, file containing Planck mask data
    ellmax: int, maximum ell for which to compute results
    ells_per_bin: int, number of ells to put in a bin
    pol: Bool, whether maps are I,Q,U or just I
    save_only: Bool, set to True if only want to save spectra but not plot them
    plot_only: Bool, set to True if only want to plot already save spectra, but not re-compute them
    
    RETURNS
    -------
    None
    ''' 

    mask = hp.read_map(mask_file, field=(3)) #70% fsky
    mask = hp.ud_grade(mask, nside)

    if not plot_only:
        spectra = []
        for i in range(3):
            ell_eff, cl = get_mask_deconvolved_spectrum(nside, mask, ells_per_bin, maps[i], map2=None, ellmax=ellmax)
            spectra.append(cl)

        pickle.dump(spectra, open(f'{output_dir}/mask_deconvolved_spectra.p', 'wb'))
        pickle.dump(ell_eff, open(f'{output_dir}/ell_eff.p', 'wb'))
    
    else:
        try:
            spectra = pickle.load(open(f'{output_dir}/mask_deconvolved_spectra.p', 'rb'))
            ell_eff = pickle.load(open(f'{output_dir}/ell_eff.p', 'rb'))
        except FileNotFoundError:
            print('Need to save mask-deconvolved spectra before plotting. Re-run with plot_only=False.')
            return

    if save_only:
        return
    
    #plot
    freqs = [220, 150, 90]
    types = ['I', 'Q', 'U']
    for freq in range(3):
        for t in range(3):
            plt.clf()
            if not pol:
                hp.mollview(maps[freq]*mask, title=f'Masked {freqs[freq]} GHz {types[t]} Map')
            else:
                hp.mollview(maps[freq][t]*mask, title=f'Masked {freqs[freq]} GHz {types[t]} Map')
            plt.savefig(f'{plot_dir}/gal_mask_{freqs[freq]}.png')
            if not pol: break
    plt.clf()
    to_dl = ell_eff*(ell_eff+1)/2/np.pi
    for freq in range(3):
        if not pol:
            plt.plot(ell_eff, to_dl*spectra[freq], label=f'{freqs[freq]} GHz')
        else:
            plt.plot(ell_eff, to_dl*spectra[freq][0], label=f'{freqs[freq]} GHz TT')
            plt.plot(ell_eff, to_dl*spectra[freq][1], label=f'{freqs[freq]} GHz EE')
            plt.plot(ell_eff, to_dl*spectra[freq][2], label=f'{freqs[freq]} GHz BB')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
    plt.grid()
    plt.legend()
    plt.savefig(f'{plot_dir}/gal_mask_deconvolved_power_spectra.png')

    return

if __name__ == '__main__':

    nside = 8192
    output_dir = f'outputs_nside{nside}'
    plot_dir = f'plots_nside{nside}'
    maps = pickle.load(open(f'{output_dir}/beam_convolved_maps.p', 'rb'))
    mask_file = 'HFI_Mask_GalPlane-apo5_2048_R2.00.fits'
    pol = False
    ellmax = 10000
    ells_per_bin = 50

    plot_and_save_mask_deconvolved_spectra(nside, output_dir, plot_dir, maps, mask_file, ellmax, ells_per_bin, pol=pol)
    

