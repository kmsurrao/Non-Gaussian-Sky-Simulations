import numpy as np
import healpy as hp
import pickle
import pymaster as nmt
import matplotlib.pyplot as plt

def get_mask_deconvolved_spectrum(mask, ells_per_bin, map1, map2=None, ellmax=None):
    '''
    ARGUMENTS
    ---------
    mask: 1D numpy array in healpix format containing mask
    ells_per_bin: int, number of ells to put in a bin
    map1: 1D numpy array in healpix format containing map
    map2: 1D numpy array in healpix format containing map2 (if different from map1)
    ellmax: int, maximum ell for which to compute results (if not give, goes to 3*nside)
    
    RETURNS
    -------
    ell_arr: array of effective ell values for bins
    cl_12: array containing binned mask-deconvolved power spectrum
    ''' 
    if ellmax:
        b = nmt.NmtBin.from_nside_linear(ellmax//3, ells_per_bin, is_Dell=True)
    else:
        nside = hp.get_nside(map1)
        b = nmt.NmtBin.from_nside_linear(nside, ells_per_bin, is_Dell=True)
    f_1 = nmt.NmtField(mask, [map1])
    if map2:
        f_2 = nmt.NmtField(mask, [map2])
    else:
        f_2 = f_1
    dl_12 = nmt.compute_full_master(f_1, f_2, b)
    ell_arr = b.get_effective_ells()
    to_dl = ell_arr*(ell_arr+1)/2/np.pi
    return ell_arr, dl_12/to_dl


if __name__ == '__main__':

    nside = 8192
    output_dir = f'outputs_nside{nside}'
    plot_dir = f'plots_nside{nside}'
    maps = pickle.load(open(f'{output_dir}/beam_convolved_maps.p', 'rb'))
    mask_file = 'HFI_Mask_GalPlane-apo5_2048_R2.00.fits'

    mask = hp.read_map(mask_file, field=(5)) #90% fsky
    mask = hp.ud_grade(mask, nside)

    ellmax = 7000
    ells_per_bin = 100
    spectra = []
    for i in range(3):
        ell_eff, cl = get_mask_deconvolved_spectrum(mask, ells_per_bin, maps[i], map2=None, ellmax=ellmax)
        spectra.append(cl)

    pickle.dump(spectra, open(f'{output_dir}/mask_deconvolved_spectra.p', 'wb'))
    pickle.dump(ell_eff, open(f'{output_dir}/ell_eff.p', 'wb'))

    #plot
    freqs = [220, 150, 90]
    for freq in range(3):
        plt.clf()
        hp.mollview(maps[freq]*mask, title=f'Masked {freqs[freq]} GHz')
        plt.savefig(f'{plot_dir}/gal_mask_{freqs[freq]}.png')
    plt.clf()
    to_dl = ell_eff*(ell_eff+1)/2/np.pi
    for freq in range(3):
        plt.plot(ell_eff, to_dl*spectra[freq], label=f'{freqs[freq]} GHz')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
    plt.grid()
    plt.legend()
    plt.savefig(f'{plot_dir}/gal_mask_deconvolved_power_spectra.png')
