import numpy as np
import healpy as hp
import pickle
import pymaster as nmt

def get_mask_deconvolved_spectrum(mask, ells_per_bin, map1, map2=None):
    '''
    ARGUMENTS
    ---------
    mask: 1D numpy array in healpix format containing mask
    ells_per_bin: int, number of ells to put in a bin
    map1: 1D numpy array in healpix format containing map
    map2: 1D numpy array in healpix format containing map2 (if different from map1)
    
    RETURNS
    -------
    ell_arr: array of effective ell values for bins
    cl_12: array containing binned mask-deconvolved power spectrum
    ''' 
    nside = hp.get_nside(map1)
    b = nmt.NmtBin.from_nside_linear(nside, ells_per_bin, is_Dell=True)
    f_1 = nmt.NmtField(mask, [map1])
    if map2:
        f_2 = nmt.NmtField(mask, [map2])
    else:
        f_2 = f_1
    cl_12 = nmt.compute_full_master(f_1, f_2, b)
    ell_arr = b.get_effective_ells()
    return ell_arr, cl_12


if __name__ == '__main__':

    output_dir = 'outputs'
    maps = f'{output_dir}/beam_convolved_maps.p'
    mask_file = '/scratch/09334/ksurrao/HFI_Mask_GalPlane-apo5_2048_R2.00.fits'

    mask = hp.read_map(mask_file, field=(5)) #90% fsky
    mask = hp.ud_grade(hp.get_nside(maps[0]))

    ellmax = 10000
    ells_per_bin = 50
    spectra = []
    for i in range(3):
        ell_eff, cl = get_mask_deconvolved_spectrum(mask, ells_per_bin, maps[i], map2=None)
        to_dl = ell_eff*(ell_eff+1)/2/np.pi
        spectra.append(cl/to_dl)

    pickle.dump(spectra, open(f'{output_dir}/mask_deconvolved_spectra.p'))
    pickle.dump(ell_eff, open(f'{output_dir}/ell_eff.p'))
