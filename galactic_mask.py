import numpy as np
import healpy as hp
import pickle
import pymaster as nmt
import matplotlib.pyplot as plt
import argparse
from input import Info

def compute_master(inp, mask, map1, map2=None):
    '''
    Use this function to compute mask-deconvolved spectra if computing them multiple times
    with the same mask

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
        Must already have a wsp (NaMaster workspace) attribute added,
        for which the coupling matrix has already been compupted
    f_a: NaMaster field object for first field
    f_b: NaMaster field object for second field

    RETURNS
    -------
    ell_arr: array of effective ell values for bins
    cl: array containing binned mask-deconvolved power spectrum
        1D if not pol, 3D if pol (TT, EE, BB)
    '''
    if map2 is None:
        map2 = map1
    if inp.pol:
        map1_T, map2_T = map1[0], map2[0]
    else:
        map1_T, map2_T = map1, map2

    f_a = nmt.NmtField(mask, [map1_T])
    f_b = nmt.NmtField(mask, [map2_T])
    dl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    dl_decoupled = inp.wsp.decouple_cell(dl_coupled)
    
    ell_arr =inp.b.get_effective_ells()
    to_dl = ell_arr*(ell_arr+1)/2/np.pi
    cl_decoupled = dl_decoupled/to_dl

    if not inp.pol:
        return ell_arr, cl_decoupled[0]
    
    f_a_pol = nmt.NmtField(mask, map1[1:])
    f_b_pol = nmt.NmtField(mask, map2[1:])

    dl_coupled_tempxpol = nmt.compute_coupled_cell(f_a, f_b_pol)
    dl_decoupled_tempxpol = inp.wsp2.decouple_cell(dl_coupled_tempxpol)
    cl_decoupled_tempxpol = dl_decoupled_tempxpol/np.tile(to_dl, (2,1))

    dl_coupled_pol = nmt.compute_coupled_cell(f_a_pol, f_b_pol)
    dl_decoupled_pol = inp.wsp3.decouple_cell(dl_coupled_pol)
    cl_decoupled_pol = dl_decoupled_pol/np.tile(to_dl, (4,1))

    TT = cl_decoupled[0]
    EE, EB, BE, BB = cl_decoupled_pol
    TE, TB = cl_decoupled_tempxpol

    return ell_arr, np.array([TT, EE, BB, TE, EB, TB])


def plot_and_save_mask_deconvolved_spectra(inp, maps, save_only=False, plot_only=False):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    maps: ndarray containing beam-convolved maps
        index as maps[freq][I,Q,U][pixel] if pol
        index as maps[freq][pixel] if not pol
    save_only: Bool, set to True if only want to save spectra but not plot them
    plot_only: Bool, set to True if only want to plot already save spectra, but not re-compute them
    
    RETURNS
    -------
    None
    ''' 

    mask = hp.read_map(inp.mask_file, field=(3)) #70% fsky
    mask = hp.ud_grade(mask, inp.nside)

    if not plot_only:
        spectra = []
        for i in range(3):
            ell_eff, cl = compute_master(inp, mask, maps[i], map2=None)
            spectra.append(cl)

        pickle.dump(spectra, open(f'{inp.output_dir}/mask_deconvolved_spectra.p', 'wb'))
        pickle.dump(ell_eff, open(f'{inp.output_dir}/ell_eff.p', 'wb'))
    
    else:
        try:
            spectra = pickle.load(open(f'{inp.output_dir}/mask_deconvolved_spectra.p', 'rb'))
            ell_eff = pickle.load(open(f'{inp.output_dir}/ell_eff.p', 'rb'))
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
            if not inp.pol:
                hp.mollview(maps[freq]*mask, title=f'Masked {freqs[freq]} GHz {types[t]} Map')
            else:
                hp.mollview(maps[freq][t]*mask, title=f'Masked {freqs[freq]} GHz {types[t]} Map')
            plt.savefig(f'{inp.plot_dir}/gal_mask_{freqs[freq]}.png')
            if not inp.pol: break
    plt.clf()
    to_dl = ell_eff*(ell_eff+1)/2/np.pi
    for freq in range(3):
        if not inp.pol:
            plt.plot(ell_eff, to_dl*spectra[freq], label=f'{freqs[freq]} GHz')
        else:
            plt.plot(ell_eff, to_dl*spectra[freq][0], label=f'{freqs[freq]} GHz TT')
            plt.plot(ell_eff, to_dl*spectra[freq][1], label=f'{freqs[freq]} GHz EE')
            plt.plot(ell_eff, to_dl*spectra[freq][2], label=f'{freqs[freq]} GHz BB')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
    plt.grid()
    plt.legend()
    plt.savefig(f'{inp.plot_dir}/gal_mask_deconvolved_power_spectra.png')

    return

