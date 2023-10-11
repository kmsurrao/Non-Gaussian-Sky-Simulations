# Script to check masked spectra of final CAR maps (with added noise), using DR6 masks

import numpy as np
import argparse
import os
import subprocess
from pixell import enmap
import pymaster as nmt
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append('..')  
from input import Info  
from utils import get_CAR_shape_and_wcs
from galactic_mask import compute_master


def compute_coupling_matrices_CAR(inp, wcs, mask):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    wcs: wcs pixell object (same as noise maps if adding noise)
    mask: mask in CAR format

    RETURNS
    -------
    None (modifies inp in-place)
    '''
    inp.wsp_car = nmt.NmtWorkspace()
    inp.b = nmt.NmtBin.from_lmax_linear(inp.ellmax, inp.ells_per_bin, is_Dell=True)
    f_tmp = nmt.NmtField(mask, [mask], wcs=wcs)
    inp.wsp_car.compute_coupling_matrix(f_tmp, f_tmp, inp.b)
    if inp.pol:
        inp.wsp2_car = nmt.NmtWorkspace() #temperature x pol
        inp.wsp3_car = nmt.NmtWorkspace() #pol x pol
        f_tmp_pol = nmt.NmtField(mask, [mask, mask], wcs=wcs)
        inp.wsp2_car.compute_coupling_matrix(f_tmp, f_tmp_pol, inp.b)
        inp.wsp3_car.compute_coupling_matrix(f_tmp_pol, f_tmp_pol, inp.b)
    return 


def main():

    # main input file containing most specifications 
    parser = argparse.ArgumentParser(description="Non-gaussian full sky simulations.")
    parser.add_argument("--config", default="../example_yaml_files/moto.yaml")
    args = parser.parse_args()
    input_file = args.config

    # read in the input file and set up relevant info object
    inp = Info(input_file)

    # set up output directory
    if not os.path.isdir(inp.output_dir):
        env = os.environ.copy()
        subprocess.call(f'mkdir {inp.output_dir}', shell=True, env=env)

    # set up output directory for plots
    if inp.plot_dir and not os.path.isdir(inp.plot_dir):
        env = os.environ.copy()
        subprocess.call(f'mkdir {inp.plot_dir}', shell=True, env=env)

    # paths to ACT masks (on terremoto here)
    act_masks_path = '/moto/hill/users/kms2320/repositories/ACT-Simulations/ACT_masks'
    act_mask_220 = enmap.read_map(f'{act_masks_path}/kspace_mask_dr6_pa4_f220.fits')
    act_mask_150 = enmap.read_map(f'{act_masks_path}/kspace_mask_dr6_pa5_f150.fits')
    act_mask_90 = enmap.read_map(f'{act_masks_path}/kspace_mask_dr6_pa6_f090.fits')
    print('act_mask_220 wcs: ', act_mask_220.wcs, flush=True)
    print('act_mask_150 wcs: ', act_mask_150.wcs, flush=True)
    print('act_mask_90 wcs: ', act_mask_90.wcs, flush=True)
    print('act_mask_220 shape: ', act_mask_220.shape, flush=True)

    # get CAR shape and wcs
    shape, wcs = get_CAR_shape_and_wcs(inp)
    print('noise map wcs: ', wcs, flush=True)
    print('noise map shape: ', shape, flush=True)
    print('Computed coupling matrix for mask deconvolution', flush=True)

    # compute and save all mask-deconvolved spectra
    freqs = [220, 150, 90]
    spectra = []
    for i, mask in enumerate([act_mask_220, act_mask_150, act_mask_90]):
        compute_coupling_matrices_CAR(inp, wcs, mask)
        if inp.pol:
            workspace = [inp.wsp_car, inp.wsp2_car, inp.wsp3_car]
        else:
            workspace = inp.wsp_car
        map1 = enmap.read_map(f'{inp.output_dir}/sim_{freqs[i]}GHz_split0')
        ell_eff, Cl = compute_master(inp, mask, map1, wcs=wcs, workspace=workspace)
        pickle.dump(Cl, open(f'{inp.output_dir}/CAR_mask_deconvolved_spectra_{freqs[i]}.p', 'wb'))
        pickle.dump(ell_eff, open(f'{inp.output_dir}/ell_eff.p', 'wb'))
        print(f'saved {inp.output_dir}/CAR_mask_deconvolved_spectra_{freqs[i]}.p and ell_eff.p', flush=True)
        spectra.append(Cl)
    
    # make plots
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
    plt.savefig(f'{inp.plot_dir}/CAR_mask_deconvolved_power_spectra.png')

    return


if __name__ == '__main__':
    main()
