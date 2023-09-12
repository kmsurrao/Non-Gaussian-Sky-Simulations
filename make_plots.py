import numpy as np
import healpy as hp
import pickle
import matplotlib.pyplot as plt
from pixell import enmap, enplot, curvedsky
import os
import subprocess
from galactic_mask import plot_and_save_mask_deconvolved_spectra

def plot_outputs(output_dir, plot_dir, ellmax, pol, nside, mask_file=None, ells_per_bin=None, plots_to_make='all'):
    '''
    Produces and saves plots of various outputs 

    ARGUMENTS
    ---------
    output_dir: str, directory in which output files were put
    plot_dir: str, directory in which to save plots
    ellmax: int, maximum ell for which to compute power spectra
    pol: Bool, False if only calculting intensity maps, True if also calculating polarization maps
    plots_to_make: list of str, plots_to_make plots to make
        'passband': make plots of passbands 
        'gal_and_extragal_comps': galactic and extragalactic component maps
        'freq_maps_no_beam': healpix frequency maps before applying beam
        'beam_convolved_maps': healpix beam-convolved frequency maps
        'CAR_maps': CAR beam-convolved frequency maps
        'all_comp_spectra': power spectra of all components
        Alternatively plots_to_make can be set to a single string 'all' to make all plots.

    '''

    #Passbands
    if plots_to_make == 'all' or 'passband' in plots_to_make:
        plt.clf()
        all_bandpass_freqs = pickle.load(open(f'{output_dir}/all_bandpass_freqs.p', 'rb'))
        all_bandpass_weights = pickle.load(open(f'{output_dir}/all_bandpass_weights.p', 'rb'))
        for i, central_freq in enumerate([220, 150, 90]):
            bandpass_freqs, bandpass_weights = all_bandpass_freqs[i], all_bandpass_weights[i]
            if central_freq == 220:
                band_name = 'PA4 220'
            elif central_freq == 150:
                band_name = 'PA5 150'
            elif central_freq == 90:
                band_name = 'PA6 90'
            plt.plot(bandpass_freqs, bandpass_weights/np.amax(bandpass_weights), label=band_name)
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Bandpass Weight')
        plt.grid()
        plt.legend()
        plt.savefig(f'{plot_dir}/passbands.png')


    #Galactic and Extragalactic Component Maps
    if plots_to_make == 'all' or 'gal_and_extragal_comps' in plots_to_make:
        all_maps = pickle.load(open(f'{output_dir}/gal_and_extragal_before_beam.p', 'rb'))
        freqs = [220, 150, 90]
        map_types = ['I', 'Q', 'U']
        comp_types = ['Galactic Components', 'Extragalactic Components']
        for freq in range(3):
            for c, comp_type in enumerate(comp_types):
                if not pol:
                    plt.clf()
                    comp_type_no_space = comp_type.replace(' ', '')
                    hp.mollview(all_maps[freq,c], title=f'{freqs[freq]} GHz {comp_type}')
                    plt.savefig(f'{plot_dir}/{comp_type_no_space}_{freqs[freq]}.png')
                else:
                    for map_type in range(3):
                        plt.clf()
                        comp_type_no_space = comp_type.replace(' ', '')
                        hp.mollview(all_maps[freq,c,map_type], title=f'{freqs[freq]} GHz {map_types[map_type]} {comp_type} Map')
                        plt.savefig(f'{plot_dir}/{comp_type_no_space}_{map_types[map_type]}_{freqs[freq]}.png')


    #Frequency Maps and Power Spectra without Beam
    if plots_to_make == 'all' or 'freq_maps_no_beam' in plots_to_make:
        freq_maps = np.sum(all_maps, axis=1)
        for freq in range(3):
            if not pol:
                plt.clf()
                hp.mollview(freq_maps[freq], title=f'{freqs[freq]} GHz')
                plt.savefig(f'{plot_dir}/no_beam_{freqs[freq]}.png')
            else:
                for map_type in range(3):
                    plt.clf()
                    hp.mollview(freq_maps[freq,map_type], title=f'{freqs[freq]} GHz {map_types[map_type]} Map')
                    plt.savefig(f'{plot_dir}/no_beam_{freqs[freq]}_{map_types[map_type]}.png')
        plt.clf()
        ells = np.arange(ellmax+1)
        to_dl = ells*(ells+1)/2/np.pi
        for freq in range(3):
            if not pol:
                plt.plot(ells[2:], (to_dl*hp.anafast(freq_maps[freq], lmax=ellmax))[2:], label=f'{freqs[freq]} GHz')
            else:
                spectra = hp.anafast(freq_maps[freq], lmax=ellmax, pol=True)
                spectra_types = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
                for t in range(6):
                    plt.plot(ells[2:], (to_dl*spectra[t])[2:], label=f'{freqs[freq]} GHz {spectra_types[t]}')       
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
        plt.grid()
        plt.legend()
        plt.savefig(f'{plot_dir}/no_beam_power_spectra.png')



    #Frequency Maps and Power Spectra with Beam
    if plots_to_make == 'all' or 'beam_convolved_maps' in plots_to_make:
        beam_convolved_maps = pickle.load(open(f'{output_dir}/beam_convolved_maps.p', 'rb'))
        for freq in range(3):
            if not pol:
                plt.clf()
                hp.mollview(beam_convolved_maps[freq], title=f'{freqs[freq]} GHz')
                plt.savefig(f'{plot_dir}/beam_{freqs[freq]}.png')
            else:
                for map_type in range(3):
                    plt.clf()
                    hp.mollview(beam_convolved_maps[freq,map_type], title=f'{freqs[freq]} GHz {map_types[map_type]} Map')
                    plt.savefig(f'{plot_dir}/beam_{freqs[freq]}_{map_types[map_type]}.png')
        plt.clf()
        for freq in range(3):
            if not pol:
                plt.plot(ells[2:], (to_dl*hp.anafast(beam_convolved_maps[freq], lmax=ellmax))[2:], label=f'{freqs[freq]} GHz')
            else:
                spectra = hp.anafast(beam_convolved_maps[freq], lmax=ellmax, pol=True)
                spectra_types = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
                for t in range(6):
                    plt.plot(ells[2:], (to_dl*spectra[t])[2:], label=f'{freqs[freq]} GHz {spectra_types[t]}')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
        plt.grid()
        plt.legend()
        plt.savefig(f'{plot_dir}/beam_power_spectra.png')


    #Final CAR Maps and Power Spectra
    if plots_to_make == 'all' or 'CAR_maps' in plots_to_make:
        def eshow(x,**kwargs): enplot.show(enplot.plot(x,**kwargs))
        for i, freq in enumerate([220, 150, 90]):
            for map_type in range(3):
                plt.clf()
                map_ = enmap.read_map(f'sim_{freq}GHz')
                eshow(map_)
                plt.savefig(f'{plot_dir}/CAR_{map_types[map_type]}map_{freq}.png')
                if not pol: break
        plt.clf()
        for i, freq in enumerate([220, 150, 90]):
            map_ = enmap.read_map(f'sim_{freq}GHz')
            alm = curvedsky.map2alm(map_, lmax=ellmax)
            cl = curvedsky.alm2cl(alm)
            if not pol:
                plt.plot(ells, ells*(ells+1)/2/np.pi*cl[0], label=f'{freq} GHz')
            else:
                spectra_types = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
                for t in range(6):
                    plt.plot(ells[2:], (to_dl*cl[t])[2:], label=f'{freqs[freq]} GHz {spectra_types[t]}')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
        plt.grid()
        plt.legend()
        plt.savefig(f'{plot_dir}/CAR_beam_power_spectra.png')
    

    #Power Spectra of all Components
    if plots_to_make == 'all' or 'all_comp_spectra' in plots_to_make:
        for i, freq in enumerate([220, 150, 90]):
            ells = np.arange(ellmax+1)
            to_dl = ells*(ells+1)/2/np.pi
            gal_spectra = pickle.load( open(f'{output_dir}/gal_comp_spectra_{freq}.p', 'rb'))
            extragal_spectra = pickle.load( open(f'{output_dir}/extragal_comp_spectra_{freq}.p', 'rb'))
            gal_comps = ['Dust', 'Synchrotron', 'AME', 'Free-free']
            extragal_comps = ['CMB', 'Late-time kSZ', 'tSZ', 'CIB', 'Radio', 'Reionization kSZ']
            modes = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
            if not pol:
                plt.clf()
                for c in range(len(gal_spectra)):
                    plt.plot(ells[2:], (to_dl*gal_spectra[c])[2:], label=gal_comps[c])
                for c in range(len(extragal_spectra)):
                    plt.plot(ells[2:], (to_dl*extragal_spectra[c])[2:], label=extragal_comps[c])
                plt.yscale('log')
                plt.xlabel(r'$\ell$')
                plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
                plt.grid()
                plt.legend()
                plt.savefig(f'{plot_dir}/all_comp_spectra_{freq}.png')
            else:
                for m in range(6):
                    plt.clf()
                    for c in range(len(gal_spectra)):
                        plt.plot(ells[2:], (to_dl*gal_spectra[c,m])[2:], label=gal_comps[c])
                    for c in range(len(extragal_spectra)):
                        plt.plot(ells[2:], (to_dl*extragal_spectra[c,m])[2:], label=extragal_comps[c])
                    plt.yscale('log')
                    plt.xlabel(r'$\ell$')
                    plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
                    plt.grid()
                    plt.legend()
                    plt.savefig(f'{plot_dir}/all_comp_spectra_{freq}_{modes[m]}.png')
    
    #Galactic Mask-Deconvolved Power Spectrum at Each Frequency
    if  plots_to_make=='all' or 'mask_deconvolved_spectra' in plots_to_make:
        beam_convolved_maps = pickle.load(open(f'{output_dir}/beam_convolved_maps.p', 'rb'))
        plot_and_save_mask_deconvolved_spectra(nside, output_dir, plot_dir, beam_convolved_maps, mask_file, ellmax, ells_per_bin, pol=pol, plot_only=True)



    #Galactic Mask-Deconvolved Power Spectra of all Components
    if plots_to_make == 'all' or 'mask_deconvolved_comp_spectra' in plots_to_make:
        for i, freq in enumerate([220, 150, 90]):
            ells = np.arange(ellmax+1)
            to_dl = ells*(ells+1)/2/np.pi
            gal_spectra = pickle.load( open(f'{output_dir}/gal_comp_mask_deconvolved_spectra_{freq}.p', 'rb'))
            extragal_spectra = pickle.load( open(f'{output_dir}/extragal_comp_mask_deconvolved_spectra_{freq}.p', 'rb'))
            gal_comps = ['Dust', 'Synchrotron', 'AME', 'Free-free']
            extragal_comps = ['CMB', 'Late-time kSZ', 'tSZ', 'CIB', 'Radio', 'Reionization kSZ']
            modes = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
            if not pol:
                plt.clf()
                for c in range(len(gal_spectra)):
                    plt.plot(ells[2:], (to_dl*gal_spectra[c])[2:], label=gal_comps[c])
                for c in range(len(extragal_spectra)):
                    plt.plot(ells[2:], (to_dl*extragal_spectra[c])[2:], label=extragal_comps[c])
                plt.yscale('log')
                plt.xlabel(r'$\ell$')
                plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
                plt.grid()
                plt.legend()
                plt.savefig(f'{plot_dir}/all_comp_spectra_{freq}_mask_deconvolved.png')
            else:
                for m in range(6):
                    plt.clf()
                    for c in range(len(gal_spectra)):
                        plt.plot(ells[2:], (to_dl*gal_spectra[c,m])[2:], label=gal_comps[c])
                    for c in range(len(extragal_spectra)):
                        plt.plot(ells[2:], (to_dl*extragal_spectra[c,m])[2:], label=extragal_comps[c])
                    plt.yscale('log')
                    plt.xlabel(r'$\ell$')
                    plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
                    plt.grid()
                    plt.legend()
                    plt.savefig(f'{plot_dir}/all_comp_spectra_{freq}_{modes[m]}_mask_deconvolved.png')
            

    return
