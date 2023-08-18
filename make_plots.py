import numpy as np
import healpy as hp
import pickle
import matplotlib.pyplot as plt
from pixell import enmap, enplot, curvedsky
import os
import subprocess

def plot_outputs(output_dir, plot_dir, ellmax, pol, which='all'):
    '''
    Produces and saves plots of various outputs 

    ARGUMENTS
    ---------
    output_dir: str, directory in which output files were put
    plot_dir: str, directory in which to save plots
    ellmax: int, maximum ell for which to compute power spectra
    pol: Bool, False if only calculting intensity maps, True if also calculating polarization maps
    which: list of str, which plots to make
        'passband': make plots of passbands 
        'gal_and_extragal_comps': galactic and extragalactic component maps
        'freq_maps_no_beam': healpix frequency maps before applying beam
        'beam_convolved_maps': healpix beam-convolved frequency maps
        'CAR_maps': CAR beam-convolved frequency maps
        Alternatively which can be set to a single string 'all' to make all plots.

    '''

    #Set up output directory for plots
    if not os.path.isdir(plot_dir):
        env = os.environ.copy()
        subprocess.call(f'mkdir {plot_dir}', shell=True, env=env)

    #Passbands
    if which == 'all' or 'passband' in which:
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
    if which == 'all' or 'gal_and_extragal_comps' in which:
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
    if which == 'all' or 'freq_maps_no_beam' in which:
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
    if which == 'all' or beam_convolved_maps in which:
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
    if which == 'all' or 'CAR_maps' in which:
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

    return
