import numpy as np
import healpy as hp
import pickle
import matplotlib.pyplot as plt
from pixell import enmap, enplot, curvedsky
from galactic_mask import plot_and_save_mask_deconvolved_spectra

def plot_outputs(inp):
    '''
    Produces and saves plots of various outputs 

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications

    RETURNS
    -------
    None
    '''

    #Passbands
    if inp.plots_to_make == 'all' or 'passband' in inp.plots_to_make:
        plt.clf()
        all_bandpass_freqs = pickle.load(open(f'{inp.output_dir}/all_bandpass_freqs.p', 'rb'))
        all_bandpass_weights = pickle.load(open(f'{inp.output_dir}/all_bandpass_weights.p', 'rb'))
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
        plt.savefig(f'{inp.plot_dir}/passbands.png')
        print(f'saved {inp.plot_dir}/passbands.png', flush=True)


    #Galactic and Extragalactic Component Maps
    if inp.plots_to_make == 'all' or 'gal_and_extragal_comps' in inp.plots_to_make:
        all_maps = pickle.load(open(f'{inp.output_dir}/gal_and_extragal_before_beam.p', 'rb'))
        freqs = [220, 150, 90]
        map_types = ['I', 'Q', 'U']
        comp_types = ['Galactic Components', 'Extragalactic Components']
        for freq in range(3):
            for c, comp_type in enumerate(comp_types):
                if not inp.pol:
                    plt.clf()
                    comp_type_no_space = comp_type.replace(' ', '')
                    hp.mollview(all_maps[freq,c], title=f'{freqs[freq]} GHz {comp_type}')
                    plt.savefig(f'{inp.plot_dir}/{comp_type_no_space}_{freqs[freq]}.png')
                    print(f'saved {inp.plot_dir}/{comp_type_no_space}_{freqs[freq]}.png', flush=True)
                else:
                    for map_type in range(3):
                        plt.clf()
                        comp_type_no_space = comp_type.replace(' ', '')
                        hp.mollview(all_maps[freq,c,map_type], title=f'{freqs[freq]} GHz {map_types[map_type]} {comp_type} Map')
                        plt.savefig(f'{inp.plot_dir}/{comp_type_no_space}_{map_types[map_type]}_{freqs[freq]}.png')
                        print(f'saved {inp.plot_dir}/{comp_type_no_space}_{map_types[map_type]}_{freqs[freq]}.png', flush=True)


    #Frequency Maps and Power Spectra without Beam
    if inp.plots_to_make == 'all' or 'freq_maps_no_beam' in inp.plots_to_make:
        freq_maps = np.sum(all_maps, axis=1)
        for freq in range(3):
            if not inp.pol:
                plt.clf()
                hp.mollview(freq_maps[freq], title=f'{freqs[freq]} GHz')
                plt.savefig(f'{inp.plot_dir}/no_beam_{freqs[freq]}.png')
                print(f'saved {inp.plot_dir}/no_beam_{freqs[freq]}.png', flush=True)
            else:
                for map_type in range(3):
                    plt.clf()
                    hp.mollview(freq_maps[freq,map_type], title=f'{freqs[freq]} GHz {map_types[map_type]} Map')
                    plt.savefig(f'{inp.plot_dir}/no_beam_{freqs[freq]}_{map_types[map_type]}.png')
                    print(f'saved {inp.plot_dir}/no_beam_{freqs[freq]}_{map_types[map_type]}.png', flush=True)
        plt.clf()
        ells = np.arange(inp.ellmax+1)
        to_dl = ells*(ells+1)/2/np.pi
        for freq in range(3):
            if not inp.pol:
                plt.plot(ells[2:], (to_dl*hp.anafast(freq_maps[freq], lmax=inp.ellmax))[2:], label=f'{freqs[freq]} GHz')
            else:
                spectra = hp.anafast(freq_maps[freq], lmax=inp.ellmax, pol=True)
                spectra_types = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
                for t in range(6):
                    plt.plot(ells[2:], (to_dl*spectra[t])[2:], label=f'{freqs[freq]} GHz {spectra_types[t]}')       
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
        plt.grid()
        plt.legend()
        plt.savefig(f'{inp.plot_dir}/no_beam_power_spectra.png')
        print(f'saved {inp.plot_dir}/no_beam_power_spectra.png', flush=True)



    #Frequency Maps and Power Spectra with Beam
    if inp.plots_to_make == 'all' or 'beam_convolved_maps' in inp.plots_to_make:
        beam_convolved_maps = pickle.load(open(f'{inp.output_dir}/beam_convolved_maps.p', 'rb'))
        for freq in range(3):
            if not inp.pol:
                plt.clf()
                hp.mollview(beam_convolved_maps[freq], title=f'{freqs[freq]} GHz')
                plt.savefig(f'{inp.plot_dir}/beam_{freqs[freq]}.png')
                print(f'saved {inp.plot_dir}/beam_{freqs[freq]}.png', flush=True)
            else:
                for map_type in range(3):
                    plt.clf()
                    hp.mollview(beam_convolved_maps[freq,map_type], title=f'{freqs[freq]} GHz {map_types[map_type]} Map')
                    plt.savefig(f'{inp.plot_dir}/beam_{freqs[freq]}_{map_types[map_type]}.png')
                    print(f'saved {inp.plot_dir}/beam_{freqs[freq]}_{map_types[map_type]}.png', flush=True)
        plt.clf()
        for freq in range(3):
            if not inp.pol:
                plt.plot(ells[2:], (to_dl*hp.anafast(beam_convolved_maps[freq], lmax=inp.ellmax))[2:], label=f'{freqs[freq]} GHz')
            else:
                spectra = hp.anafast(beam_convolved_maps[freq], lmax=inp.ellmax, pol=True)
                spectra_types = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
                for t in range(6):
                    plt.plot(ells[2:], (to_dl*spectra[t])[2:], label=f'{freqs[freq]} GHz {spectra_types[t]}')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
        plt.grid()
        plt.legend()
        plt.savefig(f'{inp.plot_dir}/beam_power_spectra.png')
        print(f'saved {inp.plot_dir}/beam_power_spectra.png', flush=True)


    #Final CAR Maps and Power Spectra
    if inp.plots_to_make == 'all' or 'CAR_maps' in inp.plots_to_make:
        def eshow(x,**kwargs): enplot.show(enplot.plot(x,**kwargs))
        for i, freq in enumerate([220, 150, 90]):
            for map_type in range(3):
                plt.clf()
                map_ = enmap.read_map(f'sim_{freq}GHz')
                eshow(map_)
                plt.savefig(f'{inp.plot_dir}/CAR_{map_types[map_type]}map_{freq}.png')
                print(f'saved {inp.plot_dir}/CAR_{map_types[map_type]}map_{freq}.png', flush=True)
                if not inp.pol: break
        plt.clf()
        for i, freq in enumerate([220, 150, 90]):
            map_ = enmap.read_map(f'sim_{freq}GHz')
            alm = curvedsky.map2alm(map_, lmax=inp.ellmax)
            cl = curvedsky.alm2cl(alm)
            if not inp.pol:
                plt.plot(ells, ells*(ells+1)/2/np.pi*cl[0], label=f'{freq} GHz')
            else:
                spectra_types = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
                for t in range(6):
                    plt.plot(ells[2:], (to_dl*cl[t])[2:], label=f'{freqs[freq]} GHz {spectra_types[t]}')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
        plt.grid()
        plt.legend()
        plt.savefig(f'{inp.plot_dir}/CAR_beam_power_spectra.png')
        print(f'saved {inp.plot_dir}/CAR_beam_power_spectra.png', flush=True)
    

    #Power Spectra of all Components
    if inp.plots_to_make == 'all' or 'all_comp_spectra' in inp.plots_to_make:
        for i, freq in enumerate([220, 150, 90]):
            ells = np.arange(inp.ellmax+1)
            to_dl = ells*(ells+1)/2/np.pi
            gal_spectra = pickle.load( open(f'{inp.output_dir}/gal_comp_spectra_{freq}.p', 'rb'))
            extragal_spectra = pickle.load( open(f'{inp.output_dir}/extragal_comp_spectra_{freq}.p', 'rb'))
            gal_comps = ['Dust', 'Synchrotron', 'AME', 'Free-free']
            extragal_comps = ['CMB', 'Late-time kSZ', 'tSZ', 'CIB', 'Radio', 'Reionization kSZ']
            modes = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
            if not inp.pol:
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
                plt.savefig(f'{inp.plot_dir}/all_comp_spectra_{freq}.png')
                print(f'saved {inp.plot_dir}/all_comp_spectra_{freq}.png', flush=True)
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
                    plt.savefig(f'{inp.plot_dir}/all_comp_spectra_{freq}_{modes[m]}.png')
                    print(f'saved {inp.plot_dir}/all_comp_spectra_{freq}_{modes[m]}.png', flush=True)
    

    #Galactic Mask-Deconvolved Power Spectrum at Each Frequency
    if  inp.plots_to_make=='all' or 'mask_deconvolved_spectra' in inp.plots_to_make:
        beam_convolved_maps = pickle.load(open(f'{inp.output_dir}/beam_convolved_maps.p', 'rb'))
        plot_and_save_mask_deconvolved_spectra(inp, beam_convolved_maps, plot_only=True)


    #Galactic Mask-Deconvolved Power Spectra of Galactic Components
    if inp.plots_to_make == 'all' or 'mask_deconvolved_comp_spectra' in inp.plots_to_make:
        for i, freq in enumerate([220, 150, 90]):
            ells = pickle.load(open(f'{inp.output_dir}/ell_eff.p', 'rb'))
            to_dl = ells*(ells+1)/2/np.pi
            gal_spectra = pickle.load( open(f'{inp.output_dir}/gal_comp_mask_deconvolved_spectra_{freq}.p', 'rb'))
            gal_comps = ['Dust', 'Synchrotron', 'AME', 'Free-free']
            modes = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
            if not inp.pol:
                plt.clf()
                for c in range(len(gal_spectra)):
                    plt.plot(ells[2:], (to_dl*gal_spectra[c])[2:], label=gal_comps[c])
                plt.yscale('log')
                plt.xlabel(r'$\ell$')
                plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
                plt.grid()
                plt.legend()
                plt.savefig(f'{inp.plot_dir}/all_comp_spectra_{freq}_mask_deconvolved.png')
                print(f'saved {inp.plot_dir}/all_comp_spectra_{freq}_mask_deconvolved.png', flush=True)
            else:
                for m in range(6):
                    plt.clf()
                    for c in range(len(gal_spectra)):
                        plt.plot(ells[2:], (to_dl*gal_spectra[c,m])[2:], label=gal_comps[c])
                    plt.yscale('log')
                    plt.xlabel(r'$\ell$')
                    plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
                    plt.grid()
                    plt.legend()
                    plt.savefig(f'{inp.plot_dir}/all_comp_spectra_{freq}_{modes[m]}_mask_deconvolved.png')
                    print(f'saved {inp.plot_dir}/all_comp_spectra_{freq}_{modes[m]}_mask_deconvolved.png', flush=True)
            

    return
