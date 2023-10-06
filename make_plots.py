import numpy as np
import healpy as hp
import pickle
import matplotlib.pyplot as plt
from pixell import enmap, enplot, curvedsky
from galactic_mask import plot_and_save_mask_deconvolved_spectra

def plot_outputs(inp, save=True):
    '''
    Produces and saves plots of various outputs 

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    save: Bool, whether to save power spectra that are not already saved

    RETURNS
    -------
    None
    '''

    #Plotting definitions
    colors = ['red', 'blue', 'green']
    linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    ells = np.arange(inp.ellmax+1)
    to_dl = ells*(ells+1)/2/np.pi
    all_maps = pickle.load(open(f'{inp.output_dir}/maps_before_beam.p', 'rb'))
    freqs = [220, 150, 90]
    map_types = ['I', 'Q', 'U']
    spectra_types = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']

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
        plt.close('all')
    

    #Beams
    if inp.plots_to_make == 'all' or 'beams' in inp.plots_to_make:
        colors = ['red', 'blue', 'green']
        linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
        plt.clf()
        for freq in range(3):
            for split in range(4):
                if freq==0:
                    beamfile = f'{inp.beam_dir}/set{split}_pa4_f220_night_beam_tform_jitter_cmb.txt'
                    label = f'PA4 220, split {split}'
                elif freq==1:
                    beamfile = f'{inp.beam_dir}/set{split}_pa5_f150_night_beam_tform_jitter_cmb.txt'
                    label = f'PA5 150, split {split}'
                elif freq==2:
                    beamfile = f'{inp.beam_dir}/set{split}_pa6_f090_night_beam_tform_jitter_cmb.txt'
                    label = f'PA6 90, split {split}'
                data = np.loadtxt(beamfile)
                l = data[:,0]
                Bl = data[:,1]
                Bl /= Bl[0]
                plt.plot(l, Bl, label=label, color=colors[freq], linestyle=linestyles[split])
        plt.grid()
        plt.legend()
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$B_\ell$')
        plt.yscale('log')
        plt.savefig(f'{inp.plot_dir}/beams.png')
        print(f'saved {inp.plot_dir}/beams.png', flush=True)
        plt.close('all')


    #Frequency Maps and Power Spectra without Beam
    if inp.plots_to_make == 'all' or 'freq_maps_no_beam' in inp.plots_to_make:
        freq_maps = all_maps
        for freq in range(3):
            if not inp.pol:
                plt.clf()
                hp.mollview(freq_maps[freq], title=f'{freqs[freq]} GHz')
                plt.savefig(f'{inp.plot_dir}/no_beam_{freqs[freq]}.png')
                print(f'saved {inp.plot_dir}/no_beam_{freqs[freq]}.png', flush=True)
            else:
                plt.clf()
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9,6), sharey=True)
                axs = (ax1, ax2, ax3)
                for map_type in range(3):
                    plt.axes(axs[map_type])
                    hp.mollview(freq_maps[freq,map_type], title=f'{freqs[freq]} GHz {map_types[map_type]} Map', hold=True)
                plt.savefig(f'{inp.plot_dir}/no_beam_{freqs[freq]}.png')
                print(f'saved {inp.plot_dir}/no_beam_{freqs[freq]}.png', flush=True)
        plt.clf()
        if not inp.pol:
            for freq in range(3):
                plt.plot(ells[2:], (to_dl*hp.anafast(freq_maps[freq], lmax=inp.ellmax))[2:], label=f'{freqs[freq]} GHz', color=colors[freq])
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
            plt.grid()
            plt.legend()
            plt.savefig(f'{inp.plot_dir}/no_beam_power_spectra.png')
            print(f'saved {inp.plot_dir}/no_beam_power_spectra.png', flush=True)
        else:
            spectra_no_beam = np.zeros((3,6,inp.ellmax+1), dtype=np.float32)
            for freq in range(3):
                spectra_no_beam[freq] = hp.anafast(freq_maps[freq], lmax=inp.ellmax, pol=True)
            fig, axs = plt.subplots(2, 3, figsize=(9,6))
            axs = axs.flatten()
            for t in range(6):
                plt.axes(axs[t])
                for freq in range(3):
                    plt.plot(ells[2:], (to_dl*spectra_no_beam[freq,t])[2:], label=f'{freqs[freq]} GHz', color=colors[freq])  
                plt.grid()
                plt.xlabel(r'$\ell$')
                plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
                plt.title(f'{spectra_types[t]}')
                if t <= 2:
                    plt.yscale('log')
                elif t==3:
                    plt.ylim(-0.02e-8, 0.02e-8)
                elif t==4:
                    plt.ylim(-0.02e-10, 0.02e-10)
                elif t==5:
                    plt.ylim(-0.02e-9, 0.02e-9)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{inp.plot_dir}/no_beam_power_spectra.png')
            print(f'saved {inp.plot_dir}/no_beam_power_spectra.png', flush=True)
        plt.close('all')
        if save:
            pickle.dump(spectra_no_beam, open(f'{inp.output_dir}/spectra_no_beam.p', 'wb'))



    #Frequency Maps and Power Spectra with Beam
    if inp.plots_to_make == 'all' or 'beam_convolved_maps' in inp.plots_to_make:
        beam_convolved_maps = pickle.load(open(f'{inp.output_dir}/beam_convolved_maps.p', 'rb'))
        for freq in range(3):
            for split in range(4):
                if not inp.pol:
                    plt.clf()
                    hp.mollview(beam_convolved_maps[freq, split], title=f'{freqs[freq]} GHz, Split {split}')
                    plt.savefig(f'{inp.plot_dir}/beam_{freqs[freq]}_split{split}.png')
                    print(f'saved {inp.plot_dir}/beam_{freqs[freq]}_split{split}.png', flush=True)
                else:
                    plt.clf()
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9,6), sharey=True)
                    axs = (ax1, ax2, ax3)
                    for map_type in range(3):
                        plt.axes(axs[map_type])
                        hp.mollview(beam_convolved_maps[freq, split, map_type], title=f'{freqs[freq]} GHz Split {split} {map_types[map_type]} Map', hold=True)
                    plt.savefig(f'{inp.plot_dir}/beam_{freqs[freq]}_split{split}.png')
                    print(f'saved {inp.plot_dir}/beam_{freqs[freq]}_split{split}.png', flush=True)
        if not inp.pol:
            beam_convolved_spectra = np.zeros((3, 4, inp.ellmax+1), dtype=np.float32)
        else:
            beam_convolved_spectra = np.zeros((3, 4, 6, inp.ellmax+1), dtype=np.float32)
        for freq in range(3):
            for split in range(4):
                beam_convolved_spectra[freq,split] = hp.anafast(beam_convolved_maps[freq, split], lmax=inp.ellmax, pol=inp.pol)
        if not inp.pol:
            plt.clf()
            for freq in range(3):
                for split in range(4):
                    spectra_to_plot = beam_convolved_spectra[freq,split]
                    plt.plot(ells[2:], (to_dl*spectra_to_plot)[2:], label=f'{freqs[freq]} GHz, split {split}', color=colors[freq], linestyle=linestyles[split])
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
            plt.yscale('log')
            plt.xlim(2, inp.ellmax)
            plt.grid()
            plt.legend()
            plt.savefig(f'{inp.plot_dir}/beam_convolved_map_power_spectra.png')
            print(f'saved {inp.plot_dir}/beam_convolved_map_power_spectra.png', flush=True)
        else:
            plt.clf()
            fig, axs = plt.subplots(2, 3, figsize=(9,6))
            axs = axs.flatten()
            for t in range(6):
                plt.axes(axs[t])
                for freq in range(3):
                    for split in range(4):
                        spectra_to_plot = beam_convolved_spectra[freq,split,t]
                        plt.plot(ells[2:], (to_dl*spectra_to_plot)[2:], label=f'{freqs[freq]} GHz, split {split}', color=colors[freq], linestyle=linestyles[split])
                plt.xlabel(r'$\ell$')
                plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
                plt.grid()
                plt.title(f'{spectra_types[t]}')
                if t <= 2:
                    plt.yscale('log')
                elif t==3:
                    plt.ylim(-0.02e-8, 0.02e-8)
                elif t==4:
                    plt.ylim(-0.02e-10, 0.02e-10)
                elif t==5:
                    plt.ylim(-0.02e-9, 0.02e-9)
                plt.xlim(2, inp.ellmax)
            handles, labels = axs[-1].get_legend_handles_labels()
            fig.legend(handles, labels, fontsize=12, bbox_to_anchor=(1.0, 0.), ncol=3)
            plt.tight_layout()
            plt.savefig(f'{inp.plot_dir}/beam_convolved_map_power_spectra.png')
            print(f'saved {inp.plot_dir}/beam_convolved_map_power_spectra.png', flush=True)
        plt.close('all')
        if save:
            pickle.dump(beam_convolved_spectra, open(f'{inp.output_dir}/beam_convolved_spectra.p', 'wb'))


    #Final CAR Maps and Power Spectra
    if inp.plots_to_make == 'all' or 'CAR_maps' in inp.plots_to_make:
        for i, freq in enumerate([220, 150, 90]):
            for split in range(4):
                map_ = enmap.read_map(f'{inp.output_dir}/sim_{freq}GHz_split{split}')
                for map_type in range(3):
                    plt.clf()
                    if inp.pol:
                        to_plot = enplot.plot(map_[map_type])
                    else:
                        to_plot = enplot.plot(map_)
                    enplot.write(f'{inp.plot_dir}/CAR_{map_types[map_type]}map_{freq}_split{split}', to_plot)
                    print(f'saved {inp.plot_dir}/CAR_{map_types[map_type]}map_{freq}_split{split}', flush=True)
                    if not inp.pol: 
                        break
        plt.clf()
        if not inp.pol:
            CAR_spectra = np.zeros((3, inp.ellmax+1), dtype=np.float32)
        else:
            CAR_spectra = np.zeros((3, 6, inp.ellmax+1), dtype=np.float32)
        if not inp.pol:
            for i, freq in enumerate([220, 150, 90]):
                map_ = enmap.read_map(f'{inp.output_dir}/sim_{freq}GHz_split0')
                alm = curvedsky.map2alm(map_, lmax=inp.ellmax)
                cl = curvedsky.alm2cl(alm)
                CAR_spectra[i] = cl[0]
                plt.plot(ells, to_dl*cl[0], label=f'{freq} GHz')
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
            plt.xlim(2, inp.ellmax)
            plt.grid()
            plt.legend()
            plt.savefig(f'{inp.plot_dir}/CAR_beam_power_spectra.png')
            print(f'saved {inp.plot_dir}/CAR_beam_power_spectra.png', flush=True)
            
        else:
            fig, axs = plt.subplots(2, 3, figsize=(9,6))
            axs = axs.flatten()
            spectra_types_here = ['T', 'E', 'B']
            axis_mapping = {(0,0):0, (0,1):3, (0,2):5, (1,1):1, (1,2):4, (2,2):2}
            for i, freq in enumerate([220, 150, 90]):
                map_ = enmap.read_map(f'{inp.output_dir}/sim_{freq}GHz_split0')
                alm = curvedsky.map2alm(map_, lmax=inp.ellmax)
                cl = curvedsky.alm2cl(alm[:,None,:], alm[None,:,:])
                ax = 0
                for t1 in range(3):
                    for t2 in range(t1,3):
                        t = axis_mapping[(t1,t2)]
                        plt.axes(axs[t])
                        CAR_spectra[i, ax] = cl[t1,t2]
                        ax += 1
                        plt.plot(ells[2:], (to_dl*cl[t1,t2])[2:], label=f'{freq} GHz', color=colors[i])
                        plt.title(f'{spectra_types_here[t1]}{spectra_types_here[t2]}')
                        plt.xlabel(r'$\ell$')
                        plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
                        plt.grid()
                        plt.xlim(2, inp.ellmax)
                        if t <= 2:
                            plt.yscale('log')
                        elif t==3:
                            plt.ylim(-0.08e-8, 0.08e-8)
                        elif t==4:
                            plt.ylim(-0.01e-7, 0.01e-7)
                        elif t==5:
                            plt.ylim(-0.01e-7, 0.01e-7)
                        plt.xscale('log')
            handles, labels = axs[-1].get_legend_handles_labels()
            fig.legend(handles, labels, fontsize=12, bbox_to_anchor=(1.0, 0.), ncol=3)
            plt.tight_layout()
            plt.savefig(f'{inp.plot_dir}/CAR_beam_power_spectra.png')
            print(f'saved {inp.plot_dir}/CAR_beam_power_spectra.png', flush=True)
        plt.close('all')
        if save:
            pickle.dump(CAR_spectra, open(f'{inp.output_dir}/CAR_spectra.p', 'wb'))
    

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
                plt.xlim(2, inp.ellmax)
                plt.legend()
                plt.savefig(f'{inp.plot_dir}/all_comp_spectra_{freq}.png')
                print(f'saved {inp.plot_dir}/all_comp_spectra_{freq}.png', flush=True)
            else:
                plt.clf()
                fig, axs = plt.subplots(2, 3, figsize=(9,6))
                axs = axs.flatten()
                for m in range(6):
                    plt.axes(axs[m])
                    for c in range(len(gal_spectra)):
                        plt.plot(ells[2:], (to_dl*gal_spectra[c,m])[2:], label=gal_comps[c])
                    for c in range(len(extragal_spectra)):
                        plt.plot(ells[2:], (to_dl*extragal_spectra[c,m])[2:], label=extragal_comps[c])
                    if m < 3:
                        plt.yscale('log')
                    ylims = [0.01e-8, 0.02e-8, 0.03e-8]
                    if m==3:
                        plt.ylim(-ylims[i], ylims[i])
                    elif m==4:
                        plt.ylim(-ylims[i]*10**(-2), ylims[i]*10**(-2))
                    elif m==5:
                        plt.ylim(-ylims[i]*10**(-1), ylims[i]*10**(-1))
                    plt.xlabel(r'$\ell$')
                    plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
                    plt.grid()
                    plt.xlim(2, inp.ellmax)
                    plt.title(f'{modes[m]}')
                handles, labels = axs[-1].get_legend_handles_labels()
                fig.legend(handles, labels, fontsize=12, bbox_to_anchor=(1.0, 0.), ncol=3)
                plt.suptitle(f'{freq} GHz', fontsize=20)
                plt.tight_layout()
                plt.savefig(f'{inp.plot_dir}/all_comp_spectra_{freq}.png')
                print(f'saved {inp.plot_dir}/all_comp_spectra_{freq}.png', flush=True)
        plt.close('all')


    #Galactic Mask-Deconvolved Power Spectrum at Each Frequency
    if  inp.plots_to_make=='all' or 'mask_deconvolved_spectra' in inp.plots_to_make:
        beam_convolved_maps = pickle.load(open(f'{inp.output_dir}/beam_convolved_maps.p', 'rb'))
        plot_and_save_mask_deconvolved_spectra(inp, beam_convolved_maps, plot_only=True)
        plt.close('all')


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
                    if m < 3:
                        plt.yscale('log')
                    plt.xlabel(r'$\ell$')
                    plt.ylabel(r'$D_\ell$ [$\mathrm{K}^2$]')
                    plt.grid()
                    plt.legend()
                    plt.savefig(f'{inp.plot_dir}/all_comp_spectra_{freq}_{modes[m]}_mask_deconvolved.png')
                    print(f'saved {inp.plot_dir}/all_comp_spectra_{freq}_{modes[m]}_mask_deconvolved.png', flush=True)
        plt.close('all')
            

    return
