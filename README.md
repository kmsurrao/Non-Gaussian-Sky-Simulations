# Non-Gaussian-Sky-Simulations
Generation of bandpass-integrated, beam-convolved full-sky non-Gaussian simulations for ACT likelihood validation 

## Running
Create or modify one of the yaml files in the example_yaml_files directory.  
```python main.py --config = example_yaml_files/[FILENAME].yaml```   

## Dependencies
healpy     
pixell     
h5py   
pysm3    
NaMaster  

## Operations
1. Get passbands.  
    - Computed in [bandpass_integration.py](bandpass_integration.py) and saved as all_bandpass_freqs.p and all_bandpass_weights.p.    
2. Get bandpass-integrated galactic component maps from pysm.  
    - Computed in [bandpass_integration.py](bandpass_integration.py) and saved as gal_and_extragal_before_beam.p.  
3. Get bandpass-integrated extragalactic component maps from agora simulations, as well as reionization/patchy kSZ. Apply initial flux cut to radio and CIB agora maps by inpainting extremely bright pixels (> 100 mJy).  
    - Computed in [bandpass_integration.py](bandpass_integration.py), [apply_flux_cut.py](apply_flux_cut.py), and [diffusive_inpaint.py](diffusive_inpaint.py), and saved as gal_and_extragal_before_beam.p.
    - NOT IMPLEMENTED: Harmonic transform and apply beam convolution. Transform back to pixel space and apply flux cut mask (> 15 mJy at 150 GHz) with 5 arcmin holes, using inpainting.    
4. Optionally compute power spectra and galactic mask-deconvolved power spectrum of each component.
    - Computed in [bandpass_integration.py](bandpass_integration.py) and [galactic_mask.py](galactic_mask.py) and saved as gal_comp_spectra_{central_freq}.p and gal_comp_mask_deconvolved_spectra_{central_freq}.p.  
5. Apply beams to frequency maps. 
    - Computed in [beams.py](beams.py) and saved as beam_convolved_maps.p.  
6. Optionally compute galactic mask-deconvolved power spectrum of each frequency map.  
    - Computed in [galactic_mask.py](galactic_mask.py) and saved as mask_deconvolved_spectra.p and $\ell$ bins saved in ell_eff.p.    
7. Reproject from HEALPIX to CAR.  
    - Computed in [reprojection.py](reprojection.py) and saved as sim_{freq}GHz.  
8. Make plots. 
    - Computed in [make_plots.py](make_plots.py) and saved in plot_dir.  

## Outputs (not including plots)  
The following outputs are saved in the output directory listed in the yaml file:  
1. all_bandpass_freqs.p  
    - list of 3 numpy arrays of frequencies (GHz) in the passband (first array has central frequency 220 GHz, followed by 150, and then 90)  
2. all_bandpass_weights.p  
    - list of 3 numpy arrays of bandpass weights with $\nu^2$ divided out (first array contains weights for central frequency 220 GHz, followed by 150, and then 90)  
3. gal_and_extragal_before_beam.p  
    - numpy array containing galactic and extragalactic component maps in healpix format before beam convolution. If generating polarized maps, gal_and_extragal_before_beam has shape (Nfreqs, 2 for gal and extragal, 3 for I/Q/U, Npix), where freqs are indexed in decreasing order. If generating only a temperature map, gal_and_extragal_before_beam has shape (Nfreqs, 2 for gal or extragal, Npix).  
4. beam_convolved_maps.p  
    - numpy array containing beam-convolved maps in healpix format. If generating polarized maps, beam_convolved_maps has shape (Nfreqs, 3 for I/Q,U, Npix). Otherwise, has shape (Nfreqs, Npix).  
5. sim_90GHz, sim_150GHz, sim_220_GHz  
    - final CAR maps
6. (Optional) gal_comp_spectra_{central_freq}.p and extragal_comp_spectra_{central_freq}.p  
    - numpy array containing power spectrum of each galactic component at central_freq. If generating polarized maps, has shape (N_(extra)galactic_comps, 6, $\ell_{\mathrm{max}}$), where 6 is for TT, EE, BB, TE, EB, TE. If only generating temparature maps, has shape (N_(extra)galactic_comps, $\ell_{\mathrm{max}}$).  
7. (Optional) ell_eff.p  
    - numpy array containing central $\ell$ values in each bin used for mask deconvolution.  
8. (Optional) gal_comp_mask_deconvolved_spectra_{central_freq}.p  
    - numpy array containing Planck 70% fsky galactic mask-deconvolved power spectrum of each galactic component at central_freq. If generating polarized maps, has shape (N_galactic_components, 6, Nbins), where 6 is for TT, EE, BB, TE, EB, TE, and Nbins is the number of values in ell_eff. If generating only temperature maps, has shape (N_galactic_components, Nbins).  
9. (Optional) mask_deconvolved_spectra.p  
    - numpy array containing Planck 70% fsky galactic mask-deconvolved power spectrum of each frequency map containing all galactic and extragalactic components. If generating polarized maps, has shape (Nfreqs, 6, Nbins), where 6 is for TT, EE, BB, TE, EB, TE, and Nbins is the number of values in ell_eff. If generating only temperature maps, has shape (Nfreqs, Nbins).  