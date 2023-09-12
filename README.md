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
4. Compute power spectra and, optionally, galactic mask-deconvolved power spectrum of each component.
    - Computed in [bandpass_integration.py](bandpass_integration.py) and [galactic_mask.py](galactic_mask.py) and saved as gal_comp_spectra_{central_freq}.p and gal_comp_mask_deconvolved_spectra_{central_freq}.p.  
5. Apply beams to frequency maps. 
    - Computed in [beams.py](beams.py) and saved as beam_convolved_maps.p.  
6. Optionally compute galactic mask-deconvolved power spectrum of each frequency map.  
    - Computed in [galactic_mask.py](galactic_mask.py) and saved as mask_deconvolved_spectra.p and ell bins saved in ell_eff.p.    
7. Reproject from HEALPIX to CAR.  
    - Computed in [reprojection.py](reprojection.py) and saved as sim_{freq}GHz.  
8. Make plots. 
    - Computed in [make_plots.py](make_plots.py) and saved in plot_dir.  