import yaml
import numpy as np
import os

##########################
# simple function for opening the file
def read_dict_from_yaml(yaml_file):
    assert(yaml_file != None)
    with open(yaml_file) as f:
        config = yaml.safe_load(f)
    return config
##########################

##########################
"""
class that contains input info
"""
class Info(object):
    def __init__(self, input_file):
        self.input_file = input_file
        p = read_dict_from_yaml(self.input_file)

        self.nside = p['nside']
        assert type(self.nside) is int and (self.nside & (self.nside-1) == 0) and self.n != 0, "nside must be integer power of 2"
        self.ellmax = p['ellmax']
        assert type(self.ellmax) is int and self.ellmax >= 2, "ellmax must be integer >= 2"
        assert self.ellmax <= 3*self.nside-1, "ellmax must be less than 3*nside-1"
        self.ells_per_bin = p['ells_per_bin']
        if self.ells_per_bin:
            assert type(self.ells_per_bin) is int and 1 <= self.ells_per_bin <= self.ellmax-2, "ells_per_bin must be int with 1 <= ells_per_bin <= ellmax-2"
        self.galactic_components = p['galactic_components']
        self.pol = p['pol']

        self.passband_file = p['passband_file']
        assert type(self.passband_file) is str, "TypeError: passband_file must be str"
        assert os.path.isfile(self.passband_file), "Passband file does not exist"
        self.agora_sims_dir = p['agora_sims_dir']
        assert type(self.agora_sims_dir) is str, "TypeError: agora_sims_dir must be str"
        assert os.path.isdir(self.agora_sims_dir), "agora_sims_dir does not exist"
        self.ksz_reionization_file = p['ksz_reionization_file']
        if self.ksz_reionization_file:
            assert type(self.ksz_reionization_file) is str, "TypeError: ksz_reionization_file must be str"
            assert os.path.isfile(self.ksz_reionization_file), "ksz_reionization_file does not exist"
        self.mask_file = p['mask_file']
        if self.mask_file:
            assert type(self.mask_file) is str, "TypeError: mask_file must be str"
            assert os.path.isfile(self.mask_file), "mask_file does not exist"
        self.beam_dir = p['beam_dir']
        assert type(self.beam_dir) is str, "TypeError: beam_dir must be str"
        assert os.path.isdir(self.beam_dir), "beam_dir does not exist"

        self.output_dir = p['output_dir']
        assert type(self.output_dir) is str, "TypeError: output_dir must be str"
        self.plot_dir = p['plot_dir']
        assert type(self.plot_dir) is str or not self.plot_dir, "TypeError: plot_dir must be str"

        self.plots_to_make = p['plots_to_make']
        assert type(self.plots_to_make) is str or type(self.plots_to_make) is list, "TypeError: plots_to_make must be the string 'all' or a list of strings"
        if type(self.plots_to_make) is str:
            return
        if not self.plot_dir:
            assert not self.plots_to_make, "Must define plot_dir if plots_to_make is not an emptry list"
        assert set(self.plots_to_make).issubset({'passband', 'gal_and_extragal_comps', 'freq_maps_no_beam', 
              'beam_convolved_maps', 'CAR_maps', 'all_comp_spectra', 'mask_deconvolved_spectra',
              'mask_deconvolved_comp_spectra'}), \
              "plots_to_make must be a subset of 'passband', 'gal_and_extragal_comps', 'freq_maps_no_beam', 'beam_convolved_maps', 'CAR_maps', 'all_comp_spectra', 'mask_deconvolved_spectra', 'mask_deconvolved_comp_spectra'"
        if 'mask_deconvolved_spectra' in self.plots_to_make or 'mask_deconvolved_comp_spectra' in self.plots_to_make:
            assert self.ells_per_bin is not None and self.mask_file is not None, "mask_file and ells_per_bin must be defined to plot mask-deconvolved spectra"