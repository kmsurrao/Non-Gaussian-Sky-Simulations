import numpy as np
import healpy as hp
from tqdm import tqdm


def apply_beam(map_, beamfile, pol):
    '''
    ARGUMENTS
    ---------
    map_: numpy array, either [I,Q,U] healpix maps if pol, or just one intensity healpix map if not pol
    beamfile: str, path of file containing ACT beam information
    pol: Bool, whether [I,Q,U] or just I map is provided
    
    RETURNS
    -------
    beam_convolved_map: ndarray of beam-convolved [I,Q,U] if pol, otherwise just I
    '''
    data = np.loadtxt(beamfile)
    ll = data[:,0]
    Bl = data[:,1]
    Bl /= Bl[0]
    ellmax = min(int(ll[-1]), int(3*hp.get_nside(map_)-1))
    beam_convolved_map = hp.smoothing(map_, beam_window=Bl, pol=pol, lmax=ellmax)
    beam_convolved_map = np.array(beam_convolved_map, dtype=np.float32)
    return beam_convolved_map


def apply_beam_alm(alm, beamfile, pol):
    '''
    ARGUMENTS
    ---------
    alm: numpy array, either an array representing one alm, or a sequence of arrays for I/Q/U
    beamfile: str, path of file containing ACT beam information
    pol: Bool, whether [I,Q,U] or just I map is provided
    
    RETURNS
    -------
    beam_convolved_alm: ndarray of beam-convolved [I,Q,U] alm if pol, otherwise just I
    '''
    data = np.loadtxt(beamfile)
    ll = data[:,0]
    Bl = data[:,1]
    Bl /= Bl[0]
    ellmax = min(int(ll[-1]), int(hp.Alm.getlmax(len(alm))))
    beam_convolved_alm = hp.smoothalm(alm, beam_window=Bl, pol=pol, lmax=ellmax)
    return beam_convolved_alm


def get_all_beam_convolved_maps(inp, all_maps):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input specifications
    all_maps: ndarray of shape (Nfreqs, Npix) if not pol, 
            or shape (Nfreqs, 3 for I/Q/U, Npix) if pol (freqs in descending order)
            contains healpix maps of galactic and extragalactic components before beam convolution
             
    RETURNS
    -------
    beam_convolved_maps: ndarray of shape (3 for PA4 PA5 PA6, 2 for freqs in array in ascending order, Nsplits, Npix) if not pol,
        or shape (3 for PA4 PA5 PA6, 2 for freqs in array in ascending order, Nsplits, 3 for I/Q/U, Npix) if pol
        contains beam-convolved healpix maps for each frequency and split
    '''
    
    freq_maps = all_maps #shape (Nfreqs, Npix) if not pol or (Nfreqs, 3, Npix) if pol (freqs in decreasing order)
    if not inp.pol: #index as beam_convolved_maps[PA, freq, split, pixel], freqs in ascending order
        beam_convolved_maps = np.zeros((3, 2, 4, 12*inp.nside**2), dtype=np.float32)
    else: #index as beam_convolved_maps[PA, freq, split, I/Q/U, pixel], freqs in ascending order
        beam_convolved_maps = np.zeros((3, 2, 4, 3, 12*inp.nside**2), dtype=np.float32)
    pa_freq_dict = {4:[150,220], 5:[90,150], 6:[90,150]}
    all_maps_freq_idx = {220:0, 150:1, 90:2} #map frequency to index in all_maps
    for p, pa in enumerate([4, 5, 6]):
        for i, freq in enumerate(pa_freq_dict[pa]):
            for split in range(4):
                beamfile = f'{inp.beam_dir}/set{split}_pa{pa}_f{freq:0>3}_night_beam_tform_jitter_cmb.txt'
                beam_convolved_maps[p, i, split] = apply_beam(freq_maps[all_maps_freq_idx[freq]], beamfile, inp.pol)
    return beam_convolved_maps


def get_all_beam_convolved_alm(inp, all_maps):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input specifications
    all_maps: ndarray of shape (Nfreqs, Npix) if not pol, 
            or shape (Nfreqs, 3 for I/Q/U, Npix) if pol (freqs in descending order)
            contains healpix maps of galactic and extragalactic components before beam convolution
             
    RETURNS
    -------
    beam_convolved_maps: ndarray of shape (3 for PA4 PA5 PA6, 2 for freqs in array in ascending order, Nsplits, Npix) if not pol,
        or shape (3 for PA4 PA5 PA6, 2 for freqs in array in ascending order, Nsplits, 3 for I/Q/U, Npix) if pol
        contains beam-convolved healpix maps for each frequency and split
    '''
    print('Applying beams...', flush=True)
    freq_maps = all_maps #shape (Nfreqs, Npix) if not pol or (Nfreqs, 3, Npix) if pol (freqs in decreasing order)
    freq_alm = [hp.map2alm(map_) for map_ in freq_maps]
    alm_size = hp.Alm.getsize(3*inp.nside-1)
    if not inp.pol: #index as beam_convolved_alm[PA, freq, split, alm_idx], freqs in ascending order
        beam_convolved_alm= np.zeros((3, 2, 4, alm_size), dtype=np.complex128)
    else: #index as beam_convolved_maps[PA, freq, split, I/Q/U, alm_idx], freqs in ascending order
        beam_convolved_alm = np.zeros((3, 2, 4, 3, alm_size), dtype=np.complex128)
    pa_freq_dict = {4:[150,220], 5:[90,150], 6:[90,150]}
    all_maps_freq_idx = {220:0, 150:1, 90:2} #map frequency to index in all_maps
    for p, pa in enumerate(tqdm([4, 5, 6], desc='PA')):
        for i, freq in enumerate(tqdm(pa_freq_dict[pa], desc='Freq')):
            for split in tqdm(range(4), desc='Split'):
                beamfile = f'{inp.beam_dir}/set{split}_pa{pa}_f{freq:0>3}_night_beam_tform_jitter_cmb.txt'
                beam_convolved_alm[p, i, split] = apply_beam_alm(freq_alm[all_maps_freq_idx[freq]], beamfile, inp.pol)
    return beam_convolved_alm
