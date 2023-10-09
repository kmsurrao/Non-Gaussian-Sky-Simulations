import numpy as np
import healpy as hp
import multiprocessing as mp


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



def get_all_beam_convolved_maps(inp, all_maps, parallel):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input specifications
    all_maps: ndarray of shape (Nfreqs, Npix), 
             if not pol, or shape (Nfreqs, 3 for I/Q/U, Npix) if pol
             contains healpix maps of galactic and extragalactic components before beam convolution
    parallel: Bool, whether to run in parallel (can cause memory issues for large maps)
             
    RETURNS
    -------
    beam_convolved_maps: ndarray of shape (Nfreqs, Nsplits, Npix) if not pol,
        or shape (Nfreqs, Nsplits, 3 for I/Q/U, Npix) if pol
        contains beam-convolved healpix maps for each frequency and split
    '''
    
    freq_maps = all_maps #shape (Nfreqs, Npix) if not pol or (Nfreqs, 3, Npix) if pol
    beamfiles = []
    for freq in range(3):
        for split in range(4):
            if freq==0:
                beamfile = f'{inp.beam_dir}/set{split}_pa4_f220_night_beam_tform_jitter_cmb.txt'
            elif freq==1:
                beamfile = f'{inp.beam_dir}/set{split}_pa5_f150_night_beam_tform_jitter_cmb.txt'
            elif freq==2:
                beamfile = f'{inp.beam_dir}/set{split}_pa6_f090_night_beam_tform_jitter_cmb.txt'
            beamfiles.append(beamfile)
    
    if parallel:
        pool = mp.Pool(12)
        results = pool.starmap(apply_beam, [(freq_maps[i//4], beamfiles[i], inp.pol) for i in range(12)])
        pool.close()
        results = np.array(results, dtype=np.float32)
    else:
        results = []
        for i in range(12):
            results.append(apply_beam(freq_maps[i//4], beamfiles[i], inp.pol))

    if not inp.pol: #index as all_maps[freq, split, pixel], freqs in decreasing order
        beam_convolved_maps = np.reshape(results, (3, 4, 12*inp.nside**2))
    else: #index as all_maps[freq, split, I/Q/U, pixel], freqs in decreasing order
        beam_convolved_maps = np.reshape(results, (3, 4, 3, 12*inp.nside**2))
    return beam_convolved_maps
