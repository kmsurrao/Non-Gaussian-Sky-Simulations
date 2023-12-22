import healpy as hp
import numpy as np
from tqdm import tqdm

def rotate_healpix(map_, ellmax):
    '''
    ARGUMENTS
    ---------
    map_: 1D numpy array if not inp.pol, otherwise 3D numpy array for T,Q,U containing
        map in healpix format
    ellmax: int, maximum ell for which to compute alm

    RETURNS
    -------
    map_rotated: 1D numpy array if not inp.pol, otherwise 3D numpy array for T,Q,U containing
        map in healpix format after applying rotation from galactic to equatorial (celestial) coordinates
    '''

    # get alm and rotate in alm space, then transform back to map space
    nside = hp.get_nside(map_)
    alm = hp.map2alm(map_, lmax=ellmax)
    rot_gal2eq = hp.Rotator(coord="GC") # C stands for celestial=equatorial
    alm_rotated = rot_gal2eq.rotate_alm(alm, lmax=ellmax)
    map_rotated = hp.alm2map(alm_rotated, nside)
    return map_rotated


def rotate_healpix_alm(alm, ellmax, nside):
    '''
    ARGUMENTS
    ---------
    alm: 1D numpy array if not inp.pol, otherwise 3D numpy array for T,Q,U containing
        alm in healpix ordering
    ellmax: int, maximum ell for which to compute alm
    nside: int, nside of map to convert to

    RETURNS
    -------
    map_rotated: 1D numpy array if not inp.pol, otherwise 3D numpy array for T,Q,U containing
        map in healpix format after applying rotation from galactic to equatorial (celestial) coordinates
    '''

    alm = alm[:hp.Alm.getidx(ellmax, ellmax, ellmax)+1]
    rot_gal2eq = hp.Rotator(coord="GC") # C stands for celestial=equatorial
    alm_rotated = rot_gal2eq.rotate_alm(alm, lmax=ellmax)
    map_rotated = hp.alm2map(alm_rotated, nside)
    return map_rotated


def get_all_rotated_maps(inp, beam_convolved_alm):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input specifications
    beam_convolved_alm: ndarray of shape (3 for PA4 PA5 PA6, 2 for freqs in array in ascending order, Nsplits, alm_size) if not pol,
        or shape (3 for PA4 PA5 PA6, 2 for freqs in array in ascending order, Nsplits, 3 for I/Q/U, alm_size) if pol
        contains beam-convolved healpix alm for each frequency and split
    
    RETURNS
    -------
    rotated_maps: ndarray of shape (3 for PA4 PA5 PA6, 2 for freqs in array in ascending order, Nsplits, Npix) if not pol,
        or shape (3 for PA4 PA5 PA6, 2 for freqs in array in ascending order, Nsplits, 3 for I/Q/U, Npix) if pol
        contains beam-convolved healpix maps for each frequency and split after 
        rotating from galactic to equatorial (celestial) coordinates

    '''
    print('Rotating maps...', flush=True)
    if not inp.pol: #index as rotated_maps[PA, freq, split, pixel], freqs in ascending order
        rotated_maps = np.zeros((3, 2, 4, 12*inp.nside**2), dtype=np.float32)
    else: #index as rotated_maps[PA, freq, split, I/Q/U, pixel], freqs in ascending order
        rotated_maps = np.zeros((3, 2, 4, 3, 12*inp.nside**2), dtype=np.float32)
    pa_freq_dict = {4:[150,220], 5:[90,150], 6:[90,150]}
    for p, pa in enumerate(tqdm([4, 5, 6], desc='PA')):
        for i, freq in enumerate(tqdm(pa_freq_dict[pa], desc='Freq')):
            for split in tqdm(range(4), desc='Split'):

                # get ellmax for rotation based on max ell in beamfile
                beamfile = f'{inp.beam_dir}/set{split}_pa{pa}_f{freq:0>3}_night_beam_tform_jitter_cmb.txt'
                data = np.loadtxt(beamfile)
                ll = data[:,0]

                rotated_maps[p,i,split] = rotate_healpix_alm(beam_convolved_alm[p,i,split], min(int(ll[-1]), 3*inp.nside-1), inp.nside)

    return rotated_maps