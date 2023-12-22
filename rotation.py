import healpy as hp
import numpy as np
import multiprocessing as mp

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


def get_all_rotated_maps(inp, beam_convolved_maps):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input specifications
    beam_convolved_maps: ndarray of shape (3 for PA4 PA5 PA6, 2 for freqs in array in ascending order, Nsplits, Npix) if not pol,
        or shape (3 for PA4 PA5 PA6, 2 for freqs in array in ascending order, Nsplits, 3 for I/Q/U, Npix) if pol
        contains beam-convolved healpix maps for each frequency and split
    
    RETURNS
    -------
    rotated_maps: ndarray of shape (3 for PA4 PA5 PA6, 2 for freqs in array in ascending order, Nsplits, Npix) if not pol,
        or shape (3 for PA4 PA5 PA6, 2 for freqs in array in ascending order, Nsplits, 3 for I/Q/U, Npix) if pol
        contains beam-convolved healpix maps for each frequency and split after 
        rotating from galactic to equatorial (celestial) coordinates

    '''
    rotated_maps = np.zeros_like(beam_convolved_maps)
    pa_freq_dict = {4:[150,220], 5:[90,150], 6:[90,150]}
    for p, pa in enumerate([4, 5, 6]):
        for i, freq in enumerate(pa_freq_dict[pa]):
            for split in range(4):

                # get ellmax for rotation based on max ell in beamfile
                beamfile = f'{inp.beam_dir}/set{split}_pa{pa}_f{freq:0>3}_night_beam_tform_jitter_cmb.txt'
                data = np.loadtxt(beamfile)
                ll = data[:,0]

                rotated_maps[p,i,split] = rotate_healpix(beam_convolved_maps[p,i,split], min(int(ll[-1]), 3*inp.nside-1))

    return rotated_maps