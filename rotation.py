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


def get_all_rotated_maps(inp, beam_convolved_maps, parallel=False):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input specifications
    beam_convolved_maps: ndarray of shape (Nfreqs, Nsplits, Npix) if not pol,
        or shape (Nfreqs, Nsplits, 3 for I/Q/U, Npix) if pol
        contains beam-convolved healpix maps for each frequency and split
    parallel: Bool, whether to run reprojection in parallel (can cause memory issues)
    
    RETURNS
    -------
    rotated_maps: ndarray of shape (Nfreqs, Nsplits, Npix) if not pol,
        or shape (Nfreqs, Nsplits, 3 for I/Q/U, Npix) if pol
        contains beam-convolved healpix maps for each frequency and split after 
        rotating from galactic to equatorial (celestial) coordinates

    '''
    rotated_maps = np.zeros_like(beam_convolved_maps)

    # get ellmax for rotation based on max ell in beamfile
    ellmax_arr = [] #contains max ell in beamfile for each frequency (220, 150, 90)
    for freq in range(3):
        if freq==0:
            beamfile = f'{inp.beam_dir}/set0_pa4_f220_night_beam_tform_jitter_cmb.txt'
        elif freq==1:
            beamfile = f'{inp.beam_dir}/set0_pa5_f150_night_beam_tform_jitter_cmb.txt'
        elif freq==2:
            beamfile = f'{inp.beam_dir}/set0_pa6_f090_night_beam_tform_jitter_cmb.txt'
        data = np.loadtxt(beamfile)
        ll = data[:,0]
        Bl = data[:,1]
        Bl /= Bl[0]
        ellmax_arr.append(min(int(ll[-1]), 3*inp.nside-1))

    if parallel:
        pool = mp.Pool(12)
        results = pool.starmap(rotate_healpix, [(beam_convolved_maps[i//4, i%4], ellmax_arr[i//4]) for i in range(12)])
        pool.close()
    
    idx = 0
    for freq in range(3):
        for split in range(4):
            if parallel:
                map_ = results[idx]
            else:
                map_ = rotate_healpix(beam_convolved_maps[freq, split], ellmax_arr[freq])
            rotated_maps[freq, split] = map_
            idx += 1

    return rotated_maps