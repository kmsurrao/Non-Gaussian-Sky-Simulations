import numpy as np
import healpy as hp


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
    Bl = data[:,1]
    Bl /= Bl[0]
    beam_convolved_map = hp.smoothing(map_, beam_window=Bl, pol=pol)
    return beam_convolved_map