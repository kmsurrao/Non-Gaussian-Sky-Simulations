import numpy as np
import matplotlib.pyplot as plt

def plot_hist(inp, cmap, freq, comp):
    '''
    Plots histogram of pixel values and saves them in inp.plot_dir

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    cmap: 1D numpy array if not inp.pol, otherwise 3D numpy array for T,Q,U containing
        maps in healpix format
    freq: int, frequency of input map in GHz
    comp: str, name of component shown in the map

    RETURNS
    -------
    None
    '''
    types = ['I', 'Q', 'U']
    for t in range(3):
        plt.clf()
        plt.hist(cmap[t], density=True, alpha=0.5)
        plt.yscale('log')
        plt.savefig(f'{inp.plot_dir}/pixel_hist_{freq}ghz_{types[t]}_{comp}_before.png')
        if not inp.pol:
            break
    return