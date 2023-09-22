import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(inp, cmap, freq, comp, string='before'):
    '''
    Plots histogram of pixel values and saves them in inp.plot_dir

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    cmap: 1D numpy array if not inp.pol, otherwise 3D numpy array for T,Q,U containing
        maps in healpix format
    freq: int, frequency of input map in GHz
    comp: str, name of component shown in the map
    string: str, 'before' or 'after' based on whether plotting histogram 
                    before or after applying flux cut

    RETURNS
    -------
    None
    '''
    types = ['I', 'Q', 'U']
    for t in range(3):
        plt.clf()
        if inp.pol:
            plt.hist(cmap[t], density=True, alpha=0.5)
        else:
            plt.hist(cmap, density=True, alpha=0.5)
        plt.yscale('log')
        plt.savefig(f'{inp.plot_dir}/pixel_hist_{freq}ghz_{types[t]}_{comp}_{string}.png')
        plt.close()
        if not inp.pol:
            break
    return