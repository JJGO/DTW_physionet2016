import numpy as np

from itertools import chain
from scipy.io import loadmat


def extract(file):
    with open(file, 'r') as f:
        content = f.read().strip().split('\n')
    return content


def custom_loadmat(file):
    """
    Simple auxiliary function to load .mat files without
    the unnecesary MATLAB keys and squeezing unnecesary
    dimensions
    """
    d = loadmat(file)
    del d['__globals__']
    del d['__header__']
    del d['__version__']
    d = {k: d[k].squeeze() for k in d}
    return d


def paired_time(array, fs):
    """
    Time sequence associated with a signal sampled at fs

    Args:
        array : list
    Returns:
        time : numpy array
    """
    return np.arange(0, 1.0 / fs * (len(array)), 1.0 / fs)


def group_to_range(group):
    group = ''.join(group.split())
    sign, g = ('-', group[1:]) if group.startswith('-') else ('', group)
    r = g.split('-', 1)
    r[0] = sign + r[0]
    r = sorted(int(__) for __ in r)
    return range(r[0], 1 + r[-1])


def rangeexpand(txt, sep=','):
    ranges = chain.from_iterable(group_to_range(__) for __ in txt.split(sep))
    return sorted(set(ranges))
