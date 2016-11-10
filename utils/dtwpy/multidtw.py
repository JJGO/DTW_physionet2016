import multiprocessing as mp
import itertools
import numpy as np

from .dtw import dtw


def _dtw_wrapper(arg):
    """
    Wrapper function over dtwpy.dtw.dtw in order to unpack
    args and kwargs accordingly.
    Needed for parallelization purposes
    """
    args, kwargs = arg
    return dtw(*args, **kwargs)


def dtw_distances(X, Y=None, n_jobs=1, **kwargs):
    """
    Given an array of arbitrarily sized 1d arrays, computes the
     Dynamic Time Warping distance matrix between each pair of sequences.

    Args:
        X : {list,array} of n_samples 1d numpy arrays
            Set of 1d numpy arrays
        n_jobs : int
            Number of threads to use, if -1 uses all. Defaults to 1
        **kwargs : dict
            See dtwpy.dtw.dtw for more keyword values
    Returns:
        dist_matrix : {array} shape (n_samples, n_samples)
            Symmetric distance matrix
    """
    # If -1 use all
    n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

    N = len(X)
    if Y is None:
        # TODO use numpy.triu_indices to avoid using the indexes
        dist_matrix = np.zeros([N, N])
        indexes = range(N)
        # Symmetric zero diagonal matrix , only compute unique pairwise distances
        params = zip(itertools.combinations(X, 2), itertools.repeat(kwargs))
        with mp.Pool(processes=n_jobs) as pool:
            distances = pool.map(_dtw_wrapper, params)
        for (i, j), d in zip(itertools.combinations(indexes, 2), distances):
            dist_matrix[(i, j)] = d
            dist_matrix[(j, i)] = d
    else:
        M = len(Y)
        params = zip(itertools.product(X, Y), itertools.repeat(kwargs))
        with mp.Pool(processes=n_jobs) as pool:
            distances = pool.map(_dtw_wrapper, params)
        dist_matrix = np.array(distances).reshape(N, M)

    return dist_matrix


def dtw_medoid(X, **kwargs):
    """
    Given an array of arbitrarily sized 1d arrays, computes the
     Dynamic Time Warping medoid sequence and returns the index

    Args:
        X : {list,array} of n_samples 1d numpy arrays
            Set of 1d numpy arrays
        **kwargs : dict
            See dtwpy.multidtw.dtw_distances for more keyword values
    Returns:
        medoid_index : int
            index of medoid sequence in X
    """
    dist_matrix = dtw_distances(X, **kwargs)
    medoid_index = np.argmin(np.sum(dist_matrix, axis=0))
    return medoid_index
