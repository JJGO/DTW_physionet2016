import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .dtw import dtw
from .multidtw import dtw_distances

def plot_alignment(x, y, **kwargs):
    """
    Plot Dynamic Time Warping alignment between two sequences

    Args:
        x : 1d array_like object (N)
            first sequence
        y : 1d array_like object (M)
            second sequence
        **kwargs : dict
            See dtwpy.dtw.dtw for more keyword values
    Returns:
        fig : matplotlib.pyplot.figure
            Figure with the plot
    """
    dist, cost_matrix, (alig_x, alig_y) = dtw(x,y,dist_only=False,**kwargs)
    fig = plt.figure()
    plt.plot(x[alig_x])
    plt.plot(y[alig_y])
    return fig

def plot_cost_matrix(x, y, ratio=4, cmap=plt.cm.Reds, axis=False, **kwargs):
    """
    Plot Dynamic Time Warping cost matrix between two sequences.
    First Sequence will be aligned on top, second on the left
    
    TODO : Make the spacing relative

    Args:
        x : 1d array_like object (N)
            first sequence
        y : 1d array_like object (M)
            second sequence
        ratio : int
            ratio of matrix to sequence
        cmap : matplotlib.colors.LinearSegmentedColormap
            Defaults to plt.cm.Reds
        axis : bool
            Displays axis on the aligned sequences. Defaults to False
        **kwargs : dict
            See dtwpy.dtw.dtw for more keyword values
    Returns:
        fig : matplotlib.pyplot.figure
            Figure with the cost matrix and sequences
    """
    fig = plt.figure(figsize=(8, 8)) #TODO
    plt.subplots_adjust(hspace=0.1, wspace=0.15) #TODO
    gs = gridspec.GridSpec(2, 2, width_ratios=[1,4],height_ratios=[1,4])

    dist, cost_matrix, (alig_x, alig_y) = dtw(x,y,dist_only=False,**kwargs)

    # Sequence x, displayed above the matrix
    ax1 = plt.subplot(gs[1])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.axis('off') if not axis else None
    ax1.plot(np.arange(len(y)),y)
    # Sequence y, displayed left of the matrix
    ax2 = plt.subplot(gs[2])
    ax2.axis('off') if not axis else None
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.yaxis.set_ticks_position('right')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.plot(-x,np.arange(len(x)))
    # Cost Matrix
    ax3 = plt.subplot(gs[3])
    ax3.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    ax3.axis('off')
    ax3.plot(alig_y, alig_x,color='g')
    im = ax3.imshow(cost_matrix, origin='lower', cmap=cmap, interpolation='nearest')
    cax = fig.add_axes([0.93, 0.1315, 0.02, 0.58]) #TODO
    fig.colorbar(im, cax=cax)
    return fig

def plot_distances(X, cmap=plt.cm.Reds, axis=True, **kwargs):
    """
    Given an array of arbitrarily sized 1d arrays, plots the
     Dynamic Time Warping pairwise cost matrix between the sequences

    Args:
        X : {list,array} of n_samples 1d numpy arrays
            Set of 1d numpy arrays
        cmap : matplotlib.colors.LinearSegmentedColormap
            Defaults to plt.cm.Reds
        axis : bool
            Displays axis on the aligned sequences. Defaults to True
        **kwargs : dict
            See dtwpy.multidtw.dtw_distances for more keyword values
    Returns:
        fig : matplotlib.pyplot.figure
            Figure with the matrix of DTW distances
    """
    dist_matrix = dtw_distances(X,**kwargs)

    fig = plt.figure()
    im = plt.imshow(dist_matrix,cmap=cmap, interpolation='nearest')
    fig.axis('off') if not axis else None
    return fig

def plot_cost_matrices(X, vmax=None, cmap=plt.cm.Reds, axis=False, **kwargs):
    """
    Given an array of arbitrarily sized 1d arrays, plots the
     Dynamic Time Warping pairwise cost matrix between the sequences

    TODO : Make the spacing relative

    Args:
        X : {list,array} of n_samples 1d numpy arrays
            Set of 1d numpy arrays
        vmax : int
            Maximum distance for the cost matrix display
        cmap : matplotlib.colors.LinearSegmentedColormap
            Defaults to plt.cm.Reds
        axis : bool
            Displays axis on the aligned sequences. Defaults to False
        **kwargs : dict
            See dtwpy.multidtw.dtw_distances for more keyword values
    Returns:
        fig : matplotlib.pyplot.figure
            Figure with the grid of sequences and DTW cost matrices
    """
    VMAX = max( len(s) for s in X)//2
    vmax = VMAX if not vmax else vmax

    N = len(X)
    dist_matrix = np.zeros([N,N])
    indexes = range(N)
    
    fig = plt.figure(figsize=(2.5*N, 2.5*N)) #TODO
    plt.subplots_adjust(hspace=0.1, wspace=0.1) #TODO

    # Plot all the sequence on the top row and leftmost column
    for i,s in enumerate(X):
        t = range(len(s))
        k = np.ravel_multi_index((0,i+1),(N+1,N+1))
        ax = plt.subplot(N+1,N+1,k+1)
        ax.plot(t,s)
        ax.axis('off') if not axis else None
        k = np.ravel_multi_index((i+1,0),(N+1,N+1))
        ax = plt.subplot(N+1,N+1,k+1)
        ax.plot(-s,t)
        ax.axis('off') if not axis else None

    # For each combination (with replacement) display two cost matrices
    # (transpose of each other) and the alignment path
    for i,j in itertools.combinations_with_replacement(indexes,2):
        dist, cost_matrix, (alig_i, alig_j) = dtw(X[i],X[j],dist_only=False,**kwargs)
        ax = plt.subplot(N+1,N+1,1+np.ravel_multi_index((i+1,j+1),(N+1,N+1)))
        ax.plot(alig_j,alig_i,color='g')
        ax.axis('off') if not axis else None
        im = ax.imshow(cost_matrix,vmin=0,vmax=vmax,origin='lower', cmap=cmap, interpolation='nearest')
        ax = plt.subplot(N+1,N+1,1+np.ravel_multi_index((j+1,i+1),(N+1,N+1)))
        ax.plot(alig_i,alig_j,color='g')
        ax.axis('off') if not axis else None
        im = ax.imshow(cost_matrix.T,vmin=0,vmax=vmax,origin='lower', cmap=cmap, interpolation='nearest')
        dist_matrix[(i,j)] = dist
        dist_matrix[(j,i)] = dist
    

    ax = plt.subplot(N+1,N+1,1)
    # At the top left corner display matrix of DTW distances
    im = ax.imshow(dist_matrix, vmin=0, vmax=vmax,cmap=cmap, interpolation='nearest')
    ax.axis('off') if not axis else None
    # Make an axis for the colorbar on the right side
    cax = fig.add_axes([0.93, 0.1, 0.02, 0.7])
    plt.colorbar(im, cax=cax)
    return fig