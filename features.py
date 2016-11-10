#!/usr/bin/env python

import numpy as np
import pandas as pd
import pywt

from scipy.signal import resample, hilbert, butter, filtfilt, lfilter

from _defaults import TEMPLATE_FOLDER

from utils.dtwpy import dtw, dtw_distances
from utils.dtwpy.dtw import _normalize
from utils.speech import mfcc
from utils.segmentation import get_transitions, boundaries, get_intervals
from utils.misc import custom_loadmat


# HELPERS
# Simple auxiliary function to compute a pair of mean and std

interDTW_down = 2

ALENGTH = {
    'RR':   960,
    'S1':   150,
    'Sys':  210,
    'S2':   120,
    'Dia':  510,
}


def _mean_std(x):
    return (np.mean(x), np.std(x))


def dtw_affinity(x, y, sigma=1e2, **kwargs):
    d = dtw(x, y, **kwargs)
    return affinity(d, sigma=sigma)


def affinity(x, sigma=1):
    """
    Computes affinity with kernel width sigma
    Maps real distance value to simmilarity interval (0,1]

    Args:
        x : float
            real value to compute affinity
        sigma : float
            kernel width
    Returns:
        affinity : float
    """
    return np.exp(-1.0 / (2 * sigma**2) * (x**2))


# SIGNAL TRANSFORMS


def hilbert_transform(x):
    """
    Computes modulus of the complex valued
    hilbert transform of x
    """
    return np.abs(hilbert(x))


def homomorphic_envelope(x, fs=1000, f_LPF=8, order=3):
    """
    Computes the homomorphic envelope of x

    Args:
        x : array
        fs : float
            Sampling frequency. Defaults to 1000 Hz
        f_LPF : float
            Lowpass frequency, has to be f_LPF < fs/2. Defaults to 8 Hz
    Returns:
        time : numpy array
    """
    b, a = butter(order, 2 * f_LPF / fs, 'low')
    he = np.exp(filtfilt(b, a, np.log(np.abs(hilbert(x)))))
    return he


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def dtw_envelope(x, fs=1000, f_LPF=20, order_LPF=5):
    envelope = homomorphic_envelope(x, fs, f_LPF, order_LPF)
    return envelope


def dtw_preprocess(pcg, pre=None):
    # pcg = _normalize(pcg) # Comment this line for Fucks sake
    if pre in ['hpf', 'env']:
        pcg = butter_highpass_filter(pcg, 25, 1000, 3)
        if pre == 'env':
            pcg = homomorphic_envelope(pcg, 1000, 20, 5)
    pcg = _normalize(pcg)
    return pcg


def mfcc_coefs(pcg, transitions, interval='RR'):
    fs = 1000

    pcg = _normalize(pcg)
    intervals = get_intervals(pcg, transitions, interval=interval, resize=ALENGTH[interval])
    signal = np.concatenate(intervals)

    win = ALENGTH[interval] / fs
    mel = mfcc(signal, samplerate=fs, winlen=win, winstep=win, numcep=13, nfilt=26, nfft=1024)
    return mel


# FEATURE FUNCTIONS

def interval_features(pcg, transitions, suf=None):
    def diff(t):
        return t[1] - t[0]
    IntRR   = diff(boundaries(transitions, 'RR'))
    IntS1   = diff(boundaries(transitions, 'S1'))
    IntSys  = diff(boundaries(transitions, 'Sys'))
    IntS2   = diff(boundaries(transitions, 'S2'))
    IntDia  = diff(boundaries(transitions, 'Dia'))
    m_RR,         sd_RR     = _mean_std(IntRR)
    mean_IntS1,   sd_IntS1  = _mean_std(IntS1)
    mean_IntSys,  sd_IntSys = _mean_std(IntSys)
    mean_IntS2,   sd_IntS2  = _mean_std(IntS2)
    mean_IntDia,  sd_IntDia = _mean_std(IntDia)
    m_Ratio_SysRR,   sd_Ratio_SysRR  = _mean_std(IntSys / IntRR)
    m_Ratio_DiaRR,   sd_Ratio_DiaRR  = _mean_std(IntDia / IntRR)
    m_Ratio_SysDia,  sd_Ratio_SysDia = _mean_std(IntSys / IntDia)

    P_S1  = np.array([np.sum(np.abs(pcg[i:j])) / (j - i) for i, j in zip(*boundaries(transitions, 'S1'))])
    P_Sys = np.array([np.sum(np.abs(pcg[i:j])) / (j - i) for i, j in zip(*boundaries(transitions, 'Sys'))])
    P_S2  = np.array([np.sum(np.abs(pcg[i:j])) / (j - i) for i, j in zip(*boundaries(transitions, 'S2'))])
    P_Dia = np.array([np.sum(np.abs(pcg[i:j])) / (j - i) for i, j in zip(*boundaries(transitions, 'Dia'))])

    def _soft_div(x, y):
        # Element wise division that ignores the dimensions where y == 0 or x > y
        d = x[y != 0.0] / y[y != 0.0]
        d = d[d < 1.0]
        return d
    m_Amp_SysS1, sd_Amp_SysS1 = _mean_std(_soft_div(P_Sys, P_S1))
    m_Amp_DiaS2, sd_Amp_DiaS2 = _mean_std(_soft_div(P_Dia, P_S2))
    features = np.array([m_RR, sd_RR, mean_IntS1, sd_IntS1, mean_IntS2, sd_IntS2, mean_IntSys, sd_IntSys, mean_IntDia, sd_IntDia, m_Ratio_SysRR,
                         sd_Ratio_SysRR, m_Ratio_DiaRR, sd_Ratio_DiaRR, m_Ratio_SysDia, sd_Ratio_SysDia, m_Amp_SysS1, sd_Amp_SysS1, m_Amp_DiaS2,
                         sd_Amp_DiaS2])
    features[np.isnan(features)] = 0
    return features

interval_features.names = ["m_RR", "sd_RR", "mean_IntS1", "sd_IntS1", "mean_IntS2", "sd_IntS2", "mean_IntSys", "sd_IntSys", "mean_IntDia",
                           "sd_IntDia", "m_Ratio_SysRR", "sd_Ratio_SysRR", "m_Ratio_DiaRR", "sd_Ratio_DiaRR", "m_Ratio_SysDia", "sd_Ratio_SysDia",
                           "m_Amp_SysS1", "sd_Amp_SysS1", "m_Amp_DiaS2", "sd_Amp_DiaS2"]


def wavelet_features(pcg, transitions, interval='RR', wavelet='rbio3.9', level=3, suf=None):

    level = int(level)
    pcg = _normalize(pcg)
    intervals = get_intervals(pcg, transitions, interval=interval)
    cAs, cD1s, cD2s, cD3s = [], [], [], []
    for x in intervals:
        x = np.concatenate((x, np.zeros(2**level - len(x) % 2**level)))
        cA, cD1, cD2, cD3 = pywt.wavedec(x, wavelet, level=level)
        cAs.append(np.linalg.norm(cA)**2)
        cD1s.append(np.linalg.norm(cD1)**2)
        cD2s.append(np.linalg.norm(cD2)**2)
        cD3s.append(np.linalg.norm(cD3)**2)
    m_cA, std_cA    = _mean_std(cAs)
    m_cD1, std_cD1  = _mean_std(cD1s)
    m_cD2, std_cD2  = _mean_std(cD2s)
    m_cD3, std_cD3  = _mean_std(cD3s)
    return [m_cA, std_cA, m_cD1, std_cD1, m_cD2, std_cD2, m_cD3, std_cD3]

wavelet_features.names = ["m_cA", "std_cA", "m_cD1", "std_cD1", "m_cD2", "std_cD2", "m_cD3", "std_cD3"]


def mfcc_features(pcg, transitions, interval='RR', pre=None, suf=None):

    pcg = dtw_preprocess(pcg, pre)

    mel = mfcc_coefs(pcg, transitions, interval=interval)

    mean_mel = np.mean(mel, axis=0)
    std_mel  = np.std(mel, axis=0)

    features = np.concatenate([mean_mel, std_mel])
    return features

mfcc_features.names = ["%s_M%02d" % (q, i) for q in ['mean', 'std'] for i in range(1, 14)]


def dmfcc_features(pcg, transitions, interval='RR', pre=None, suf=None):

    pcg = dtw_preprocess(pcg, pre)

    mel = mfcc_coefs(pcg, transitions, interval=interval)

    delta_mel = np.diff(mel, axis=0)
    mean_delta_mel = np.mean(delta_mel, axis=0)
    std_delta_mel  = np.std(delta_mel, axis=0)

    features = np.concatenate([mean_delta_mel, std_delta_mel])
    return features

dmfcc_features.names = ["%s_dM%02d" % (q, i) for q in ['mean', 'std'] for i in range(1, 14)]


def finite_matrix(m):
    N, M = m.shape
    x = np.sum(np.isnan(m), axis=1)
    y = np.sum(np.isnan(m), axis=0)
    i_x = (x < M - 1)
    i_y = (y < N - 1)
    m = m[i_x, :][:, i_y]
    m[np.isinf(m)] = 10 * np.max(m[np.isfinite(m)])
    return m


def DTW_features(pcg, transitions, interval='RR', constraint='sakoe_chiba', k=0.1, norm="resample", pre=None, downsample_rate=2, suf=None, sigma=None):

    k = float(k)
    if sigma is not None:
        sigma = float(sigma)

    pcg = dtw_preprocess(pcg, pre=pre)

    resize = ALENGTH[interval] if norm == 'resample' else None
    intervals = get_intervals(pcg, transitions, interval=interval, resize=resize)
    intervals = [resample(i, len(i) // downsample_rate) for i in intervals]

    if norm not in ['path', 'resample']:
        raise ValueError("Invalid normalization {0}".format(norm))

    path_norm = norm == 'path'
    dist_matrix = dtw_distances(intervals, n_jobs=-1, constraint=constraint, k=k, path_norm=path_norm, normalize=True)
    dist_matrix = finite_matrix(dist_matrix)

    medoid_index = np.argmin(np.sum(dist_matrix, axis=0))

    # Remove the infinite distances
    if sigma:
        dist_matrix = -affinity(dist_matrix, sigma)

    medoid_distances = dist_matrix[:, medoid_index]
    medoid_distances = medoid_distances[np.isfinite(medoid_distances)]
    m_MDTW, s_MDTW = _mean_std(medoid_distances)
    Q1_MDTW, Q2_MDTW, Q3_MDTW = np.percentile(medoid_distances, [25, 50, 75])

    contiguous_distances = np.array([dist_matrix[i, i + 1] for i in np.arange(len(dist_matrix) - 1)])
    contiguous_distances = contiguous_distances[np.isfinite(contiguous_distances)]
    Q1_CDTW, Q2_CDTW, Q3_CDTW = np.percentile(contiguous_distances, [25, 50, 75])

    features = np.array([m_MDTW, s_MDTW, Q1_MDTW, Q2_MDTW, Q3_MDTW, Q1_CDTW, Q2_CDTW, Q3_CDTW])
    return features

DTW_features.names = ["m_MDTW", "s_MDTW", "Q1_MDTW", "Q2_MDTW", "Q3_MDTW", "Q1_CDTW", "Q2_CDTW", "Q3_CDTW"]


def inter_DTW_features(pcg, transitions, interval='RR', templates='random_templates.mat', pre=None, constraint='sakoe_chiba', k=0.1, suf=None, sigma=None):
    k = float(k)
    if sigma is not None:
        sigma = float(sigma)

    templates = custom_loadmat(TEMPLATE_FOLDER + templates)[interval]

    inter_DTW_features.names = ["%s_d%02d" % (q, i) for q in ['mean', 'std'] for i, _ in enumerate(templates)]

    pcg = dtw_preprocess(pcg, pre=pre)

    intervals = get_intervals(pcg, transitions, interval=interval, resize=ALENGTH[interval] // interDTW_down)
    dist_matrix = dtw_distances(intervals, templates, n_jobs=-1, constraint=constraint, k=k)
    dist_matrix = finite_matrix(dist_matrix)
    if sigma:
        dist_matrix = -affinity(dist_matrix, sigma)

    mean_dtw = np.mean(dist_matrix, axis=0)
    std_dtw = np.mean(dist_matrix, axis=0)

    features = np.concatenate([mean_dtw, std_dtw])
    return features

inter_DTW_features.names = []


def cinter_DTW_features(pcg, transitions, templates='random_templates', pre=None, constraint='sakoe_chiba', k=0.1, suf=None, sigma=None):
    k = float(k)
    if sigma is not None:
        sigma = float(sigma)

    templates = custom_loadmat(TEMPLATE_FOLDER + templates)

    inter_DTW_features.names = ["%s_d%02d" % (q, i) for q in ['mean', 'std'] for i, _ in enumerate(templates)]

    pcg = dtw_preprocess(pcg, pre=pre)

    distances = []
    for interval in ['RR', 'S1', 'Sys', 'S2', 'Dia']:
        templates_i = templates[interval]

        intervals = get_intervals(pcg, transitions, interval=interval, resize=ALENGTH[interval] // interDTW_down)
        intervals = [resample(i, ALENGTH[interval] // interDTW_down) for i in intervals][:50]

        dist_matrix = dtw_distances(intervals, templates_i, n_jobs=-1, constraint=constraint, k=k)

        dist_matrix = finite_matrix(dist_matrix)
        if sigma:
            dist_matrix = -affinity(dist_matrix, sigma)
        distances.append(dist_matrix)

    RR_mean, S1_mean, Sys_mean, S2_mean, Dia_mean = [np.mean(d) for d in distances]
    RR_std, S1_std, Sys_std, S2_std, Dia_std = [np.std(d) for d in distances]

    features = [RR_mean, S1_mean, Sys_mean, S2_mean, Dia_mean, RR_std, S1_std, Sys_std, S2_std, Dia_std]
    return features

cinter_DTW_features.names = ['RR_mean', 'S1_mean', 'Sys_mean', 'S2_mean', 'Dia_mean', 'RR_std', 'S1_std', 'Sys_std', 'S2_std', 'Dia_std']

feature_dict = {
    "interval": interval_features,
    "dtw":      DTW_features,
    "wavelet":  wavelet_features,
    "mfcc":     mfcc_features,
    "dmfcc":    dmfcc_features,
    "interdtw": inter_DTW_features,
    "cinterdtw": cinter_DTW_features,
}
