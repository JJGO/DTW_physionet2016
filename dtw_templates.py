"""Script to compute and store PCG features

Usage:
  dtw_templates.py [-p PCGS] [-s STATES] [-n SIZE] [-q FRAC] [-b] (-o OUTFILE) random  SPLITS...
  dtw_templates.py [-p PCGS] [-s STATES] [-n SIZE] [-q FRAC] [-b] (-o OUTFILE) cluster MEDOIDS SIGMA SPLITS...
  dtw_templates.py [-p PCGS] [-s STATES] build TEMPLATES
  dtw_templates.py (-h | --help)

Arguments:
  SPLITS                Record files to use for the template search
  TEMPLATES             File containing the records and intervals to use for building
  MEDOIDS               Medoid folder to use for clustering
  SIGMA                 Sigma value for the affinity computation

Options:
  -h, --help            Show this screen.
  -b                    Build after getting the files
  -p PCGS               Specify .mat file containing the PCG recordings
                        Defaults to data/PCG_training.mat
  -s STATES             Specify .mat file containing the associated states
                        Defaults to data/states_training.mat
  -o OUTFILE            File to save the features to
  -n SIZE               Number of templates to get
  -q FRAC               For each split a FRAC*sqrt(size of split) is used

"""

from collections import defaultdict

import numpy as np
import pandas as pd
from docopt import docopt
from scipy.io import savemat
from tqdm import tqdm

from _defaults import *
from features import affinity, ALENGTH, interDTW_down
from utils.segmentation import get_intervals, get_transitions
from utils.misc import custom_loadmat, extract

from sklearn.cluster import AgglomerativeClustering, SpectralClustering


def random_choice(pcgs, states, records, size):
    t_records = list(np.random.choice(records, size, replace=False))
    indexes = []
    for r in t_records:
        pcg = pcgs[r]
        transitions = get_transitions(states[r])
        intervals = get_intervals(pcg, transitions)
        i = np.random.choice(range(len(intervals)))
        indexes.append(i)
    return t_records, indexes


def compile_random(splits, pcgs, states, size):
    t_records = []
    indexes = []
    s = size
    for split in tqdm(splits):
        records = extract(SPLITS_FOLDER + split)
        if isinstance(size, float):
            s = int(np.sqrt(len(records)) * size)
        r, i = random_choice(pcgs, states, records, s)
        t_records.extend(r)
        indexes.extend(i)
    return t_records, indexes


def cluster(aff_matrix, records, n_clusters, medoid_indexes):
    Cluster = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    labels = Cluster.fit_predict(aff_matrix)

    medoid_indexes = medoid_indexes.loc[records]

    t_records = []
    indexes = []

    for i in range(n_clusters):
        labels_i = np.where(labels == i)[0]
        sub_aff_matrix = aff_matrix[labels_i, :][:, labels_i]
        medoid_index   = np.argmax(np.prod(sub_aff_matrix, axis=0))
        absolute_index = labels_i[medoid_index]
        r = medoid_indexes.index[absolute_index]
        t_records.append(r)
        i = medoid_indexes.iloc[absolute_index].values[0]
        indexes.append(i)
    return t_records, indexes


def compile_cluster(splits, medoid_folder, sigma, size):
    t_records = []
    indexes = []
    s = size
    for split in tqdm(splits):
        records = extract(SPLITS_FOLDER + split)
        if isinstance(size, float):
            s = int(np.sqrt(len(records)) * size)
        matrix = np.load(medoid_folder + split + '.npy')
        aff_matrix = affinity(matrix, sigma)
        medoid_indexes = pd.read_csv(medoid_folder + 'index.csv', index_col=0, header=None)
        r, i = cluster(aff_matrix, records, s, medoid_indexes)
        t_records.extend(r)
        indexes.extend(i)
    return t_records, indexes


def build_templates(pcgs, states, records, indexes):
    templates = defaultdict(list)
    for r, i in tqdm(zip(records, indexes), total=len(records)):
        pcg = pcgs[r]
        for interval in ['RR', 'S1', 'Sys', 'S2', 'Dia']:
            transitions = get_transitions(states[r])
            inter = get_intervals(pcg, transitions, interval=interval, resize=ALENGTH[interval] // interDTW_down)[i]
            templates[interval].append(inter)
    return dict(templates)


def save_templates(records, indexes, filename, build=False, pcgs=None, states=None):
    df = pd.DataFrame(data=indexes, index=records)
    df.to_csv(TEMPLATE_FOLDER + filename + '.csv', header=None)

    if build:
        templates = build_templates(pcgs, states, records, indexes)
        filename = TEMPLATE_FOLDER + filename + '.mat'
        savemat(filename, templates)


if __name__ == '__main__':
    arguments = docopt(__doc__, version='1.0')

    if arguments['-p']:
        PCG_FILE = arguments['-p']
    pcgs = custom_loadmat(PCG_FILE)

    if arguments['-s']:
        STATE_FILE = arguments['-s']
    states = custom_loadmat(STATE_FILE)

    size = 3
    if arguments['-n']:
        size = int(arguments['-n'])
    elif arguments['-q']:
        size = float(arguments['-q'])

    filename = arguments['-o']

    if arguments['random']:
        records, indexes = compile_random(arguments['SPLITS'], pcgs, states, size)
        save_templates(records, indexes, filename, build=arguments['-b'], pcgs=pcgs, states=states)

    elif arguments['cluster']:
        medoid_folder = arguments['MEDOIDS']
        if not medoid_folder.endswith('/'):
            medoid_folder += '/'
        sigma = float(arguments['SIGMA'])
        records, indexes = compile_cluster(arguments['SPLITS'], medoid_folder, sigma, size)
        save_templates(records, indexes, filename, build=arguments['-b'], pcgs=pcgs, states=states)

    elif arguments['build']:
        template_file = arguments['TEMPLATES']
        df = pd.read_csv(template_file, header=None, index_col=0)
        records, indexes = df.index, df.values.squeeze()
        templates = build_templates(pcgs, states, records, indexes)

        if not filename:
            filename = template_file.strip('.csv') + '.mat'

        savemat(filename, templates)
