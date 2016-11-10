"""Script to preprocess PCGs

Usage:
  pcgutil.py [-p PCGS] [-r RECORDS] (--hpf | --env) [-o OUTFILE] filter
  pcgutil.py [-p PCGS] [-s STATES] [-r RECORDS] [-o OUTFILE] medoids
  pcgutil.py dtwmatrix MEDOIDS [SPLITS]...
  pcgutil.py (-h | --help)

Arguments:
  filter                Filter and store PCG file
  medoids               Compute and store DTW medoids
  dtwmatrix             Computes and store the DTW distance matrix for the given records medoids
  MEDOIDS               Medoids to use for the matix computation
  SPLITS                Record files to use for the matrix computation


Options:
  -h, --help            Show this screen.
  -p PCGS               Specify .mat file containing the PCG recordings
                        Defaults to data/PCG_training.mat
  -s STATES             Specify .mat file containing the associated states
                        Defaults to data/states_training.mat
  -r RECORDS            Text file with the record ids
                        Defaults to data/RECORDS
  -o OUTFILE            File to save the features to
  --hpf                 Apply high pass filter
  --env                 Use homomorphic envelope
"""

import os
import time

import numpy as np
from scipy.io import savemat
from tqdm import tqdm
from docopt import docopt

from _defaults import *
from features import ALENGTH, dtw_preprocess, interDTW_down

from utils.misc import custom_loadmat, extract
from utils.dtwpy import dtw_medoid, dtw_distances
from utils.segmentation import get_intervals, get_transitions

dtw_params = {
    'n_jobs': -1,
    'constraint': 'sakoe_chiba',
    'k': 0.1,
    'path_norm': False,
    'normalize': True
}

if __name__ == '__main__':
    arguments = docopt(__doc__, version='1.0')

    if arguments['-p']:
        PCG_FILE = arguments['-p']
    pcgs = custom_loadmat(PCG_FILE)

    if arguments['-s']:
        STATE_FILE = arguments['-s']
    states = custom_loadmat(STATE_FILE)

    if arguments['-r']:
        RECORD_FILE = arguments['-r']
    records = extract(RECORD_FILE)

    filename = PCG_FILE
    if arguments['filter']:

        if arguments['--hpf']:
            filename = filename.replace('PCG', 'fPCG')
            pre = 'hpf'

        if arguments['--env']:
            filename = filename.replace('PCG', 'ePCG')
            pre = 'env'

        fpcgs = {r: dtw_preprocess(pcgs[r], pre=pre) for r in tqdm(records)}

        if arguments['-o']:
            filename = arguments['-o']

        savemat(filename, fpcgs)
        print('Saved to {0}'.format(filename))

    elif arguments['medoids']:
        filename = filename.replace('PCG', 'medoids')

        inters = {r: get_intervals(pcgs[r], get_transitions(states[r]), interval='RR',
                  resize=ALENGTH['RR'] // interDTW_down) for r in tqdm(records)}

        medoid_indexes = [dtw_medoid(inters[r], **dtw_params) for r in tqdm(records)]

        medoids = {r: inters[r][m]  for r, m in zip(records, medoid_indexes)}

        if arguments['-o']:
            filename = arguments['-o']

        savemat(filename, medoids)
        print('Saved to {0}'.format(filename))

        directory = filename.replace('.mat', '/')

        os.makedirs(directory, exist_ok=True)

        with open(directory + 'index.csv', 'w') as fp:
            for r, m in zip(records, medoid_indexes):
                print("{0},{1}".format(r, m), file=fp)

    elif arguments['dtwmatrix']:

        medoids = custom_loadmat(arguments['MEDOIDS'])
        directory = arguments['MEDOIDS'].replace('.mat', '/')

        os.makedirs(directory, exist_ok=True)

        for split in arguments['SPLITS']:
            start = time.time()
            records = extract(SPLITS_FOLDER + split)
            n = len(records)
            print("{0} : Processing {1} records ({2})".format(split, n, (n * (n - 1)) // 2))
            medoids_i = [medoids[r] for r in records]
            dist_matrix = dtw_distances(medoids_i, **dtw_params)
            np.save(directory + split, dist_matrix)
            print("{0} : Saved to {1}{0} - T {2:.3f}s".format(split, directory, time.time() - start))
