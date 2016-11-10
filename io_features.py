
"""Script to compute and store PCG features

Usage:
  io_features.py [-p PCGS] [-s STATES] [-r RECORDS] [-o OUTFILE] FEATURE [(ARG VALUE)]...
  io_features.py (-h | --help)

Arguments:
  FEATURE               Feature identifier : inter, DTW, wavelet
  ARG                   Name of the keyword argument to pass to the feature function
  VALUE                 Value of the keyword argument to pass to the feature function
  PARAMS                List of key-value params for the given feature function

Options:
  -h, --help            Show this screen.
  -p PCGS               Specify .mat file containing the PCG recordings
                        Defaults to data/PCG_training.mat
  -s STATES             Specify .mat file containing the associated states
                        Defaults to data/states_training.mat
  -r RECORDS            Text file with the record ids
                        Defaults to data/RECORDS
  -o OUTFILE            File to save the features to

"""


from docopt import docopt
from tqdm import tqdm

from features import *
from _defaults import *
from utils.misc import custom_loadmat

import logging


def load_features(features_files):
    """
    Auxiliary getter that stores a cache of the feature files
    fetched so far to optimize the IO operations

    Args:
        feature_files : str iterable
                        Iterable with the filenames to read
    Returns:
        features : pandas DataFrame
                   Concatenated dataframe with all the fetched features
    """
    for f in features_files:
        if f not in load_features.cache:
            load_features.cache[f] = pd.read_csv(f, index_col=0)

    features = pd.concat([load_features.cache[f] for f in features_files], join='inner', axis=1)
    return features

load_features.cache = {}


def save_features(feat_fun, pcgs, states, records, params):
    """
    Saves the specified features into an intermediate csv file
    Args:
      feat_fun : feature function to be used
      pcgs : dict of numpy arrays with the stored pcgs
      states : dict of numpy arrays with the stored states
      records : list of records to compute the feature to
    Returns:
      df_features : pandas DataFrame containing the features
    """
    feat_fun(pcgs[records[0]], get_transitions(states[records[0]]), **params)
    header = feat_fun.names
    data = np.zeros((len(records), len(header)))
    for i, r in tqdm(enumerate(records), total=len(records)):
        logging.info("Processing {0}/{1} - {2}".format(i + 1, len(records), r))
        pcg, transitions = pcgs[r], get_transitions(states[r])
        data[i, :] = feat_fun(pcg, transitions, **params)
        feats = ",".join([str(round(f, 3)) for f in data[i, :]])
        logging.info("{0},{1}".format(r, feats))

    df_features = pd.DataFrame(data=data, index=records, columns=header)
    return df_features

if __name__ == '__main__':
    arguments = docopt(__doc__, version='1.0')

    if arguments['-p']:
        PCG_FILE = arguments['-p']
    pcgs = custom_loadmat(PCG_FILE)

    if arguments['-s']:
        STATE_FILE = arguments['-s']
    states = custom_loadmat(STATE_FILE)

    feature = arguments['FEATURE']
    feat_fun = feature_dict[feature]
    feat_id = '{0}_{1}'.format(feature, '_'.join(arguments['VALUE']))

    logging.basicConfig(filename=LOG_FOLDER + "io_feature_%s.log" % feat_id, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logging.info(" ******* START *******")
    logging.info("Computing {0} feature".format(feature))

    filename = FEAT_FOLDER + feat_id.strip('_')

    records = RECORD_FILE
    if arguments['-r']:
        records = arguments['-r']

    with open(records, 'r') as f:
        logging.info('Using records in {0}'.format(records))
        records = f.read().strip().split('\n')

    params = dict(zip(arguments['ARG'], arguments['VALUE']))
    logging.info("Using params {0}".format(params))

    df_features = save_features(feat_fun, pcgs, states, records, params=params)

    filename += '.csv'
    if arguments['-o']:
        filename = arguments['-o']

    df_features.to_csv(filename)

    logging.info('Saved to %s' % filename)
    logging.info(" ******* END *******")
