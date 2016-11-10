"""Build list of experiments

Usage:
  build_experiments.py FEAT_FILE CLF_FILE EXP_IDS [-o OUTFILE]
  build_experiments.py (-h | --help)

Arguments:
  FEAT_FILE          CSV file containing the features to use in the format
                        ID,FEATURES...
  CLF_FILE           CSV file cotaining the Classifiers ids
                        ID,CLF
  EXP_IDS            Range of ids to use. It supports comma separator and ranges
                     Eg "101,102-104,109"

Options:
  -o OUTFILE                Output file to append the results. Defaults to stdout

"""
import sys
import os

import pandas as pd

from docopt import docopt

from utils.misc import rangeexpand


def get_features(df, i):
    feat_names = df.loc[i].values[0].split(' ')
    feat_files = []
    for f in feat_names:
        if f.startswith('&'):
            r = rangeexpand(f[1:], sep=';')
            l = [get_features(df, i) for i in r]
            feat_files.extend([item for sublist in l for item in sublist])
        else:
            feat_files.append(f)
    unique_feat_files = []
    for i in feat_files:
        if i not in unique_feat_files:
            unique_feat_files.append(i)
    return unique_feat_files

if __name__ == '__main__':
    arguments = docopt(__doc__)
    HEADER = ['ID', 'CLF', 'FEATURES']
    experiments = []

    dtype = {'ID': int}

    df_feat = pd.read_csv(arguments['FEAT_FILE'], index_col=0, dtype=dtype)
    df_clf  = pd.read_csv(arguments['CLF_FILE'],  index_col=0, dtype=dtype)

    for i in rangeexpand(arguments['EXP_IDS']):
        clf_id = (i // 1000) * 1000
        if clf_id not in df_clf.index:
            continue
        clf = df_clf.loc[clf_id].values[0]
        feat_id = i % 1000
        if feat_id not in df_feat.index:
            continue
        feat_files = get_features(df_feat, feat_id)
        feat_files = " ".join(feat_files)
        experiments.append([i, clf, feat_files])

    df_exp = pd.DataFrame(data=experiments, columns=HEADER)

    if arguments['-o']:
        outfile = arguments['-o']
        if os.path.isfile(outfile):
            df_exp = pd.read_csv(outfile).append(df_exp, ignore_index=True)
        df_exp.to_csv(outfile, index=False)
    else:
        df_exp.to_csv(sys.stdout, index=False)
