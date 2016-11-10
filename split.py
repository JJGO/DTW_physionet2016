import numpy as np

from sklearn.cross_validation import train_test_split

from _defaults import SPLITS_FOLDER

NUM_ITER = 20

METHODS = {
    '.3bcde.3a.2f': ([('a', 0.0, 0.3), ('b', 0.7, 0.3), ('c', 0.7, 0.3), ('d', 0.7, 0.3), ('e', 0.7, 0.3), ('f', 0.0, 0.2)], NUM_ITER),
    '.3bcde!af':    ([('a', 0.0, 1.0), ('b', 0.7, 0.3), ('c', 0.7, 0.3), ('d', 0.7, 0.3), ('e', 0.7, 0.3), ('f', 0.0, 1.0)], NUM_ITER),
    '.3bcde':       ([                 ('b', 0.7, 0.3), ('c', 0.7, 0.3), ('d', 0.7, 0.3), ('e', 0.7, 0.3),                ], NUM_ITER),
    '.3abcdef':     ([('a', 0.7, 0.3), ('b', 0.7, 0.3), ('c', 0.7, 0.3), ('d', 0.7, 0.3), ('e', 0.7, 0.3), ('f', 0.7, 0.3)], NUM_ITER),
    'f.3bcdf.4a':   ([('a', 0.0, 0.4), ('b', 0.7, 0.3), ('c', 0.7, 0.3), ('d', 0.7, 0.3), ('e', 0.7, 0.3), ('f', 1.0, 0.0)], NUM_ITER),
    'a.3bcde!f':    ([('a', 1.0, 0.0), ('b', 0.7, 0.3), ('c', 0.7, 0.3), ('d', 0.7, 0.3), ('e', 0.7, 0.3), ('f', 0.0, 1.0)], NUM_ITER),
    '!a':           ([('a', 0.0, 1.0), ('b', 1.0, 0.0), ('c', 1.0, 0.0), ('d', 1.0, 0.0), ('e', 1.0, 0.0), ('f', 1.0, 0.0)], 1),
    '!b':           ([('a', 1.0, 0.0), ('b', 0.0, 1.0), ('c', 1.0, 0.0), ('d', 1.0, 0.0), ('e', 1.0, 0.0), ('f', 1.0, 0.0)], 1),
    '!c':           ([('a', 1.0, 0.0), ('b', 1.0, 0.0), ('c', 0.0, 1.0), ('d', 1.0, 0.0), ('e', 1.0, 0.0), ('f', 1.0, 0.0)], 1),
    '!d':           ([('a', 1.0, 0.0), ('b', 1.0, 0.0), ('c', 1.0, 0.0), ('d', 0.0, 1.0), ('e', 1.0, 0.0), ('f', 1.0, 0.0)], 1),
    '!e':           ([('a', 1.0, 0.0), ('b', 1.0, 0.0), ('c', 1.0, 0.0), ('d', 1.0, 0.0), ('e', 0.0, 1.0), ('f', 1.0, 0.0)], 1),
    '!f':           ([('a', 1.0, 0.0), ('b', 1.0, 0.0), ('c', 1.0, 0.0), ('d', 1.0, 0.0), ('e', 1.0, 0.0), ('f', 0.0, 1.0)], 1),



    # CV Methods
    '.3cde.3b':     ([                 ('b', 0.0, 0.3), ('c', 0.7, 0.3), ('d', 0.7, 0.3), ('e', 0.7, 0.3),                ], NUM_ITER),
    '.3bde.3c':     ([                 ('b', 0.7, 0.3), ('c', 0.0, 0.3), ('d', 0.7, 0.3), ('e', 0.7, 0.3),                ], NUM_ITER),
    '.3bce.3d':     ([                 ('b', 0.7, 0.3), ('c', 0.7, 0.3), ('d', 0.0, 0.3), ('e', 0.7, 0.3),                ], NUM_ITER),
    '.3bcd.3e':     ([                 ('b', 0.7, 0.3), ('c', 0.7, 0.3), ('d', 0.7, 0.3), ('e', 0.0, 0.3),                ], NUM_ITER),

    '.3cdef.3b':    ([                 ('b', 0.0, 0.3), ('c', 0.7, 0.3), ('d', 0.7, 0.3), ('e', 0.7, 0.3), ('f', 0.7, 0.3)], NUM_ITER),
    '.3bdef.3c':    ([                 ('b', 0.7, 0.3), ('c', 0.0, 0.3), ('d', 0.7, 0.3), ('e', 0.7, 0.3), ('f', 0.7, 0.3)], NUM_ITER),
    '.3bcef.3d':    ([                 ('b', 0.7, 0.3), ('c', 0.7, 0.3), ('d', 0.0, 0.3), ('e', 0.7, 0.3), ('f', 0.7, 0.3)], NUM_ITER),
    '.3bcdf.3e':    ([                 ('b', 0.7, 0.3), ('c', 0.7, 0.3), ('d', 0.7, 0.3), ('e', 0.0, 0.3), ('f', 0.7, 0.3)], NUM_ITER),
    '.3bcde.3f':    ([                 ('b', 0.7, 0.3), ('c', 0.7, 0.3), ('d', 0.7, 0.3), ('e', 0.7, 0.3), ('f', 0.0, 0.3)], NUM_ITER),

    '.3bcde.3a':    ([('a', 0.0, 0.3), ('b', 0.7, 0.3), ('c', 0.7, 0.3), ('d', 0.7, 0.3), ('e', 0.7, 0.3),                ], NUM_ITER),
    '.3acde.3b':    ([('a', 0.7, 0.3), ('b', 0.0, 0.3), ('c', 0.7, 0.3), ('d', 0.7, 0.3), ('e', 0.7, 0.3),                ], NUM_ITER),
    '.3abde.3c':    ([('a', 0.7, 0.3), ('b', 0.7, 0.3), ('c', 0.0, 0.3), ('d', 0.7, 0.3), ('e', 0.7, 0.3),                ], NUM_ITER),
    '.3abce.3d':    ([('a', 0.7, 0.3), ('b', 0.7, 0.3), ('c', 0.7, 0.3), ('d', 0.0, 0.3), ('e', 0.7, 0.3),                ], NUM_ITER),
    '.3abcd.3e':    ([('a', 0.7, 0.3), ('b', 0.7, 0.3), ('c', 0.7, 0.3), ('d', 0.7, 0.3), ('e', 0.0, 0.3),                ], NUM_ITER),


    # CLEAN TRAINING AND REGULAR TEST
    '.3bcde.3a.2f_n': ([                 ('b_c', 0.7, 0.3), ('c_c', 0.7, 0.3), ('d_c', 0.7, 0.3), ('e_c', 0.7, 0.3),
                      ('a',   0.0, 0.3), ('b_n', 0.0, 0.3), ('c_n', 0.0, 0.3), ('d_n', 0.0, 1.0), ('e_n', 0.0, 0.3), ('f',   0.0, 0.2)], NUM_ITER),
    '.3bcde!af_n':    ([                 ('b_c', 0.7, 0.3), ('c_c', 0.7, 0.3), ('d_c', 0.7, 0.3), ('e_c', 0.7, 0.3),
                      ('a',   0.0, 1.0), ('b_n', 0.0, 0.3), ('c_n', 0.0, 0.3), ('d_n', 0.0, 1.0), ('e_n', 0.0, 0.3), ('f',   0.0, 1.0)], NUM_ITER),
    '.3bcde_n':       ([                 ('b_c', 0.7, 0.3), ('c_c', 0.7, 0.3), ('d_c', 0.7, 0.3), ('e_c', 0.7, 0.3),
                                         ('b_n', 0.0, 0.3), ('c_n', 0.0, 0.3), ('d_n', 0.0, 1.0), ('e_n', 0.0, 0.3),                  ], NUM_ITER),
    '.3abcdef_n':   ([('a_c', 0.7, 0.3), ('b_c', 0.7, 0.3), ('c_c', 0.7, 0.3), ('d_c', 0.7, 0.3), ('e_c', 0.7, 0.3), ('f_c', 0.7, 0.3),
                      ('a_n', 0.0, 1.0), ('b_n', 0.0, 0.3), ('c_n', 0.0, 0.3), ('d_n', 0.0, 1.0), ('e_n', 0.0, 0.3), ('f_n', 0.0, 0.3)], NUM_ITER),

    '!a_n':         ([('a',   0.0, 1.0), ('b_c', 1.0, 0.0), ('c_c', 1.0, 0.0), ('d_c', 1.0, 0.0), ('e_c', 1.0, 0.0), ('f_c', 1.0, 0.0)], 1),
    '!b_n':         ([('a_c', 1.0, 0.0), ('b',   0.0, 1.0), ('c_c', 1.0, 0.0), ('d_c', 1.0, 0.0), ('e_c', 1.0, 0.0), ('f_c', 1.0, 0.0)], 1),
    '!c_n':         ([('a_c', 1.0, 0.0), ('b_c', 1.0, 0.0), ('c',   0.0, 1.0), ('d_c', 1.0, 0.0), ('e_c', 1.0, 0.0), ('f_c', 1.0, 0.0)], 1),
    '!d_n':         ([('a_c', 1.0, 0.0), ('b_c', 1.0, 0.0), ('c_c', 1.0, 0.0), ('d',   0.0, 1.0), ('e_c', 1.0, 0.0), ('f_c', 1.0, 0.0)], 1),
    '!e_n':         ([('a_c', 1.0, 0.0), ('b_c', 1.0, 0.0), ('c_c', 1.0, 0.0), ('d_c', 1.0, 0.0), ('e',   0.0, 1.0), ('f_c', 1.0, 0.0)], 1),
    '!f_n':         ([('a_c', 1.0, 0.0), ('b_c', 1.0, 0.0), ('c_c', 1.0, 0.0), ('d_c', 1.0, 0.0), ('e_c', 1.0, 0.0), ('f',   0.0, 1.0)], 1),

    # CV Methods
    '.3cde.3b_n':   ([                   ('b_c', 0.0, 0.3), ('c_c', 0.7, 0.3), ('d_c', 0.7, 0.3), ('e_c', 0.7, 0.3)       ], NUM_ITER),
    '.3bde.3c_n':   ([                   ('b_c', 0.7, 0.3), ('c_c', 0.0, 0.3), ('d_c', 0.7, 0.3), ('e_c', 0.7, 0.3)       ], NUM_ITER),
    '.3bce.3d_n':   ([                   ('b_c', 0.7, 0.3), ('c_c', 0.7, 0.3), ('d_c', 0.0, 0.3), ('e_c', 0.7, 0.3)       ], NUM_ITER),
    '.3bcd.3e_n':   ([                   ('b_c', 0.7, 0.3), ('c_c', 0.7, 0.3), ('d_c', 0.7, 0.3), ('e_c', 0.0, 0.3)       ], NUM_ITER),
}

CV_METHODS = {
    '.3bcde.3a.2f':     ['.3cde.3b', '.3bde.3c', '.3bce.3d', '.3bcd.3e'] * 2,
    '.3bcde!af':        ['.3cde.3b', '.3bde.3c', '.3bce.3d', '.3bcd.3e'] * 2,
    '.3bcde':           ['.3bcde'] * 5,
    '.3abcdef':         ['.3abcdef'] * 5,
    '!a':               [      '!b', '!c', '!d', '!e', '!f'],
    '!b':               ['!a',       '!c', '!d', '!e', '!f'],
    '!c':               ['!a', '!b',       '!d', '!e', '!f'],
    '!d':               ['!a', '!b', '!c',       '!e', '!f'],
    '!e':               ['!a', '!b', '!c', '!d',       '!f'],
    '!f':               ['!a', '!b', '!c', '!d', '!e',     ],
    'f.3bcdf.4a':       ['.3cdef.3b', '.3bdef.3c', '.3bcef.3d', '.3bcdf.3e', '.3bcde.3f'] * 2,
    'a.3bcde!f':        ['.3bcde.3a', '.3acde.3b', '.3abde.3c', '.3abce.3d', '.3abcd.3e'] * 2,

    '.3bcde.3a.2f_n':   ['.3cde.3b_n', '.3bde.3c_n', '.3bce.3d_n', '.3bcd.3e_n'] * 2,
    '.3bcde!af_n':      ['.3cde.3b', '.3bde.3c', '.3bce.3d', '.3bcd.3e'] * 2,
    '.3bcde_n':         ['.3bcde'] * 5,
    '.3abcdef_n':       ['.3abcdef'] * 5,  # TODO define for just clean
    '!a_n':             [        '!b_n', '!c_n', '!d_n', '!e_n', '!f_n'],
    '!b_n':             ['!a_n',         '!c_n', '!d_n', '!e_n', '!f_n'],
    '!c_n':             ['!a_n', '!b_n',         '!d_n', '!e_n', '!f_n'],
    '!d_n':             ['!a_n', '!b_n', '!c_n',         '!e_n', '!f_n'],
    '!e_n':             ['!a_n', '!b_n', '!c_n', '!d_n',         '!f_n'],
    '!f_n':             ['!a_n', '!b_n', '!c_n', '!d_n', '!e_n',       ],
}


def _load_records(file):
    if file not in _load_records.cache:
        with open(SPLITS_FOLDER + file, 'r') as fp:
            _load_records.cache[file] = fp.read().strip().split('\n')
    return _load_records.cache[file]

_load_records.cache = {}


def split_labels(labels, splits):
    """
    Make X and y training and test splits based on a given scheme.
    Uses a collection of splits and returns the union of all the splits.

    Args:
        labels   : pandas DataFrame containing the labels
        splits   : itereable( ( str, float, float ) )
                   Collection of population splits. Has three arguments
                    file : str of the filename cotaining the records to split
                    train_size : int or float
                                 float - fraction to use for the training set
                                 int - absolute
                    test_size : float representing the fraction to use for the test set
    Returns:
        train_ix : numpy ndarray with the training features
        test_ix  : numpy ndarray with the test features
    """
    train_ix, test_ix = [], []
    for file, train_size, test_size in splits:
        records = _load_records(file)
        records = [r for r in records if r in labels.index]
        indexes = np.array([labels.index.get_loc(r) for r in records if r in labels.index])
        train_ix_, test_ix_ = [], []
        y = labels.loc[records].values
        if train_size == 1.0:
            train_ix_ = indexes
        elif test_size == 1.0:
            test_ix_ = indexes
        elif train_size == 0.0:
            _, test_ix_, _, _ = train_test_split(indexes, y, test_size=test_size, stratify=y)
        elif test_size == 0.0:
            _, train_ix_, _, _ = train_test_split(indexes, y, test_size=train_size, stratify=y)
        else:
            train_ix_, test_ix_, _, _ = train_test_split(indexes, y, test_size=test_size, train_size=train_size, stratify=y)
        train_ix.extend(train_ix_)
        test_ix.extend(test_ix_)

    train_ix, test_ix = np.array(train_ix), np.array(test_ix)
    return train_ix, test_ix


def cv_split_labels(labels, method):
    cv_splits = []
    for cv_method in CV_METHODS[method]:
        splits, _ = METHODS[cv_method]
        cv_splits.append(split_labels(labels, splits))
    return cv_splits


def train_test_unpack(features, labels, train_ix, test_ix):
    X_train = features.iloc[train_ix].values
    X_test  = features.iloc[test_ix].values
    y_train = labels.iloc[train_ix].values.squeeze()
    y_test  = labels.iloc[test_ix].values.squeeze()
    return X_train, X_test, y_train, y_test
