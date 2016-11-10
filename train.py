#!/usr/bin/env python

"""Train a classifier given csv feature files

Usage:
  train.py [-r RECORDS] [--cv] [-p PARAMS] [-o FILE] [--thr THR] CLF LABELS FEATURES...
  train.py (-h | --help)

Arguments:
  CLF           Classifier Type to use [RSVM, LSVC, LSVM, RF]
  LABELS        CSV file  containing the labels for each records
  FEATURES      CSV files containing the features for each records

Options:
  --cv          Cross validate the classifier before training
  -p PARAMS     json file with the params to use
  -o FILE       Output file to save the classifier to
  -r RECORDS    File Specifying the record subset to train on
  -h, --help    Show this screen.
  --thr THR     Force to use arguments of threshold and radius

"""

import os
import json
import numpy as np
import pandas as pd

from docopt import docopt

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

from _defaults import CV_PARAMS_FILE
from io_features import load_features
from classifiers import classifier_dict
from predict import predict
from utils.cinc_scores import cinc_confusion_matrix, cinc_score, cinc_best_threshold, cinc_confidence_interval, cinc_cv_scorer


def train_classifier(clf, X_train, y_train, cv_params=None, cv_iter=5, verbose=False, thr=None):
    """
    Trains a given classifier based on a train set of features and labels.
    Supports preliminary cross validation.

    Args:
        clf : estimator object (sklearn.base.ClassifierMixin) to train
        X_train : numpy array with the training features
        y_train : numpy array with the training labels
        cv_params : dict(list) of parameters for the cross validation.
                    Defaults to None. If not provided, no CV is performed
        cv_iter : int, cross-validation generator or an iterable, optional
                  Determines the cross-validation splitting strategy. Possible inputs for cv are:
                  - None, to use the default 5-fold cross-validation,
                  - integer, to specify the number of folds.
                  - An iterable yielding train/test splits.
        verbose : bool that determines the verbosity of the training

    Returns:
        clf : The trained estimator object
        parameters : dict
                     best parameters after cross validation. Empty dict if cv_params is None
        (t, r) : (float, float) the best threshold and radius values for the unsure labels

    """

    if cv_params:
        print("Cross Validating the Model") if verbose else None
        gs_clf = GridSearchCV(clf, cv_params, n_jobs=-1, verbose=int(verbose), cv=cv_iter, scoring=cinc_cv_scorer)
        gs_clf = gs_clf.fit(X_train, y_train)
        clf = gs_clf.best_estimator_
        parameters = gs_clf.best_params_
        if verbose:
            print('Best parameters are:')
            for p, v in parameters.items():
                print(p, v)
    else:
        parameters = {}
        clf.fit(X_train, y_train)

    try:
        y_train_pred = clf.decision_function(X_train)
        if not thr:
            t, r = cinc_best_threshold(y_train, y_train_pred)
        else:
            t, r = thr
    except:
        y_train_pred = clf.predict(X_train)
        t, r = 0, 0

    if verbose:
        print('Threshold values t = %.02f, r = %.02f' % (t, r))
        score_train = cinc_confidence_interval(y_train, y_train_pred, t, r, N=100)
        cfm_train   = cinc_confusion_matrix(y_train, y_train_pred, t, r)
        print("=== TRAIN ==")
        print("CinC Score : %.4f (%.4f, %.4f)" % (score_train[0], score_train[1][0], score_train[1][1]))
        print("CinC Confusion Matrix \n %s" % cfm_train)
    return clf, parameters, (t, r)

if __name__ == '__main__':
    arguments = docopt(__doc__)
    print(arguments)
    print('Using features : %r' % arguments['FEATURES'])
    # Argument parsing
    clf       = classifier_dict[arguments['CLF']]
    features  = load_features(arguments['FEATURES'])
    labels    = pd.read_csv(arguments['LABELS'], index_col=0)
    cv_params = None

    if arguments['--cv']:
        cv_params = json.load(open(CV_PARAMS_FILE, 'r'))[arguments['CLF']]

    if arguments['-p']:
        with open(arguments['-p'], 'r') as fp:
            params = json.load(fp)
            clf.set_params(**params)

    if arguments['-r']:
        with open(arguments['-r'], 'r') as fp:
            records = fp.read().strip().split('\n')
        features = features.loc[records]
        labels = labels.loc[records]

    thr = None
    if arguments['--thr']:
        t, r = arguments['--thr'].split(',')
        thr = float(t), float(r)

    X_train, y_train = features.values, labels.values.squeeze()
    clf, parameters, (t, r) = train_classifier(clf, X_train, y_train, cv_params, verbose=True, thr=thr)

    if arguments['-o']:
        if arguments['-p'] and not arguments['--cv']:
            parameters = params
        model = {
            "clf_type": arguments['CLF'],
            "parameters": parameters,
            "thr__t": t,
            "thr__r": r,
        }
        if arguments['-r']:
            model['trainset'] = arguments['-r']

        clf_out = arguments['-o'] + '/'
        os.mkdir(clf_out)
        joblib.dump(clf, clf_out + 'classifier.pkl')

        with open(clf_out + 'model.json', 'w') as fp:
            json.dump(model, fp)

        with open('data/RECORDS_validation', 'r') as fp:
            val_rec = fp.read().strip().split('\n')
        features  = load_features(arguments['FEATURES'])
        X_test = features.loc[val_rec].values

        y_pred = predict(clf, X_test, t, r).astype(int)

        df_val = pd.DataFrame(data=y_pred, index=val_rec)
        with open(clf_out + 'answers.txt', 'w') as fp:
            df_val.to_csv(fp, header=None)
