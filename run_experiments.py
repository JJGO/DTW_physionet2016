#!/usr/bin/env python

"""Run experiments and

Usage:
  run_experiments.py [-m METHODSFILE] [-c CACHE_FILE] [--cv-test] [-o OUTFILE] [--ignore] EXP_FILE EXP_IDS
  run_experiments.py (-h | --help)

Arguments:
  EXP_FILE           Text file containing the experiments to run in the format
                        ID,CLF,FEATURES...
  EXP_IDS            Range of ids to use. It supports comma separator and ranges
                     Eg "101,102-104,109"
  CV_CACHE_FILE

Options:
  -m METHODSFILE            File containing the methods to use for the experiments
  -c CACHE_FILE             Use and update a cache for parameter finding
  --cv-test                 Perform a Cross Validation for the parameters applied to the test sets
  -o OUTFILE                Output file to append the results. Defaults to stdout
  --ignore                        Ignore the cache reads

"""
import sys
import os
import json
import logging

import numpy as np
import pandas as pd

from docopt import docopt
from tqdm import tqdm

from sklearn.grid_search import ParameterGrid
from sklearn.externals.joblib import Parallel, delayed
from sklearn.cross_validation import _fit_and_score
from sklearn.base import clone

from _defaults import *
from io_features import load_features
from classifiers import classifier_dict
from train import train_classifier
from predict import predict
from split import split_labels, cv_split_labels, train_test_unpack, METHODS

from utils.cinc_scores import cinc_confidence_interval, cinc_cv_scorer
from utils.misc import rangeexpand

HEADER = ['EXP_ID', 'TEST']
HEADER += ['SCORE_TRAIN', 'SCORE_TRAIN_L', 'SCORE_TRAIN_H']
HEADER += ['SCORE_TEST', 'SCORE_TEST_L', 'SCORE_TEST_H']
HEADER += ['SCORE_TEST_GRID']


def search_test_params(base_clf, cv_params, X, y, train, test, scoring):
    parameter_iterable = ParameterGrid(cv_params)
    grid_scores = Parallel(n_jobs=-1)(
        delayed(_fit_and_score)(clone(base_clf), X, y, scoring,
                                train, test, 0, parameters,
                                None, return_parameters=True)
            for parameters in parameter_iterable)
    # grid_scores = [_fit_and_score(clone(base_clf), X, y, scoring, train, test, 0, parameters, None, return_parameters=True) for parameters in parameter_iterable]
    grid_scores = sorted(grid_scores, key=lambda x: x[0], reverse=True)
    scores, _, _, parameters = grid_scores[0]
    return scores, parameters


def run_experiments(experiments, methods, cache={}, grid_test=False, ignore_cache=False):
    """
    TODO
    """

    def _check_cache(cache, mode, clf_name, feature_hash, method):
        if mode not in cache:
            cache[mode] = {}

        if clf_name not in cache[mode]:
            cache[mode][clf_name] = {}

        if feature_hash not in cache[mode][clf_name]:
            cache[mode][clf_name][feature_hash] = {}

        if method in cache[mode][clf_name][feature_hash]:
            return cache[mode][clf_name][feature_hash][method]

        return None

    def _most_common_dict(param_list):
        param_list = [frozenset(p.items()) for p in param_list]
        mode = max(set(param_list), key=param_list.count)
        return dict(mode)

    # Defaults
    labels = pd.read_csv(LABEL_FILE, index_col=0)
    with open(CV_PARAMS_FILE, 'r') as fp:
        cv_params_dict = json.load(fp)

    results = []

    for exp_id, clf_name, feature_files in tqdm(experiments, total=len(experiments)):
        feature_files = sorted(feature_files.split())
        features = load_features(feature_files)

        logger.info('Experiment {0} ({1} - {2})'.format(exp_id, clf_name, feature_files))

        for method in tqdm(methods, total=len(methods), leave=False):

            splits, iterations = METHODS[method]
            base_clf = classifier_dict[clf_name]
            clf = clone(base_clf)
            feature_hash = ":".join(feature_files)
            cv_param_grid = cv_params_dict[clf_name]

            logger.info('CV ParamGrid {0}'.format(cv_param_grid))

            logger.info('Method {0}'.format(method))

            # Check CV cache

            scores = []
            cv_params = []
            cv_cache = _check_cache(cache, 'cv', clf_name, feature_hash, method)

            cv_cache = not ignore_cache and cv_cache

            if cv_cache:
                cv_params = _check_cache(cache, 'cv', clf_name, feature_hash, method)
                logger.info('CV_CACHE Params: {0}'.format(cv_params))

            test_params = []
            test_cache = _check_cache(cache, 'test', clf_name, feature_hash, method)

            test_cache = not ignore_cache and test_cache

            if test_cache:
                test_params = _check_cache(cache, 'test', clf_name, feature_hash, method)
                logger.info('TEST_CACHE Params: {0}'.format(test_params))

            for j in tqdm(range(iterations), total=iterations, leave=False):
                parameters = _check_cache(cache, 'cv', clf_name, feature_hash, method)

                train_ix, test_ix = split_labels(labels, splits)
                X_train, X_test, y_train, y_test = train_test_unpack(features, labels, train_ix, test_ix)

                if isinstance(cv_params, dict):
                    clf.set_params(**cv_params)
                    clf, parameters, (t, r) = train_classifier(clf, X_train, y_train)
                else:
                    logger.info('It {0}: CV_START search'.format(j))

                    train_labels = labels.iloc[train_ix]
                    cv_iter = cv_split_labels(train_labels, method)

                    clf, parameters, (t, r) = train_classifier(clf, X_train, y_train, cv_param_grid, cv_iter)
                    cv_params.append(parameters)

                    logger.info('It {0}: CV_END search. Params: {1}'.format(j, parameters))

                logger.info('TRAIN_THR t:{0}, r:{1}'.format(t, r))
                y_pred = predict(clf, X_train, t, r)
                sc_train, (sc_train_low, sc_train_up) = cinc_confidence_interval(y_train, y_pred, t, r, 100)
                y_pred = predict(clf, X_test,  t, r)
                sc_test,  (sc_test_low, sc_test_up)   = cinc_confidence_interval(y_test,  y_pred, t, r, 100)

                if grid_test:
                    X, y = features.values, labels.values.squeeze()
                    if isinstance(test_params, dict):
                        sc_test_best, _, _ = _fit_and_score(clf, X, y, cinc_cv_scorer, train_ix, test_ix, 0, test_params, None)

                    else:
                        logger.info('It {0}: TEST_START search'.format(j))
                        cv_param_grid = cv_params_dict[clf_name]
                        sc_test_best, parameters = search_test_params(base_clf, cv_param_grid, X, y,
                                                                      train_ix, test_ix,
                                                                      scoring=cinc_cv_scorer)
                        test_params.append(parameters)

                        logger.info('It {0}: TEST_END search. Params: {1}'.format(j, parameters))
                else:
                    sc_test_best = np.nan

                scores.append(np.round([sc_train, sc_train_low, sc_train_up, sc_test, sc_test_low, sc_test_up, sc_test_best], 3))
                logger.info('It {0}/{1} - Scores {2}'.format(j, iterations - 1, scores[-1]))

            scores = np.mean(np.array(scores), axis=0)

            if not cv_cache:
                best_cv_params = _most_common_dict(cv_params)
                cache['cv'][clf_name][feature_hash][method] = best_cv_params
                logger.info('BEST_CV params: {0}'.format(best_cv_params))

            if not test_cache:
                best_test_params = _most_common_dict(test_params)
                cache['test'][clf_name][feature_hash][method] = best_test_params
                logger.info('BEST_TEST params: {0}'.format(best_test_params))

            cv_test_gap = np.round(scores[-1] - scores[-4], 3)
            scores = np.array([exp_id, method] + list(scores))
            results.append(scores)

            logger.info('Mean Scores: {0}'.format(scores))
            if grid_test:
                logger.info('CV_TEST gap: {0}'.format(cv_test_gap))

    df_results = pd.DataFrame(data=results, columns=HEADER)
    return df_results, cache


if __name__ == '__main__':

    arguments = docopt(__doc__)

    # Logging config
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler('logs/run_experiments_{0}.log'.format(arguments['EXP_IDS']))
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    logging.captureWarnings(True)

    logger.info('\n======== START ========\n\n')
    logger.info('{0}'.format(arguments))
    exp_df      = pd.read_csv(arguments['EXP_FILE'])
    exp_ids     = rangeexpand(arguments['EXP_IDS'])
    experiments = exp_df.loc[exp_df['ID'].isin(exp_ids)].values

    methods = METHODS.keys()
    if arguments['-m']:
        with open(arguments['-m'], 'r') as fp:
            methods = fp.read().strip().split('\n')

    cache_file = arguments['-c']
    if os.path.isfile(cache_file):
        with open(cache_file, 'r') as fp:
            cache = json.load(fp)
    else:
        cache = {}

    df_results, cache = run_experiments(experiments,
                                        methods,
                                        cache,
                                        grid_test=arguments['--cv-test'],
                                        ignore_cache=arguments['--ignore'])

    if cache_file:
        with open(cache_file, 'w') as fp:
            json.dump(cache, fp)

    if arguments['-o']:
        outfile = arguments['-o']
        if os.path.isfile(outfile):
            df_results = pd.read_csv(outfile).append(df_results, ignore_index=True)
        df_results.to_csv(outfile, index=False)

    logger.info('\n\n\n======== END ========\n\n')
