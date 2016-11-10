#!/usr/bin/env python3

# python3 predict.py <RECORD_NAME>

import sys
import json

import numpy as np

from sklearn.externals import joblib

from features import feature_dict
from utils.segmentation import get_transitions
from utils.misc import custom_loadmat

FEATURES = [
    ("mfcc",        {'interval': 'S1'}),
    ("mfcc",        {'interval': 'Sys'}),
    ("mfcc",        {'interval': 'S2'}),
    ("mfcc",        {'interval': 'Dia'}),
    ("dtw",         {'interval': 'RR', 'constraint': 'sakoe_chiba', 'k': 0.1, 'norm': 'resample', 'pre': 'env'}),
]

def predict(clf, X, t=0.0, r=0.0):
    try:
        y_pred = clf.decision_function(X)
        y_pred[y_pred > t + r] = 1
        y_pred[y_pred < t - r] = -1
        y_pred[np.abs(y_pred - t) <= r] = 0
    except:
        y_pred = clf.predict(X)
    return y_pred

if __name__ == '__main__':
    # Parse arguments
    if len(sys.argv) < 2:
        raise ValueError("Insufficient arguments")
    record_name = sys.argv[1]
    clf_folder = sys.argv[2]

    # Load matlab output and get transitions of states
    variables = custom_loadmat('segmentation/%s.mat' % record_name)
    pcg = variables['PCG_resampled']
    states = variables['assigned_states']
    transitions = get_transitions(states)

    # Load Classifier
    clf = joblib.load(clf_folder+"/classifier.pkl")
    with open(clf_folder + '/model.json', 'r') as fp:
        model = json.load(fp)
        t, r = model['thr__t'], model['thr__r']
    
    # Compute features for the record and predict the label
    X = np.concatenate([feature_dict[f](pcg, transitions, **kwargs) for f,kwargs in FEATURES])
    X = X.reshape(1,-1)
    # print(X)
    y_pred = predict(clf, X, t, r)

    # Append the result to the output file
    with open('answers.txt','a') as f:
        print('%s,%d' % (record_name, y_pred), file = f)
