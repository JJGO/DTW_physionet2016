import numpy as np

# Calculate score based on the CinC requirement

import logging


def cinc_confusion_matrix(y_true, y_cont, t=0.0, r=0.0):
    y_pred = np.copy(y_cont)
    y_pred[y_pred > t + r] = 1
    y_pred[y_pred < t - r] = -1
    y_pred[np.abs(y_pred - t) <= r] = 0
    cfm = np.zeros((2, 3))
    for i, true_l in enumerate([-1, 1]):
        for j, pred_l in enumerate([-1, 0, 1]):
            cfm[i][j] = np.sum(np.logical_and(y_true == true_l, y_pred == pred_l))
    return cfm


def cinc_score(y_true, y_pred, t=0.0, r=0.0, weight=0.5):
    cfm = cinc_confusion_matrix(y_true, y_pred, t, r)
    sensitity = (cfm[1, 2] + weight * cfm[1, 1]) / np.sum(cfm[1])
    specificity = (cfm[0, 0] + weight * cfm[0, 1]) / np.sum(cfm[0])
    return (specificity + sensitity) / 2


def cinc_best_threshold(y_true, y_pred, t_range=np.linspace(-1, 1, 41), r_range=np.linspace(0, 0.2, 20)):
    scores = [(cinc_score(y_true, y_pred, t, r), (t, r)) for t in t_range for r in r_range]
    best = sorted(scores, reverse=True)[0]
    return best[1]


def cinc_confidence_interval(y_true, y_pred, t=0.0, r=0.0, N=100, ci=(2.5, 97.5)):
    score = cinc_score(y_true, y_pred, t, r)
    performances = np.zeros(N, dtype='float64')

    for i in range(N):
        sample = np.random.choice(len(y_true), len(y_true))
        performances[i] = cinc_score(y_true[sample], y_pred[sample], t, r)

    lower = np.percentile(performances, ci[0])
    upper = np.percentile(performances, ci[1])

    return score, (lower, upper)


def cinc_cv_scorer(clf, X, y_true):
    try:
        y_cont = clf.decision_function(X)
        t, r = cinc_best_threshold(y_true, y_cont)
        score = cinc_score(y_true, y_cont, t, r)
    except:
        y_pred = clf.predict(X)
        score = cinc_score(y_true, y_pred)
    return score


def cinc_confusion_matrix_new(y_true, y_pred, quality):
    cfm = np.zeros((4, 3))

    for i, true_l in enumerate([1, -1]):
        for j, qual in enumerate([1, 0]):
            for k, pred_l in enumerate([1, 0, -1]):
                cfm[2 * i + j][k] = np.sum(np.logical_and(np.logical_and(y_true == true_l, quality == qual), y_pred == pred_l))
    return cfm


def cinc_score_new(y_true, y_pred, quality):

    cfm = cinc_confusion_matrix_new(y_true, y_pred, quality)

    wa1 = np.sum(cfm[0]) / np.sum(cfm[:2])
    wa2 = np.sum(cfm[1]) / np.sum(cfm[:2])
    wn1 = np.sum(cfm[2]) / np.sum(cfm[2:])
    wn2 = np.sum(cfm[3]) / np.sum(cfm[2:])

    Se = (wa1 * cfm[0, 0]) / np.sum(cfm[0]) + (wa2 * (cfm[1, 0] + cfm[1, 1])) / np.sum(cfm[1])
    Sp = (wn1 * cfm[2, 2]) / np.sum(cfm[2]) + (wa2 * (cfm[3, 2] + cfm[3, 1])) / np.sum(cfm[3])

    return (Se + Sp) / 2
