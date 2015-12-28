#!/usr/bin/env python
# Author: David C. Lambert [dcl -at- panix -dot- com]
# Copyright(c) 2013
# License: Simple BSD
"""
==========================================================
Prediction utility for trained EnsembleSelectionClassifier
==========================================================

Get predictions from trained EnsembleSelectionClassifier given
svm format data file.

Can output predicted classes or probabilities from the full
ensemble or just the best model.

Expects to find a trained ensemble in the sqlite db specified.

usage: ensemble_predict.py [-h] [-s {best,ens}] [-p] db_file data_file

Get EnsembleSelectionClassifier or EnsembleSelectionRegressor predictions

positional arguments:
  db_file        sqlite db file containing model
  data_file      testing data in svm format

optional arguments:
  -h, --help     show this help message and exit
  -s {best,ens}  choose source of prediction ["best", "ens"]
  -p             predict probabilities
  -T --type {'Regression','Classification'
                 type of method to implement (regression or classification)
"""
from __future__ import print_function

import numpy as np

from argparse import ArgumentParser

from sklearn.datasets import load_svmlight_file

from ensemble import EnsembleSelectionClassifier, EnsembleSelectionRegressor


def parse_args():
    desc = 'Get EnsembleSelectionClassifier predictions'
    parser = ArgumentParser(description=desc)

    method_choices = ['Regression', 'Classification']
    dflt_fmt = '(default: %(default)s)'

    help_fmt = 'method of estimation %s' % dflt_fmt
    parser.add_argument('-T', dest='meth', nargs='+', choices=method_choices, help=help_fmt, default=['Regression'])

    parser.add_argument('db_file', help='sqlite db file containing model')
    parser.add_argument('data_file', help='testing data in svm format')

    help_fmt = 'choose source of prediction ["best", "ens"] (default "ens")'
    parser.add_argument('-s', dest='pred_src',
                        choices=['best', 'ens'],
                        help=help_fmt, default='ens')

    parser.add_argument('-p', dest='return_probs',
                        action='store_true', default=False,
                        help='predict probabilities')

    return parser.parse_args()


def predictMan(res):
    X, _ = load_svmlight_file(res.data_file)
    X = X.toarray()

    if res.meth[0] == 'Classification':
        ens = EnsembleSelectionClassifier(db_file=res.db_file, models=None)
    elif res.meth[0] == 'Regression':
        ens = EnsembleSelectionRegressor(db_file=res.db_file, models=None)
    else:
        msg = "Invalid method passed (-T does not conform to ['Regression','Classification']"
        raise ValueError(msg)

    if (res.pred_src == 'best'):
        preds = ens.best_model_predict_proba(X)
    else:
        preds = ens.predict_proba(X)

    if res.meth[0] == 'Classification':
        if (not res.return_probs):
            preds = np.argmax(preds, axis=1)

    for p in preds:
        if (res.return_probs):
            mesg = " ".join(["%.5f" % v for v in p])
        else:
            mesg = p
        print
        mesg
    return mesg




if (__name__ == '__main__'):
    res = parse_args()
    predictMan(res)
    '''X, _ = load_svmlight_file(res.data_file)
    X = X.toarray()

    if res.meth[0] == 'Classification':
        ens = EnsembleSelectionClassifier(db_file=res.db_file, models=None)
    elif res.meth[0] == 'Regression':
        ens = EnsembleSelectionRegressor(db_file=res.db_file, models=None)
    else:
        msg = "Invalid method passed (-T does not conform to ['Regression','Classification']"
        raise ValueError(msg)



    if (res.pred_src == 'best'):
        preds = ens.best_model_predict_proba(X)
    else:
        preds = ens.predict_proba(X)

    if res.meth[0] == 'Classification':
        if (not res.return_probs):
            preds = np.argmax(preds, axis=1)

    for p in preds:
        if (res.return_probs):
            mesg = " ".join(["%.5f" % v for v in p])
        else:
            mesg = p

        print(mesg)
'''
