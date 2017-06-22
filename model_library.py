# -*- coding: utf8
# Author: David C. Lambert [dcl -at- panix -dot- com]
# Copyright(c) 2013
# License: Simple BSD
"""Utility module for building model library"""

from __future__ import print_function

import numpy as np

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.grid_search import ParameterGrid
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.kernel_approximation import Nystroem
from xgboost.sklearn import XGBClassifier as XGBoostingClassifier
from xgboost.sklearn import XGBRegressor as XGBoostingRegressor

# generic model builder
def build_models(model_class, param_grid):
    print('Building %s models' % str(model_class).split('.')[-1][:-2])

    return [model_class(**p) for p in ParameterGrid(param_grid)]


def build_randomForestClassifiers(random_state=None):
    param_grid = {
        'n_estimators': [20, 50, 100, 200, 500],
        'criterion':  ['gini', 'entropy'],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'max_depth': [1, 2, 4, 7, 10],
        #'min_density': [0.25, 0.5, 0.75, 1.0],
        'random_state': [random_state],
        'n_jobs': [-1],
    }

    return build_models(RandomForestClassifier, param_grid)

def build_randomForestRegressors(random_state=None):
    param_grid = {
        'n_estimators': [20, 50, 100, 200, 500],
        'criterion':  ['mse'],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'max_depth': [1, 2, 4, 7, 10],
        #'min_density': [0.25, 0.5, 0.75, 1.0],
        'random_state': [random_state],
        'n_jobs': [-1],
    }

    return build_models(RandomForestRegressor, param_grid)


def build_xgBoostingRegressors(random_state=None):
    import multiprocessing as mp

    if mp.cpu_count() <= 4:
        n_thread = 2
    else:
        n_thread = 4

    param_grid = {
        'max_depth': [1, 2, 5, 10],
        'n_estimators': [50, 100, 200, 500],
        'subsample': np.linspace(0.2, 1.0, 5),
        # 'max_features': np.linspace(0.2, 1.0, 5),
        'max_depth': [1, 2, 4, 7, 10],
        'min_child_weight': [1, 2],
        'nthread': [n_thread],
        'seed': [random_state],
        'max_delta_step': [1],
    }

    return build_models(XGBoostingRegressor, param_grid)


def build_gradientBoostingClassifiers(random_state=None):
    param_grid = {
        'max_depth': [1, 2, 5, 10],
        'n_estimators': [10, 20, 50, 100],
        'subsample': np.linspace(0.2, 1.0, 5),
        'max_features': np.linspace(0.2, 1.0, 5),
    }

    return build_models(GradientBoostingClassifier, param_grid)


def build_xgBoostingClassifiers(random_state=None):
    import multiprocessing as mp
    from math import ceil
    if mp.cpu_count() <= 4:
        n_thread = 2
    else:
        n_thread = 4
    param_grid = {
        'max_depth': [1, 2, 5, 10],
        'n_estimators': range(25, 175, 25),
        'subsample': np.linspace(0.2, 1.0, 5),
        # 'max_depth': [1,3,5,7,10],
        'min_child_weight': range(1, 5, 2),  # np.linspace(1,3,5),
        'scale_pos_weight': [1],
        #'reg_alpha' : [1e-5,1e-2,0.1,1],
        'colsample_bytree': [0.8],
        'max_delta_step': [1],
        'seed': [random_state],
        'nthread': [n_thread],  # [int(ceil(mp.cpu_count() / 2))],
    }

    return build_models(XGBoostingClassifier, param_grid)


def build_gradientBoostingRegressors(random_state=None):
    param_grid = {
        'max_depth': [1, 2, 5, 10],
        'n_estimators': [10, 20, 50, 100],
        'subsample': np.linspace(0.2, 1.0, 5),
        'max_features': np.linspace(0.2, 1.0, 5),
    }

    return build_models(GradientBoostingRegressor, param_grid)


def build_sgdClassifiers(random_state=None):
    param_grid = {
        'loss': ['log', 'modified_huber'],
        'penalty': ['elasticnet'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'optimal'],
        'n_iter': [2, 5, 10],
        'eta0': [0.001, 0.01, 0.1],
        'l1_ratio': np.linspace(0.0, 1.0, 3),
    }

    return build_models(SGDClassifier, param_grid)


def build_decisionTreeClassifiers(random_state=None):
    rs = check_random_state(random_state)

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'max_depth': [None, 1, 2, 5, 10],
        'min_samples_split': [1, 2, 5, 10],
        'random_state': [rs.random_integers(100000) for i in xrange(3)],
    }

    return build_models(DecisionTreeClassifier, param_grid)

def build_decisionTreeRegressors(random_state=None):
    rs = check_random_state(random_state)

    param_grid = {
        'criterion': ['mse'],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'max_depth': [None, 1, 2, 5, 10],
        'min_samples_split': [1, 2, 5, 10],
        'random_state': [rs.random_integers(100000) for i in xrange(3)],
    }

    return build_models(DecisionTreeRegressor, param_grid)

def build_extraTreesClassifiers(random_state=None):
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'n_estimators': [5, 10, 20],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'max_depth': [None, 1, 2, 5, 10],
        'min_samples_split': [2, 5, 10],
        'random_state': [random_state],
        'n_jobs': [-1],
    }

    return build_models(ExtraTreesClassifier, param_grid)

def build_extraTreesRegressors(random_state=None):
    param_grid = {
        'criterion': ['mse'],
        'n_estimators': [5, 10, 20],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'max_depth': [None, 1, 2, 5, 10],
        'min_samples_split': [2, 5, 10],
        'random_state': [random_state],
        'n_jobs': [-1],
    }

    return build_models(ExtraTreesRegressor, param_grid)

def build_svcs(random_state=None):
    print('Building SVM models')

    Cs = np.logspace(-7, 2, 10)
    gammas = np.logspace(-6, 2, 9, base=2)
    coef0s = [-1.0, 0.0, 1.0]

    models = []

    for C in Cs:
        models.append(SVC(kernel='linear', C=C, probability=True,
                          cache_size=1000))

    for C in Cs:
        for coef0 in coef0s:
            models.append(SVC(kernel='sigmoid', C=C, coef0=coef0,
                              probability=True, cache_size=1000))

    for C in Cs:
        for gamma in gammas:
            models.append(SVC(kernel='rbf', C=C, gamma=gamma,
                              cache_size=1000, probability=True))

    param_grid = {
        'kernel': ['poly'],
        'C': Cs,
        'gamma': gammas,
        'degree': [2],
        'coef0': coef0s,
        'probability': [True],
        'cache_size': [1000],
    }

    for params in ParameterGrid(param_grid):
        models.append(SVC(**params))

    return models


def build_kernPipelines(random_state=None):
    print('Building Kernel Approximation Pipelines')

    param_grid = {
        'n_components': xrange(5, 105, 5),
        'gamma': np.logspace(-6, 2, 9, base=2)
    }

    models = []

    for params in ParameterGrid(param_grid):
        nys = Nystroem(**params)
        lr = LogisticRegression()
        models.append(Pipeline([('nys', nys), ('lr', lr)]))

    return models


def build_kmeansPipelines(random_state=None):
    print('Building KMeans-Logistic Regression Pipelines')

    param_grid = {
        'n_clusters': xrange(5, 205, 5),
        'init': ['k-means++', 'random'],
        'n_init': [1, 2, 5, 10],
        'random_state': [random_state],
    }

    models = []

    for params in ParameterGrid(param_grid):
        km = KMeans(**params)
        lr = LogisticRegression()
        models.append(Pipeline([('km', km), ('lr', lr)]))

    return models


models_dict = {
    'svc': build_svcs,
    'sgd': build_sgdClassifiers,
    'gbc': build_gradientBoostingClassifiers,
    'xgbc': build_xgBoostingClassifiers,
    'xgbr': build_xgBoostingRegressors,
    'dtree': build_decisionTreeClassifiers,
    'forest': build_randomForestClassifiers,
    'extra': build_extraTreesClassifiers,
    'kmp': build_kmeansPipelines,
    'kernp': build_kernPipelines,
    'gb_reg': build_gradientBoostingRegressors,
    'dtree_reg': build_decisionTreeRegressors,
    'forest_reg': build_randomForestRegressors,
    'extra_reg': build_extraTreesRegressors
}


def build_model_library(model_types=['dtree'], random_seed=None):
    models = []
    for m in model_types:
        models.extend(models_dict[m](random_state=random_seed))
    return models
