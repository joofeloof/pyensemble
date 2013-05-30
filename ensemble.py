# -*- coding: utf8
# Author: David C. Lambert [dcl -at- panix -dot- com]
# Copyright(c) 2013
# License: Simple BSD
"""
The :mod:`ensemble` module implements the ensemble selection
technique of Caruana et al [1][2].

Currently supports accuracy, rmse, and mean cross entropy scores
for hillclimbing.  Based on numpy, scipy, sklearn and sqlite.

Work in progress.

References
----------
.. [1] Caruana, et al, "Ensemble Selection from Libraries of Rich Models",
       Proceedings of the 21st International Conference on Machine Learning
       (ICML `04).
.. [2] Caruana, et al, "Getting the Most Out of Ensemble Selection",
       Proceedings of the 6th International Conference on Data Mining
       (ICDM `06).
"""
import os
import sys
import sqlite3
import numpy as np
from math import sqrt
from cPickle import loads, dumps
from collections import Counter

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer


class EnsembleSelectionClassifier(BaseEstimator, ClassifierMixin):
    """Caruana-style ensemble selection [1][2]

    Parameters:
    -----------
    `db_name` : string
        Name of file for sqlite db backing store

    `models` : list or None
        List of classifiers following sklearn fit/predict API, if None
        fitted models are loaded from the specified database

    `n_best` : int (default: 5)
        Number of top models in initial ensemble

    `n_folds` : int (default: 3)
        Number of internal cross-validation folds

    `bag_fraction` : float (default: 0.25)
        Fraction of (post-pruning) models to randomly select for each bag

    `prune_fraction` : float (default: 0.8)
        Fraction of worst models to prune before ensemble selection

    `score_metric` : string (default: 'accuracy')
        Score metric to use when hillclimbing.  Must be one of
        'accuracy', 'xentropy', 'rmse', 'f1'.

    `epsilon` : float (default: 0.01)
        Minimum score improvement to add model to ensemble

    `max_models` : int (default: 50)

    `verbose` : boolean (default: False)
        Turn on verbose messages

    `random_state`  : int, RandomState instance or None (default=None)
        Control the pseudo random number generator used to select
        candidates for each bag

    References
    ----------
    .. [1] Caruana, et al, "Ensemble Selection from Libraries of Rich Models",
           Proceedings of the 21st International Conference on Machine Learning
           (ICML `04).
    .. [2] Caruana, et al, "Getting the Most Out of Ensemble Selection",
           Proceedings of the 6th International Conference on Data Mining
           (ICDM `06).
    """

    # db setup script
    _createTablesScript = """
        create table models (
            model_idx      integer UNIQUE NOT NULL,
            pickled_model  blob NOT NULL
        );

        create table fitted_models (
            model_idx      integer NOT NULL,
            fold_idx       integer NOT NULL,
            pickled_model  blob NOT NULL
        );

        create table model_scores (
            model_idx      integer UNIQUE NOT NULL,
            score          real NOT NULL,
            probs          blob NOT NULL
        );

        create table ensemble (
            model_idx      integer NOT NULL,
            weight         integer NOT NULL
        );
    """

    def _rmse(y, y_bin, probs):
        """return 1-rmse since we're maximizing the score for hillclimbing"""

        return 1.0 - sqrt(mean_squared_error(y_bin, probs))

    def _mxe(y, y_bin, probs):
        """return negative mean cross entropy since we're maximizing the score
        for hillclimbing"""

        # clip away from extremes to avoid under/overflows
        probs = np.clip(probs, 0.00001, 0.99999, probs)

        #return -np.mean(np.sum(y_bin * np.log(probs), axis=1)
        #+ np.sum((1-y_bin)*np.log(1.0-probs), axis=1))
        return -np.mean(np.sum(y_bin * np.log(probs), axis=1))

    def _acc(y, y_bin, probs):
        """return accuracy score"""

        return accuracy_score(y, np.argmax(probs, axis=1))

    def _f1(y, y_bin, probs):
        """return f1 score"""

        return f1_score(y, np.argmax(probs, axis=1))

    _metrics = {
        'f1': _f1,
        'rmse': _rmse,
        'accuracy': _acc,
        'xentropy': _mxe,
    }

    def __init__(self, db_name=None,
                 models=None, n_best=5, n_folds=3,
                 n_bags=20, bag_fraction=0.25,
                 prune_fraction=0.8,
                 score_metric='accuracy',
                 epsilon=0.01, max_models=50,
                 verbose=False,
                 random_state=None):

        self.db_name = db_name
        self.models = models
        self.n_best = n_best
        self.n_bags = n_bags
        self.n_folds = n_folds
        self.bag_fraction = bag_fraction
        self.prune_fraction = prune_fraction
        self.score_metric = score_metric
        self.epsilon = epsilon
        self.max_models = max_models
        self.verbose = verbose
        self.random_state = random_state

        self._check_params()

        self._folds = None
        self._n_models = 0
        self._n_classes = 0
        self._metric = None
        self._ensemble = Counter()
        self._model_scores = []
        self._scored_models = []
        self._fitted_models = []

        self._init_db(models)

    def _check_params(self):
        if (not self.db_name):
            msg = "db_name parameter is required"
            raise ValueError(msg)

        if (self.epsilon < 0.0):
            msg = "epsilon must be >= 0.0"
            raise ValueError(msg)

        if (self.score_metric not in self._metrics.keys()):
            msg = "score_metric not in 'accuracy', 'xentropy', 'rmse', 'f1'"
            raise ValueError(msg)

        if (self.n_best < 1):
            msg = "n_best must be >= 1"
            raise ValueError(msg)

        if (self.max_models < self.n_best):
            msg = "max_models must be >= n_best"
            raise ValueError(msg)

    def _init_db(self, models):
        """Initialize database"""

        if (models):
            # nuke old database
            try:
                os.remove(self.db_name)
            except OSError, e:
                pass

        db_conn = sqlite3.connect(self.db_name)
        with db_conn:
            db_conn.execute("pragma journal_mode = off")

        if (models):
            # build database
            with db_conn:
                db_conn.executescript(self._createTablesScript)

            # populate model table
            insert_stmt = """insert into models (model_idx, pickled_model)
                             values (?, ?)"""
            with db_conn:
                vals = ((i, buffer(dumps(m))) for i, m in enumerate(models))
                db_conn.executemany(insert_stmt, vals)
                create_stmt = "create index models_index on models (model_idx)"
                db_conn.execute(create_stmt)

            self._n_models = len(models)

        else:
            curs = db_conn.cursor()
            curs.execute("select count(*) from models")
            self._n_models = curs.fetchone()[0]

            curs.execute("select model_idx, weight from ensemble")
            for k, v in curs.fetchall():
                self._ensemble[k] = v

            # clumsy hack to get n_classes
            curs.execute("select probs from model_scores limit 1")
            r = curs.fetchone()
            probs = loads(str(r[0]))
            self._n_classes = probs.shape[1]

        db_conn.close()

    def fit(self, X, y):
        """Perform model fitting and ensemble building"""

        self.fit_models(X, y)
        self.build_ensemble(X, y)

    def fit_models(self, X, y):
        """Perform internal cross-validation fit"""

        if (self.verbose):
            sys.stderr.write('\nfitting models\n')

        self._folds = list(StratifiedKFold(y, n_folds=self.n_folds))

        select_stmt = "select pickled_model from models where model_idx = ?"
        insert_stmt = """insert into fitted_models
                             (model_idx, fold_idx, pickled_model)
                         values (?,?,?)"""

        db_conn = sqlite3.connect(self.db_name)
        curs = db_conn.cursor()

        for model_idx in xrange(self._n_models):

            curs.execute(select_stmt, [model_idx])
            pickled_model = curs.fetchone()[0]
            model = loads(str(pickled_model))
            model_folds = []

            for fold_idx, fold in enumerate(self._folds):
                train_inds, _ = fold
                model.fit(X[train_inds], y[train_inds])

                pickled_model = buffer(dumps(model))
                model_folds.append((model_idx, fold_idx, pickled_model))

            with db_conn:
                db_conn.executemany(insert_stmt, model_folds)

            if (self.verbose):
                if ((model_idx + 1) % 50 == 0):
                    sys.stderr.write('%d\n' % (model_idx + 1))
                else:
                    sys.stderr.write('.')

        if (self.verbose):
            sys.stderr.write('\n')

        with db_conn:
            stmt = """create index fitted_models_index
                      on fitted_models (model_idx, fold_idx)"""

            db_conn.execute(stmt)

        db_conn.close()

    def _score_models(self, db_conn, X, y, y_bin):
        """Get cross-validated test scores for each model"""

        self._metric = self._metrics[self.score_metric]

        if (self.verbose):
            sys.stderr.write('\nscoring models\n')

        insert_stmt = """insert into model_scores (model_idx, score, probs)
                         values (?,?,?)"""

        select_stmt = """select pickled_model
                         from fitted_models
                         where model_idx = ? and fold_idx = ?"""

        # nuke existing scores
        with db_conn:
            stmt = """drop index if exists model_scores_index;
                      delete from model_scores;"""
            db_conn.executescript(stmt)

        curs = db_conn.cursor()

        # build probs array using the test sets for each internal CV fold
        for model_idx in xrange(self._n_models):
            probs = np.zeros((len(X), self._n_classes))

            for fold_idx, fold in enumerate(self._folds):
                _, test_inds = fold

                curs.execute(select_stmt, [model_idx, fold_idx])
                res = curs.fetchone()
                model = loads(str(res[0]))

                probs[test_inds] = model.predict_proba(X[test_inds])

            score = self._metric(y, y_bin, probs)

            with db_conn:
                vals = (model_idx, score, buffer(dumps(probs)))
                db_conn.execute(insert_stmt, vals)

            if (self.verbose):
                if ((model_idx + 1) % 50 == 0):
                    sys.stderr.write('%d\n' % (model_idx + 1))
                else:
                    sys.stderr.write('.')

        if (self.verbose):
            sys.stderr.write('\n')

        with db_conn:
            stmt = """create index model_scores_index
                      on model_scores (model_idx)"""
            db_conn.execute(stmt)

    def _get_ensemble_score(self, db_conn, ensemble, y, y_bin):
        """Get score for model ensemble"""

        n_models = float(sum(ensemble.values()))
        ensemble_probs = np.zeros((len(y), self._n_classes))

        curs = db_conn.cursor()
        select_stmt = """select model_idx, probs
                         from model_scores
                         where model_idx in %s"""

        curs.execute(select_stmt % str(tuple(ensemble)))

        for row in curs.fetchall():
            model_idx, probs = row
            probs = loads(str(probs))
            weight = ensemble[model_idx]
            ensemble_probs += probs * weight/n_models

        score = self._metric(y, y_bin, ensemble_probs)
        return score

    def _ensemble_from_candidates(self, db_conn, candidates, X, y, y_bin):
        """Build an ensemble from a list of candidate models"""

        ensemble = Counter(candidates[:self.n_best])
        ensemble_score = self._get_ensemble_score(db_conn, ensemble, y, y_bin)
        if (self.verbose):
            ensemble_count = sum(ensemble.values())
            sys.stderr.write('%d/%.3f ' % (ensemble_count, ensemble_score))

        last_ensemble_score = -100.0
        while(ensemble_count < self.max_models):
            new_scores = []
            for model_idx in candidates:
                ens = ensemble + Counter({model_idx: 1})
                score = self._get_ensemble_score(db_conn, ens, y, y_bin)
                new_scores.append({'new_idx': model_idx, 'score': score})

            new_scores.sort(key=lambda x: x['score'], reverse=True)

            last_ensemble_score = ensemble_score
            ensemble_score = new_scores[0]['score']

            # if score improvement is less than epsilon,
            # don't add the new model and stop
            score_diff = ensemble_score - last_ensemble_score
            if (score_diff < self.epsilon):
                break

            ensemble.update({new_scores[0]['new_idx']: 1})

            ensemble_count = sum(ensemble.values())
            if (self.verbose):
                if ((ensemble_count - self.n_best) % 8 == 0):
                    sys.stderr.write("\n         ")
                msg = '%d/%.3f ' % (ensemble_count, ensemble_score)
                sys.stderr.write(msg)

        if (self.verbose):
            sys.stderr.write('\n')

        return ensemble

    def _get_best_model(self, curs):
        """perform query for best scoring model"""

        select_stmt = """select model_idx, pickled_model
                         from models
                         where model_idx =
                               (select model_idx
                                from model_scores
                                order by score desc
                                limit 1)"""
        curs.execute(select_stmt)
        row = curs.fetchone()

        return row[0], loads(str(row[1]))

    def best_model(self):
        """Returns best model found after CV scoring"""

        db_conn = sqlite3.connect(self.db_name)
        _, model = self._get_best_model(db_conn.cursor())
        db_conn.close()
        return model

    def _print_best_results(self, curs, best_model_score):
        """Show best model and score"""

        sys.stderr.write('Best model CV score: %.5f\n' % best_model_score)

        _, best_model = self._get_best_model(curs)
        sys.stderr.write('Best model: %s\n\n' % repr(best_model))

    def build_ensemble(self, X, y, rescore=True):
        """Generate bagged ensemble"""

        self._n_classes = len(np.unique(y))

        db_conn = sqlite3.connect(self.db_name)
        curs = db_conn.cursor()

        if (self._n_classes > 2):
            y_bin = LabelBinarizer().fit_transform(y)
        else:
            y_bin = np.column_stack((1-y, y))

        # get CV scores for fitted models
        if (rescore):
            self._score_models(db_conn, X, y, y_bin)

        # get number of best models to take
        n_models = int(self._n_models * (1.0 - self.prune_fraction))
        bag_size = int(self.bag_fraction * n_models)
        if (self.verbose):
            sys.stderr.write('%d models left after pruning\n' % n_models)
            sys.stderr.write('leaving %d candidates per bag\n\n' % bag_size)

        # get indices and scores from DB
        select_stmt = """select model_idx, score
                         from model_scores
                         order by score desc
                         limit %d"""
        curs.execute(select_stmt % n_models)
        ranked_model_scores = [(r[0], r[1]) for r in curs.fetchall()]

        # print best performing model results
        best_model_score = ranked_model_scores[0][1]
        if (self.verbose):
            self._print_best_results(curs, best_model_score)
            sys.stderr.write("Ensemble scores for each bag (size/score):\n")

        ensembles = []

        rs = check_random_state(self.random_state)
        for i in xrange(self.n_bags):
            # get bag_size elements at random
            cand_indices = rs.permutation(n_models)[:bag_size]
            #cand_indices = sorted(rs.permutation(n_models)[:bag_size])

            # sort by rank
            candidates = [ranked_model_scores[ci][0] for ci in cand_indices]

            if (self.verbose):
                sys.stderr.write('Bag %02d): ' % (i+1))

            # build an ensemble with current candidates
            ensemble = self._ensemble_from_candidates(db_conn,
                                                      candidates,
                                                      X, y, y_bin)
            ensembles.append(ensemble)

        # combine ensembles from each bag
        for e in ensembles:
            self._ensemble += e

        # push to DB
        insert_stmt = "insert into ensemble(model_idx, weight) values (?, ?)"
        with db_conn:
            val_gen = ((mi, w) for mi, w in self._ensemble.most_common())
            db_conn.executemany(insert_stmt, val_gen)

        if (self.verbose):
            score = self._get_ensemble_score(db_conn, self._ensemble, y, y_bin)
            fmt = "\nFinal ensemble (%d components) CV score: %.5f\n\n"

            sys.stderr.write(fmt % (sum(self._ensemble.values()), score))

        db_conn.close()

    def _model_predict_proba(self, X, model_idx=0):
        """Get probability predictions for a model given its index"""

        db_conn = sqlite3.connect(self.db_name)
        curs = db_conn.cursor()
        select_stmt = """select pickled_model
                         from fitted_models
                         where model_idx = ? and fold_idx = ?"""

        probs = np.zeros((len(X), self._n_classes))

        for fold_idx in xrange(self.n_folds):
            curs.execute(select_stmt, [model_idx, fold_idx])

            res = curs.fetchone()
            model = loads(str(res[0]))

            probs += model.predict_proba(X)/float(self.n_folds)

        db_conn.close()

        return probs

    def best_model_predict_proba(self, X):
        """Probability estimates for all classes (ordered by class label)
        using best model"""

        db_conn = sqlite3.connect(self.db_name)
        best_model_idx, _ = self._get_best_model(db_conn.cursor())
        db_conn.close()

        return self._model_predict_proba(X, best_model_idx)

    def best_model_predict(self, X):
        """Predict class labels for samples in X using best model"""
        return np.argmax(self.best_model_predict_proba(X), axis=1)

    def predict_proba(self, X):
        """Probability estimates for all classes (ordered by class label)"""

        n_models = float(sum(self._ensemble.values()))

        probs = np.zeros((len(X), self._n_classes))

        for model_idx, weight in self._ensemble.items():
            probs += self._model_predict_proba(X, model_idx) * weight/n_models

        return probs

    def predict(self, X):
        """Predict class labels for samples in X."""
        return np.argmax(self.predict_proba(X), axis=1)