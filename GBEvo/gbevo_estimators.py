# -*- coding: utf-8 -*-
"""
@author: Mikołaj Mizera
"""
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np


class GBEvo(MetaEstimatorMixin, BaseEstimator):

    """Evolutionary optimization of parameter values for an estimator.

    The parameters of the estimator are optimized by cross-validated search
    using Covariance Matrix Adaptation Evolution Strategy.

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_ranges : list of dictionaries
        List of dictionaries with parameters names (string) as keys and tuples
        with lower bound and upper bound (lb, ub) of valid hyperparameters
        as values.

    feature_selection : bool, default: True
        Determines feature selection along with hyperparameters optimization.

    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's score method is used.

    n_jobs : int or None, optional, default: -1
        Number of jobs to run in parallel.
        ``-1`` means using all processors.

    cv : int, cross-validation generator or an iterable, optional, default: 5
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    refit : boolean, string, or callable, default: True
        Refit an estimator using the best found parameters on the whole
        dataset.

    verbose : integer, default: 0
        Controls the verbosity: the higher, the more messages.

"""
    def __init__(self, estimator, params_ranges, feature_selection=True,
                 scoring=None, n_jobs=None, cv=5, refit=True, verbose=0):

        self.estimator = estimator
        self.params_ranges = params_ranges
        self.feature_selection = feature_selection
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = cv
        self.refit = refit
        self.verbose = verbose

    def fit(self, X, y, strat_vec=None):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.X_ = X
        self.y_ = y
        
        if type(self.cv) is int:
            cv_splits = KFold(self.cv).split(self.X_, self.y_)
        elif (type(self.cv) is StratifiedKFold) and not (strat_vec is None):
            cv_splits = self.cv.split(self.X_, strat_vec)

        self.estimator = clone(self.estimator)

        # Code for evolution here

        return self

    def predict(self, X):

        # Check if fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        return self.base_estimator.predict(X[:, self.var_mask])


class GradientBoostingFM(MetaEstimatorMixin, BaseEstimator):

    def __init__(self, estimator, params={}, feature_mask=None):
        self.estimator = estimator
        self.params = params
        self.feature_mask = feature_mask

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.X_ = X
        self.y_ = y

        self.estimator = clone(self.estimator).set_params(**self.params)

        # Do not mask any features if the mask is not specified
        if self.feature_mask is None:
            self.feature_mask = np.ones((self.X_.shape[1]), dtype=bool)

        # Fit the estimator with feature mask applied
        self.estimator.fit(self.X_[:, self.feature_mask], self.y_)

        # Return the estimator
        return self

    def predict(self, X):

        # Check if fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        return self.estimator.predict(X[:, self.var_mask])

    def get_params(self, deep=True):
        return {'estimator': self.estimator,
                'params_ranges': self.params_ranges,
                'feature_mask': self.feature_mask}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
