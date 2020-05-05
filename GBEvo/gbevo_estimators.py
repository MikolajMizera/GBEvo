# -*- coding: utf-8 -*-
"""
@author: Mikołaj Mizera
"""
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np

from .optimizers import CMAOpt

class GBEvoOptimizer(MetaEstimatorMixin, BaseEstimator):

    """Evolutionary optimization of parameter values for an estimator.

    The parameters of the estimator are optimized by cross-validated search
    using Covariance Matrix Adaptation Evolution Strategy.

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.

    param_ranges : list of dictionaries
        List of dictionaries with parameters names (string) as keys and ranges
        of valid hyperparameters as values. The type of a parameter is inffered
        from the first value in tuple.
        Example: {'n_estimators' : np.arange(10, 100, 10),
                  'max_bin' : 2**np.arange(3, 8),
                  'reg_lambda' : np.linspace(0, 5, 100),
                  'learning_rate': np.logspace(-5, -1, 100)}
        In the above example, `n_estimators` and `max_bin` parameters are of
        integer type, while `reg_lambda` and `learning_rate` are floats.

    feature_selection : bool, default: True
        Determines feature selection along with hyperparameters optimization.
        If True, then estimator will be retturned wrapped by FMEstimator
        object.

    metric : callable, default: None
        A callable which takes y_true and y_pred array as arguments and returns
        a score value (the higher value the better score).

    n_jobs : int or None, optional, default: -1
        Number of jobs to run in parallel.
        ``-1`` means using all processors.

    cv : int, cross-validation generator or an iterable, optional, default: 5
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - An iterable yielding (train, test) splits as arrays of indices.

    refit : boolean, string, or callable, default: True
        Refit an estimator using the best found parameters on the whole
        dataset.

    verbose : integer, default: 0
        Controls the verbosity: the higher, the more messages.

"""
    def __init__(self, estimator, param_ranges, feature_selection=True,
                 metric=None, n_jobs=None, cv=5, refit=True, verbose=0,
                 optimizer_settings={'population_size': 25,
                                     'n_epochs': 100,
                                     'n_populations': 5,
                                     'migration_frequency': 5}):

        self.estimator = estimator
        self.param_ranges = param_ranges
        self.feature_selection = feature_selection
        self.metric = metric
        self.n_jobs = n_jobs
        self.cv = cv
        self.refit = refit
        self.verbose = verbose
        self.optimizer_settings = optimizer_settings

    def fit(self, X, y, strat_vec=None):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.X_ = X
        self.y_ = y

        if type(self.cv) is int:
            cv_splits = KFold(self.cv).split(self.X_, self.y_)
        elif (type(self.cv) is StratifiedKFold) and not (strat_vec is None):
            cv_splits = self.cv.split(self.X_, strat_vec)
        else:
            cv_splits = self.cv.split(self.X_, self.y_)

        self.estimator = clone(self.estimator)

        es_optimizer = CMAOpt(self.estimator, self.param_ranges, 
                             **self.optimizer_settings, n_jobs=1, 
                             metric=self.metric, 
                             feature_selection=self.feature_selection,
                             verbose=self.verbose)
        es_optimizer.optimze(self.X_, self.y_, cv_splits)
        
        return self

    def predict(self, X):

        # Check if fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        return self.base_estimator.predict(X[:, self.var_mask])


class FMEstimator(MetaEstimatorMixin, BaseEstimator):

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
