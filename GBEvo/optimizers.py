# -*- coding: utf-8 -*-
"""
@author: Mikołaj Mizera
"""
import numpy as np
from sklearn.base import clone

import cma
from cma import restricted_gaussian_sampler as rgs
from dask.distributed import Client, LocalCluster

extend_settings = rgs.GaussVDSampler.extend_cma_options
AdaptSigma = cma.sigma_adaptation.CMAAdaptSigmaMedianImprovement


# This function is global for remote-execution-ready
def eval_split(estimator, X, y, split):
    """
    This fits a model using provided train-test split. The predictions and the
    fitted estimator are returned.
    """
    train_idx, test_idx = split
    X_train, y_train = X[train_idx], y[train_idx]
    X_test = X[test_idx]

    estimator = estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    return y_pred, estimator


class CMAOpt:

    def __init__(self, params_ranges, population_size, n_epochs, n_populations,
                 bounds=(0, 1), cv=5, n_jobs=1, scheduler='local',
                 feature_selection=False):

        self.params_ranges = params_ranges
        self.population_size = population_size
        self.n_epochs = n_epochs
        self.n_populations = n_populations
        self.bounds = bounds
        self.cv = cv
        self.feature_selection = feature_selection

        # Helper functions for serial or parallel execution
        if n_jobs == 1:
            def prep_f(x): return x
            def exec_f(func, *args): return func(*args)
            def postprocess_f(x): return x
        else:
            if scheduler == 'local':
                cluster = LocalCluster(processes=n_jobs)
                self.client = Client(cluster)
            prep_f = self.client.scatter
            exec_f = self.client.map
            def postprocess_f(x): return x.results()

        self.prep_f = prep_f
        self.exec_f = exec_f
        self.postprocess_f = postprocess_f

    def _strategy_init(self, individual_size):
        random_vec = np.random.uniform(0.75, 0.25, individual_size)
        return cma.CMAEvolutionStrategy(random_vec, 0.5, self.settings)

    def _migrate(self, populations):

        pops_idx = np.arange(len(populations))[::2]

        # Exchange 5% random individuals between consecutive populations
        n = int(np.round(0.05) * self.population_size)
        if n < 1:
            n = 1

        for pop_a_idx, pop_b_idx in zip(pops_idx, pops_idx-1):
            rand_a = np.random.randint(0, len(populations[pop_a_idx]), n)
            rand_b = np.random.randint(0, len(populations[pop_b_idx]), n)
            samples_a = populations[pop_a_idx][rand_a]
            samples_b = populations[pop_b_idx][rand_b]
            populations[pop_a_idx][rand_a] = samples_b
            populations[pop_b_idx][rand_b] = samples_a
        return populations

    def optimze(self, X, y, splits):

        self.X_ = X
        self.y_ = y
        self.splits_ = splits

        self.settings = extend_settings({'seed': np.random.randint(1, 4400),
                                         'popsize': self.population_size,
                                         'bounds': self.bounds,
                                         'verbose': -9,
                                         'AdaptSigma': AdaptSigma})
        n_features = self.X_.shape[1]

        self.strategies = [self._strategy_init(n_features)
                           for _ in range(self.n_populations)]

    def _scale_and_type(param, param_range):
        """Scales 0-1 bounded values from ES to the proper range of the
        parameters, than type varibles to the type of parametr."""

        name, range = param_range

        dtype = type(param[0])
        scaled_param = param*range[1]
        if dtype is int:
            scaled_param = dtype(np.abs(scaled_param))

        return (name, scaled_param)

    def _create_estimator(self, x):
        """Creates estimator from values optimized by ES algortithm."""

        scaled_params = [self._scale_and_type(param, range) for param, range
                         in zip(x, self.params_ranges.items())]
        estimator_params = {k: v for k, v in scaled_params}

        estimator = clone(self.base_estimator).set_params(**estimator_params)

        return estimator

    def _run_single_gen(self, migrate=False):
        """Runs single generation of ES (N - number of strategies, M -
        population size):
            1. generate populations,
            2. exchange individuals between populations,
            3. create and evaluate estimators based on individuals,
            4. update strategies with evaluated fitness functions."""

        populations = [np.array(strategy.ask())
                       for strategy in self.strategies]
        if migrate:
            populations = self._migrate(populations)

        estimators = [[self._create_estimator(x) for x in population]
                      for population in populations]

        # Prepare arguments of fitness function for (possible) remote execution
        # The population is evaluated in CV split-wise chunks
        estimators_args = [[self.prep_f(x) for x in pop] for pop in estimators]
        splits_args = [self.prep_f(split) for split in self.splits_]

        pop_results = []
        for estimators_pop in estimators_args:
            results = []
            for estimator in estimators_pop:
                part_ind_res = [self.exec_f(eval_split,
                                            estimator, self.X, self.y, split)
                                for split in splits_args]
                results.append(part_ind_res)
            pop_results.append(results)
