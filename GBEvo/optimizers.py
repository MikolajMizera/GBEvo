# -*- coding: utf-8 -*-
"""
@author: Mikołaj Mizera
"""
import numpy as np

import cma
from cma import restricted_gaussian_sampler as rgs
from dask.distributed import Client, LocalCluster

extend_settings = rgs.GaussVDSampler.extend_cma_options
AdaptSigma = cma.sigma_adaptation.CMAAdaptSigmaMedianImprovement


def eval_split(x, X, y, split):
    pass


class CMAOpt:

    def __init__(self, population_size, n_epochs, n_populations,
                 bounds=(0, 1), cv=5, n_jobs=1, scheduler='local'):
        self.population_size = population_size
        self.n_epochs = n_epochs
        self.n_populations = n_populations
        self.bounds = bounds
        self.cv = cv

        # Serial or parallel execution
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
        # Exchange 4 random individuals between consecutive populations
        for pop_a_idx, pop_b_idx in zip(pops_idx, pops_idx-1):
            rand_a = np.random.randint(0, len(populations[pop_a_idx]), 4)
            rand_b = np.random.randint(0, len(populations[pop_b_idx]), 4)
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

    def _run_single_gen(self, migrate=False):
        populations = [np.array(strategy.ask())
                       for strategy in self.strategies]
        if migrate:
            populations = self._migrate(populations)

        # Prepare arguments of fitness function
        populations_args = [[self.prep_f(x) for x in pop]
                            for pop in populations]
        splits_args = [self.prep_f(split) for split in self.splits_]

        pop_results = []
        for population in populations_args:
            results = []
            for x in population:
                part_ind_res = [self.exec_f(eval_split,
                                            x, self.X, self.y, split)
                                for split in splits_args]

                results.append(part_ind_res)
            pop_results.append(results)

