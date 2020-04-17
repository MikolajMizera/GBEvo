# -*- coding: utf-8 -*-
"""
@author: Mikołaj Mizera
"""
import numpy as np

import cma
from cma import restricted_gaussian_sampler as rgs

extend_settings = rgs.GaussVDSampler.extend_cma_options
AdaptSigma = cma.sigma_adaptation.CMAAdaptSigmaMedianImprovement


class CMAOpt:

    def __init__(self, population_size, n_epochs, n_populations,
                 bounds=(0, 1)):
        self.settings = extend_settings({'seed': np.random.randint(1, 4400),
                                         'popsize': population_size,
                                         'bounds': bounds,
                                         'verbose': -9,
                                         'AdaptSigma': AdaptSigma})
        self.n_epochs = n_epochs
        self.n_populations = n_populations

    def strategy_init(self, individual_size):
        random_vec = np.random.uniform(0.75, 0.25, individual_size)
        return cma.CMAEvolutionStrategy(random_vec, 0.5, self.settings)

    def optimze(self, X, y):

        self.X_ = X
        self.y_ = y

        n_features = self.X_.shape[1]

        self.strategies = [self.strategy_init(n_features)
                           for _ in range(self.n_pops)]

    def _run_single_gen(self, migrate=False):
        pass
