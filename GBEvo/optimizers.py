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
    """Base optimizer for hyperparameters of estimaotr compatible with 
    scikit-learn stack.

    The parameters of the estimator are optimized by cross-validated search
    using Covariance Matrix Adaptation Evolution Strategy.

    Parameters
    ----------
    base_estimator : estimator object.
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
        
    population_size : int
        Number of individuals in each population.
    
    n_epochs: int
        Epochs of evolutions to evaluate.
    
    n_populations: int
        Number of populations (evolutional strategies).
        
    migration_frequency: int, optional, default: 0
        Will randomly migrate individuals between adjacent populations every
        `migration_frequency` epochs.

    n_jobs : int or None, optional, default: -1
        Number of jobs to run in parallel.
        ``-1`` means using all processors.
        
    scheduler : int or None, optional, default: -1
        If not None, and n_job > 1, will use schdeuler for parallel execution.
        Avaiable options:
            `local` : starts Dask LocalCluster with n_jobs
            `slurm` : currently not avaiable

    metric : callable, default: None
        A callable which takes y_true and y_pred array as arguments and returns
        a score value (the higher value the better score).
    
    feature_selection : bool, default: True
        Determines feature selection along with hyperparameters optimization.

    verbose : integer, default: 0
        Controls the verbosity: the higher, the more messages.

"""
    
    def __init__(self, base_estimator, param_ranges, population_size, 
                 n_epochs, n_populations, migration_frequency=0, 
                 n_jobs=1, scheduler='local', metric=None, 
                 feature_selection=False, verbose=1):

        self.base_estimator = base_estimator
        self.param_ranges = param_ranges
        self.population_size = population_size
        self.n_epochs = n_epochs
        self.n_populations = n_populations
        self.migration_frequency = migration_frequency
        self.metric = metric
        self.feature_selection = feature_selection
        self.verbose = verbose

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
        random_vec = np.random.uniform(0, 1, individual_size)
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

        self.X_ = self.prep_f(X)
        self.y_ = self.prep_f(y)
        self.splits_ = list(splits)

        self.settings = {'seed': np.random.randint(1, 4400),
                         'popsize': self.population_size,
                         'bounds': [0, 1],
                         'verbose': -9,
                         'AdaptSigma': AdaptSigma}
        
        individual_size = len(self.param_ranges)
        if individual_size > 10:
            self.settings = extend_settings(self.settings)
        
        self.strategies = [self._strategy_init(individual_size)
                           for _ in range(self.n_populations)]
        
        optimization_results = []
        for epoch in range(self.n_epochs):
            migrate = (epoch > 0) and (epoch%self.migration_frequency == 0)
            migrate &= (self.migration_frequency>0)
            epoch_results = self._run_single_gen(migrate)
            optimization_results.append(epoch_results)
            if self.verbose:
                print('Epoch %d, best score: %.3f'%(epoch, epoch_results[0]))
        return optimization_results

    def _scale(self, param, param_range):
        """Scales 0-1 bounded values from ES to the proper range of the
        parameters."""

        name, range = param_range
        scaled_param = range[int(np.ceil(param*(len(range)-1)))]
        return (name, scaled_param)

    def _create_estimator(self, x):
        """Creates estimator from values optimized by ES algortithm."""

        scaled_params = [self._scale(param, range) for param, range
                         in zip(x, self.param_ranges.items())]
        estimator_params = {k: v for k, v in scaled_params}

        estimator = clone(self.base_estimator).set_params(**estimator_params)

        return estimator
    
    def _evaluate_score(self, y_splitted_preds):
        
        avg_score = []
        for y_preds, (_, test_idx) in zip(y_splitted_preds, self.splits_):
            score = self.metric(self.y_[test_idx], y_preds)
            avg_score.append(score)
        return np.mean(avg_score)

    def _run_single_gen(self, migrate=False):
        """Runs single generation of ES:
            1. generates populations,
            2. exchanges individuals (parameter sets) between populations,
            3. creates estimators based on optimized parameters and evaluates,
            4. updates strategies with evaluated fitness functions."""

        populations = [np.array(strategy.ask())
                       for strategy in self.strategies]
        if migrate:
            populations = self._migrate(populations)

        estimators = [[self._create_estimator(x) for x in population]
                      for population in populations]

        # Prepare arguments of fitness function for (possible) remote execution
        all_pops_estimators = [[self.prep_f(x) for x in pop] 
                               for pop in estimators]
        splits_args = [self.prep_f(split) for split in self.splits_]
        
        # The populations are evaluated in CV split-wise chunks to maximise
        # parallel execution efficiency.
        all_pops_results = []
        for single_pop_estimators in all_pops_estimators:
            
            single_pop_results = []
            for estimator in single_pop_estimators:
                
                estimator_results = [self.exec_f(eval_split, estimator,
                                                 self.X_, self.y_, split)
                                     for split in splits_args]  # y_pred, estimator = part_ind_res
                
                single_pop_results.append(estimator_results)
            all_pops_results.append(single_pop_results)
        
        # zip results to process each strategy separately
        results = zip(self.strategies, populations, all_pops_results)
        
        # update startegies with fitnesses and get best estimator
        best_results = []
        for strategy, population, single_pop_results in results:
            
            # unzip predictions and estimators fitted for each split separately
            split_unzip = [list(zip(*[self.postprocess_f(r) for r in rr])) 
                           for rr in single_pop_results]
            predictions, lgbms = list(zip(*split_unzip))
            
            # calculate scores, update strategies and select best estimator
            # for a population
            scores = [self._evaluate_score(splitted_predictions)
                      for splitted_predictions in predictions]
            scores = np.array(scores)
            strategy.tell(population, -scores)
            best_estimator_id = np.argmax(scores)
            
            best_results.append((scores[best_estimator_id],
                                lgbms[best_estimator_id],
                                population[best_estimator_id]))
    
        best_population_id = np.argmax([r[0] for r in best_results])
        
        return best_results[best_population_id]