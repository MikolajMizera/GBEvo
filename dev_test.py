import numpy as np
from lightgbm import LGBMRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

from GBEvo.gbevo_estimators import GBEvoOptimizer

neg_mse = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)

X, y = load_boston(return_X_y=True)

base_estimator = LGBMRegressor()
base_score = cross_val_score(base_estimator, X, y, cv=5,
                             scoring=make_scorer(mean_squared_error)).mean()
print('Base score: %.3f'%base_score)

param_ranges = {'n_estimators' : np.arange(100, 500, 10),
                'max_bin' : 2**np.arange(3, 8),
                'reg_lambda' : np.linspace(0, 5, 100),
                'learning_rate': np.logspace(-3, -1, 100)}

est = GBEvoOptimizer(LGBMRegressor(), param_ranges, feature_selection=False,
                     metric=neg_mse, n_jobs=1, verbose=1,
                     optimizer_settings={'population_size': 15,
                                         'n_epochs': 100,
                                         'n_populations': 5,
                                         'migration_frequency': 5})
est.fit(X, y)