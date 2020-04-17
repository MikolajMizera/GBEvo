# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import GridSearchCV
import numpy as np

class GBEvo(MetaEstimatorMixin, BaseEstimator):
    
    def __init__(self, estimator, params_ranges={}, mask_features=True,
                 scoring=None, n_jobs=None, verbose=0, error_score=np.nan):
        
        self.estimator = estimator
        self.params_ranges = params_ranges
        self.mask_features = mask_features
        
    def fit(self, X, y):
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        self.X_ = X
        self.y_ = y
        
        self.estimator = clone(self.estimator)
        
        ### Code for evolution here
        
        return self
    
    def predict(self, X):
        
        # Check if fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        return self.base_estimator.predict(X[:,self.var_mask])

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
        self.estimator.fit(self.X_[:,self.feature_mask], self.y_)
        
        # Return the estimator
        return self
    
    def predict(self, X):
        
        # Check if fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        return self.estimator.predict(X[:,self.var_mask])
            
    def get_params(self, deep=True):
        return {'estimator':self.estimator,
                'params_ranges':self.params_ranges, 
                'feature_mask':self.feature_mask}
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self