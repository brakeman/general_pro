from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import time
from functools import wraps
import pickle
from scipy.stats import mstats


class StableEnc(BaseEstimator, TransformerMixin):
    '''Pvalue > 10% 就认为 不拒绝H0: 同分布
        '''
    def __init__(self, cols, test, thresh=0.1): # no *args and **kwargs
        super().__init__()
        self.cols = cols
        self.test = test
        self.thresh = thresh
    
    def KS_check(self, train, test_, col):
        from scipy.stats import ks_2samp
        return ks_2samp(train[col], test_[col])[1]
    
    
    def fit(self, x, y=None):
        self.col_dics = {}
        if self.cols is None:
            self.cols = x.columns
        for col in self.cols:
            self.col_dics[col] = self.KS_check(x, self.test, col)
        self.P_value_df = pd.DataFrame.from_dict(self.col_dics, orient='index')
        self.P_value_df.columns = ['P_value']
        self.P_value_df = self.P_value_df.sort_values('P_value', ascending='False')
        print(self.P_value_df)
        return self
    
    def transform(self, x):
        self.stable_cols = self.P_value_df[self.P_value_df.P_value>self.thresh].index.tolist()
        return x[self.stable_cols]
