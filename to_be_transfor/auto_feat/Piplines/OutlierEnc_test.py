from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import time
from functools import wraps
import pickle
from scipy.stats import mstats


class Outlier_Enc(BaseEstimator, TransformerMixin):
    '''
        '''
    def __init__(self, cols, min_ratio=0.05, max_ratio=0.95): # no *args and **kwargs
        super().__init__()
        self.cols = cols
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
    
    def fit(self, x, y=None):
        self.col_dics = {}
        if self.cols is None:
            self.cols = x.columns
        self.max_dic = x[self.cols].apply(lambda x: np.percentile(x, self.max_ratio, interpolation='nearest')).to_dict()
        self.min_dic = x[self.cols].apply(lambda x: np.percentile(x, self.min_ratio, interpolation='nearest')).to_dict()
        return self
    
    def transform(self, x):
        df = pd.DataFrame()
        for col in self.cols:
            max, min = self.max_dic[col], self.min_dic[col]
            df['Clip('+str(col)+')'] = x[col].clip(lower=min, upper=max)
        return df



    
if __name__ == '__main__':
    da = range(10000000)
    daa = np.array([da, da, da])
    daa = pd.DataFrame(daa).T
    Count = Outlier_Enc(cols=[0,1,2])
    Count.fit(daa)
    tra_rc = Count.transform(daa)

