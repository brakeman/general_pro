from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import random
import time
import itertools
from functools import wraps

def timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running function: %s with %s seconds" %
            (function.__name__, str(t1-t0)))
        return result
    return function_timer

class AutoCombine(BaseEstimator, TransformerMixin):
    def __init__(self, cols, order, null_value):
        super().__init__()
        self.cols = cols
        self.order = order
        self.col_dicts = {}
        self.cache = None
        self.null_value = null_value
        self.combine_list = [i for i in itertools.combinations(cols, order)]

    @timer
    def fit(self, x, y=None):
        DF = pd.DataFrame()
        for idx, col_names in enumerate(self.combine_list):
            new_name = 'comb('+','.join(col_names)+')'
            print('processing col: {}, {}/{}'.format(new_name, idx, len(self.combine_list)))
            DF[new_name] = (x[list(col_names)].astype(str)+'|').sum(axis=1)
            self.col_dicts[new_name] =  DF[new_name].unique()
        self.cache = DF
        return self   
    
    @timer
    def transform(self, x, train):
        if train:
            return self.cache
        DF = pd.DataFrame()
        for col_names in self.combine_list:
            new_name = 'comb('+','.join(col_names)+')'
            tra_unique = self.col_dicts[new_name]
            DF[new_name] = (x[list(col_names)].astype(str)+'|').sum(axis=1)
            # 凡是 test中存在， train中不存在的，变为null
            DF[new_name][~DF[new_name].isin(tra_unique)] = self.null_value
        return DF
    
    
if __name__ == '__main__':
    data_path = './data'
    # test.csv  train.csv  train_target.csv
    tra_x = pd.read_csv(data_path + '/train.csv')
    tra_y = pd.read_csv(data_path + '/train_target.csv')
    final = tra_x.merge(tra_y,on='id')
    final['dist']= final.dist.apply(lambda x: int(x/100))
    random.seed(1)
    tra_id = set(random.sample(range(final.shape[0]),70000))
    val_id = set(range(final.shape[0])) - tra_id
    tra_id = [i for i in tra_id]
    val_id = [i for i in val_id]
    Train = final.iloc[tra_id,:]
    Valid = final.iloc[val_id,:]
    tra_x, tra_y = Train.drop('target', axis=1), Train.target
    val_x, val_y = Valid.drop('target', axis=1), Valid.target
    tbr = AutoCombine(cols=['edu', 'age', 'gender'], order=2, null_value=-999)
    tbr.fit(tra_x)
    z = tbr.transform(tra_x, True)
    z2 = tbr.transform(val_x, False)