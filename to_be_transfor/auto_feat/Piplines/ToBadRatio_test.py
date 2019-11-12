from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import random
import time
from functools import wraps
# import ipdb


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

class ToBadRatio(BaseEstimator, TransformerMixin):
    '''
    对于离散变量，希望能够通过层级坏率做排序，对于异常值，也作为一层级，当作正常值对待，因此该类会取消异常值；
    该类会产生nan值，因为分母有可能为0
    另外，当不存在
    '''
    def __init__(self, cols): # no *args and **kwargs
        super().__init__()
        self.cols = cols
        self.col_dics = {}

    def _bad_ratio(self, Series, col, y):
        ser = Series.copy()
        ct_df = pd.crosstab(ser, y) 
        dic = {k:v for v,k in (ct_df[1]/(ct_df.sum(axis=1))+0.0001).sort_values().reset_index()[[col]].to_dict()[col].items()}
        return dic

    
    @timer
    def fit(self, df, y):
        self.col_dics = {}
        for col in self.cols:
            self.col_dics[col] = self._bad_ratio(df[col], col, y)
        return self
    
    def transform(self, x):
        df = pd.DataFrame()
        for col in self.cols:
            if col not in self.col_dics:
                raise Exception('col:{} not in col_dics'.format(col))
            new_name = 'TBR('+col+')'
#             ipdb.set_trace()
            df[new_name] = x[col].map(self.col_dics[col])
        return df
    
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
    tbr = ToBadRatio(cols=['edu'])
    tbr.fit(tra_x, Train.target)
    z = tbr.transform(val_x)
    
    