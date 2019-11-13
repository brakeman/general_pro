from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import random
import time
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

class CountEnc(BaseEstimator, TransformerMixin):
    '''
    排序分箱；
    处理了两端异常；
    '''
    def __init__(self, cols): # no *args and **kwargs
        super().__init__()
        self.cols = cols
        self.col_dics = {}
    
    @timer
    def fit(self, df, y=None):
        self.col_dics = {}
        for col in self.cols:
            Ser = df[col].copy()
            self.col_dics[col] = Ser.value_counts().to_dict()
        return self
    
    def transform(self, x):
        df = pd.DataFrame()
        for col in self.cols:
            if col not in self.col_dics:
                raise Exception('col:{} not in col_dics'.format(col))
            col_dic = self.col_dics[col]
            new_name = 'Count('+col+')'
            df[new_name] = x[col].map(col_dic)
        return df
    
if __name__ == '__main__':
    data_path = './data/'
    # test.csv  train.csv  train_target.csv
    tra_x = pd.read_csv(data_path + '/train.csv')
    tra_y = pd.read_csv(data_path + '/train_target.csv')
    tes_x = pd.read_csv(data_path + '/test.csv')
    final = tra_x.merge(tra_y,on='id')

    final['certValidStop'] = final.certValidStop.astype(int)
    final.fillna(-999,inplace=True)

    tra_id = set(random.sample(range(final.shape[0]), 70000))
    val_id = set(range(final.shape[0])) - tra_id
    tra_id = [i for i in tra_id]
    val_id = [i for i in val_id]
    Train = final.iloc[tra_id,:].set_index(keys='id')
    Valid = final.iloc[val_id,:].set_index(keys='id')
    tra_x, tra_y = Train.drop('target', axis=1), Train.target
    val_x, val_y = Valid.drop('target', axis=1), Valid.target
    disc_vars = ['ethnic', 'job', 'linkRela']
    Count = CountEnc(cols=disc_vars)
    Count.fit(tra_x)
    tra_rc = Count.transform(tra_x)
    val_rc = Count.transform(val_x)
    