from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import random
import time
from functools import wraps
import ipdb


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

class ExpandMeanEnc(BaseEstimator, TransformerMixin):
    '''
    cumsum = df_tr.groupby(col)['target'].cumsum() - df_tr['target']
    cumcnt = df_tr.groupby(col).cumcount()
    train_new[col + '_mean_target'] = cusum/cumcnt
    '''
    def __init__(self, cols, target_col_name): # no *args and **kwargs
        super().__init__()
        self.cols = cols
        self.col_dics = {}
        self.target_col_name= target_col_name
    
    @timer
    def fit(self, df, y):
        self.col_dics = {}
        for col in self.cols:
#             ipdb.set_trace()
            tmp_df = pd.concat([df[col], y], axis=1)
            cumsum = tmp_df.groupby(col)[self.target_col_name].cumsum() - tmp_df[self.target_col_name]
            cumcnt = tmp_df.groupby(col).cumcount()
            tmp_df['enc'] = cumsum/cumcnt
            self.col_dics[col] = tmp_df.groupby(col)[col,'enc'].tail(1).set_index(col).to_dict()
        return self
    
    def transform(self, x):
        df = pd.DataFrame()
        for col in self.cols:
            if col not in self.col_dics:
                raise Exception('col:{} not in col_dics'.format(col))
#             ipdb.set_trace()
            col_dic = self.col_dics[col]
            new_name = 'ExpMeanEnc('+col+')'
            df[new_name] = x[col].map(col_dic['enc'])
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
    EME = ExpandMeanEnc(cols=disc_vars, target_col_name='target')
    EME.fit(tra_x, tra_y)
    tra_rc = EME.transform(tra_x)
    val_rc = EME.transform(val_x)
    