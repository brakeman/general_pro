from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import time
from functools import wraps
import ipdb
import pickle

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

class IdContAgg(BaseEstimator, TransformerMixin):
    '''
    '''
    def __init__(self, cols, cont_col, agg_types=['mean', 'max', 'min', 'std']): # no *args and **kwargs
        super().__init__()
        self.cols = cols
        self.col_dics = {}
        self.cont_col = cont_col
        self.agg_types = agg_types
    
    def fit(self, x, y=None):
        print(self.cols)
        for col in self.cols:
            self.col_dics[col] = {}
            for agg_type in self.agg_types:
                self.col_dics[col][agg_type] = x.groupby([col])[self.cont_col].agg([agg_type]).to_dict()
        return self
    
    def transform(self, x):
        DF = pd.DataFrame(index=x.index)
        for col in self.cols:
            for agg_type in self.agg_types:
                new_col_name = '_'.join([col, self.cont_col, agg_type])
#                 ipdb.set_trace()
                DF[new_col_name] = x[col].map(self.col_dics[col][agg_type][agg_type])
        return DF
    
if __name__ == '__main__':

    data_path = './data'
    # test.csv  train.csv  train_target.csv
    tra_x = pd.read_csv(data_path + '/train.csv')
    tra_y = pd.read_csv(data_path + '/train_target.csv')
    tes_x = pd.read_csv(data_path + '/test.csv')
    final = tra_x.merge(tra_y,on='id')

    final['certValidStop'] = final.certValidStop.astype(int)
    final.fillna(-999,inplace=True)

    file = open('/data-0/qibo/pickle_files/cv_idx_dic.pickle', 'rb')
    idx_dic = pickle.load(file)
    tra_id, val_id = idx_dic['cv_0']['train_idx'], idx_dic['cv_0']['valid_idx']

    Train = final.iloc[tra_id,:].set_index(keys='id')
    Valid = final.iloc[val_id,:].set_index(keys='id')
    tra_x, tra_y = Train.drop('target', axis=1), Train.target
    val_x, val_y = Valid.drop('target', axis=1), Valid.target
    disc_vars = ['ethnic', 'job', 'linkRela']
    idcontAgg = IdContAgg(cols=disc_vars, cont_col='lmt')
    idcontAgg.fit(tra_x)
    tra_rc = idcontAgg.transform(tra_x)
    val_rc = idcontAgg.transform(val_x)
    