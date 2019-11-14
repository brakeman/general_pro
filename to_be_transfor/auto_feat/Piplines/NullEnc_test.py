from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import random
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


class NullEnc(BaseEstimator, TransformerMixin):
    '''
    '''
    def __init__(self, null_value, normalize, add_ori): # no *args and **kwargs
        super().__init__()
        self.null_value = null_value
        self.normalize = normalize
        self.add_ori = add_ori
    
    @timer
    def fit(self, x, y=None):
        if self.normalize:
            new_df = pd.DataFrame(index = x.index) 
            null_df = x==self.null_value
            new_df['null_num'] = null_df.sum(axis=1)
            new_df['null_rate'] = null_df.sum(axis=1)/null_df.shape[1]
            self.SS = StandardScaler()
            self.SS.fit(new_df)
        return self
    
    def transform(self, x):
        x_null = pd.DataFrame(index = x.index) 
        null_df = x==self.null_value
        x_null['null_num'] = null_df.sum(axis=1)
        x_null['null_rate'] = null_df.sum(axis=1)/null_df.shape[1]
        if self.normalize:
            x_null = pd.DataFrame(self.SS.transform(x_null), index=x.index, columns=['null_num','null_rate'])
        
        if self.add_ori:
            x_null = x_null.join(x)
        return x_null
    
if __name__ == '__main__':
    data_path = './data/'
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
    NE = NullEnc(null_value=-999, normalize=True, add_ori=False)
    NE.fit(tra_x)
    tra_rc = NE.transform(tra_x)
    val_rc = NE.transform(val_x)