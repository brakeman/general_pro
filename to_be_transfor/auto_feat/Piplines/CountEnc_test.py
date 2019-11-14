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

class CountEnc(BaseEstimator, TransformerMixin):
    '''
    '''
    def __init__(self, cols, normalize, only_rank): # no *args and **kwargs
        super().__init__()
        self.cols = cols
        self.col_dics = {}
        self.normalize = normalize
        assert only_rank in ['rank', 'count', 'both']
        self.only_rank = only_rank
    

    def _fit(self, x, y=None):
        self.col_dics = {}
        if self.cols is None:
            self.cols = x.columns
        for col in self.cols:
#             ipdb.set_trace()
            Ser = x[col].copy()
            df = pd.DataFrame()
            df['count_']=Ser.value_counts()
            df['rank_']=range(df.shape[0])
            self.col_dics[col] = {}
            self.col_dics[col]['count']=df.count_.to_dict()
            self.col_dics[col]['rank']=df.rank_.to_dict()
        return self
    
    def fit(self, x, y=None):
        self._fit(x, y)
        if self.normalize:
            self.SS = StandardScaler()
            self.SS.fit(self._transform(x))
        return self
    
    def _transform(self, x):
        df = pd.DataFrame()
        for col in self.cols:
            if col not in self.col_dics:
                raise Exception('col:{} not in col_dics'.format(col))
            col_dic = self.col_dics[col]                
            new_name1 = 'Count('+col+')'
            new_name2 = 'CountRank('+col+')'
            if self.only_rank=='rank':
                df[new_name2] = x[col].map(col_dic['rank'])
            elif self.only_rank=='count':
                df[new_name1] = x[col].map(col_dic['count'])
            else:
                df[new_name2] = x[col].map(col_dic['rank'])
                df[new_name1] = x[col].map(col_dic['count'])
        return df
    
    def transform(self, x):
        df = self._transform(x)
        if self.normalize:
            columns = df.columns
            index = x.index
            df = pd.DataFrame(self.SS.transform(df), columns = columns, index=index)
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

    file = open('/data-0/qibo/pickle_files/cv_idx_dic.pickle', 'rb')
    idx_dic = pickle.load(file)
    tra_id, val_id = idx_dic['cv_0']['train_idx'], idx_dic['cv_0']['valid_idx']

    Train = final.iloc[tra_id,:].set_index(keys='id')
    Valid = final.iloc[val_id,:].set_index(keys='id')
    tra_x, tra_y = Train.drop('target', axis=1), Train.target
    val_x, val_y = Valid.drop('target', axis=1), Valid.target
    disc_vars = ['ethnic', 'job', 'linkRela']
    Count = CountEnc(cols=disc_vars, normalize=False, only_rank='both')
    Count.fit(tra_x)
    tra_rc = Count.transform(tra_x)
    val_rc = Count.transform(val_x)
    