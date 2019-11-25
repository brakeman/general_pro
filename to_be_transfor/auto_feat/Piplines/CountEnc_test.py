from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import time
from functools import wraps
import ipdb
import pickle


class CountEnc(BaseEstimator, TransformerMixin):
    '''
    '''
    def __init__(self, cols): # no *args and **kwargs
        super().__init__()
        self.cols = cols
        self.col_dics = {}

    

    def fit(self, x, y=None):
        self.col_dics = {}
        if self.cols is None:
            self.cols = x.columns
        for col in self.cols:
            self.col_dics[col] = x[col].value_counts()
        return self
    
    
    def transform(self, x):
        df = pd.DataFrame()
        for col in self.cols:
            if col not in self.col_dics:
                raise Exception('col:{} not in col_dics'.format(col))
            col_dic = self.col_dics[col]                
            new_name1 = 'Count('+col+')'
            df[new_name1] = x[col].map(col_dic)
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
    Count = CountEnc(cols=disc_vars)
    Count.fit(tra_x)
    tra_rc = Count.transform(tra_x)
    val_rc = Count.transform(val_x)
    