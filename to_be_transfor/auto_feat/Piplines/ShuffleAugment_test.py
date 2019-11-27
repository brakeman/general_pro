from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import time
import sys
sys.path.append('../')
from auto_feat.Piplines import CountEnc_test
from functools import wraps
import ipdb
import pickle
import lightgbm as lgb


class ShuffleAugment(BaseEstimator, TransformerMixin):
    '''
    '''
    def __init__(self, cols, fake_ratio, target_name): # no *args and **kwargs
        super().__init__()
        self.cols = cols
        self.target_name = target_name
        self.fake_ratio = fake_ratio
        
    def fit(self, x, y):
        temp_DF = x.join(y)
        self.pos_df, self.neg_df = temp_DF[temp_DF[self.target_name]==0], temp_DF[temp_DF[self.target_name]==1]
        assert (self.pos_df.shape[0]>0) & (self.neg_df.shape[0]>0)
        return self
    
    def transform(self, x):
        fake_pos, fake_neg = pd.DataFrame(), pd.DataFrame()
        for col in self.cols:
            fake_pos[col] = self.pos_df.sample(frac=self.fake_ratio)[col].tolist()
            fake_neg[col] = self.neg_df.sample(frac=self.fake_ratio)[col].tolist()
        fake_pos[self.target_name] = 0
        fake_neg[self.target_name] = 1
        return fake_pos.append(fake_neg)
    
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
    disc_vars = ['job', 'linkRela']
        
    SA = ShuffleAugment(cols=disc_vars, fake_ratio=1, target_name='target')
    SA.fit(tra_x, tra_y)
    fake =SA.transform(val_x)