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


    

class Ind_Count_Score(BaseEstimator, TransformerMixin):
    '''
    每列有 原始特征+CountEnc特征, 两个变量对吧？
    - 用两列变量 跑lgb + lgb_predict 得到 200个logit 
    - 最终200个logit 综合跑lgb 
    '''
    def __init__(self, cols, val_x, val_y): # no *args and **kwargs
        super().__init__()
        self.cols = cols
        self.col_dics = {}
        self.val_y = val_y
        self.val_x = val_x
        
    def auc_impo(self, tra_x, tra_y, val_x, val_y, n_job, params=None, verbose=1):
        print(' train:{}   valid:{}'.format(tra_x.shape, val_x.shape))
        if params is None:
            params = {'task': 'train',
                  'boosting_type': 'gbdt',
                  'objective': 'binary',
                  'metric': 'auc',
                  'num_threads': 10,
    #               'max_depth': 3,
                  'num_leaves': 20,  # 31,
                  'learning_rate': 0.008,  # 0.002
                  'feature_fraction': 1,
                  'lambda_l2': 140,
                  'bagging_fraction': 0.4,
                  'bagging_freq': 5,
                  'num_threads':n_job}

        cv_train = lgb.Dataset(tra_x, tra_y.astype('int'))        
        cv_valid = lgb.Dataset(val_x, val_y.astype('int'))        
        gbm = lgb.train(params,       # 参数字典
                        cv_train,       # 训练集
                        num_boost_round=2000,       # 迭代次数
                        valid_sets=cv_valid,        # 验证集
                        early_stopping_rounds = 100,
                        verbose_eval=verbose)
        print('with best auc:{}'.format(gbm.best_score['valid_0']))
        return gbm
    
    def fit(self, x, y):
        self.CountEnc = CountEnc_test.CountEnc(cols=self.cols, normalize=False, only_rank='count')
        self.CountEnc.fit(x)
        x_count = self.CountEnc.transform(x)
        val_count = self.CountEnc.transform(self.val_x)
        for var in self.cols:
            model = self.auc_impo(tra_x=np.hstack([x[var].values.reshape(-1,1), x_count['Count('+var+')'].values.reshape(-1,1)]), 
                             tra_y=y,
                             val_x=np.hstack([self.val_x[var].values.reshape(-1,1), val_count['Count('+var+')'].values.reshape(-1,1)]), 
                             val_y=self.val_y,
                             n_job=10)
    
            self.col_dics[var] = model
        return self

    
    def transform(self, x):
        DF = pd.DataFrame(index=x.index)
        x_count = self.CountEnc.transform(x)
        for var in self.col_dics:
            print(var)
            model = self.col_dics[var]
            name = 'ICS('+var+')'
            DF[name] =  model.predict(np.hstack([x[var].values.reshape(-1,1), 
                                         x_count['Count('+var+')'].values.reshape(-1,1)]))
        return DF
    
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
    ICS = Ind_Count_Score(cols=disc_vars, val_x=val_x, val_y=val_y)
    ICS.fit(tra_x, tra_y)
    tra_rc = ICS.transform(tra_x)
    val_rc = ICS.transform(val_x)
        