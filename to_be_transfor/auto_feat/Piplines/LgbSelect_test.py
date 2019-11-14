from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import time
from functools import wraps
import ipdb
import pickle
import lightgbm as lgb

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

class LgbSelect(BaseEstimator, TransformerMixin):
    '''
    '''
    def __init__(self, verbose, njob, val_x, val_y, topk, importance_type): # no *args and **kwargs
        super().__init__()
        self.verbose = verbose
        self.njob = njob
        self.val_x = val_x
        self.val_y = val_y
        self.topk=topk
        assert importance_type in ['split', 'gain']
        self.importance_type = importance_type
    
    
    def _fit_lgb(self, tra_x, tra_y, val_x, val_y, params=None):
        print(' train:{}   valid:{}'.format(tra_x.shape, val_x.shape))
        if params is None:
            params = {'task': 'train',
                  'boosting_type': 'gbdt',
                  'objective': 'binary',
                  'metric': 'auc',
                  'num_threads': 10,
                  'num_leaves': 3,  # 31,
                  'learning_rate': 0.008,  # 0.002
                  'feature_fraction': 0.5,
                  'lambda_l2': 140,
                  'bagging_fraction': 0.5,
                  'bagging_freq': 5,
                  'num_threads':self.njob}

        cv_train = lgb.Dataset(tra_x, tra_y.astype('int'))        
        cv_valid = lgb.Dataset(val_x, val_y.astype('int'))        
        gbm = lgb.train(params,       # 参数字典
                        cv_train,       # 训练集
                        num_boost_round=2000,       # 迭代次数
                        valid_sets=cv_valid,        # 验证集
                        early_stopping_rounds = 100,
                        verbose_eval=self.verbose)
        print('with best auc:{}'.format(gbm.best_score['valid_0']))
        return gbm

    def _topk_feat(self, tra_x, gbm):
        z = pd.DataFrame()
        z['col'] = tra_x.columns
        z['imp'] = gbm.feature_importance(self.importance_type) 
        col_names = z.nlargest(self.topk, 'imp').col.tolist()
        return col_names

    
    def fit(self, x, y):
        gbm = self._fit_lgb(x, y, self.val_x, self.val_y)
        self.top_col_names = self._topk_feat(x, gbm)
        return self
    
    def transform(self, x):
        return x[self.top_col_names]
    
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

    lgbselect = LgbSelect(verbose=0, njob=38, val_x=val_x, val_y=val_y, topk=50, importance_type='split')
    lgbselect.fit(tra_x, tra_y)
    tra_rc = lgbselect.transform(tra_x)
    val_rc = lgbselect.transform(val_x)
    