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

class RankCut(BaseEstimator, TransformerMixin):
    '''
    排序分箱；
    必须 return_numeric 才能后接CountEnc; 
    处理了两端异常；
    '''
    def __init__(self, cols, bins, null_value, return_numeric): # no *args and **kwargs
        super().__init__()
        self.cols = cols
        self.null_value = null_value
        self.bins = bins
        self.return_numeric = return_numeric
        self.col_dics = {}

        
    def _rank_cut(self, Series, min_, max_, bins):
        if isinstance(bins, int):
            labels = range(bins)
        elif isinstance(bins, list):
            labels = range(len(bins)-1)
        else:
            raise Exception('bins type not allowed!')
        Ser = Series.copy()
        Ser[(Ser<min_)&(Ser!=-self.null_value)] = min_
        Ser[(Ser>=max_)&(Ser!=-self.null_value)] = max_-0.0001
        if self.return_numeric:
            Ser[Ser!=self.null_value], new_bins = pd.cut(Ser[Ser!=self.null_value], 
                                                          bins, right=False, 
                                                          retbins=True, labels=labels)
        else:
            Ser[Ser!=self.null_value], new_bins = pd.cut(Ser[Ser!=self.null_value], 
                                                          bins, right=False, retbins=True)
        return Ser, new_bins
    
    @timer
    def fit(self, df, y=None):
        self.col_dics = {}
        self.cache=pd.DataFrame()
        for col in self.cols:
            Ser = df[col].copy()
            self.col_dics[col] = {}
            self.col_dics[col]['min'] = min_ = Ser[Ser!=self.null_value].min()
            self.col_dics[col]['max'] = max_ = Ser[Ser!=self.null_value].max()
            new_name = 'RC('+col+')'
            self.cache[new_name], self.col_dics[col]['bins'] = self._rank_cut(Ser, min_, max_, self.bins)
        return self
    
    def transform(self, x, train):
        df = pd.DataFrame()
        if train:
            return self.cache
        for col in self.cols:
            if col not in self.col_dics:
                raise Exception('col:{} not in col_dics'.format(col))
            min_, max_, bins = self.col_dics[col]['min'], self.col_dics[col]['max'], self.col_dics[col]['bins']
            new_name = 'RC('+col+')'
            df[new_name], _ = self._rank_cut(x[col], min_, max_, list(bins))
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
    cont_vars = ['certValidStop', 'certValidBegin', 'lmt',]
    RC = RankCut_test.RankCut(cols=cont_vars, bins=20, null_value=-999, return_numeric=True)
    RC.fit(tra_x)
    tra_rc = RC.transform(tra_x, train=True)
    val_rc = RC.transform(val_x, train=False)