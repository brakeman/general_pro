from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import random
import time
import itertools
from functools import wraps
from multiprocessing import Pool
import ipdb
import pickle
from functools import reduce


class AutoMerge(BaseEstimator, TransformerMixin):
    '''多进程k阶类别特征组合'''
    def __init__(self, cols, order, op, null_value, num_process, verbose, max_comb=4000):
        super().__init__()
        self.cols = cols
        self.op = op
        self.order = order
        self.null_value = null_value
        self.verbose=verbose
        self.num_process=num_process
        self.max_comb = max_comb
        
    def _trans_single(self, x, col_names):
        DF_new = pd.DataFrame(index=x.index)
        col_names = list(col_names)
        tra_cols = [x[i] for i in col_names]
        if self.op=='add':
            return x[col_names].sum(axis=1)       
        elif self.op =='multiply':
            return reduce(lambda x, y: x*y, tra_cols)
        elif self.op == 'sub':
            return reduce(lambda x, y: x-y, tra_cols)
        else:
            raise Exception('op error')
            
    def _transform(self, x, tuple_list):
        DF = pd.DataFrame()
        for idx, tuple_ in enumerate(tuple_list):
            new_name = '{}('.format(self.op)+','.join(list(tuple_))+')'
            length = len(tuple_list)
            if idx%(length//2)==0 and self.verbose==1:
                print('processing col: {}, {}/{}'.format(new_name, idx, length))
            DF[new_name] = self._trans_single(x, tuple_)
        return DF
    
    def fit(self, x, y=None):
        if self.cols is None:
            self.cols = x.columns
            print(len(self.cols))
        self.combine_list = [i for i in itertools.combinations(self.cols, self.order)]
        if len(self.combine_list)>self.max_comb:
            print('clip since reach max_comb:{}/{}'.format(len(self.combine_list), self.max_comb))
            self.combine_list = random.sample(self.combine_list, self.max_comb)
        else:
            print('totally {} combinations'.format(len(self.combine_list)))
        if self.num_process!=None:
            self.sub_len = len(self.combine_list)//self.num_process+1
            self.sub_comb_list = [self.combine_list[x:x+self.sub_len] for x in range(0, len(self.combine_list), self.sub_len)]
        return self
    
    def transform(self, x):
#         ipdb.set_trace()
        if self.num_process!=None:
            st = time.time()
            print('------------------------{}-{}----------------------------'.format(self.__class__.__name__, 'transform()'))
            print('program is going to use multiprocessing with {} Ps'.format(self.num_process))
            p2 = Pool(self.num_process)
            rst = []
            for i in range(self.num_process):
#                 ipdb.set_trace()
                aa = p2.apply_async(self._transform, args=(x, self.sub_comb_list[i])) 
                rst.append(aa)       
            p2.close()
            p2.join()
            new_x = pd.concat([i.get() for i in rst], axis=1)
            print('------------------------use:{} s----------------------------'.format(time.time()-st))
            return new_x
        else:
            return self._transform(x, self.combine_list)
            
    
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
    zz = tra_x.apply(lambda x: len(x.unique())).sort_values(ascending=False)
    small_cats = zz[zz<5].index.tolist()
    st=time.time()
    tbr = AutoMerge(cols=small_cats[:30], order=2, op='sub', null_value=-999, num_process=None, verbose=1)
    tbr.fit(tra_x)
    z1 = tbr.transform(tra_x)
    z2 = tbr.transform(val_x)
    print(time.time()-st)