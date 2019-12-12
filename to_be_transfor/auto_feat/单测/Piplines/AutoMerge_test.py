from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import random
import time
import itertools
from multiprocessing import Pool
from functools import reduce


class AutoMerge(BaseEstimator, TransformerMixin):
    '''多进程k阶 +-*/'''
    def __init__(self, cols, order, op_list, null_value, num_process, verbose, max_comb=7200):
        super().__init__()
        self.cols = cols
        self.op_list = op_list
        self.order = order
        self.null_value = null_value
        self.verbose=verbose
        self.num_process=num_process
        self.max_comb = max_comb
        
    def _trans_single(self, x, col_names, op):
        # DF_new = pd.DataFrame(index=x.index)
        col_names = list(col_names)
        tra_cols = [x[i] for i in col_names]
        if op=='add':
            return reduce(lambda x, y: x+y, tra_cols)    
        elif op =='multiply':
            return reduce(lambda x, y: x*y, tra_cols)
        elif op == 'sub':
            return reduce(lambda x, y: x-y, tra_cols)
        elif op == 'divide':
            return reduce(lambda x, y: x/y, tra_cols)
        else:
            raise Exception('op error')

    # def get_feature_names(self):
    #     if self.num_process is not None:
    #         raise Exception('multiprocessing not give consistent order of all columns')
    #     return self.COL_NAMES            
        
    def _transform(self, x, tuple_list):
        DF = pd.DataFrame()
        for idx, tuple_ in enumerate(tuple_list):
            # if self.op in ['add', 'multiply', 'sub', 'divide']:
            #     new_name = '{}('.format(self.op)+','.join(list(tuple_))+')'
            #     length = len(tuple_list)
            #     if idx%(length//2)==0 and self.verbose==1:
            #         print('processing col: {}, {}/{}'.format(new_name, idx, length))
            #     DF[new_name] = self._trans_single(x, tuple_, self.op)
            # elif self.op == 'all':
            for op in self.op_list:
            # ['add', 'multiply', 'sub', 'divide']:
                new_name = '{}('.format(op)+','.join(list(tuple_))+')'
                length = len(tuple_list)
                DF[new_name] = self._trans_single(x, tuple_, op)
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
            self.num_process = min(self.num_process, len(self.sub_comb_list))
            print('program is going to use multiprocessing with {} Ps'.format(self.num_process))
        return self
    
    def transform(self, x):
        if self.num_process!=None:
            st = time.time()
            print('------------------------{}-{}----------------------------'.format(self.__class__.__name__, 'transform()'))
            print('program is going to use multiprocessing with {} Ps'.format(self.num_process))
            p2 = Pool(self.num_process)
            rst = []
            for i in range(self.num_process):
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
    data_path = './data'
    tra_x = pd.read_csv('../data/train.csv', encoding="utf-8").set_index('ID')
    test = pd.read_csv('../data/test.csv', encoding="utf-8").set_index('ID')
    Y = pd.read_csv('../data/train_label.csv', encoding="utf-8").set_index('ID')
    submission = pd.read_csv('../data/submission.csv')
    id_cols = ['企业类型', '登记机关', '行业代码', '行业门类', '企业类别', '管辖机关']
    AM = AutoMerge(cols=id_cols[:4], order=3, op_list=['sub', 'divide'], null_value=np.nan, num_process=3, verbose=1)
    tmp_x = AM.fit_transform(tra_x)
    tmp_test = AM.transform(test)
    print(tmp_test.head())