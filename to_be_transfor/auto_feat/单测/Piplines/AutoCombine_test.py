from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import random
import time
import itertools
from multiprocessing import Pool


class AutoCombine(BaseEstimator, TransformerMixin):
    '''多进程k阶类别特征组合'''
    def __init__(self, cols, order, null_value, num_process, verbose, max_comb=4000):
        super().__init__()
        self.cols = cols
        self.order = order
        self.col_dicts = {}
        self.null_value = null_value
        self.combine_list = [i for i in itertools.combinations(cols, order)]
        self.verbose=verbose
        if len(self.combine_list)>max_comb:
            print('clip since reach max_comb:{}/{}'.format(len(self.combine_list), max_comb))
            self.combine_list = random.sample(self.combine_list, max_comb)
        else:
            print('totally {} combinations'.format(len(self.combine_list)))
        self.num_process=num_process
        if num_process!=None:
            self.sub_len = len(self.combine_list)//num_process+1
            self.sub_comb_list = [self.combine_list[x:x+self.sub_len] for x in range(0, len(self.combine_list), self.sub_len)]

        
    def _fit(self, x, combine_list):
        DF = pd.DataFrame()
        col_dicts={}
        for idx, col_names in enumerate(combine_list):
            new_name = 'comb('+','.join(col_names)+')'
            length=len(combine_list)
            if length!=1 and idx%(length//2)==0 and self.verbose==1:
                print('processing col: {}, {}/{}'.format(new_name, idx, length))
            DF[new_name] = (x[list(col_names)].astype(str)+'|').sum(axis=1)
            col_dicts[new_name] =  DF[new_name].unique()
        return DF, col_dicts
    
    def fit(self, x, y=None):
        st = time.time()
        if self.num_process:
            print('------------------------{}-{}----------------------------'.format(self.__class__.__name__, 'fit()'))
            self.num_process = min(self.num_process, len(self.sub_comb_list))
            print('program is going to use multiprocessing with {} Ps'.format(self.num_process))
            p = Pool(self.num_process)
            rst = []
            for i in range(self.num_process):
                a = p.apply_async(self._fit, args=(x, self.sub_comb_list[i])) 
                rst.append(a)
            p.close()
            p.join()
            new_x=rst[0].get()[0]
            for i in rst[1:]:
                new_x=new_x.join(i.get()[0])
            list_dic = [i.get()[1] for i in rst]
            for d in list_dic:
                self.col_dicts.update(d)
        else:
            new_x, col_dicts =  self._fit(x, self.combine_list)
            self.col_dicts =col_dicts
            
        print('------------------------use:{} s----------------------------'.format(time.time()-st))
        return self
        
    def _transform(self, x, combine_list):
        DF = pd.DataFrame()
        for idx, col_names in enumerate(combine_list):
            new_name = 'comb('+','.join(col_names)+')'
            length = len(combine_list)
            if length!=1 and idx%(length//2)==0 and self.verbose==1:
                print('processing col: {}, {}/{}'.format(new_name, idx, length))
            tra_unique = self.col_dicts[new_name]
#             ipdb.set_trace()
            DF[new_name] = (x[list(col_names)].astype(str)+'|').sum(axis=1)
            # 凡是 test中存在， train中不存在的，变为null
            DF[new_name][~DF[new_name].isin(tra_unique)] = self.null_value
        return DF
    
    def transform(self, x):
        if self.num_process:
            st = time.time()
            print('------------------------{}-{}----------------------------'.format(self.__class__.__name__, 'transform()'))
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
            print('will not use multi processing')
            return self._transform(x, self.combine_list)
    
    
if __name__ == '__main__':
    data_path = './data'
    tra_x = pd.read_csv('../data/train.csv', encoding="utf-8").set_index('ID')
    test = pd.read_csv('../data/test.csv', encoding="utf-8").set_index('ID')
    Y = pd.read_csv('../data/train_label.csv', encoding="utf-8").set_index('ID')
    submission = pd.read_csv('../data/submission.csv')
    id_cols = ['企业类型', '登记机关', '行业代码', '行业门类', '企业类别', '管辖机关']
    AC = AutoCombine(cols=id_cols[:5], order=3, null_value=np.nan, num_process=4, verbose=None)
    tmp_x = AC.fit_transform(tra_x)
    tmp_test = AC.transform(test)
    print(tmp_test.head())
