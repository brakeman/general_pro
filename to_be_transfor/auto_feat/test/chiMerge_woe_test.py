from scipy.stats import chi2
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import random

# https://blog.csdn.net/weixin_32549789/article/details/84937921
def cat2bad_rate(df, col='edu', target='target', null_value=-999):
    ct_df = pd.crosstab(df[col], df['target'])
    new_dic = {k:v for v,k in (ct_df[1]/ct_df.sum(axis=1)).sort_values().reset_index()[[col]].to_dict()[col].items()}
    new_dic[null_value] = null_value
    return Train[col].map(new_dic)


class WOE_enc(BaseEstimator, TransformerMixin):
    def __init__(self, cols): # no *args and **kwargs
        super().__init__()
        self.cols = cols
        self.cols_info = {}
    
    def fit(self, df, y):
        woe_dics = {}
        for col in self.cols:
            woe_dics[col]={}
            woe, iv = self._calWOE(df, col, y)
            woe_dics[col]['woe'] = woe
            woe_dics[col]['iv'] = iv
        self.woe_dics = woe_dics
        return self
    
    def transform(self, x):
        df = pd.DataFrame()
        for col in self.cols:
            if col not in self.woe_dics:
                raise Exception('col:{} not in woe_dics'.format(col))
            df[col] = x[col].map(self.woe_dics[col]['woe'])
        return df
    
    def _calWOE(self, df, var, y):
        '''
            计算WOE编码
            param df：数据集pandas.dataframe
            param var：已分组的列名，无缺失值
            param target：响应变量（0,1）
            return：编码字典
            '''
        eps = 0.000001  #避免除以0
        gbi = pd.crosstab(df[var], y) + eps
        gb = y.value_counts() + eps
        gbri = gbi/gb
        gbri['woe'] = np.log(gbri[1]/gbri[0])
        tmp = (gbri[1] - gbri[0])*gbri['woe']
        iv = tmp.sum()
        return gbri['woe'].to_dict(), iv


class ChiMerge(BaseEstimator, TransformerMixin):
    def __init__(self, cols, null_value, max_groups, threshold=None): # no *args and **kwargs
        self.null_value = null_value
        self.max_groups = max_groups
        self.threshold = threshold
        self.cols = cols
        super().__init__()
    
    
    def _chi_value(self, arr):
        '''
            计算一组相邻区间的卡方值
            arr:频数统计表,二维numpy数组。
            '''
        assert(arr.ndim==2)
        R_N = arr.sum(axis=1)
        C_N = arr.sum(axis=0)
        N = arr.sum()
        # 计算期望频数 C_i * R_j / N。
        E = np.ones(arr.shape)* C_N / N
        E = (E.T * R_N).T
        square = (arr-E)**2 / E
        #期望频数为0时，做除数没有意义，不计入卡方值
        square[E==0] = 0
        #卡方值
        v = square.sum()
        return v
    
    
    def fit(self, df, y):
        dic = {}
        for col in self.cols:
            freq_tab = pd.crosstab(df[col], y)
            null_flag = False
            if self.null_value in freq_tab.index:
                freq_tab = freq_tab[freq_tab.index!=self.null_value]
                null_flag = True
            #转成numpy数组用于计算。
            freq = freq_tab.values
            #初始分组切分点，每个变量值都是切分点。每组中只包含一个变量值.
            #分组区间是左闭右开的，如cutoffs = [1,2,3]，则表示区间 [1,2) , [2,3) ,[3,3+)。
            cutoffs = freq_tab.index.values
            #如果没有指定最大分组
            if self.max_groups is None:
                #如果没有指定卡方阈值，就以95%的置信度（自由度为类数目-1）设定阈值。
                if self.threshold is None:
                    #类数目
                    cls_num = freq.shape[-1]
                    threshold = chi2.isf(0.05, df= cls_num - 1)
    
    while True:
        minvalue = None
            minidx = None
                #从第1组开始，依次取两组计算卡方值，并判断是否小于当前最小的卡方
                for i in range(len(freq) - 1):
                    v = self._chi_value(freq[i:i+2])
                    if minvalue is None or (minvalue > v): #小于当前最小卡方，更新最小值
                        minvalue = v
                        minidx = i
            #如果最小卡方值小于阈值，则合并最小卡方值的相邻两组，并继续循环
            if  (self.max_groups is not None and self.max_groups< len(freq)) or (self.threshold is not None and minvalue < self.threshold):
                #minidx后一行合并到minidx
                tmp  = freq[minidx] + freq[minidx+1]
                freq[minidx] = tmp
                #删除minidx后一行
                freq = np.delete(freq,minidx+1,0)
                    #删除对应的切分点
                    cutoffs = np.delete(cutoffs,minidx+1,0)
                else: #最小卡方值不小于阈值，停止合并。
                    break
    if null_flag:
        cutoffs = np.append(cutoffs, [self.null_value])
            dic[col]=cutoffs
    self.cols_dic = dic
        return self

def _value2group(self, x, cutoffs):
    
    '''
        将变量的值转换成相应的组。
        x: 需要转换到分组的值
        cutoffs: 各组的起始值。
        return: x对应的组，如group1。从group1开始。
        '''
            
            #切分点从小到大排序。
            cutoffs = sorted(cutoffs)
            num_groups = len(cutoffs)
            
            #异常情况：小于第一组的起始值。这里直接放到第一组。
            #异常值建议在分组之前先处理妥善。
            if x <= cutoffs[0]:
                return cutoffs[0]
                    
                    for i in range(1,num_groups):
                        if cutoffs[i-1] <= x < cutoffs[i]:
                            return '['+','.join([str(cutoffs[i-1]), str(cutoffs[i])])+')'
                                
                                #最后一组，也可能会包括一些非常大的异常值。
                                return '['+str(cutoffs[-1])+',_)'

def transform(self, X, y=None):
    df = pd.DataFrame()
    for col, cutoffs in self.cols_dic.items():
        df[col] = X[col].apply(self._value2group,args=(cutoffs,))
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
    cm = ChiMerge(cols=['basicLevel', 'age'], null_value=-999, max_groups=10)
    _ = cm.fit(Train, Train.target)
    bb_tra = cm.transform(Train)
    bb_val = cm.transform(Valid)
    
    woe = WOE_enc(cols=['basicLevel', 'age'])
    bb_tra['target']=Train.target
    _ = woe.fit(bb_tra, Train.target)
    b = woe.transform(bb_val)
    
    from sklearn.pipeline import Pipeline
    from sklearn import svm
    cm = ChiMerge(cols=['basicLevel', 'age'], null_value=-999, max_groups=10)
    woe = WOE_enc(cols=['basicLevel', 'age'])
    clf = svm.SVC(kernel='linear')
    pip = Pipeline([('cm', cm), ('woe', woe)])
    X, y = tra_x, tra_y
    pip.fit(X=Train, y=Train.target)
    pip.transform(Valid)
