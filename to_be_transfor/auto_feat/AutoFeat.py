# 针对非时序表做特征工程，核心依赖 cat列之间的交互
# v0 半自动，一组cat列交互之后 看lgb auc + top k features
# v1 自动，给出全部cat列，有限资源下，按阶数做 lgb auc + top k features, 不需要再人工参与筛选了


import pandas as pd
import numpy as np
import random
import itertools
import lightgbm as lgb
from functools import reduce

class autofeat_draft:
    '''
    1. k_order_comb generation 特征组合生成
    2. dummy_lgb_topk 特征降维
    3. auto_transform 小微特征加减乘 生成
    '''
    
    def __init__(self, global_id = 'id'):
        self.global_id = global_id
        
    
    def k_order_comb(self, tra, val, column_names, order=2):
        # 确认输入是否都是cat
        assert tra.index.name == self.global_id
        DF_tra, DF_val = pd.DataFrame(index=tra.index), pd.DataFrame(index=val.index)
        col_names_list = [i for i in itertools.combinations(column_names, order)]
    #     print(col_names_list)
        for col_names in col_names_list:
            tra_cols = [tra[i] for i in col_names]
            val_cols = [val[i] for i in col_names]
            tra_new, uniq = self._auto_combine(tra_cols, col_names)
            val_new = self._auto_combine_transform(val_cols, uniq, col_names)
            DF_tra[tra_new.name] = tra_new
            DF_val[tra_new.name] = val_new
        print('after {} order feature combination, there are totally {} new features generated'.format(order, DF_tra.shape[1]))
        return DF_tra, DF_val
    
    def dummy_lgb_topk(self, tra_x, tra_y, val_x, val_y, col_names_list, max_unique=20000):
        unique_num = 0
        tmp_tra_df = pd.DataFrame(index = tra_x.index)
        tmp_val_df = pd.DataFrame(index = val_x.index)
        final_tra, final_val = [], []
        for idx, col in enumerate(col_names_list):
            if unique_num <= max_unique:
                print('processing col:{}/{}unique num reach max_unique, start go lgb_topk selection'.format(idx, len(col_names_list)))
                unique_num += len(tra_x[col].unique())
                tmp_tra_df = pd.get_dummies(tra_x[col], prefix=col+'_').join(tmp_tra_df)
                tmp_val_df = pd.get_dummies(val_x[col], prefix=col+'_').join(tmp_val_df)
            tra_1, val_1, col_names, gbm = self._auto_topk(self, tra_x, tra_y, val_x, val_y)
            final_tra.append(tra_1)
            final_val.append(val_1)
        return pd.concat(final_tra, axis=1), pd.concat(final_val, axis=1)
    
    
    def auto_transform(self, tra, val, column_names, order=2, op='add'):
        assert op in ['add', 'multiply', 'sub']
        assert tra.index.name == self.global_id == val.index.name
        DF_tra, DF_val = pd.DataFrame(index=tra.index), pd.DataFrame(index=val.index)
        col_names_list = [i for i in itertools.combinations(column_names, order)]
        for col_names in col_names_list:
            tra_cols = [tra[i] for i in col_names]
            val_cols = [val[i] for i in col_names]
            new_name = '&'.join(col_names) + '&op={}'.format(op)
            if op=='add':
                DF_tra[new_name] = reduce(lambda x, y: x+y, tra_cols)
                DF_val[new_name] = reduce(lambda x, y: x+y, val_cols)           
            elif op =='multiply':
                DF_tra[new_name] = reduce(lambda x, y: x*y, tra_cols)
                DF_val[new_name] = reduce(lambda x, y: x*y, val_cols)
            elif op == 'sub':
                DF_tra[new_name] = reduce(lambda x, y: x-y, tra_cols)
                DF_val[new_name] = reduce(lambda x, y: x-y, val_cols)
            else:
                raise Exception('op error')
        return DF_tra, DF_val
    
    def _auto_topk(self, tra_x, tra_y, val_x, val_y):
        gbm = self._auc_impo(tra_x, tra_y, val_x, val_y)
        tra_1, col_names = self._topk_feat(tra_x, gbm, 20)
        val_1 = self._topk_feat_transfrom(val_x, col_names)
        return tra_1, val_1, col_names, gbm
    
    def _auc_impo(self, tra_x, tra_y, val_x, val_y, params=None):
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
                  'bagging_freq': 5}

        cv_train = lgb.Dataset(tra_x, tra_y.astype('int'))        
        cv_valid = lgb.Dataset(val_x, val_y.astype('int'))        
        gbm = lgb.train(params,                     # 参数字典
                        cv_train,                  # 训练集
                        num_boost_round=2000,       # 迭代次数
                        valid_sets=cv_train,        # 验证集
                        early_stopping_rounds = 100,
                        verbose_eval=0)
        print('with best auc:{}'.format(gbm.best_score['training']['auc']))
        return gbm

    def _topk_feat(self, tra_x, gbm, k):
        z = pd.DataFrame()
        z['col'] = tra_x.columns
        z['imp'] = gbm.feature_importance() 
        col_names = z.nlargest(k, 'imp').col.tolist()
        return tra_x[col_names], col_names

    def _topk_feat_transfrom(self, val_x, col_names):
        return val_x[col_names]


    def _auto_combine(self, cols, col_names, max_limit = 10000, null_rate=0.66, test_transform=False):
        # 多种类别形 直接 字符串相加 变成新列;
        # cols: List[series]
        # col_names: List[str]
        # return: pd.Series with name;
        name, val, unique_cat = '', '', 0
        for i in range(len(col_names)):
            if unique_cat > max_limit and not test_transform:
                break            
            name += col_names[i]+'|'
            val += col_names[i]+'='+cols[i].astype('str')+'|'
            unique_cat = len(val.value_counts())

        new_col = pd.Series(val, name=name)
        if test_transform:
            return new_col

        tmp = new_col.value_counts(normalize=True).cumsum()
        unique_cats = tmp[np.less(tmp, null_rate)].index.tolist()
        new_col[~new_col.isin(unique_cats)] = -999
        print('generated new column:{}\n.................with unique category length:{}\n'.format(name, unique_cat))
        print('after null ratio control, there are totally {} categories left'.format(new_col.value_counts().shape[0]))
        return new_col, unique_cats


    def _auto_combine_transform(self, test_cols, unique_cats, col_names):
        test_new_cols = self._auto_combine(test_cols, col_names, test_transform=True)
        test_new_cols[~test_new_cols.isin(unique_cats)] = -999
        print('after null ratio control, there are totally {} categories left'.format(test_new_cols.value_counts().shape[0]))
        return test_new_cols

    
class PrePostProcess:
    '''  
    1. 连续变量处理 排序分箱；
    2. null 处理; 造两列空置率；     
    '''   
    
    def _to_bins(self, col, q, labels=None):
        new_col, bins = pd.cut(col, bins=q, labels=labels, retbins=True)
        return new_col, bins

    def _to_bins_trans(self, col, bins, labels=None):
        new_col, bins = pd.cut(col, bins=bins, retbins=True, labels=labels)
        return new_col, bins

    def add_null_rate(self, df, null_val):
        assert 'null_num' not in df.columns
        new_df = pd.DataFrame(index=df.index)
        null_df = new_df==null_val
        new_df['null_num'] = null_df.sum(axis=1)
        new_df['null_rate'] = null_df.sum(axis=1)/null_df.shape[1]
        return new_df
    
    def rank_cut(self, tra, val, cont_vars):
        #  列 rank_cut
        tmp_tra, tmp_val = pd.DataFrame(index=tra.index), pd.DataFrame(index=val.index)
        for col_name in cont_vars:
            a1, bins1 = self._to_bins(tra[col_name], q=10, labels=range(10))
            assert col_name+'_rankcut' not in tra.columns
            tmp_tra[col_name+'_rankcut'] = a1
            b1, _ = self._to_bins_trans(val[col_name], bins=bins1, labels=range(10))
            tmp_val[col_name+'_rankcut'] = b1
        return tmp_tra, tmp_val

        

