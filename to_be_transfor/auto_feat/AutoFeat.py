import pandas as pd
import numpy as np
import random
import itertools
import lightgbm as lgb
from functools import reduce
from utils import merge_box
import sys
sys.path.append('../')
from utils import _merge_box, k_order_comb
import itertools

class autofeat_draft:
    '''
    1.  k_order_comb generation 特征组合生成
    1.5. merge_box 组合特征后可以接一下merge_box
    2.  dummy_lgb_topk 特征降维
    3.  auto_transform 小微特征加减乘 生成
    '''
    def __init__(self, global_id = 'id'):
        self.global_id = global_id
    
    def k_order_comb(self, tra, val, column_names, order, merge_tail, thresh):
        assert tra.index.name == self.global_id
        return k_order_comb(tra, val, column_names, order, merge_tail, thresh)
    
    def merge_box(self, tra, val, columns, thresh):
        return merge_box(tra, val, columns, thresh)
    
    
    def dummy_lgb_topk(self, tra_x, tra_y, val_x, val_y, col_names_list, max_unique, top_k):
        # 每囤够max_unique个dummy 就跑一次 lgb top k 
        tmp_tra_df, tmp_val_df = pd.DataFrame(index = tra_x.index), pd.DataFrame(index = val_x.index)
        final_tra, final_val, unique_num = [], [], 0
        for idx, col in enumerate(col_names_list):
            print('dummy column: {}/{}'.format(idx, len(col_names_list)))
            unique_num += len(tra_x[col].unique())
            if unique_num <= max_unique + top_k*idx: # 因为是滚动式的，每提取出一次 dummy_lgb_topk 就在此基础上；
                tmp_tra_df = pd.get_dummies(tra_x[col], prefix=col+'_').join(tmp_tra_df)
                tmp_val_df = pd.get_dummies(val_x[col], prefix=col+'_').join(tmp_val_df)
                tmp_tra_df = tmp_tra_df[tmp_val_df.columns] # 防止 tra, val 列数因为dummy 不一致
            else:
                print('dummy reach: {}/{}, take only: {} columns'.format(unique_num, max_unique, top_k))
                tra_1, val_1, col_names, gbm = self._auto_topk(tmp_tra_df, tra_y, tmp_val_df, val_y, top_k)
                final_tra.append(tra_1)
                final_val.append(val_1)
                unique_num = 0
                tmp_tra_df = tra_1
                tmp_val_df = val_1
                assert tmp_val_df.shape[1] == top_k
        if len(final_tra)==0:
            tra_1, val_1, col_names, gbm = self._auto_topk(tmp_tra_df, tra_y, tmp_val_df, val_y, top_k)
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
            new_name = '{}('.format(op) + ','.join(col_names) + ')'
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

    
    def _auto_topk(self, tra_x, tra_y, val_x, val_y, top_k):
        gbm = self._auc_impo(tra_x, tra_y, val_x, val_y)
        tra_1, col_names = self._topk_feat(tra_x, gbm, top_k)
        val_1 = self._topk_feat_transfrom(val_x, col_names)
        return tra_1, val_1, col_names, gbm
    
    def _auc_impo(self, tra_x, tra_y, val_x, val_y, params=None, verbose=0):
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
                  'bagging_freq': 5}

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

    def _topk_feat(self, tra_x, gbm, k, importance_type='gain'):
        assert importance_type in ['split', 'gain']
        z = pd.DataFrame()
        z['col'] = tra_x.columns
        z['imp'] = gbm.feature_importance(importance_type) 
        col_names = z.nlargest(k, 'imp').col.tolist()
        return tra_x[col_names], col_names

    def _topk_feat_transfrom(self, val_x, col_names):
        return val_x[col_names]


    
class PrePostProcess:
    '''  
    1. rank_cut 排序分箱；
    2. null 处理; 造两列空置率； 
    3. to be done: 根据分布一致性删除列；
    4. merge_box: 合箱子；
    '''   
    
    def group_simi_cols(self, tra_x, val_x, simi_cols):
        tra_all, val_all = [], []
        for i in simi_cols:
            tra_tmp, val_tmp= self.auto_transform(tra_x, val_x, column_names=list(i), order=2, op='sub')
            tra_all.append(tra_tmp)
            val_all.append(val_tmp)
        return pd.concat(tra_all, axis=1), pd.concat(val_all, axis=1)
    
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
    

    def merge_box(self, tra, val, columns, thresh):
        return merge_box(tra, val, columns, thresh)

    def auto_transform(self, tra, val, column_names, order=2, op='add'):
        # 确认输入是否都是cat
        assert op in ['add', 'multiply', 'sub']
    #     assert tra.index.name == self.global_id == val.index.name
        DF_tra, DF_val = pd.DataFrame(index=tra.index), pd.DataFrame(index=val.index)
        col_names_list = [i for i in itertools.combinations(column_names, order)]
        for col_names in col_names_list:
            tra_cols = [tra[i] for i in col_names]
            val_cols = [val[i] for i in col_names]
            new_name = '{}('.format(op) + ','.join(col_names) + ')'
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
