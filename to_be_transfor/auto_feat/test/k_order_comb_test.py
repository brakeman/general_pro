import pandas as pd
import numpy as np
import itertools
import random
import sys
sys.path.append('../../')
from auto_feat.AutoFeat import _auto_combine, _auto_combine_transform


def k_order_comb(tra, val, column_names, order=2):
    # 确认输入是否都是cat
    DF_tra, DF_val = pd.DataFrame(index=tra.index), pd.DataFrame(index=val.index)
    col_names_list = [i for i in itertools.combinations(column_names, order)]
#     print(col_names_list)
    for col_names in col_names_list:
        tra_cols = [tra[i] for i in col_names]
        val_cols = [val[i] for i in col_names]
        tra_new, uniq = _auto_combine(tra_cols, col_names)
        val_new = _auto_combine_transform(val_cols, uniq, col_names)
        DF_tra[tra_new.name] = tra_new
        DF_val[tra_new.name] = val_new
    print('after {} order feature combination, there are totally {} new features generated'.format(order, DF_tra.shape[1]))
    return DF_tra, DF_val
    

if __name__ == '__main__':
    data_path = '/home/chenxiaotian/Projects/xiamen_match/data'
    # test.csv  train.csv  train_target.csv
    tra_x = pd.read_csv(data_path + '/train.csv')
    tra_y = pd.read_csv(data_path + '/train_target.csv')
    final = tra_x.merge(tra_y,on='id')
    final['dist']= final.dist.apply(lambda x: int(x/100))

    tra_id = set(random.sample(range(final.shape[0]),70000))
    val_id = set(range(final.shape[0])) - tra_id
    tra_id = [i for i in tra_id]
    val_id = [i for i in val_id]
    Train = final.iloc[tra_id,:]
    Valid = final.iloc[val_id,:]
    tra_x, tra_y = Train.drop('target', axis=1), Train.target
    val_x, val_y = Valid.drop('target', axis=1), Valid.target

    DF_tra, DF_val = k_order_comb(tra_x, val_x, column_names=tra_x.columns[4:8], order=2)
    DF_tra2, DF_val2 = k_order_comb(tra_x, val_x, column_names=tra_x.columns[4:8], order=3)