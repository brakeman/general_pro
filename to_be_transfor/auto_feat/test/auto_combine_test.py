import pandas as pd
import numpy as np
import random
import sys
sys.path.append('../../')
from utils import _merge_box


def auto_combine(tra, val, col_names, merge_tail, thresh):
    '''
    多种类别形 直接 字符串相加 变成新列;
    -----------------------------------------
    tra: train_df;
    val: valid_df;
    col_names: list(str);
    merge_tail: bool; 是否合并每列尾部的微小类别；
    thresh: float; 如果merge_tail, 则需要给出阈值；
    -----------------------------------------
    return:
        tra: pd.Series;
        val: pd.Series;
    '''
    tra_tmp, val_tmp = pd.DataFrame(index=tra.index), pd.DataFrame(index=val.index)
    new_name = 'comb('+','.join(col_names)+')'
    tra_tmp[new_name] = (tra[col_names].astype(str)+'|').sum(axis=1)
    val_tmp[new_name] = (val[col_names].astype(str)+'|').sum(axis=1)
    if merge_tail:
        return _merge_box(tra_tmp, val_tmp, new_name, thresh)
    else:
        drop_tes = list(set(val_tmp[new_name].unique()) - set(tra_tmp[new_name].unique()))
        val_tmp[new_name][val_tmp[new_name].isin(drop_tes)] = -888        
        return tra_tmp[new_name], val_tmp[new_name]


if __name__ == '__main__':
    data_path = './data/'
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
    
    tes_col = ['gender', 'edu', 'job']
    a, b = auto_combine(tra_x, val_x, col_names=tes_col, merge_tail=True, thresh=0.95)