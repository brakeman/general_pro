import pandas as pd
import numpy as np
import random
import itertools
import sys
sys.path.append('../../')
from utils import auto_combine


def k_order_comb(tra, val, column_names, order, merge_tail, thresh):
    '''
    自动做k阶特征组合；
    -----------------------------------------
    tra: train_df;
    val: valid_df;
    column_names: list(str); 从这里挑选k阶排列组合;
    order: k阶特征组合;
    merge_tail: bool; 是否合并每个 新生成*组合列* 尾部的微小类别；
    thresh: float; 如果merge_tail, 则需要给出阈值；
    -----------------------------------------
    return:
        tra: df;
        val: df;
    '''
    DF_tra, DF_val = pd.DataFrame(index=tra.index), pd.DataFrame(index=val.index)
    col_names_list = [i for i in itertools.combinations(column_names, order)]
    for col_names in col_names_list:
        tra_new, val_new = auto_combine(tra, val, list(col_names), merge_tail, thresh)
        DF_tra[tra_new.name] = tra_new
        DF_val[tra_new.name] = val_new
    print('after {} order feature combination, there are totally {} new features generated'.format(order, DF_tra.shape[1]))
    return DF_tra, DF_val



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
    a, b = k_order_comb(tra_x, val_x, tes_col, 2, merge_tail=True, thresh=0.95)