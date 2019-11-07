import numpy as np
import pandas as pd
import random

def _merge_box(tra, val, column, thresh):
    tra_df, val_df = tra[column].copy(), val[column].copy()
    tmp = tra_df.value_counts(normalize=True).cumsum()
    drop_cats = tmp[np.greater(tmp, thresh)].index.tolist()
    if len(drop_cats)==len(tra[column].unique()):
        drop_cats = tmp[np.greater(tmp, thresh)].index[1:]
    tra_df[tra_df.isin(drop_cats)] = -888
    val_df[val_df.isin(drop_cats)] = -888
    return tra_df, val_df

def merge_box(tra, val, columns, thresh):
    tra_df, val_df = pd.DataFrame(index=tra.index), pd.DataFrame(index=val.index)
    for col in columns:
        name = 'mb('+col+')'
        tra_tmp, val_tmp = _merge_box(tra, val, col, thresh)
        tra_df[name], val_df[name] = tra_tmp, val_tmp
    return tra_df, val_df

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
    tra_x, tra_y = Train.drop('target', axis=1).set_index(keys='id'), Train[['id','target']].set_index(keys='id')
    val_x, val_y = Valid.drop('target', axis=1).set_index(keys='id'), Valid[['id','target']].set_index(keys='id')

    tra_df, val_df =  merge_box(tra_x, val_x, columns=mid_cats, thresh=0.92)
