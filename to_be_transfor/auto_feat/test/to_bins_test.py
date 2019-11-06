import pandas as pd
import random

def to_bins(col, q, labels=None):
    new_col, bins = pd.cut(col, bins=q, labels=labels, retbins=True)
    return new_col, bins

def to_bins_trans(col, bins, labels=None):
    new_col, bins = pd.cut(col, bins=bins, retbins=True, labels=labels)
    return new_col, bins

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
    
    new_col, bins = to_bins(tra_x.lmt, q=6,  labels=range(6))
    test_trans_col, same_bins = to_bins_trans(val_x.lmt, bins=bins,  labels=range(6))