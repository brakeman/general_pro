import sys 
sys.path.append('../../')
import pandas as pd
import random
from auto_feat.AutoFeat import PrePostProcess, autofeat_draft

def dummy_lgb_topk(tra_x, tra_y, val_x, val_y, col_names_list, max_unique=3000, top_k=200):
    # 每囤够max_unique个dummy 就跑一次 lgb top k 
    tmp_tra_df = pd.DataFrame(index = tra_x.index)
    tmp_val_df = pd.DataFrame(index = val_x.index)
    final_tra, final_val, unique_num = [], [], 0
    for idx, col in enumerate(col_names_list):
        print('dummy column: {}/{}'.format(idx, len(col_names_list)))
        unique_num += len(tra_x[col].unique())
        if unique_num <= max_unique:
            tmp_tra_df = pd.get_dummies(tra_x[col], prefix=col+'_').join(tmp_tra_df)
            tmp_val_df = pd.get_dummies(val_x[col], prefix=col+'_').join(tmp_val_df)
        else:
            print('dummy reach: {}/{}, take only: {} columns'.format(unique_num, max_unique, top_k))
            tra_1, val_1, col_names, gbm = FFF._auto_topk(tmp_tra_df, tra_y, tmp_val_df, val_y, top_k)
            final_tra.append(tra_1)
            final_val.append(val_1)
            unique_num = 0
            tmp_tra_df = pd.DataFrame(index = tra_x.index)
            tmp_val_df = pd.DataFrame(index = val_x.index)
    if len(final_tra)==0:
        tra_1, val_1, col_names, gbm = FFF._auto_topk(tmp_tra_df, tra_y, tmp_val_df, val_y, top_k)
        final_tra.append(tra_1)
        final_val.append(val_1)
    return pd.concat(final_tra, axis=1), pd.concat(final_val, axis=1)


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

    cols = random.sample(tra_x.columns.tolist(), 20)
    FFF = autofeat_draft(global_id='id')
    a, b = FFF.k_order_comb(tra_x, val_x, column_names=cols, order=2)
    