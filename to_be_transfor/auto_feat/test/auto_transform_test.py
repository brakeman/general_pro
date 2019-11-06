import pandas as pd
import random 
import itertools
from functools import reduce

def auto_transform(tra, val, column_names, order=2, op='add'):
    # 确认输入是否都是cat
    assert op in ['add', 'multiply', 'sub']
#     assert tra.index.name == self.global_id == val.index.name
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
    tra, val = auto_transform(tra_x, val_x, column_names = ['gender' ,'age', 'edu','basicLevel','x_0','x_1','x_2'], order=3, op='multiply')
