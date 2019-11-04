import pandas as pd
import numpy as np


def _auto_combine(cols, col_names, max_limit = 10000, null_rate=0.66, fill='null', test_transform=False):
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
        print('generated new column:{}\n.................with unique category length:{}\n'.format(name, unique_cat))
    new_col = pd.Series(val, name=name)
    if test_transform:
        return new_col

    tmp = new_col.value_counts(normalize=True).cumsum()
    unique_cats = tmp[np.less(tmp, null_rate)].index.tolist()
    new_col[~new_col.isin(unique_cats)] = fill
    print('after null ratio control, there are totally {} categories left'.format(new_col.value_counts().shape[0]))
    return new_col, unique_cats


def _auto_combine_transform(test_cols, unique_cats, col_names, fill):
    test_new_cols = _auto_combine(test_cols, col_names, test_transform=True)
    test_new_cols[~test_new_cols.isin(unique_cats)] = fill
    print('after null ratio control, there are totally {} categories left'.format(test_new_cols.value_counts().shape[0]))
    return test_new_cols


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
    
    tes_col = ['gender', 'edu', 'job', 'dist']
    col_names = tes_col
    cols = [tra_x[i] for i in col_names]
    test_cols = [val_x[i] for i in col_names]
    train_col, uniq = _auto_combine(cols, col_names)
    test_new = _auto_combine_transform(test_cols, uniq, col_names, fill='null')