import numpy as np
import pandas as pd
import random
import itertools
ALL =  ['count_enc', 'label_enc', 'null_enc', 'merge_box', '_merge_box', 'auto_combine', 'k_order_comb']


def label_enc(Ser):
    dic = {k:v for v, k in dict(enumerate(Ser.unique())).items()}
    return Ser.map(dic)

def count_enc(Ser):
    return Ser.map(Ser.value_counts().to_dict())

def null_enc(Ser, null_val):
    return (Ser == null_val)*1

# def one_hot_topk_enc(Ser, )
 

# 隐患是tra, val 可能保留不同的 dummy
# 方案是val 中出现了tra 中没出现的就直接-888
def merge_box(tra, val, columns, thresh):
    
    '''
    group merge big cats columns
    tra: train df
    val: valid df
    columns: list(str)
    thresh: 尾部类别被忽略为-888
    '''
    tra_df, val_df = pd.DataFrame(index=tra.index), pd.DataFrame(index=val.index)
    for col in columns:
        name = 'mb('+col+')'
        tra_tmp, val_tmp = _merge_box(tra, val, col, thresh)
        tra_df[name], val_df[name] = tra_tmp, val_tmp
    return tra_df, val_df


def _merge_box(tra, val, column, thresh):
    '''
    merge a single column with big cats;
    tra: train df
    val: valid df
    column: str
    thresh: 尾部类别被忽略为-888
    '''
    assert isinstance(tra, pd.DataFrame) and isinstance(val, pd.DataFrame)
    tra_df, val_df = tra[column].copy(), val[column].copy()
    tmp = tra_df.value_counts(normalize=True).cumsum()
    drop_cats = tmp[np.greater(tmp, thresh)].index.tolist()
    if len(drop_cats)==len(tra[column].unique()):
        drop_cats = tmp[np.greater(tmp, thresh)].index[1:]
    # 隐患是tra, val 可能保留不同的 dummy
    drop_tes = list((set(val_df.unique()) - set(tra_df.unique())).union(set(drop_cats)))
    tra_df[tra_df.isin(drop_cats)] = -888
    val_df[val_df.isin(drop_tes)] = -888
    return tra_df, val_df


# 优化空间： 算3阶的时候，重复计算了2阶;
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

