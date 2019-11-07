import pandas as pd
import random
import lightgbm as lgb

def auc_impo(tra_x, tra_y, val_x, val_y, params=None):
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
    gbm = lgb.train(params,                     # 参数字典
                    cv_train,                  # 训练集
                    num_boost_round=2000,       # 迭代次数
                    valid_sets=cv_valid,        # 验证集
                    early_stopping_rounds = 100,
                    verbose_eval=0)
    print('with best auc:{}'.format(gbm.best_score['training']['auc']))
    return gbm

def topk_feat(tra_x, gbm, k):
    z = pd.DataFrame()
    z['col'] = tra_x.columns
    z['imp'] = gbm.feature_importance() 
    col_names = z.nlargest(k, 'imp').col.tolist()
    return tra_x[col_names], col_names
    
def topk_feat_transfrom(val_x, col_names):
    return val_x[col_names]

def auto_topk(tra_x, tra_y, val_x, val_y):
    gbm = auc_impo(tra_x, tra_y, val_x, val_y)
    tra_1, col_names = topk_feat(tra_x, gbm, 20)
    val_1 = topk_feat_transfrom(val_x, col_names)
    return tra_1, val_1, col_names, gbm

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
    tra_1, val_1, col_names, gbm= auto_topk(tra_x, tra_y, val_x, val_y)
    assert tra_1.columns.tolist() == val_1.columns.tolist()
