import glob
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
import time
import numpy as np

# 过滤nan
non_list = label[label.default_20.isnull()].index.tolist()
save_list = label[~label.default_20.isnull()].index.tolist()

# 实验用小数据；
temp_train_list = save_list[:1000]
temp_valid_list = save_list[1000:1500]
temp_test_list = save_list[1500:2000]


Final_x = feat[temp_train_list,:-1]
Final_y = label.values[temp_train_list,-1]

val_x = feat[temp_valid_list,:-1]
val_y = label.values[temp_valid_list,-1]

cv_train = lgb.Dataset(Final_x, Final_y.astype('int'))
cv_valid = lgb.Dataset(val_x, val_y.astype('int'))

params = {'task': 'train',
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'auc',
          'num_threads': 10,
          'num_leaves': 4,  # 31,
          'learning_rate': 0.002,  # 0.002
          'feature_fraction': 0.5,
          'lambda_l2': 150,
          'bagging_fraction': 0.5,
          'bagging_freq': 5}


print('Start training...')
t0 = time.time()
gbm = lgb.train(params,
    cv_train,
    num_boost_round=1000,
    valid_sets=[cv_train, cv_valid],
#     early_stopping_rounds=5000,
    verbose_eval=100
    )
t1 = time.time()
print('total time:', t1 - t0)


train_leaf = gbm.predict(Final_x, pred_leaf=True)
val_leaf = gbm.predict(val_x, pred_leaf=True)