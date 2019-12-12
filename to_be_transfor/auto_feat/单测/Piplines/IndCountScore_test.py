import sys

sys.path.append('../')
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from Piplines import CountEnc_test
from multiprocessing import Pool
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold


class IndCountScoreSingle(BaseEstimator, TransformerMixin):
    '''
    - ICS 在fit时候偷看了标签，重用是不允许的，非要重用就无法遵循fit_trans框架，但是AUC是有效的（没偷看）
    '''

    def __init__(self, cols, num_process):  # no *args and **kwargs
        super().__init__()
        self.cols = cols
        self.col_dics = {}
        self.res_dics = {}
        self.num_process = num_process

    @property
    def get_result(self):
        res = pd.DataFrame.from_dict(self.res_dics, 'index', columns=['auc'])
        res = res.sort_values('auc', ascending=False)
        res.index = ['ICS2({})'.format(i) for i in res.index]
        return res

    def auc_impo(self, tra_x, tra_y, val_x, val_y, n_job, params=None):
        print(' train:{}   valid:{}'.format(tra_x.shape, val_x.shape))
        if params is None:
            params = {'task': 'train',
                      'boosting_type': 'gbdt',
                      'objective': 'binary',
                      'metric': 'auc',
                      'num_leaves': 8,  # 31,
                      'learning_rate': 0.01,  # 0.002
                      'feature_fraction': 1,
                      'lambda_l2': 140,
                      'num_threads': n_job}

        cv_train = lgb.Dataset(tra_x, tra_y.astype('int'))
        cv_valid = lgb.Dataset(val_x, val_y.astype('int'))
        gbm = lgb.train(params,  # 参数字典
                        cv_train,  # 训练集
                        num_boost_round=1000,  # 迭代次数
                        valid_sets=[cv_train, cv_valid],  # 验证集
                        early_stopping_rounds=100,
                        verbose_eval=50)
        print('with best auc:{}'.format(gbm.best_score['valid_1']['auc']))
        return gbm

    def _fit(self, x, sub_list, y, skf):
        col_dics = {}
        imp_dics = {}
        if self.num_process is not None:
            n_job = 1
        else:
            n_job = -1
        for var in sub_list:
            col_dics[var] = []
            imp_dics[var] = 0
            for index, (train_index, val_index) in enumerate(skf.split(x, y)):
                tra_x, val_x, tra_y, val_y = x.iloc[train_index], x.iloc[val_index], y.values[train_index], y.values[
                    val_index]
                model = self.auc_impo(tra_x=tra_x[var].values.reshape(-1, 1),
                                      tra_y=tra_y.flatten(),
                                      val_x=val_x[var].values.reshape(-1, 1),
                                      val_y=val_y.flatten(),
                                      n_job=n_job)
                imp_dics[var] += model.best_score['valid_1']['auc'] / skf.n_splits
                col_dics[var].append(model)
        return col_dics, imp_dics

    def fit(self, x, y):
        if self.cols is None:
            self.cols = x.columns
        # self.CountEnc = CountEnc_test.CountEnc(cols=self.cols, test=self.test, thresh=0.008)
        # self.CountEnc.fit(x)
        self.K = 2
        skf = StratifiedKFold(n_splits=self.K, random_state=2019, shuffle=True)
        # x_count = self.CountEnc.transform(x)
        # x = x.join(x_count)
        if self.num_process != None:
            sub_len = len(self.cols) // self.num_process + 1
            self.sub_list = [self.cols[x:x + sub_len] for x in range(0, len(self.cols), sub_len)]
            self.num_process = min(self.num_process, len(self.sub_list))
            print('program is going to use multiprocessing with {} Ps'.format(self.num_process))
            p2 = Pool(self.num_process)
            res = []
            for i in range(self.num_process):
                aa = p2.apply_async(self._fit, args=(x, self.sub_list[i], y, skf))
                res.append(aa)
            p2.close()
            p2.join()
            for i in res:
                self.col_dics.update(i.get()[0])
                self.res_dics.update(i.get()[1])
        else:
            self.col_dics, self.res_dics = self._fit(x, self.cols, y, skf)
        return self

    def transform(self, x):
        if self.num_process != None:
            sub_len = len(self.cols) // self.num_process + 1
            self.sub_list = [self.cols[x:x + sub_len] for x in range(0, len(self.cols), sub_len)]
            self.num_process = min(self.num_process, len(self.sub_list))
            print('program is going to use multiprocessing with {} Ps'.format(self.num_process))
            p2 = Pool(self.num_process)
            rst = []
            for i in range(self.num_process):
                aa = p2.apply_async(self._transform, args=(x, self.sub_list[i]))
                rst.append(aa)
            p2.close()
            p2.join()
            new_x = pd.concat([i.get() for i in rst], axis=1)
            return new_x
        else:
            return self._transform(x, self.cols)

    def _transform(self, x, sub_list):
        DF = pd.DataFrame(index=x.index)
        # x_count = self.CountEnc.transform(x)
        for var in sub_list:
            print(var)
            model_list = self.col_dics[var]
            name = 'ICS2(' + var + ')'
            DF[name] = 0
            for model in model_list:
                DF[name] += model.predict(x[var].values.reshape(-1, 1)) / self.K
        return DF


class IndCountScore(BaseEstimator, TransformerMixin):
    '''
    每列有 原始特征+CountEnc特征, 两个变量对吧？
    - 用两列变量 跑lgb + lgb_predict 得到 200个logit
    - 最终200个logit 综合跑lgb
    '''

    def __init__(self, cols, num_process, count_thresh, count_test):  # no *args and **kwargs
        super().__init__()
        self.cols = cols
        self.col_dics = {}
        self.res_dics = {}
        self.num_process = num_process
        self.count_thresh =count_thresh
        self.count_test = count_test

    @property
    def get_result(self):
        res = pd.DataFrame.from_dict(self.res_dics, 'index', columns=['auc'])
        res = res.sort_values('auc', ascending=False)
        res.index = ['ICS({})'.format(i) for i in res.index]
        return res

    def auc_impo(self, tra_x, tra_y, val_x, val_y, n_job, params=None):
        print(' train:{}   valid:{}'.format(tra_x.shape, val_x.shape))
        if params is None:
            params = {'task': 'train',
                      'boosting_type': 'gbdt',
                      'objective': 'binary',
                      'metric': 'auc',
                      'num_leaves': 8,  # 31,
                      'learning_rate': 0.01,  # 0.002
                      'feature_fraction': 1,
                      'lambda_l2': 140,
                      'num_threads': n_job}

        cv_train = lgb.Dataset(tra_x, tra_y.astype('int'))
        cv_valid = lgb.Dataset(val_x, val_y.astype('int'))
        gbm = lgb.train(params,  # 参数字典
                        cv_train,  # 训练集
                        num_boost_round=1000,  # 迭代次数
                        valid_sets=[cv_train, cv_valid],  # 验证集
                        early_stopping_rounds=100,
                        verbose_eval=50)
        print('with best auc:{}'.format(gbm.best_score['valid_1']['auc']))
        return gbm

    def _fit(self, x, sub_list, y, skf):
        # y = np.array(y).flatten()
        col_dics = {}
        imp_dics = {}
        if self.num_process is not None:
            n_job = 1
        else:
            n_job = self.num_process
        for var in sub_list:
            col_dics[var] = []
            imp_dics[var] = 0
            for index, (train_index, val_index) in enumerate(skf.split(x, y)):
                tra_x, val_x, tra_y, val_y = x.iloc[train_index], x.iloc[val_index], y.values[train_index], y.values[
                    val_index]
                model = self.auc_impo(tra_x=np.hstack(
                    [tra_x[var].values.reshape(-1, 1), tra_x['Count(' + var + ')'].values.reshape(-1, 1)]),
                                      tra_y=tra_y.flatten(),
                                      val_x=np.hstack([val_x[var].values.reshape(-1, 1),
                                                       val_x['Count(' + var + ')'].values.reshape(-1, 1)]),
                                      val_y=val_y.flatten(),
                                      n_job=n_job)
                imp_dics[var] += model.best_score['valid_1']['auc'] / skf.n_splits
                col_dics[var].append(model)
        return col_dics, imp_dics

    def fit(self, x, y):
        if self.cols is None:
            self.cols = x.columns
        self.CountEnc = CountEnc_test.CountEnc(cols=self.cols, thresh=self.count_thresh, test=self.count_test)
        self.CountEnc.fit(x)
        self.K = 2
        skf = StratifiedKFold(n_splits=self.K, random_state=2019, shuffle=True)
        x_count = self.CountEnc.transform(x)
        x = x.join(x_count)
        if self.num_process != None:
            sub_len = len(self.cols) // self.num_process + 1
            self.sub_list = [self.cols[x:x + sub_len] for x in range(0, len(self.cols), sub_len)]
            self.num_process = min(self.num_process, len(self.sub_list))
            print('program is going to use multiprocessing with {} Ps'.format(self.num_process))
            p2 = Pool(self.num_process)
            res = []
            for i in range(self.num_process):
                aa = p2.apply_async(self._fit, args=(x, self.sub_list[i], y, skf))
                res.append(aa)
            p2.close()
            p2.join()
            for i in res:
                self.col_dics.update(i.get()[0])
                self.res_dics.update(i.get()[1])
        else:
            self.col_dics, self.res_dics = self._fit(x, self.cols, y, skf)
        return self

    def transform(self, x):
        if self.num_process != None:
            sub_len = len(self.cols) // self.num_process + 1
            self.sub_list = [self.cols[x:x + sub_len] for x in range(0, len(self.cols), sub_len)]
            self.num_process = min(self.num_process, len(self.sub_list))
            print('program is going to use multiprocessing with {} Ps'.format(self.num_process))
            p2 = Pool(self.num_process)
            rst = []
            for i in range(self.num_process):
                aa = p2.apply_async(self._transform, args=(x, self.sub_list[i]))
                rst.append(aa)
            p2.close()
            p2.join()
            new_x = pd.concat([i.get() for i in rst], axis=1)
            return new_x
        else:
            return self._transform(x, self.cols)

    def _transform(self, x, sub_list):
        DF = pd.DataFrame(index=x.index)
        x_count = self.CountEnc.transform(x)
        for var in sub_list:
            print(var)
            model_list = self.col_dics[var]
            name = 'ICS(' + var + ')'
            DF[name] = 0
            for model in model_list:
                DF[name] += model.predict(np.hstack([x[var].values.reshape(-1, 1),
                                                     x_count['Count(' + var + ')'].values.reshape(-1, 1)])) / self.K
        return DF


if __name__ == '__main__':
    data_path = './data'
    tra_x = pd.read_csv('../data/train.csv', encoding="utf-8").set_index('ID')
    test = pd.read_csv('../data/test.csv', encoding="utf-8").set_index('ID')
    Y = pd.read_csv('../data/train_label.csv', encoding="utf-8").set_index('ID')
    submission = pd.read_csv('../data/submission.csv')
    id_cols = ['企业类型', '登记机关', '行业代码', '行业门类', '企业类别', '管辖机关']
    ICS = IndCountScoreSingle(cols=id_cols, num_process=4)
    tra_rc = ICS.fit_transform(tra_x, Y)
    val_rc = ICS.transform(test)

    ICS2 = IndCountScore(cols=id_cols, num_process=4, count_test=test, count_thresh=0.01)
    tra_rc = ICS2.fit_transform(tra_x, Y)
    val_rc2 = ICS2.transform(test)

    print(val_rc.head())
    print(val_rc2.head())