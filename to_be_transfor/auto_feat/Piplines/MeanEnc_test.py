import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from itertools import product
# import ipdb
 
class MeanEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, categorical_features, n_splits=5):
        """
        categorical_features: list of str, the name of the categorical columns to encode
        n_splits: the number of splits used in mean encoding
        """
        super().__init__()
        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}
        self.target_values = []
        self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))
 

    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()
#         ipdb.set_trace()
        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            X_train['pred_temp'] = (y_train == target).astype(int)  # classification
        prior = X_train['pred_temp'].mean()
        col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg({'mean': 'mean', 'beta': 'size'})
        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
        col_avg_y[nf_name] = col_avg_y['beta'] * prior + (1 - col_avg_y['beta']) * col_avg_y['mean']
        col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)
        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values
        return nf_train, nf_test, prior, col_avg_y
 
    def fit(self, X, y):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :param y: pandas Series or numpy array, n_samples
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
        skf = StratifiedKFold(self.n_splits)

        self.target_values = sorted(set(y))
        self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
                              product(self.categorical_features, self.target_values)}
        for variable, target in product(self.categorical_features, self.target_values):
            nf_name = '{}_pred_{}'.format(variable, target)
            X_new.loc[:, nf_name] = np.nan
            for large_ind, small_ind in skf.split(y, y):
                nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                    X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target, self.prior_weight_func)
                X_new.iloc[small_ind, -1] = nf_small
                self.learned_stats[nf_name].append((prior, col_avg_y))
        return self
 
    def transform(self, X):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
        for variable, target in product(self.categorical_features, self.target_values):
            nf_name = '{}_pred_{}'.format(variable, target)
            X_new[nf_name] = 0
            for prior, col_avg_y in self.learned_stats[nf_name]:
                X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                    nf_name]
            X_new[nf_name] /= self.n_splits
        return X_new

if __name__ == '__main__':
    data_path = './data/'
    # test.csv  train.csv  train_target.csv
    tra_x = pd.read_csv(data_path + '/train.csv')
    tra_y = pd.read_csv(data_path + '/train_target.csv')
    tes_x = pd.read_csv(data_path + '/test.csv')
    final = tra_x.merge(tra_y,on='id')

    final['certValidStop'] = final.certValidStop.astype(int)
    final.fillna(-999,inplace=True)

    tra_id = set(random.sample(range(final.shape[0]), 70000))
    val_id = set(range(final.shape[0])) - tra_id
    tra_id = [i for i in tra_id]
    val_id = [i for i in val_id]
    Train = final.iloc[tra_id,:].set_index(keys='id')
    Valid = final.iloc[val_id,:].set_index(keys='id')
    tra_x, tra_y = Train.drop('target', axis=1), Train.target
    val_x, val_y = Valid.drop('target', axis=1), Valid.target
    ME = MeanEncoder(categorical_features=['certId'])
    ME.fit(tra_x, tra_y)
    ME.transform(tra_x)