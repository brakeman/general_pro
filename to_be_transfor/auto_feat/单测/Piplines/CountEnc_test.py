from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class CountEnc(BaseEstimator, TransformerMixin):
    '''
    为适应比赛，直接append(test)
    '''
    def __init__(self, cols, test, thresh):  # no *args and **kwargs
        super().__init__()
        self.cols = cols
        self.col_dics = {}
        self.test = test
        self.thresh = thresh

    def fit(self, x, y=None):
        self.col_dics = {}
        if self.cols is None:
            self.cols = x.columns.tolist()
        x = x[self.cols].append(self.test[self.cols])
        x_len = len(x)
        thresh_cnt = int(x_len * self.thresh)
        for col in self.cols:
            tmp_val_counts = x[col].value_counts()
            tmp_val_counts = tmp_val_counts[tmp_val_counts > thresh_cnt]
            self.col_dics[col] = tmp_val_counts
        return self

    def transform(self, x):
        df = pd.DataFrame()
        for col in self.cols:
            if col not in self.col_dics:
                raise Exception('col:{} not in col_dics'.format(col))
            col_dic = self.col_dics[col]
            new_name1 = 'Count(' + col + ')'
            df[new_name1] = x[col].map(col_dic)
        return df


if __name__ == '__main__':
    data_path = './data'
    tra_x = pd.read_csv('../data/train.csv', encoding="utf-8").set_index('ID')
    test = pd.read_csv('../data/test.csv', encoding="utf-8").set_index('ID')
    Y = pd.read_csv('../data/train_label.csv', encoding="utf-8").set_index('ID')
    submission = pd.read_csv('../data/submission.csv')
    id_cols = ['企业类型', '登记机关']
    Count = CountEnc(cols=id_cols, test=test[id_cols], thresh=0.05)
    tra_rc = Count.fit_transform(tra_x)
    val_rc = Count.transform(test)
    print(val_rc.head())