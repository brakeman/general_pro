# 1. 修改了 val_loader & test_loader 的 batch size, 因为一个中间变量是 bs*ts*F 这么大，如果full batch 去评估，5000*5000*16 这个中间变量太大了; 不可忍受；
# 2. 修改了another_train_loader， 避免使用full_train_loader, 同样理由
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random


def get_split_idx(all_valid_idx, val_split, test_split, shuffle=True):
    '''
    :param all_valid_idx: 有效idx, 在保证 数据源x, y 的idx一一对应前提下，只取过滤掉nan后的样本idx；
    :param val_split: 切分点，前面代表train;
    :param test_split: 切分点，前面代表valid; 后面代表test;
    :param shuffle: 打乱index 后再切割;
    :return: train_idx, val_idx, test_idx;
    '''

    assert test_split > val_split
    dataset_size = len(all_valid_idx)
    indices = all_valid_idx
    split1 = int(val_split * dataset_size)
    split2 = int(test_split * dataset_size)
    if shuffle:
        random.shuffle(indices)
    train_idx, val_idx, test_idx = indices[:split1], indices[split1:split2], indices[split2:]
    print('len_train:{}, len_val:{}, len_test:{}'.format(len(train_idx), len(val_idx), len(test_idx)))
    return train_idx, val_idx, test_idx


class CbDataNew(Dataset):

    def __init__(self, root_dir, gbdt_model, data_idx, add_cls=False, transform=None):
        """
        Args:
            root_dir (string): Directory with all the data.
            gbdt_model: a lgb model.
            data_idx: 有效样本idx.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        x = np.load(root_dir + '/data.npy')
        y = pd.read_csv(root_dir + '/really_final_label_with_x_order.csv')
        y.columns = ['loanid', 'label']

        self.data_idx = data_idx
        x = gbdt_model.predict(x[data_idx, :-7], pred_leaf=True)  # [bs, 4000 trees]
        _, num_trees = x.shape
        leaf_num_per_tree = len(np.unique(x))
        self.num_unique_leaf = leaf_num_per_tree * num_trees
        to_add = np.arange(0, self.num_unique_leaf, step=leaf_num_per_tree)
        x = x + to_add  # 【bs, 4000】叶子序号
        if add_cls:
            to_concat = np.ones(len(x))[:, np.newaxis] * leaf_num_per_tree * num_trees  # [CLS]编码
            self.x = np.concatenate((to_concat, x), axis=1).astype(int)
        else:
            self.x = x
        self.y = y.iloc[data_idx].label.values
        self.transform = transform
        self.leaf_num_per_tree = leaf_num_per_tree
        self.num_trees = num_trees
        print('x shape:{}, num_trees:{}, leaf_num_per_tree:{}'.format(self.x.shape, num_trees, leaf_num_per_tree))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        sample = {'x': x, 'y': y}
        return sample
