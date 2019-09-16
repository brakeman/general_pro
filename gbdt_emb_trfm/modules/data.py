# 1. 修改了 val_loader & test_loader 的 batch size, 因为一个中间变量是 bs*ts*F 这么大，如果full batch 去评估，5000*5000*16 这个中间变量太大了; 不可忍受；
# 2. 修改了another_train_loader， 避免使用full_train_loader, 同样理由；
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler


def get_split_idx(dataset, val_split, test_split, random_seed, shuffle_dataset=True):
    '''
    :param dataset: could be torch dataset or numpy array; [samples, features]
    :param val_split: 切分点，前面代表train;
    :param test_split: 切分点，前面代表valid; 后面代表test;
    :param random_seed: for shuffle
    :param shuffle_dataset: default True
    :return: train_idx, val_idx, test_idx;
    '''
    assert test_split > val_split
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split1 = int(val_split * dataset_size)
    split2 = int(test_split * dataset_size)
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_idx, val_idx, test_idx = indices[:split1], indices[split1:split2], indices[split2:]
    print('len_train:{}, len_val:{}, len_test:{}'.format(len(train_idx), len(val_idx), len(test_idx)))
    return train_idx, val_idx, test_idx


def train_val_test_split(dataset, bs_train, pre_defined_idx, val_split=0.8, test_split=0.9,
                         shuffle_dataset=True, random_seed=666, return_idx=True):
    '''
    :param dataset: pytorch dataset object;
    :param bs_train: training batch size
    :param pre_defined_idx: a tuple as (train_idx, val_idx, test_idx), select which part of dataset as train/val/test
    :param val_split: train split ratio like 0.8 means 80% as train
    :param test_split: valid split ratio like 0.9 means 10% as valid as 10% as test;
    :param shuffle_dataset: if True, dataset will be shuffled, only works when pre_defined_idx is None;
    :param random_seed: using for shuffle;
    :param return_idx: if true, return a tuple as (train_idx, val_idx, test_idx)
    :return: if return_idx is False:full_train_loader, train_loader, val_loader, test_loader;
            else return full_train_loader, train_loader, val_loader, test_loader, (train_idx, val_idx, test_idx)
    '''

    if pre_defined_idx is None:
        train_idx, val_idx, test_idx = get_split_idx(dataset,
                                                     shuffle_dataset=shuffle_dataset,
                                                     val_split=val_split,
                                                     test_split=test_split,
                                                     random_seed=random_seed)
    else:
        train_idx, val_idx, test_idx = pre_defined_idx

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_loader = torch.utils.data.DataLoader(dataset, bs_train, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, bs_train, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, bs_train, sampler=test_sampler)
    another_train_loader = torch.utils.data.DataLoader(dataset, bs_train, sampler=train_sampler)
    print('len_train:{}, len_val:{}, len_test:{}'.format(len(train_idx), len(val_idx), len(test_idx)))
    if return_idx:
        return another_train_loader, train_loader, val_loader, test_loader, (train_idx, val_idx, test_idx)
    return another_train_loader, train_loader, val_loader, test_loader


class CbData(Dataset):

    def __init__(self, root_dir, gbdt_model, transform=None):
        """
        Args:
            root_dir (string): Directory with all the data.
            gbdt_model: a lgb model.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super()
        self.root_dir = root_dir
        self.x = np.load(root_dir + '/data.npy')
        self.y = pd.read_csv(root_dir + '/final_label.csv')
        self.y.index = range(len(self.y))
        self.keep_rows = self.y[~self.y.default_20.isnull()].index.tolist()
        self.x = self.x[self.keep_rows, :-1]

        x = gbdt_model.predict(self.x, pred_leaf=True)  # [bs, 4000 trees]
        _, num_trees = x.shape
        leaf_num_per_tree = len(np.unique(x))
        to_add = np.arange(0, leaf_num_per_tree * num_trees, step=leaf_num_per_tree)
        x = x + to_add  # 【bs, 4000】叶子序号
        to_concat = np.ones(len(x))[:, np.newaxis] * leaf_num_per_tree * num_trees  # [CLS]编码
        self.x = np.concatenate((to_concat, x), axis=1).astype(int)
        self.y = self.y.values[self.keep_rows, -1]
        self.transform = transform
        self.leaf_num_per_tree = leaf_num_per_tree
        self.num_trees = num_trees

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
