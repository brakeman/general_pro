import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def train_val_test_split(dataset, bs_train, val_split=0.8, test_split=0.9, shuffle_dataset=True, random_seed=666):
    '''pytorch best data split and return loader object'''
    assert test_split > val_split
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split1 = int(val_split*dataset_size)
    split2 = int(test_split*dataset_size)
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_idx, val_idx, test_idx = indices[:split1], indices[split1:split2], indices[split2:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    bs_val = len(val_idx)
    bs_test = len(test_idx)
    train_loader = torch.utils.data.DataLoader(dataset, bs_train, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, bs_val, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, bs_test, sampler=test_sampler)
    return train_loader, val_loader, test_loader


class CbData(Dataset):

    def __init__(self, root_dir, num_trees, leaf_num_per_tree, gbdt_model, transform=None):
        """
        Args:
            root_dir (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.x = np.load(root_dir + '/data.npy')
        self.y = pd.read_csv(root_dir + '/final_label.csv')
        self.y.index = range(len(self.y))
        self.keep_rows = self.y[~self.y.default_20.isnull()].index.tolist()
        self.x = self.x[self.keep_rows, :-1]

        x = gbdt_model.predict(self.x, pred_leaf=True)  # [bs, 4000 trees]
        to_add = np.arange(0, leaf_num_per_tree * num_trees, step=leaf_num_per_tree)
        x = x + to_add  # 【bs, 4000】叶子序号
        to_concat = np.ones(len(x))[:, np.newaxis] * leaf_num_per_tree * num_trees  # [CLS]编码
        self.x = np.concatenate((to_concat, x), axis=1).astype(int)

        self.y = self.y.values[self.keep_rows, -1]
        self.transform = transform  # gbdt_model
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


if __name__ == '__main__':
    cb = CbData(root_dir='...', )
    loader = DataLoader(dataset=cb, batch_size=10000)
    for batch_ndx, sampl in enumerate(loader):
        print(sampl['y'])