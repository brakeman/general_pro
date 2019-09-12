import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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