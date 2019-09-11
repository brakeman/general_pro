import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CbData(Dataset):

    def __init__(self, root_dir, transform=None):
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
        self.y = self.y.values[self.keep_rows, -1]
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.x[idx]
        y = self.y[idx]
        sample = {'x': x, 'y': y}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    cb = CbData(root_dir='...')
    loader = DataLoader(dataset=cb, batch_size=10000)
    for batch_ndx, sampl in enumerate(loader):
        print(sampl['y'])