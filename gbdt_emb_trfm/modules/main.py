from modules.data import train_val_test_split, CbData
from modules.layer import TRFM
from modules.auc import roc_auc_compute
from tqdm import tqdm
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")


class GbdtTrfm:

    def __init__(self, gbdt_model, pre_defined_idx, device, root_dir):
        self.gbdt_model = gbdt_model
        self.pre_defined_idx = pre_defined_idx  # 更训练gbdt 时的样本保持一致；相同的train,val,test;
        self.full_dataset = CbData(root_dir, gbdt_model=gbdt_model)
        self.num_trees = self.full_dataset.num_trees
        self.leaf_num_per_tree = self.full_dataset.leaf_num_per_tree
        self.device = device
        self.model = TRFM(num_uniq_leaf=self.num_trees*self.leaf_num_per_tree+1, dim_leaf_emb=32, num_heads=8,
                          dim_q_k=100, dim_v=200, dim_m=32, layers=6, dropout=0.3).to(device)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epoch, batch_size, lr):
        torch.backends.cudnn.enabled = False  # RAM leak;
        full_train_loader, train_loader, val_loader, test_loader = train_val_test_split(self.full_dataset,
                                                                                        batch_size,
                                                                                        self.pre_defined_idx,
                                                                                        return_idx=False)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        for epoch_id in range(epoch):
            self._train_epoch(train_loader, optimizer)
            self._val_epoch(epoch_id, full_train_loader, val_loader)

    def _val_epoch(self, epoch_id, full_train_loader, valid_loader):
        train_auc, train_loss, val_auc, val_loss = None, None, None, None
        for data in tqdm(full_train_loader):
            inputs = data['x'].to(self.device)
            target = data['y'].to(self.device)
            logits = self.model(inputs)
            train_loss = self.criterion(logits, target)
            train_auc = roc_auc_compute(target, logits)

        for data in tqdm(valid_loader):
            inputs = data['x'].to(self.device)
            target = data['y'].to(self.device)
            logits = self.model(inputs)
            val_loss = self.criterion(logits, target)
            val_auc = roc_auc_compute(target, logits)
            print('epoch_num:{} train_loss:{} train_auc:{} valid:loss:{} valid_auc:{}'.format(epoch_id, train_loss,
                                                                                              train_auc, val_loss,
                                                                                              val_auc))

    def _train_epoch(self, train_loader, optimizer):
        epoch_loss = []
        for data in tqdm(train_loader):
            inputs = data['x'].to(self.device)
            target = data['y'].to(self.device)
            optimizer.zero_grad()
            score = self.model(inputs)
            loss = self.criterion(score, target)
            # print('epoch_num:{} batch_num:{} batch_loss:{}'.format(epoch_idx, ii, loss))
            epoch_loss.append(loss)
            loss.backward()
            optimizer.step()

    def predict(self, data_loader):
        pass
