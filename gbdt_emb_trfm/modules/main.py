# 添加with with torch.no_grad(), model.eval()
# 添加numpy 求均值损失
from modules.data import train_val_test_split, CbData
from modules.layer import TRFM
from modules.auc import roc_auc_compute
import torch
import torch.nn as nn
import numpy as np
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
        self.model = TRFM(num_uniq_leaf=self.num_trees * self.leaf_num_per_tree + 1, dim_leaf_emb=16, num_heads=3,
                          dim_q_k=20, dim_v=20, dim_m=16, layers=3, dropout=0.3).to(device)

        self.criterion = nn.BCELoss()

    def train(self, epoch, batch_size, lr):
        another_train_loader, train_loader, val_loader, test_loader = train_val_test_split(self.full_dataset,
                                                                                           batch_size,
                                                                                           self.pre_defined_idx,
                                                                                           return_idx=False)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        for epoch_id in range(epoch):
            self._train_epoch(epoch_id, train_loader, another_train_loader, val_loader, optimizer)

    def _train_epoch(self, epoch_id, train_loader, another_train_loader, valid_loader, optimizer):
        for batch_idx, data in enumerate(train_loader):
            inputs = data['x'].to(self.device)
            target = data['y'].to(self.device)
            optimizer.zero_grad()
            score = self.model(inputs)
            loss = self.criterion(score, target.float())
            if batch_idx % 10 == 0:
                self._val_epoch(epoch_id, batch_idx, another_train_loader, valid_loader)
            loss.backward()
            optimizer.step()

    def _val_epoch(self, epoch_id, batch_idx, another_train_loader, valid_loader):
        self.model.eval()
        with torch.no_grad():
            Train_Targets = []
            Train_Logits = []
            Train_Loss = []
            for idx, data in enumerate(another_train_loader):
                inputs = data['x'].to(self.device)
                target = data['y'].to(self.device)
                logits = self.model(inputs)
                val_loss = self.criterion(logits, target.float())
                Train_Targets.append(target)
                Train_Logits.append(logits)
                Train_Loss.append(val_loss.tolist())
                if idx == 8:
                    break
            train_loss = np.mean(Train_Loss)
            train_auc = roc_auc_compute(torch.cat(Train_Targets), torch.cat(Train_Logits))

            Targets = []
            Logits = []
            Loss = []
            for idx, data in enumerate(valid_loader):
                inputs = data['x'].to(self.device)
                target = data['y'].to(self.device)
                logits = self.model(inputs)
                val_loss = self.criterion(logits, target.float())
                Targets.append(target)
                Logits.append(logits)
                Loss.append(val_loss.tolist())
                if idx == 8:
                    break
            val_loss = np.mean(Loss)
            val_auc = roc_auc_compute(torch.cat(Targets), torch.cat(Logits))
        print('epoch_num:{} batch_num:{} train_loss:{} train_auc:{} valid:loss:{} valid_auc:{}'.format(epoch_id,
                                                                                                       batch_idx,
                                                                                                       train_loss,
                                                                                                       train_auc,
                                                                                                       val_loss,
                                                                                                       val_auc))

    def predict(self, data_loader):
        pass
