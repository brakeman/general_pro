from FM.data import CbData, train_val_test_split
from FM.layers import WideDeep, DeepFM, NFM, DeepCross
from FM.utils import roc_auc_compute
import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class Main:

    def __init__(self, gbdt_model, pre_defined_idx, device, root_dir, emb_size, model='WideDeep'):
        assert model in ['WideDeep', 'DeepFM', 'NFM', 'DeepCross']
        self.gbdt_model = gbdt_model
        self.pre_defined_idx = pre_defined_idx  # 更训练gbdt 时的样本保持一致；相同的train,val,test;
        self.full_dataset = CbData(root_dir, add_CLS=False, gbdt_model=gbdt_model)
        self.num_trees = self.full_dataset.num_trees
        self.leaf_num_per_tree = self.full_dataset.leaf_num_per_tree
        self.device = device
        if model == 'WideDeep':
            self.model = WideDeep(self.leaf_num_per_tree * self.num_trees, self.num_trees, emb_size).to(device)
        elif model == 'DeepFM':
            self.model = DeepFM(self.leaf_num_per_tree * self.num_trees, self.num_trees, emb_size).to(device)
        elif model == 'NFM':
            self.model = NFM(self.leaf_num_per_tree * self.num_trees, emb_size).to(device)
        elif model == 'DeepCross':
            self.model = DeepCross(self.leaf_num_per_tree * self.num_trees, self.num_trees, emb_size, num_layers=1).to(
                device)

        self.criterion = nn.BCELoss()

    def train(self, epoch, batch_size, lr):
        another_train_loader, train_loader, val_loader, test_loader = train_val_test_split(self.full_dataset,
                                                                                           batch_size,
                                                                                           self.pre_defined_idx,
                                                                                           return_idx=False)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        for epoch_id in range(epoch):
            self._train_epoch(epoch_id, train_loader, another_train_loader, val_loader, test_loader, optimizer)

    def _train_epoch(self, epoch_id, train_loader, another_train_loader, valid_loader, test_loader, optimizer):
        for batch_idx, data in enumerate(train_loader):
            with autograd.detect_anomaly():
                inputs = data['x'].to(self.device)
                target = data['y'].to(self.device)
                optimizer.zero_grad()
                score = self.model(inputs)
                loss = self.criterion(score, target.float())
                if batch_idx % 10 == 0:
                    self._val_epoch(epoch_id, batch_idx, another_train_loader, valid_loader, test_loader)
                loss.backward()
                optimizer.step()

    def _val_epoch(self, epoch_id, batch_idx, another_train_loader, valid_loader, test_loader):
        self.model.eval()
        with torch.no_grad():
            train_auc, train_loss, val_auc, val_loss = None, None, None, None
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
                if idx == 10:
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
                if idx == 10:
                    break
            val_loss = np.mean(Loss)
            val_auc = roc_auc_compute(torch.cat(Targets), torch.cat(Logits))

            Test_Targets = []
            Test_Logits = []
            Test_Loss = []
            for idx, data in enumerate(test_loader):
                inputs = data['x'].to(self.device)
                target = data['y'].to(self.device)
                logits = self.model(inputs)
                temp_loss = self.criterion(logits, target.float())
                Test_Targets.append(target)
                Test_Logits.append(logits)
                Test_Loss.append(temp_loss.tolist())
                if idx == 10:
                    break
            test_loss = np.mean(Test_Loss)
            test_auc = roc_auc_compute(torch.cat(Test_Targets), torch.cat(Test_Logits))

        print(
            '\n--------------------------------batch_num:{}----------------------------------------'.format(batch_idx))
        print('train_loss:{}  valid:loss:{} test_loss:{}\ntrain_auc:{} valid_auc:{}  test_auc:{}'.format(train_loss,
                                                                                                         val_loss,
                                                                                                         test_loss,
                                                                                                         train_auc,
                                                                                                         val_auc,
                                                                                                         test_auc))

    def predict(self, data_loader):
        pass
