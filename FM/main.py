from FM.data import CbData, train_val_test_split, CbData_test
from FM.layers import WideDeep, DeepFM, NFM, DeepCross, AFM
from FM.utils import roc_auc_compute
import torch
from torch import autograd
import torch.nn as nn
import numpy as np
import warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")


class Main:

    def __init__(self, gbdt_model, pre_defined_idx, device, root_dir, emb_size, model_save_dir, model='WideDeep'):
        assert model in ['WideDeep', 'DeepFM', 'NFM', 'DeepCross', 'AFM']
        self.gbdt_model = gbdt_model
        self.pre_defined_idx = pre_defined_idx  # 更训练gbdt 时的样本保持一致；相同的train,val,test;
        self.full_dataset = CbData(root_dir, add_CLS=False, gbdt_model=gbdt_model)
        self.num_trees = self.full_dataset.num_trees
        self.leaf_num_per_tree = self.full_dataset.leaf_num_per_tree
        self.device = device
        self.model_save_dir = model_save_dir
        if model == 'WideDeep':
            self.model = WideDeep(self.leaf_num_per_tree * self.num_trees, self.num_trees, emb_size).to(device)
        elif model == 'DeepFM':
            self.model = DeepFM(self.leaf_num_per_tree * self.num_trees, self.num_trees, emb_size).to(device)
        elif model == 'NFM':
            self.model = NFM(self.leaf_num_per_tree * self.num_trees, emb_size).to(device)
        elif model == 'DeepCross':
            self.model = DeepCross(self.leaf_num_per_tree * self.num_trees, self.num_trees, emb_size, num_layers=1).to(
                device)
        elif model == 'AFM':
            self.model = AFM(self.leaf_num_per_tree * self.num_trees, self.num_trees, emb_size).to(device)
        self.criterion = nn.BCELoss()

    def predict(self, new_x, load_path=None):
        if load_path is not None:
            model = torch.load(load_path)
        else:
            model = self.model
        model.eval()
        new_x_da = CbData_test(new_x, self.gbdt_model)
        loader = torch.utils.data.DataLoader(new_x_da, len(new_x_da))
        for x in loader:
            inp = x['x']
            logits = model(inp)
        return logits

    def eval_(self, x, y):
        logits = self.predict(x)
        return roc_auc_score(y, logits.detach().numpy().flatten())

    def train(self, epoch, batch_size, lr):
        another_train_loader, train_loader, val_loader, test_loader = train_val_test_split(self.full_dataset,
                                                                                           batch_size,
                                                                                           self.pre_defined_idx,
                                                                                           return_idx=False)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        for epoch_id in range(epoch):
            valid_auc = self._train_epoch(epoch_id, train_loader, another_train_loader, val_loader, test_loader,
                                          optimizer)
            torch.save(self.model, self.model_save_dir + 'epoch_{}_valid_auc:{}'.format(epoch_id, valid_auc))
            print('model with valid_acu {} saved as {}'.format(valid_auc,
                                                               self.model_save_dir + 'epoch_{}_valid_auc:{}'.format(
                                                                   epoch_id, valid_auc)))

    def _train_epoch(self, epoch_id, train_loader, another_train_loader, valid_loader, test_loader, optimizer):
        for batch_idx, data in enumerate(train_loader):
            with autograd.detect_anomaly():
                inputs = data['x'].to(self.device)
                target = data['y'].to(self.device)
                optimizer.zero_grad()
                score = self.model(inputs)
                loss = self.criterion(score, target.float())
                if batch_idx % 10 == 0:
                    _ = self._val_epoch(epoch_id, batch_idx, another_train_loader, valid_loader, test_loader)
                loss.backward()
                optimizer.step()
        return self._val_epoch(epoch_id, batch_idx, another_train_loader, valid_loader, test_loader)

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
                inputs_test = data['x'].to(self.device)
                target_test = data['y'].to(self.device)
                logits_test = self.model(inputs_test)
                temp_loss = self.criterion(logits_test, target_test.float())
                Test_Targets.append(target_test)
                Test_Logits.append(logits_test)
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
        return val_auc