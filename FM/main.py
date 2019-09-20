from FM.data import CbDataNew
from FM.layers import WideDeep, DeepFM, NFM, DeepCross, AFM, xDeepFM
from FM.utils import roc_auc_compute
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import warnings
import time
import os
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")


class Main:

    def __init__(self, gbdt_model, pre_defined_idx, device, data_or_dataroot, emb_size, model_save_dir,
                 model_name, concat_wide):
        assert model_name in ['WideDeep', 'DeepFM', 'NFM', 'DeepCross', 'AFM', 'xDeepFM']
        self.model_name = model_name
        self.emb_size = emb_size
        self.concat_wide = concat_wide
        self.pre_defined_idx = pre_defined_idx  # 更训练gbdt 时的样本保持一致；相同的train,val,test;
        self.gbdt_model = gbdt_model
        if isinstance(data_or_dataroot, str):
            self.train_dataset = CbDataNew(root_dir=data_or_dataroot,
                                           add_CLS=False,
                                           gbdt_model=gbdt_model,
                                           data_idx=pre_defined_idx[0])

            self.val_dataset = CbDataNew(root_dir=data_or_dataroot,
                                         add_CLS=False,
                                         gbdt_model=gbdt_model,
                                         data_idx=pre_defined_idx[1])
        else:
            print('use given dataset')
            self.train_dataset, self.val_dataset = data_or_dataroot
        self.num_trees = self.val_dataset.num_trees
        self.leaf_num_per_tree = self.val_dataset.leaf_num_per_tree
        self.device = device
        self.model_save_dir = model_save_dir
        if model_name == 'WideDeep':
            self.model = WideDeep(num_uniq_leaf=self.leaf_num_per_tree * self.num_trees,
                                  num_trees=self.num_trees,
                                  dim_leaf_emb=emb_size).to(device)

        elif model_name == 'DeepFM':
            self.model = DeepFM(self.leaf_num_per_tree * self.num_trees,
                                self.num_trees,
                                emb_size,
                                concat_wide).to(device)

        elif model_name == 'NFM':
            self.model = NFM(self.leaf_num_per_tree * self.num_trees,
                             self.num_trees,
                             emb_size,
                             concat_wide).to(device)

        elif model_name == 'DeepCross':
            self.model = DeepCross(self.leaf_num_per_tree * self.num_trees,
                                   self.num_trees,
                                   emb_size,
                                   num_layers=1).to(device)

        elif model_name == 'AFM':
            self.model = AFM(self.leaf_num_per_tree * self.num_trees, emb_size).to(device)

        elif model_name == 'xDeepFM':
            self.model = xDeepFM(num_layers=2,
                                 layer_filters=[3] * 2,
                                 num_uniq_leaf=self.leaf_num_per_tree * self.num_trees,
                                 num_trees=self.num_trees,
                                 dim_leaf_emb=emb_size).to(device)
        print(self.model)
        self.criterion = nn.BCELoss()

    def train(self, epoch, batch_size, lr, weight_decay, verbose, save_model, save_log, eval_full_epoch, early_stop):
        if early_stop:
            self.early_stop = True
        else:
            self.early_stop = False
        self.best_val_auc = -1
        self.early_stop_tolerate = 0
        self.stop_flag = False
        if verbose == 1:
            self.verbose = 1
        else:
            self.verbose = 0
        if save_log:
            dir_path = os.getcwd() + '/run_log_dir/'
            print("train log will be saved in directory：" + "run_log_dir")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            file = open(dir_path + '{}_logs{}.txt'.format(self.model_name,
                                                          time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())), 'w')
            file.writelines(
                '------------------class init paras------------------\nemb_size{}\nmodel_name:{}\nmodel_save_dir:{}\nconcat_wide:{}\n'.format(
                    self.emb_size, self.model_name, self.model_save_dir,
                    self.concat_wide))

            file.writelines(
                '\n------------------train paras------------------\nepoch:{}\nbatch_size:{}\nlr:{}\nweight_decay:{}\nverbose:{}\nsave_model:{}\neval_full_epoch:{}\n'.format(
                    epoch, batch_size,
                    lr, weight_decay, verbose,
                    save_model, eval_full_epoch))
        else:
            file = None
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size)
        if eval_full_epoch:
            self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=len(self.val_dataset))
        else:
            self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=batch_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,
                                     betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
        print('start training ......')
        for epoch_id in range(epoch):
            if self.stop_flag is True:
                break
            t0 = time.time()
            train_loss, val_loss, train_auc, val_auc = self._train_epoch(optimizer=optimizer, log_file=file)

            if verbose == 1:
                if save_log:
                    file.writelines(
                        '\n-----------------------  epoch_id:{}/{} use:{} seconds  -------------------------\n'.format(
                            epoch_id, epoch, time.time() - t0))
                    file.writelines(
                        'train_loss:{}  valid:loss:{} \n\ntrain_auc:{} valid_auc:{}\n'.format(train_loss, val_loss,
                                                                                              train_auc, val_auc))

                print('\n-----------------------  epoch_id:{}/{} use:{} seconds  --------------------------'.format(
                    epoch_id, epoch, time.time() - t0))
                print(
                    'train_loss:{}  valid:loss:{} \n\ntrain_auc:{} valid_auc:{}'.format(train_loss, val_loss, train_auc,
                                                                                        val_auc))

            if save_model:
                if save_log:
                    file.writelines('\nmodel saved in {}'.format(
                        self.model_save_dir + '{}_epoch_{}_auc_{}'.format(self.model_name, epoch_id, val_auc)))

                print('\nmodel saved in {}'.format(
                    self.model_save_dir + '{}_epoch_{}_auc_{}'.format(self.model_name, epoch_id, val_auc)))
                torch.save(self.model,
                           self.model_save_dir + '{}_epoch_{}_auc_{}'.format(self.model_name, epoch_id, val_auc))
        if save_log:
            file.close()

    def _train_epoch(self, optimizer, log_file=None):
        train_targets_list, train_logit_list, train_loss_list, batch_idx = [], [], [], None
        for batch_idx, data in enumerate(self.train_loader):
            if self.stop_flag is True:
                break
            inputs = data['x'].to(self.device)
            target = data['y'].to(self.device)
            optimizer.zero_grad()
            score = self.model(inputs)
            loss = self.criterion(score, target.float())
            loss.backward()
            optimizer.step()
            #                 print('batch_idx:{} use:{} seconds'.format(batch_idx, time.time() - t0))
            train_targets_list.append(target)
            train_logit_list.append(score)
            train_loss_list.append(loss.tolist())
            if batch_idx % 10 == 0:
                self.train_loss = np.mean(train_loss_list)
                self.train_auc = roc_auc_compute(torch.cat(train_targets_list), torch.cat(train_logit_list).detach())
                self._eval_epoch(batch_idx=batch_idx, log_file=log_file)
        return self._eval_epoch(batch_idx=batch_idx, log_file=log_file)

    def _eval_epoch(self, batch_idx, log_file=None):
        self.model.eval()
        with torch.no_grad():
            val_target_list, val_logit_list, val_loss_list = [], [], []
            for idx, data in enumerate(self.val_loader):
                inputs = data['x'].to(self.device)
                target = data['y'].to(self.device)
                logits = self.model(inputs)
                val_loss = self.criterion(logits, target.float())
                val_target_list.append(target)
                val_logit_list.append(logits)
                val_loss_list.append(val_loss.tolist())
                if idx == 30:
                    break
            val_loss = np.mean(val_loss_list)
            val_auc = roc_auc_compute(torch.cat(val_target_list), torch.cat(val_logit_list))
            if val_auc >= self.best_val_auc:
                self.best_val_auc = val_auc
            else:
                self.early_stop_tolerate += 1

            if (self.early_stop_tolerate == 5) and (self.early_stop is True):
                print('early_stop_tolerate = 5, with best val_auc:{}'.format(self.best_val_auc))
                if log_file:
                    log_file.writelines('\nearly_stop_tolerate = 5, with best val_auc:{}'.format(self.best_val_auc))
                self.stop_flag = True

        if self.verbose != 1:
            print('\n--------------------------------batch_num:{}/{}----------------------------------------'.format(
                batch_idx, len(self.train_loader.dataset) // self.train_loader.batch_size))
            print('train_loss:{}  valid:loss:{}\ntrain_auc:{} valid_auc:{}'.format(self.train_loss, val_loss,
                                                                                   self.train_auc, val_auc))
            if log_file is not None:
                log_file.writelines(
                    '\n------------------------------batch_num:{}/{}-------------------------------------\n'.format(
                        batch_idx, len(self.train_loader.dataset) // self.train_loader.batch_size))
                log_file.writelines('train_loss:{}  valid:loss:{}\ntrain_auc:{} valid_auc:{}'.format(self.train_loss,
                                                                                                     val_loss,
                                                                                                     self.train_auc,
                                                                                                     val_auc))
        return self.train_loss, val_loss, self.train_auc, val_auc

    def predict(self, new_x, load_path=None):
        if load_path:
            model = torch.load(load_path)
        else:
            model = self.model
        model.eval()
        logits = model(new_x)
        return logits

    def eval_(self, x, y, load_path=None):
        x = torch.tensor(x)
        output = self.predict(x, load_path)
        return roc_auc_score(y, output.detach().numpy().flatten())
