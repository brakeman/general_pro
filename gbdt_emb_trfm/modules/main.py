from modules.data import train_val_test_split
from modules.layer import TRFM
import tqdm
import torch
from torchnet import meter


class GbdtTrfm:
    def __init__(self, gbdt_model, num_trees, leaf_num_per_tree, full_dataset):
        self.gbdt_model = gbdt_model
        self.num_trees = num_trees
        self.leaf_num_per_tree = leaf_num_per_tree
        self.full_dataset = full_dataset

    def val_epoch(self):
        pass
    
    def train_epoch(self, train_loader, device, optimizer, criterion, model):
        for ii, (data, label) in tqdm(enumerate(train_loader)):
            # train model
            input = data.to(device)
            target = label.to(device)
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()


    def train(self, epoch, batch_size, lr, device):
        # step1: data
        train_loader, val_loader, test_loader = train_val_test_split(self.full_dataset, batch_size)

        # step2: model
        model = TRFM(num_uniq_leaf=self.num_trees*self.leaf_num_per_tree+1 , dim_leaf_emb=32, num_heads=8,
                 dim_q_k=100 , dim_v=200, dim_m=32, dropout=0.3)
        model = model.to(device)

        # step3: criterion and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        # train
        for epoch in range(epoch):
            res = self.train_epoch(train_loader)

            for ii, (data, label) in tqdm(enumerate(train_loader)):
                # train model
                input = data.to(device)
                target = label.to(device)
                optimizer.zero_grad()
                score = model(input)
                loss = criterion(score, target)
                loss.backward()
                optimizer.step()

            val_loss, val_auc = self.val_epoch(model, val_loader)

    def predict(self, data_loader):
        pass
