import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torch.nn as nn

from torch01 import device


class CONVIDDataset(Dataset):
    def __init__(self,path,mode,target_only = False):
        self.mode = mode
        data = pd.read_csv(path)
        data = np.array(data[1:])[:,1:].astype(float)
        if not(target_only):
            feats = list(range(93))
        if mode == 'test':
            data = data[:,feats]
            self.data = torch.FloatTensor(data)
        else:
            target = data[:,-1]
            data = data[:,feats]
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]

            self.data = torch.FloatTensor(data)
            self.target = torch.FloatTensor(target)
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]
        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))
    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.data[index], self.target[index]
    def __len__(self):
        return len(self.data)

def pre_dataloader(path,mode,batch_size,n_jobs=0,target_only=False):
    dataset = CONVIDDataset(path,mode,target_only)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=(mode == "train"),drop_last=False,num_workers=n_jobs,pin_memory=True)
    return dataloader

## 定义神经网络
class NeuralNet(nn.Module):
    def __init__(self,input_dim):
        super(NeuralNet,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64,1),
        )
        self.criterion = nn.MSELoss(reduction='mean')
    def forward(self,x):
        return self.net(x).squeeze(1)
    def cal_loss(self, pred, target):
        return self.criterion(pred, target)

def train(tr_set,dv_set,model,config,target):
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(),**config['optim_hparas'])
    min_mse = 1000
    loss_record = {'train': [], 'dev': []}
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()
        for x,y in tr_set:
            optimizer.zero_grad()
            x,y = x.to(device),y.to(device)
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
            mse_loss.backward()
            optimizer.step()
            loss_record['train'].append(mse_loss.detach().cpu().item())
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                  .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record
def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:                         # iterate through the dataloader
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)




