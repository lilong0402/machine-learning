# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch import optim
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder

# This is for the progress bar.
from tqdm.auto import tqdm
import os



# 自定义数据集
class ImgDateset(Dataset):
    def __init__(self, root):
        self.root = root
        self.classes = os.listdir(self.root)
        self.images = []

        for i , cls in enumerate(self.classes):
            class_path = os.path.join(self.root, cls)
            for img in os.listdir(class_path):
                self.images.append(os.path.join(class_path, img))
    def __getitem__(self, index):
        return self.images[index]
    def __len__(self):
        return len(self.images)
# 创建数据加载器
def pre_dataload(dataset,batch_size,mode,n_job):
    img_dataload   = DataLoader(dataset,batch_size=batch_size,shuffle=(mode == 'train'),num_workers=n_job)
    return img_dataload

tr_transforms = transforms.Compose([
    transforms.Resize((128*128)),
    transforms.ToTensor()
])
# 定义神经网络
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x

def train(tr_dataload,model,n_epoch,lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loss_record = {'train':[]}
    for epoch in range(n_epoch):
        model.train()
        for img,label in tr_dataload:
            optimizer.zero_grad()
            logits = model(img)
            cross_loss = loss_fn(logits, label)
            cross_loss.backward()
            optimizer.step()
            loss_record['train'].append(cross_loss.detach().cpu().item())
    print(f"epoch {epoch+1}/{n_epoch} , loss = {loss_record['train']}")
    print('Finished Training')
    return loss_record
if __name__ == '__main__':
    config ={
        "n_epoch":500,
        "batch_size":64,
        "lr":0.001,
        "mode":"train",
        "n_job":3,

    }
    model = Classifier()
    # model = model.to(config['device'])
    tr_dataset = ImgDateset(r"D:\Data\pythonData\machine-learning\pytorch\hongyi\h_3_food\food-11\training\labeled")
    tr_loader = pre_dataload(tr_dataset,config['batch_size'],config['mode'],config['n_job'])
    train(tr_loader,model,config['n_epoch'],config['lr'])



