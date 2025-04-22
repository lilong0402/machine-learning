from torch.utils.data import Dataset
import torch
class MyDataset(Dataset):
    def __init__(self, data,mode, target=None):
        self.mode = mode
        if mode == "test":
            self.data = torch.tensor(data).float()
        elif mode == "train":
            self.data = torch.tensor(data).float()
            self.target = torch.tensor(target).float()
    def __getitem__(self, index):
        if self.mode == "test":
            return self.data[index]
        return self.data[index],self.target[index]
    def __len__(self):
        return len(self.data)