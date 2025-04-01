import  torch
# torch.manual_seed(2)
import numpy as np
import pandas as pd

data = pd.read_csv("covid.train.csv")

data = np.array(data[1:])[:,1:].astype(np.float32)
# data = data[:,-1]
data = data[:,range(93)]
# print((data.shape))