

import torch

def __init__(self,model):
    torch.save(model.state_dict(),"./model/"+model+".pth")