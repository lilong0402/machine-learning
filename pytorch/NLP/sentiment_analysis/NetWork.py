
import torch
from transformers import BertModel
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(0.1)  # 定义Dropout层
        self.fc = nn.Linear(768, 3)
        self.bertModel = BertModel.from_pretrained("bert-base-chinese")
    def forward(self,input_ids,attention_mask,token_type_ids):
        #冻结Bert模型的参数，让其不参与训练
        with torch.no_grad():
            out = self.bertModel(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        #增量模型参与训练
        out = self.fc(out.last_hidden_state[:, 0])
        # out = out.softmax(dim=1)
        return out
