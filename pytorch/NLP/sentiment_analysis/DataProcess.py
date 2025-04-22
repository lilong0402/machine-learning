
# from torch import BertTokenizer,BertModel
import pandas as pd
import torch


class DataProcess(object):
    def __init__(self,path_root,mode):
        text = pd.read_table(path_root)
        labels = None
        data = text.iloc[:,1].values
        # print(data)
        if mode == "train" or mode == "dev":
            labels = text.iloc[:,2].values
            print(f"labels{labels}")
        # 将文本转为bert输入模式
        # data = tokenizer(data.tolist(),max_length=max_length,padding=True,return_tensors='pt',truncation=True)
        #
        # data = tokenizer.batch_decode_plus(
        #     batch_text_or_text_pairs = data,
        #     truncation = True,
        #     return_tensors = 'pt',
        #     padding = True,
        #     return_length = True,
        #     max_length= max_length,
        # )
        self.data = data
        self.labels = labels
    def get_data(self):
        # print(type(self.data))
        return self.data
    def get_labels(self):
        return self.labels



