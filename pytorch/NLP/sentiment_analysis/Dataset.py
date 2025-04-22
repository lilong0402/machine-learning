from torch.utils.data import Dataset
import pandas as pd

class SentimentAnalysisDataset(Dataset):
    def __init__(self, dataProcess,mode, dev_num = 10  , transform=None):
        self.transform = transform
        data = dataProcess.get_data()
        self.mode = mode
        if mode == "test":
            self.data = data
        else:
            labels = dataProcess.get_labels()
            if mode == "train":
                indices = [i for i in range(len(data)) if i % dev_num != 0 ]
            elif mode == "dev":
                indices = [i for i in range(len(data)) if i % dev_num == 0 ]
            self.data = data[indices]
            self.labels = labels[indices]
            # self.input_ids = dataProcess.get_data()["input_ids"][indices]
            # self.attention_mask = dataProcess.get_data()["attention_mask"][indices]
            # self.token_type_ids = dataProcess.get_data()["token_type_ids"][indices]

    def __getitem__(self, item):
        # return self.data[item],self.labels[item]
        # return {"input_ids": self.input_ids[item],
        #         "attention_mask": self.attention_mask[item],
        #         "token_type_ids": self.token_type_ids[item],
        #         "labels": self.labels[item]}
        if self.mode == "test":
            return self.data[item]
        return self.data[item], self.labels[item]
    # def __getitem__(self):
        # return self.data[item],self.labels[item]
        # return [{"input_ids": self.input_ids,
        #         "attention_mask": self.attention_mask,
        #         "token_type_ids": self.token_type_ids,
        #         "labels": self.labels}]
    def __len__(self):
        return len(self.data)











