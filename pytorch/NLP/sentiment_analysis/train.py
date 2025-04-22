import  Dataset
# from Dataset import SentimentAnalysisDataset
import torch
import DataProcess
import numpy as np
from transformers import BertTokenizer,BertForSequenceClassification
from torch.utils.data import DataLoader
import  tqdm
import NetWork
import Safemode
import pandas as pd


def collate_fn(batch):
    # batch 是 [{"input_ids": ..., "attention_mask": ..., "labels": ...}, ...]
    sentences = [i[0] for i in batch]
    labels = [i[1] for i in batch]
    # print(f"labels:{labels}")
    data = tokenizers(sentences,max_length=config["max_length"],padding=True,return_tensors='pt',truncation=True)
    return {
        "input_ids": data['input_ids'].to(device),
        "attention_mask": data['attention_mask'].to(device),
        "token_type_ids": data['token_type_ids'].to(device),
        "labels": torch.tensor(labels).to(device)
    }

path_root = "train_sentiment.txt"
config ={
    "BertModel": "bert-base-chinese",
    "mode":"train",
    "max_length":128,
    "dev_num":10,
    "batch_size":128,
    "learning_rate":0.001,
    "epochs":5,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config = pd.read_json(path_root,encoding="utf-8")
# path_root = str(config.path_root)
# config = config.config


tokenizers  = BertTokenizer.from_pretrained(config["BertModel"])
tr_dp = DataProcess.DataProcess(path_root,"dev")
tv_dp = DataProcess.DataProcess(path_root,"dev")

tr_ds = Dataset.SentimentAnalysisDataset(tr_dp,config["mode"],config["dev_num"])
tv_ds = Dataset.SentimentAnalysisDataset(tv_dp, "dev",config["dev_num"])
tr_sentiment_dataloder = DataLoader(tr_ds,batch_size=config["batch_size"],shuffle=(config["mode"] == "train"),collate_fn=collate_fn)
tv_sentiment_dataloder = DataLoader(tv_ds,batch_size=config["batch_size"],collate_fn=collate_fn)
# bertModel = BertForSequenceClassification.from_pretrained(config["BertModel"])
model = NetWork.Net().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=config["learning_rate"])
criterion = torch.nn.CrossEntropyLoss()
model.train()
all_preds = []
for epoch in range(config["epochs"]):
    for i,batch in enumerate(tqdm.tqdm(tr_sentiment_dataloder)):
        # batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["labels"]
        outputs = model(input_ids,attention_mask,token_type_ids)
        loss = criterion(outputs,labels).to(device)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print("epoch:",epoch+1,"loss:",loss.item())
model.eval()
all_labels = []
for epoch in range(config["epochs"]):
    acc = 0
    total = 0
    for i,batch in enumerate(tqdm.tqdm(tv_sentiment_dataloder)):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["labels"]
        outputs = model(input_ids,attention_mask,token_type_ids)
        outputs = outputs.argmax(dim=1)
        all_preds.append(outputs)
        all_labels.append(labels)
        acc  += (outputs == batch["labels"]).sum().item()
        total += len(labels)
    print("准确率：",acc/total)
    print(all_preds)
    print(all_labels)
model_filename = "sentiment_model2.pth"
torch.save(model.state_dict(), "./model/" + model_filename)


