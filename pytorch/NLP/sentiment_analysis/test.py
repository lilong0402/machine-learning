import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import DataProcess
import Dataset
from tqdm import tqdm
import NetWork

# 配置
path_root = "test.sentiment2.txt"
config = {
    "BertModel": "bert-base-chinese",
    "max_length": 128,
    "batch_size": 8,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# 初始化
tokenizer = BertTokenizer.from_pretrained(config["BertModel"])

def collate_fn(batch):
    sentences = [i[0] for i in batch]
    data = tokenizer(sentences,
                    max_length=config["max_length"],
                    padding=True,
                    truncation=True,
                    return_tensors='pt')
    return {k: v.to(config["device"]) for k, v in data.items()}

# 数据准备
tr_dp = DataProcess.DataProcess(path_root, "test")
tr_ds = Dataset.SentimentAnalysisDataset(tr_dp, "test")
tr_dataloader = DataLoader(tr_ds,
                          batch_size=config["batch_size"],
                          shuffle=False,
                          collate_fn=collate_fn)

# 模型加载
model = NetWork.Net().to(config["device"])
try:
    model.load_state_dict(torch.load('./model/sentiment_model.pth'))
except:
    # 如果保存的是整个模型而非state_dict
    model = torch.load('./model/sentiment_model.pth').to(config["device"])
model.eval()

# 测试
all_predictions = []
with torch.no_grad():
    for batch in tqdm(tr_dataloader, desc="Testing"):
        outputs = model(**batch)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_predictions.extend(preds)

# 输出结果
print(f"Total test samples: {len(all_predictions)}")
print("Sample predictions:", all_predictions[:10])  # 打印前10个预测样例