# 基于bert的情感分析实战
from torch.utils.data import DataLoader,Dataset

# 自定义数据集
class SentimentAnalysis(Dataset):
    def __init__(self, data_path, transform=None):
        super(SentimentAnalysis, self).__init__()
        self.data_path = data_path
