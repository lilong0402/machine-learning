import pandas as pd

# 假设我们有一个CSV文件包含待处理文本
df = pd.read_csv('text_data.csv')
texts = df['text_column'].tolist()
from transformers import BertTokenizer, BertModel
import torch

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state

embeddings = [get_embeddings(text) for text in texts]
from sklearn.metrics.pairwise import cosine_similarity


def get_keyword_scores(text, embedding):
    # 获取各个词的嵌入
    word_embeddings = [get_embeddings(token) for token in tokenizer.tokenize(text)]
    # 计算句子的嵌入
    sentence_vec = embedding.mean(dim=1)

    scores = cosine_similarity(sentence_vec.numpy(), word_embeddings)
    return scores

keywords = []

for i, text in enumerate(texts):
    scores = get_keyword_scores(text, embeddings[i])
    keywords.append(sorted(zip(tokenizer.tokenize(text), scores[0]), key=lambda x: x[1], reverse=True)[:5])

