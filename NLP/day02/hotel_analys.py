"""
目标：检查标签的均匀性
训练集和测试集
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import jieba
# import torch
# 数据读取
train_data = pd.read_csv("./data/train.tsv",sep='\t')
dev_data = pd.read_csv("./data/dev.tsv",sep='\t')
print(train_data)

# 去重
word_list = []
sentences = train_data['sentence'].tolist()
for sentence in sentences:
    local_words = jieba.lcut(sentence)
    word_list.extend(local_words)
print(word_list)