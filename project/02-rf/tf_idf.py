"""
使用tf-idf 词频逆文档频率对新闻标题进行向量化

导入数据
实例化sklearn类
拿到词向量
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from config import Config
import pandas as pd
# 导入数据
config_obj = Config()
# 列名使用逗号隔开
data = pd.read_csv(config_obj.processing_train_path,sep=',')
print(data.columns.tolist())
# print(data)
# 实例化tf-idf类
# 读取中止词列表
with open(config_obj.stop_words_path,'r',encoding='utf8') as f:
    stop_words = f.read().split('\n')
# print(stop_words)
tfidf = TfidfVectorizer(stop_words=stop_words)
# 拿到词向量
x = tfidf.fit_transform(data['words'].tolist())
print(x)
print(x.shape)