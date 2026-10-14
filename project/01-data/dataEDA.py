"""
做探索性数据分析：
1. 数据量
2. 检查标签数据分布
3. 文本长度
步骤
1. 导包
2. 实例化配置类
3. 数据加载
4. 探索性数据分析
    4.1 数据量
    4.2 标签分布是否均匀
    4.3 文本长度 - 平均长度，最大长度，最小长度，长度标准差
"""
# 1. 导包
from collections import Counter
from config import Config
import pandas as pd
import numpy as np
# 2. 实例化配置类
config_obj = Config()
# 3. 数据加载 给数据设置一个表头不然会缺行
train_data = pd.read_csv(config_obj.train_data_path,sep='\t',names=['title','label'])
print(train_data)
# 4. 探索性数据分析
#     4.1 数据量
print(f"数据量:{len(train_data)}")
print(f"{train_data.shape}")

#     4.2 标签分布是否均匀
## 思路一 只用dict
# label_dict = {}
# for label in train_data["label"]:
#     label_dict[label] = label_dict.get(label,0) + 1
# print(label_dict)
## 思路二 使用counter 同时将数字标签找到对应的索引
with open(config_obj.class_data_path,'r',encoding="utf8") as f:
    label_lsit = f.read().split()
    # print(label_lsit)
for label,count in Counter(train_data['label']).items():
    print(f"标签{label_lsit[label]}出现了{count}次")

#     4.3 文本长度 - 平均长度，最大长度，最小长度，长度标准差
## 思路一 使用for循环 得到句子的平均长度
len_list = []
for title in train_data['title']:
    len_list.append(len(title))
len_list = np.array(len_list)
## 思路二 使用map内置函数
# len_list = np.array(list(map(lambda x:len(x), train_data['title'])))
## 思路三 使用datafram的apply函数
# len_list = train_data['title'].apply(lambda x: len(x))

print(f"句子的平均长度:{len_list.mean()}")
print(f"句子的最大长度:{len_list.max()}")
print(f"句子的最小长度:{len_list.min()}")
print(f"句子的长度标准差:{len_list.std()}")