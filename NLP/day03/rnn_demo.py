"""
学习rnn的torch的api
"""
import torch.nn as nn
import torch
IMPUT_DIM = 2           # 向量维度
HIDDEN_DIM = 6          # 隐藏层维度
NUM_LAYERS = 3          # 隐藏层的层数
SEQ_LENGTH = 8          # 词序列长度
BATCH_SIZE = 3          # 批次大小
# 定义rnn网络
rnn = nn.RNN(IMPUT_DIM,HIDDEN_DIM,NUM_LAYERS)
# 定义输入的词向量
x = torch.randn(SEQ_LENGTH,BATCH_SIZE,IMPUT_DIM)
# 定义上一个隐藏状态
h0 = torch.randn(NUM_LAYERS,BATCH_SIZE,HIDDEN_DIM)
# 每经过一个rnn网络得到两个输出：预测值和当前隐藏状态
y,hn = rnn(x,h0)
print(f"输出数据的规模：{y.shape}")
print(f"当前隐藏层的规模：{hn.shape}")