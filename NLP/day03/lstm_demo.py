"""
学习如何使用lstm进行推理
"""
import torch
lstm = torch.nn.LSTM(10,5,6)
# 定义输入数据
x = torch.randn(11,5,10)
h0 = torch.randn(6,5,5)
c0 = torch.randn(6,5,5)
# 推理 隐藏层和细胞状态要打包传入
y,(hn,cn) = lstm(x,(h0,c0))
print(f"输出形状：{y.shape}\n隐藏层形状：{hn.shape}\n细胞状态形状：{cn.shape}")