"""
PyTorch里面的数据必须是数值类型的 不能是字符串 因为字符串的长度是可以改变的
在P有Torch里面 所有的数据都可以叫做张量 包括标量 向量 矩阵 张量
案例:
    演示张量的创建.

涉及到的函数如下:
    torch.tensor()        根据指定数据创建张量.
    torch.Tensor()        根据形状创建张量, 也可以用来创建指定数据的张量.
    torch.IntTensor(), FloatTensor(), DoubleTensor()      创建指定类型的张量

张量解释:
    在PyTorch框架中, 所有的数据(标量, 矢量, 向量)都会被封装成 张量(Tensor)
    PyTorch是深度学习框架, 属于第三方包, 用之前需要 安装一下.
        pip install torch

细节:
    PyTorch中的张量必须是: 数值型.

(建议)掌握:
    torch.tensor()
"""

import torch
import numpy as np
# 使用torch.tensor 来创建张量 主要掌握这个 根据指定的数据创建张量
data1 =  torch.tensor([10])         # 创建向量
data2 = torch.tensor(10)            # 创建标量
data3 = torch.tensor([10,20,30])    # 创建向量
data4 = torch.tensor([[10.5,20.4,30],[40,50,60]])    # 创建矩阵
data5 = torch.tensor([[[10,20,30],[40,50,60]],[[70,80,90],[100,110,120]]])    # 创建三维矩阵
data7 = np.random.randint(0,100,100)
t1 = torch.tensor(data7)
print(f"t1:{t1}")
print(data1)
print(data2)
print(data3)
print(data4)
print(data5)
# 使用torch.Tensor 是根据形状创建张量 也可以和torch.tensor 一样指定数据
# 使用torch.Tensor 可能会有数据类型的转换 比如torch.tensor([10]) 会转换为torch.tensor([10.])
data6 = torch.Tensor([10])
print(data6)

# 使用torch.tensor 创建张量 如果数据是字符串 会报错
# t2 = torch.tensor("str")
# print(t2)
