"""
案例：
    演示Pytorch中如何创建 线性 和 随机张量
涉及到的函数
    torch.arange(), linspace()             创建线性张量
    torch.random.intial_seed(), manual_seed()  随机种子设置
    torch.rand(), randn()      创建随机浮点类型张量
    torch.randint(low, high, size=())   创建随机整数类型张量

掌握：
    arrange(), manual_seed(), randn(), randint()
"""
import torch
# 创建线性或随机张量
# 参数：start,end,step
t3 = torch.arange(1,10,3)
print(f"t3:{t3}")
# 参数：start,end,step（创建多少个） 
t4 = torch.linspace(0,10,2)
print(f"t4:{t4}")
# 随机浮点型的张量 2行3列
print(torch.rand(2,3))
# 随机正态分布张量 2行4列
print(torch.randn(2,4))
# 整型的张量 8行1列 取值范围1-8
print(torch.randint(1,8,(8,1)))