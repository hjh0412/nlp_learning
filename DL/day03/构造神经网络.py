import torch
import torch.nn as nn                               # 神经网络模块
import torch.nn.functional as F                     # 激活函数模块
from torchsummary import summary
# 创建神经元类
class Model(nn.Module):
    # 初始化属性值 即定义输入输出和隐藏层
    def __init__(self):
        super(Model, self).__init__()       # 调用父类的初始化方法
        # 定义 第一个隐藏层 3输入4输出
        self.linear1 = nn.Linear(3,4)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.kaiming_normal_(self.linear1.weight,nonlinearity='relu')
        nn.init.zeros_(self.linear1.bias)
        # 定义 第二个隐藏层 4输入3输出
        self.linear2 = nn.Linear(4,3)
        nn.init.kaiming_normal_(self.linear2.weight,nonlinearity='relu')
        nn.init.zeros_(self.linear2.bias)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        # 初始化输出 3输入2输出
        self.out = nn.Linear(3,2)
    # 向前传播 得到梯度等
    def forward(self,x):
        # 数据经过第一个神经层
        x = self.linear1(x)
        # 激活函数
        x = torch.sigmoid(x)
        # 经过第二个隐藏层
        x = self.linear2(x)
        # 激活函数
        x = torch.relu(x)
        # 数据经过输出层
        x = self.out(x)
        # 激活函数
        x = torch.softmax(x,dim=-1)
        return x


if __name__=='__main__':
    # 神经网络模型训练
    # 实例化对象
    my_model = Model()
    # 产生数据
    data = torch.randn(6,3)
    print(f"测试数据的规模{data.shape}")
    # 模型训练
    ouput = my_model(data)
    print(f"预测数据的规模{ouput.shape}")
    # 计算每层神经的权重和偏置的总和
    # 参数: 神经模型 输入数据的规模 批处理大小
    # 计算参数的时候 不要忘记还有一个偏置参数!
    summary(my_model, input_size=(3,),batch_size=5)
    for name, parameter in my_model.named_parameters():
        print(name, parameter)