# 导包
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


# 创建训练的测试集
def create_dataset():
    # 导入数据集
    data = pd.read_csv('手机价格预测.csv',encoding='gbk')
    # print(data)
    # 准备划分特征值和标签值
    x = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    # print(f"x:{x}")
    # print(f"y:{y}")
    x = x.astype(np.float32)
    y = y.astype(np.int64)
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)
    # 将x_train和y_train一起打包成训练集 转换为张量 因为x_train自身是df类型无法之间转换所以用x_train.values
    train_dataset = TensorDataset(torch.from_numpy(x_train.values),torch.from_numpy(y_train.values))
    test_dataset = TensorDataset(torch.from_numpy(x_test.values),torch.from_numpy(y_test.values))
    # 返回测试集和选练集 输入特征数量和输出的标签数量 np.unique(y)是对y的值进行去重排序得到[0,1,2,3]
    return train_dataset,test_dataset,x_train.shape[1],len(np.unique(y))
# 创建神经网络
class Phone_Pred(nn.Module):
    # 定义神经网络 对其进行标准化
    def __init__(self, input_dim, out_dim):
        # 继承父类的方法
        super(Phone_Pred,self).__init__()
        # 构建神经网络 这里构建三层
        # 第一层
        self.ln1 = nn.Linear(input_dim,32)
        self.bn1 = nn.BatchNorm1d(32)
        nn.init.kaiming_normal_(self.ln1.weight,nonlinearity='relu')
        # 第二层
        self.ln2 = nn.Linear(32,64)
        self.bn2 = nn.BatchNorm1d(64)
        nn.init.kaiming_normal_(self.ln2.weight,nonlinearity='relu')
        # 第三层
        self.ln3 = nn.Linear(64,128)
        self.bn3 = nn.BatchNorm1d(128)
        nn.init.kaiming_normal_(self.ln3.weight,nonlinearity='relu')
        # 输出层
        self.output = nn.Linear(128,out_dim)
    # 向前传播
    def forward(self,x):
        x = torch.relu(self.bn1(self.ln1(x)))
        x = torch.relu(self.bn2(self.ln2(x)))
        x = torch.relu(self.bn3(self.ln3(x)))
        x = self.output(x)
        return x
    
# 模型训练
def train(train_dataset,input_dim,out_dim):
    # 准备数据集
    dataloader = DataLoader(train_dataset,128,True)
    # 模型创建
    model = Phone_Pred(input_dim,out_dim)
    # 构建损失函数 多分类的问题 使用交叉熵损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 构建优化函数
    optimizer = optim.Adam(model.parameters(),0.001)
    # 定义循环次数
    epochs = 50
    # 训练模型
    for epoch in range(epochs):
        # 训练有关的参数
        total_loss = 0
        batch_size = 0
        corrent_num = 0
        start = time.time()
        for x,y in dataloader:
            # 向前传播
            y_pred = model(x)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss = loss_fn(y_pred,y)
            loss.backward()
            total_loss += loss.item()
            # 更新权重
            optimizer.step()
            # 得到最大值的下标 方便后续找到正确预测的个数
            y_pred = torch.argmax(y_pred,-1)
            corrent_num += (y_pred == y).sum().item()
            batch_size += len(y)
        print(f"Epoch:{epoch},Loss:{total_loss/batch_size},准确率:{corrent_num/len(train_dataset)*100},time:{time.time()-start}") 
    # 保存模型
    torch.save(model.state_dict(),"phone_pred.pth")
    print("训练完成")       
    
# 模型评估
def model_eva(test_dataset,input_dim,out_dim):
    # 准备数据集
    dataloader = DataLoader(test_dataset,128)
    # 模型构建
    model = Phone_Pred(input_dim,out_dim)
    # 导入模型
    model.load_state_dict(torch.load("phone_pred.pth"))
    # 定义总正确数量
    corrent_num = 0
    # 模型测评
    for x,y in dataloader:
        # 切换模型模式
        model.eval()
        # 模型训练
        y_pred = model(x)
        # 找到最大值的索引下标
        y_pred = torch.argmax(y_pred,-1)
        # 统计正确的数量
        corrent_num += (y_pred == y).sum().item()
    print(f"准确率：{corrent_num / len(test_dataset)*100}")    
# 测试训练
if __name__=='__main__':
    train_dataset,test_dataset,input_dim,out_dim = create_dataset()
    print(f"输入维度:{input_dim}")
    print(f"输出维度:{out_dim}")
    model = Phone_Pred(input_dim,out_dim)
    # summary(model=model,input_size=(input_dim,))
    train(train_dataset,input_dim,out_dim)
    model_eva(test_dataset,input_dim,out_dim)