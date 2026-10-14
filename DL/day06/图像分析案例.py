# 导包
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import time
BATCH_SIZE = 128
# 产生数据集
def create_data():
    train_dataset = CIFAR10('../data',train=True,transform=transforms.ToTensor())
    test_dataset = CIFAR10("../data",train=False,transform=transforms.ToTensor())
    return train_dataset,test_dataset
# 创建CNN神经网络
class Cnn_Model(nn.Module):
    def __init__(self):
        super(Cnn_Model,self).__init__()
        # 定义CNN
        # 定义卷积层
        self.cn1 = nn.Conv2d(3,32,3,1,1)
        # 定义池化层
        self.pool1 = nn.MaxPool2d(2,2)
        self.bn1 = nn.BatchNorm2d(32)
        nn.init.kaiming_normal_(self.cn1.weight,nonlinearity="relu")
        nn.init.zeros_(self.cn1.bias)
        # 定义卷积层
        self.cn2 = nn.Conv2d(32,64,3,1,1)
        # 定义池化层
        self.pool2 = nn.MaxPool2d(2,2)
        self.bn2 = nn.BatchNorm2d(64)
        nn.init.kaiming_normal_(self.cn2.weight,nonlinearity="relu")
        nn.init.zeros_(self.cn2.bias)
        # 定义全连接层
        self.ln1 = nn.Linear(4096,120)
        self.ln1_bn = nn.BatchNorm1d(120)
        nn.init.kaiming_normal_(self.ln1.weight,nonlinearity="relu")
        nn.init.zeros_(self.ln1.bias)
        self.dropout1 = nn.Dropout(p=0.5)
        # 第二层
        self.ln2 = nn.Linear(120,84)
        self.ln2_bn = nn.BatchNorm1d(84)
        nn.init.kaiming_normal_(self.ln2.weight,nonlinearity="relu")
        nn.init.zeros_(self.ln2.bias)
        self.dropout2 = nn.Dropout(p=0.5)
        # 输出层
        self.output = nn.Linear(84,10)
    # 函数向前传递
    def forward(self,x):
        # 让数据经过神经网络
        x = self.pool1(torch.relu(self.bn1(self.cn1(x))))
        x = self.pool2(torch.relu(self.bn2(self.cn2(x))))
        x = x.view(x.size(0),-1)
        x = torch.relu(self.ln1_bn(self.ln1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.ln2_bn(self.ln2(x)))
        x = self.dropout2(x)
        x = self.output(x)
        return x

    
# 模型训练
def train(train_dataset):
    # 数据加载
    dataloader = DataLoader(train_dataset,BATCH_SIZE,True)
    # 创建模型对象
    model = Cnn_Model()
    # 创建损失函数 多分类的问题选择交叉熵损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 创建优化器
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    # 定义循环轮次
    epochs = 15
    # 开始训练
    for epoch in range(epochs):
        # 定义循环使用的参数
        total_loss = 0
        batch_size = 0
        corrent_num = 0
        start = time.time()
        # 对数据集的数据进行处理
        for x,y in dataloader:
            # 向前传播
            y_pred = model(x)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss = loss_fn(y_pred,y)
            loss.backward()
            # 更新损失值 因为CrossEntropyLoss默认返回的是样本的平均值乘上样本量才是总的损失值
            total_loss += loss.item() * len(y)
            batch_size += len(y)
            # 更新权重
            optimizer.step()
            # 返回所有预测值最大的标签 方便后续进行统计
            y_pred = torch.argmax(y_pred,-1)
            # print(f"y_pred:{y_pred}")
            # print(f"y_true:{y}")
            # 求正确率
            corrent_num += (y_pred == y).sum().item()
        # 第一轮的训练完毕 打印相关信息
        print(f"Epoch:{epoch},Loss:{total_loss/batch_size},准确率:{corrent_num/len(train_dataset)},time:{time.time()-start}")
        # 为了节省时间所以事先加了break
        break
    # 训练完成 保存模型
    torch.save(model.state_dict(),"img_analyze.pth")
    print("训练完成")

# 模型评估
def model_eva(test_dataset):
    # 数据加载
    dataloader = DataLoader(test_dataset,BATCH_SIZE,True)
    # 创建模型
    model = Cnn_Model()
    # 加载模型
    model.load_state_dict(torch.load("img_analyze.pth"))
    # 定义参与正确率计算的参数
    corrent_num = 0
    # 模型评估
    for x,y in dataloader:
        # 切换模型为评估模式
        model.eval()
        # 模型训练
        y_pred = model(x)
        # 返回标签最大值
        y_pred = torch.argmax(y_pred,-1)
        # 统计正确数量
        corrent_num += (y_pred == y).sum().item()
    print(f"测试的准确率:{corrent_num / len(test_dataset)}")
# 运行测试
if __name__=="__main__":
    train_dataset,test_dataset = create_data()
    print(f"类别标签:{train_dataset.class_to_idx}")
    print(f"训练集的形状:{train_dataset.data.shape}")
    print(f"测试集的形状:{test_dataset.data.shape}")
    model = Cnn_Model()
    # 全连接层的参数占大头
    # summary(model,input_size=(3,32,32))
    # 模型训练
    train(train_dataset)
    model_eva(test_dataset)