"""
目的是实现酒店评论文本分类

背景：


数据：
2分类：0 负面评论 1 正面评论

步骤：
1.数据
    1.1 数据加载
    1.2 数据预处理
    1.3 特征工程
2.模型开发
    2.1 模型训练
    2.2 模型评估
3.模型部署
    3.1 模型预测

扩展：导入环境变量
from config import Config()
config_var = Config()
调用config_var.MAX_LEN...进行操作
"""
import pandas as pd
import numpy as np
import jieba
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
torch.set_num_threads(4)
# 配置超级参数
MAX_VOCAB_SIZE = 10000      # 词表的最大长度
MAX_LEN = 50                # 定义切割文本的长度
EMBED_DIM = 64              # 定义输入词的维度
HIDDEN_DIM = 64             # 隐藏层的维度
BATCH_SIZE = 64             # 批次大小
EPOCHS = 10                 # 训练轮次
# 数据加载
# def load_data(input_data_dir:str) -> Tuple[List[str], List[int]]:
def load_data():
    """
    数据加载函数，根据输入的路径加载数据
    返回文本和标签
    输入：数据路径
    输出：文本列表，标签列表
    """
    # 导入数据 分隔符是制表符
    df = pd.read_csv("../day02/data/train.tsv", sep='\t')
    # 数据空值处理
    df = df.dropna(subset=["sentence", "label"])
    # print(df)
    # 取出文本和标签 返回的列表
    texts = df["sentence"].astype(str).tolist()
    labels = df["label"].values
    return texts,labels

# 文本预处理
def text2seq(texts):
    """
    给定句子列表，找到词汇表，返回句子列表对应的词向量
    词汇表：0是pad填充词 1是ukn未知词
    步骤：
    1. 对所有的句子遍历分词，统计各个词出现的频次因为最大是1w个词
    2. 基于频次选取词作为词汇表
    3. 遍历句子拿到指定长度的向量
    4. 返回词汇表和句子列表
    """
    # 对文章进行语句切割
    tokens = [jieba.lcut(i) for i in texts]
    # print(tokens)
    word_count = {}
    # 统计出每个词出现的频率 
    for sent in tokens:
        for word in sent:
            # 看word_count的某个词出现的频率
            word_count[word] = word_count.get(word, 0) + 1
    # 词表
    vocab = {"<PAD>":0, "<UKN>":1}
    # 根据word_dount.get()做为键对word_count进行排序 得到新的列表 -2是因为词标有两个固定的值
    top_words = sorted(word_count, key=word_count.get, reverse=True)[:MAX_VOCAB_SIZE - 2]
    # print(top_words)
    # 添加词典 优先对高频词进行更新
    for w in top_words:
        vocab[w] = len(vocab)

    # 对语句进行填充和切割
    seqs = []
    for sent in tokens:
        # 找到分词后句段每个词在词表里面对应的索引
        seq = [vocab.get(w, 1) for w in sent]
        # 将词组成一个长MAX_LEN长度的句子 存入到seqs里面
        seq = seq[:MAX_LEN] + [0] * (MAX_LEN - len(seq))
        seqs.append(seq)
    # 返回词表和每个句子对应的词向量
    return np.array(seqs), vocab

# 构建数据集 把句子变成张量
class CommentDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, index):
        return torch.LongTensor(self.seqs[index]), torch.FloatTensor([self.labels[index]])
    
# lstm模型搭建
class LSTMModel(nn.Module):
    # 传入的参数是词标的大小
    def __init__(self, vocab_size):
        """
        1. 用Embedding变成稠密词向量
        2. 把稠密词向量放入lstm
        3. 把经过lstm后的最终隐藏层放入到全连接层准备进行预测
        """
        super().__init__()
        # 构建词嵌入模型
        self.embd = nn.Embedding(vocab_size, EMBED_DIM, 0)
        # 创建lstm损失
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        # 创建线性层
        self.ln = nn.Linear(HIDDEN_DIM, 1)
    # 向前传播
    def forward(self, x):
        # 进行词嵌入 嵌入数据的维度是有限制的 如果没有使用batch_first可以通过tanspose和permute来转换维度使得维度匹配
        # x = self.embd(x.transpose(0,1))
        # x = self.embd(x.permute(1,0,2))
        x = self.embd(x)
        # 通过神经网络
        out,(h,c) = self.lstm(x)
        # 全连接之后输出 取的是最后时间运行后的隐藏层
        return self.ln(h[-1])

class RNNmodel(nn.Module):
    # 传入的参数是词标的大小
    def __init__(self, vocab_size):
        super().__init__()
        # 构建词嵌入模型
        self.embd = nn.Embedding(vocab_size, EMBED_DIM, 0)
        # 创建lstm损失
        self.rnn = nn.RNN(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        # 创建线性层
        self.ln = nn.Linear(HIDDEN_DIM, 1)
    # 向前传播
    def forward(self, x):
        # 进行词嵌入
        x = self.embd(x)
        # 通过神经网络
        out,h = self.rnn(x)
        # 全连接之后输出 取的是最后时间运行后的隐藏层
        return self.ln(h[-1])

class GRUModel(nn.Module):
    # 传入的参数是词标的大小
    def __init__(self, vocab_size):
        super().__init__()
        # 构建词嵌入模型
        self.embd = nn.Embedding(vocab_size, EMBED_DIM, 0)
        # 创建lstm损失
        self.gru = nn.GRU(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        # 创建线性层
        self.ln = nn.Linear(HIDDEN_DIM, 1)
    # 向前传播
    def forward(self, x):
        # 进行词嵌入
        x = self.embd(x)
        # 通过神经网络
        out,h = self.gru(x)
        # 全连接之后输出 取的是最后时间运行后的隐藏层
        return self.ln(h[-1])

# lstm模型训练
def train(model, loader, criterion, optimizer):
    """
    输入模型 数据集 损失函数 优化器进行模型训练
    返回的是模型训练后的损失率和准确率
    """
    # 切换模型模式
    model.train()
    # 定义计算损失函数的参数
    total_loss = 0
    correct_num = 0
    # 开始训练
    for seq, label in loader:
        # 梯度清零
        optimizer.zero_grad()
        # 模型训练
        y = model(seq)
        # 反向传播
        loss = criterion(y, label)
        loss.backward()
        # 更新权重
        optimizer.step()
        # 计算总的权重
        total_loss += loss.item() * len(seq)
        # 计算准确率 使用0.5是为了让标签的概率等分 保证好坏各占一半
        pred = (torch.sigmoid(y) > 0.5).float()
        correct_num += (pred == label).sum().item()
    return total_loss / len(loader.dataset), correct_num / len(loader.dataset)


# lstm模型评估
def model_eva(model, loader):
    # 切换模型模式
    model.eval()
    # 定义评估的相关函数
    preds = []
    trues = []
    # 开始模型评估 要求不再更新权重
    with torch.no_grad():
        for seq, label in loader:
            y = model(seq)
            pred = (torch.sigmoid(y) > 0.5).float().view(-1)
            preds.extend(pred.numpy().tolist())
            # 让形状尽量与preds的相同
            trues.extend(label.view(-1).numpy().tolist())
    # 使用机器学习下的包对最后的结果进行模型评估
    return accuracy_score(trues, preds)

# lstm模型预测
def predict(model, text, vocab):
    # 对输入的文本进行分词
    words = jieba.lcut(text)
    # 语句基本处理
    seq = [vocab.get(w, 1) for w in words]
    seq = seq[:MAX_LEN] + [0]*(MAX_LEN - len(seq))
    seq = torch.LongTensor([seq])
    # 模型评估
    model.eval()
    # 对文本进行标签预测
    with torch.no_grad():
        prob = torch.sigmoid(model(seq)).item()
    label = "好评" if prob > 0.5 else "差评"
    # 返回预测的标签和是好评的概率(保留四位小数)
    return label, round(prob, 4)

if __name__=='__main__':
    # 取出文本和标签
    texts,labels = load_data()
    # print(type(texts))
    # print(f"数据集的长度：{len(labels)}")
    seqs,vocab = text2seq(texts)
    # print(seqs)
    # print(seqs.shape, len(vocab))
    # 划分数据集
    train_seq,test_seq,train_label,test_label = train_test_split(seqs, labels, test_size=0.2, random_state=42)
    # 加载训练集数据集
    train_dataloader = DataLoader(CommentDataset(train_seq,train_label),BATCH_SIZE,True)
    # 加载测试集数据集
    test_dataloader = DataLoader(CommentDataset(test_seq,test_label),BATCH_SIZE,False)
    # 创建LSTM模型
    model_lstm = LSTMModel(len(vocab))
    model_rnn = RNNmodel(len(vocab))
    model_gru = GRUModel(len(vocab))
    # for sentences, labels in train_dataloader:
    #     print(model(sentences))
    # 创建损失函数
    loss_fn = nn.BCEWithLogitsLoss()
    # 创建优化器
    optim_lstm = optim.Adam(model_lstm.parameters(),lr=0.01)
    optim_rnn = optim.Adam(model_rnn.parameters(),lr=0.001)
    optim_gru = optim.Adam(model_gru.parameters(),lr=0.01)


    # 开始训练
    start_lstm = time.time()
    for epoch in range(EPOCHS):
        loss_lstm,acc_lstm = train(model_lstm, train_dataloader, loss_fn, optim_lstm)
        # print(f"Epoch {epoch + 1:2d} | Loss {loss_lstm:.3f} | 训练准确率 {acc_lstm:.3f}")
    # 使用测试集对模型进行评估
    print(f"lstm准确率 {model_eva(model_lstm, test_dataloader):.3f},运行时长：{time.time() - start_lstm}")
    print("="*20)
    # 开始训练
    start_rnn = time.time()
    for epoch in range(EPOCHS):
        loss_rnn,acc_rnn = train(model_rnn, train_dataloader, loss_fn, optim_rnn)
        # print(f"Epoch {epoch + 1:2d} | Loss {loss_rnn:.3f} | 训练准确率 {acc_rnn:.3f},运行时长：{time.time() - start_lstm}")
    # 使用测试集对模型进行评估
    print(f"rnn准确率 {model_eva(model_rnn, test_dataloader):.3f},运行时长：{time.time() - start_rnn}")
    print("="*20)
    # 开始训练
    start_gru = time.time()
    for epoch in range(EPOCHS):
        loss_gru,acc_gru = train(model_gru, train_dataloader, loss_fn, optim_gru)
        # print(f"Epoch {epoch + 1:2d} | Loss {loss_gru:.3f} | 训练准确率 {acc_gru:.3f}")
    # 使用测试集对模型进行评估
    print(f"gru准确率 {model_eva(model_gru, test_dataloader):.3f},运行时长：{time.time() - start_gru}")
    print("="*20)



    # 对输入的文本进行预测
    # texts = [
    # "房间干净，服务很好，下次还来住",
    # "环境差，隔音不好，非常不满意",
    # "位置很好，但房间隔音太差了",
    # "早餐丰富，前台服务热情",
    # "卫生堪忧，床单有污渍"
    # ]
    # positive_texts = [
    # "房间干净，服务很好，下次还来住",
    # "位置绝佳，离地铁站很近，出行方便",
    # "床品舒适，隔音不错，睡得很安稳",
    # "前台热情周到，办理入住速度很快",
    # "早餐丰富好吃，卫生条件无可挑剔",
    # "性价比很高，设施齐全，体验超出预期",
    # "环境优雅，视野开阔，拍照很出片",
    # "客房服务响应及时，住得非常舒心"
    # ]
    # negative_texts = [
    # "环境差，隔音不好，非常不满意",
    # "房间有异味，床单不干净，再也不会来了",
    # "前台态度冷漠，办事效率极低",
    # "半夜隔壁特别吵，根本没法入睡",
    # "空调坏了没人修，热水也不热，体验极差",
    # "图片与实物严重不符，房间又小又旧",
    # "卫生堪忧，浴室有头发，毛巾发黄",
    # "退房时还要乱收费，非常坑人"
    # ]
    # for text in texts:
    #     res, prob = predict(model_lstm, text, vocab)
    #     print(f"{text}的预测结果是：{res}, 预测分数：{prob}")
