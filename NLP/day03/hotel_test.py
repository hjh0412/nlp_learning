import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import jieba
from torch.utils.data import DataLoader,Dataset

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embd = nn.Embedding(1,64,1)
        self.lstm = nn.LSTM(50,64)
        self.fc = nn.Linear(50,1)
    def forward(self,x):
        x = self.embd(x)
        y,(h,c) = self.lstm(x)
        return self.fc(h[-1])

class CommonDataset(Dataset):
    def __init__(self,seq,label):
        self.seq = seq
        self.label = label
    def __len__(self):
        return len(self.seq)
    def __getitem__(self, index):
        return torch.tensor(self.seq[index]),torch.tensor(self.label[index])

data = pd.read_csv("../day02/data/train.tsv", sep='\t')

model = LSTMModel()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.params(),0.001)
data_loader = DataLoader(CommonDataset(sentences,labels),50,True)

for sent,label in data_loader:
    y = model(sent)
    optimizer.zero_grad()
    loss = loss_fn(y,label)
    loss.backward()
    optimizer.step()
