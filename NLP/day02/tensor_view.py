"""
向量可视化的代码
"""
from torch.utils.tensorboard import SummaryWriter
import torch
writer = SummaryWriter(log_dir='./runs')
embd = torch.nn.Embedding(10, 8)
writer.add_embedding(embd.weight.data)
writer.close()