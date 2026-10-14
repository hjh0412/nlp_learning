import torch
import torch.nn as nn
mytensor = torch.tensor([[1,2,3,4],[5,6,7,8]])
embd = nn.Embedding(100,64)
embd_tensor = embd(mytensor)
print(mytensor.shape)
print(embd.weight.data.shape)
print(embd_tensor.shape)