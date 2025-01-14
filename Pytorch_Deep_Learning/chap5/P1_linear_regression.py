from torch import nn
import torch
import numpy as np
l3 = nn.Linear(2, 3) # 입력: 2, 출력: 3
torch.manual_seed(123)
l1 = nn.Linear(1, 1)
print(l1)

for param in l1.named_parameters():
    print('name: ', param[0])
    print('tensor: ', param[1])
    print('shape: ', param[1].shape)

nn.init.constant_(l1.weight, 2.0)
nn.init.constant_(l1.bias, 1.0)

print(l1.weight)
print(l1.bias)

x_np = np.arange(-2, 2.1, 1)
x = torch.tensor(x_np).float()
x = x.view(-1,1)

print(x.shape)
print(x)

y = l1(x)
print(y.shape)
print(y.data)

l2 = nn.Linear(2, 1)
nn.init.constant_(l2.weight, 1.0)
nn.init.constant_(l2.bias, 2.0)
print(l2.weight)
print(l2.bias)

x2_np = np.array([[0.,0.],
                 [0,1],
                 [1,0],
                 [1,1]])
x2 = torch.tensor(x2_np).float()

print(x2.shape)
print(x2)

y2 = l2(x2)
print(y2.shape)
print(y2.data)

nn.init.constant_(l3.weight[0,:], 1.0)
nn.init.constant_(l3.weight[1,:], 2.0)
nn.init.constant_(l3.weight[2,:], 3.0)
nn.init.constant_(l3.bias, 2.0)

print(l3.weight)
print(l3.bias)

y3 = l3(x2)
print(y3.shape)
print(y3.data)