from torch import nn
import torch

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)
    
    def forward(self, x):
        x1 = self.l1(x)
        return x1

inputs = torch.ones(100, 1)
n_input = 1
n_output = 1
net = Net(n_input, n_output)
outputs = net(inputs)

criterion = nn.MSELoss()
