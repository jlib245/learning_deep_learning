import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
l1 = nn.Linear(784, 128)
l2 = nn.Linear(128, 10)
relu = nn.ReLU(inplace=True)
inputs = torch.randn(100, 784)
m1 = l1(inputs)
m2 = relu(m1)
outputs = l2(m2)
print('입력 텐서', inputs.shape)
print('출력 텐서', outputs.shape)

net2 = nn.Sequential(
    l1,
    relu,
    l2
)
outputs2 = net2(inputs)
print('입력 텐서', inputs.shape)
print('출력 텐서', outputs.shape)

np.random.seed(123)
x = np.random.randn(100,1)
y = x**2 + np.random.randn(100, 1)*0.1

x_train = x[:50, :]
x_test = x[50:, :]
y_train = y[:50, :]
y_test = y[50:, :]

plt.scatter(x_train, y_train, c='b', label='train data')
plt.scatter(x_test, y_test, c='k', marker='x', label='validate data')
plt.legend()
plt.show()