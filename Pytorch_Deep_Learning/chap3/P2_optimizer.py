import torch
import numpy as np
def pred(X):
    return W*X+B
def mse(Yp, Y):
    loss = ((Yp-Y)**2).mean()
    return loss
sample_data_1 = np.array([
    [166, 58.7],
    [176.0, 75.7],
    [171.0, 62.1],
    [173.0, 70.4],
    [169.0, 60.1]
])
x = sample_data_1[:, 0]
y = sample_data_1[:, 1]
X = x-x.mean()
Y = y-y.mean()
X = torch.tensor(X).float()
Y = torch.tensor(Y).float()
W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()

num_epochs = 500

lr = 0.001

import torch.optim as optim
optimizer = optim.SGD([W, B], lr=lr)

history = np.zeros((0,2))
for epoch in range(num_epochs):
    Yp = pred(X)
    loss = mse(Yp, Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0 :
        item = np.array([epoch, loss.item()])
        history = np.vstack((history, item))
        print(f'epoch = {epoch} loss = {loss:.4f}')
print('W = ', W.data.numpy())
print('B = ', B.data.numpy())
print(f'초기 상태 : 손실 : {history[0,1]:.4f}')
print(f'최종 상태 : 손실 : {history[-1,1]:.4f}')