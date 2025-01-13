import torch
import numpy as np
import matplotlib.pyplot as plt

sample_data_1 = np.array([
    [166, 58.7],
    [176.0, 75.7],
    [171.0, 62.1],
    [173.0, 70.4],
    [169.0, 60.1]
])
print(sample_data_1)

x = sample_data_1[:, 0]
y = sample_data_1[:, 1]

plt.scatter(x, y, c='k', s=50)
plt.xlabel('$x$: 신장(cm)')
plt.ylabel('$y$: 체중(kg)')
plt.title('신장과 체중의 관계')
#plt.show()

X = x-x.mean()
Y = y-y.mean()

plt.clf()
plt.scatter(X, Y, c='k', s=50)
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.title('데이터 가공 후 신장과 체중의 관계')
#plt.show()

X = torch.tensor(X).float()
Y = torch.tensor(Y).float()

W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()

def pred(X):
    return W*X+B

Yp = pred(X)
print(Yp)
params = {'W':W, 'B':B}

#g = make_dot(Yp, params)
#display(g)

def mse(Yp, Y):
    loss = ((Yp-Y)**2).mean()
    return loss

loss = mse(Yp, Y)
print(loss)

#g = make_dot(loss, params)
#display(g)

loss.backward()
print(W.grad)
print(B.grad)

lr = 0.001

with torch.no_grad():
    W -= lr*W.grad
    B -= lr*B.grad
    
    #경삿값 초기화
    W.grad.zero_()
    B.grad.zero_()

print(W)
print(B)
print(W.grad)
print(B.grad)

# 반복 계산
W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()

num_epochs = 500
lr = 0.001
history = np.zeros((0, 2))

for epoch in range(num_epochs):
    Yp = pred(X)
    loss = mse(Yp, Y)
    loss.backward()
    with torch.no_grad():
        W -= lr*W.grad
        B -= lr*B.grad    
        W.grad.zero_()
        B.grad.zero_()
    if epoch%10==0:
        item = np.array([epoch, loss.item()])
        history = np.vstack((history, item))
        print(f'epoch = {epoch} loss = {loss:.4f}')

print('W = ', W.data.numpy())
print('B = ', B.data.numpy())
print(f'초기 상태 : 손실 : {history[0,1]:.4f}')
print(f'최종 상태 : 손실 : {history[-1,1]:.4f}')

plt.clf()
plt.plot(history[:,0], history[:,1], 'b')
plt.xlabel('repeat')
plt.ylabel('loss')
plt.title('training curve(loss)')
plt.show()

