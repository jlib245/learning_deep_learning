import numpy as np
import torch
import matplotlib.pyplot as plt

x_np = np.arange(-4, 4.1, 0.25)
x = torch.tensor(x_np).float()
y = torch.sigmoid(x)

plt.title('시그모이드 함수의 그래프')
plt.plot(x.data, y.data)
plt.show()

from sklearn.datasets import load_iris
iris = load_iris()
x_org, y_org = iris.data, iris.target
print('원본 데이터', x_org.shape, y_org.shape)

x_data = iris.data[:100, :2]
y_data = iris.target[:100]

print('대상 데이터', x_data.shape, y_data.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=70, test_size=30, random_state=123
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

x_t0 = x_train[y_train==0]
x_t1 = x_train[y_train==1]
plt.scatter(x_t0[:,0], x_t1[:, 1], marker='x', c='b', label='0 (setosa)')
plt.scatter(x_t1[:,0], x_t1[:, 1], marker='o', c='k', label='1 (versicolor)')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.legend()
plt.show()

# 입력 차원 수
n_input = x_train.shape[1]
# 출력 차원 수
n_output = 1

print(f"n_input: {n_input}, n_output: {n_output}")

from torch import nn
# 모델 정의
class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)
        self.sigmoid = nn.Sigmoid
        
        self.l1.weight.data.fill_(1.0)
        self.l1.bias.data.fill_(1.0)
    
    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.sigmoid(x1)
        return x2

net = Net(n_input, n_output)
print(net)

from torch import optim
criterion = nn.BCELoss()
lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=lr)


inputs = torch.tensor(x_train).float()
labels = torch.tensor(y_train).float()

labels1 = labels.view((-1, 1))

inputs_test = torch.tensor(x_test).float()
labels_test = torch.tensor(y_test).float()

labels1_test = labels_test.view((-1, 1))

num_epochs = 10000
history = np.zeros((0,5))

for epoch in range(num_epochs):
    # 훈련 페이즈
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels1)
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    predicted = torch.where(outputs < 0.5, 0, 1)
    train_acc = (predicted == labels1).sum() / len(y_train)
    
    # 예측 페이즈
    outputs_test = net(inputs_test)
    loss_test = criterion(outputs_test, labels1_test)
    val_loss = loss_test.item()
    predicted_test = torch.where(outputs_test < 0.5, 0, 1)
    val_acc = (predicted_test == labels1_test).sum()/len(y_test)
    if (epoch % 10 == 0) :
        print(f"Epoch [{epoch}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}")
        item = np.array([epoch, train_loss, train_acc, val_loss, val_acc])
        history = np.vstack((history, item)) 