import torch
import numpy as np
import matplotlib.pyplot as plt

#1 경사 계산용 변수 정의
x_np = np.arange(-2, 2.1, 0.25)
print(x_np)
x = torch.tensor(x_np, requires_grad=True,dtype=torch.float32)
print(x)

#2 텐서 변수로 계산
y = 2 * x**2 + 2
print(y)

#plt.plot(x.data, y.data)
#plt.show()
z = y.sum()

#3 계산 그래프 시각화
from torchviz import make_dot
g = make_dot(z, params={'x': x})
#display(g)

#4 경사 계산
z.backward()

#5 경삿값 가져오기
print(x.grad)

plt.plot(x.data, y.data, c='b', label='y')
plt.plot(x.data, x.grad.data, c='k', label='y.grad')
plt.legend()
plt.show()

#6 경삿값의 초기화
y = 2 * x**2 + 2
z = y.sum()
z.backward()
print(x.grad)