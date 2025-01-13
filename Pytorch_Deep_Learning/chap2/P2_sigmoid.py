import torch
import numpy as np
import matplotlib.pyplot as plt

sigmoid = torch.nn.Sigmoid()

x_np = np.arange(-2, 2.1, 0.25)
print(x_np)
x = torch.tensor(x_np, requires_grad=True,dtype=torch.float32)

y = sigmoid(x)
plt.plot(x.data, y.data)
#plt.show()

z = y.sum()

#g = make_dot(z, prams={'x':x})
#display(g)

z.backward()
print(x.grad)

plt.plot(x.data, y.data, c='b', label='y')
plt.plot(x.data, x.grad.data, c='k', label='y.grad')
plt.legend()
plt.show()