# 0계 텐서(스칼라)
import torch
r0 = torch.tensor(1.0).float()
print(type(r0))
print(r0.dtype)
print(r0.shape)
print(r0.data)

import numpy as np
# 1계 텐서(벡터)
r1_np = np.array([1, 2, 3, 4, 5])
print(r1_np.shape)
r1 = torch.tensor(r1_np).float()
print(r1.dtype)
print(r1.shape)
print(r1.data)

# 2계 텐서(행렬)
r2_np = np.array([[1,  5, 6],
                  [4, 3, 2]])
r2 = torch.tensor(r2_np).float()
print(r2.dtype)
print(r2.shape)
print(r2.data)

# 3계 텐서(난수 생성)
torch.manual_seed(32)
r3 = torch.randn((3,2,2))
print(r3.shape)
print(r3.data)

# 4계 텐서(모두 1)
r4 = torch.ones((2, 3, 2, 2))
print(r4.shape)
print(r4.data)

# 정수형 텐서
r5 = r1.long()
print(r5.dtype)
print(r5)

# numpy.reshape = tensor.view
r6 = r3.view(3, -1) # 요소수 -1로 지정 시 수를 자동으로 조정 (3계 -> 2계)
print(r6.shape)
print(r6.data)

r7 = r3.view(-1) # 3계-> 1계
print(r7.shape)
print(r7.data)

# 그 외 확인
print('requires_grad: ', r1.requires_grad)
print('device: ', r1.device)

# item 함수
item = r0.item()
print(type(item))
print(item)

# max
print(r2)
print(r2.max())
print(torch.max(r2, 1))

r2_np = r2.data.numpy()
print(type(r2_np))
print(r2_np)