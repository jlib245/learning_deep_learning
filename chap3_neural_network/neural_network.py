import numpy as np
import matplotlib.pylab as plt

# step_function
def step_function(x) :
    if x > 0 :
        return 1
    else :
        return 0
    
# step_function overloading with numpy
def step_function(x) :
    y = x > 0
    return y.astype(int) # 원하는 자료형
'''
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축 범위 지정
plt.show()
'''

# sigmoid_function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

'''
x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()
'''

# relu_function
def relu(x) :
    return np.maximum(0, x)

'''
B = np.array([[1,2],
              [3,4],
              [5,6]])
print(B)
print(np.ndim(B))
print(B.shape)
'''
'''
# 행렬 곱
A = np.array([[1,2],
              [3,4]])
print(A.shape)
B = np.array([[5,6],
              [7,8]])
print(B.shape)
print(np.dot(A,B)) # 행렬 곱 numpy함수
print(np.dot(A,[2,3]))
'''
'''
#신경망에서의 행렬 곱
X = np.array([1,2])
print(X.shape)
W = np.array([[1,3,5],
              [2,4,6]])
print(W.shape)
Y = np.dot(X, W)
print(Y)
print(Y.shape)
'''

# identity_function (항등 함수)
def identity_function(x) :
    return x

# 3층 신경망 구현
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5],
               [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
print(W1.shape)
print(X.shape)
print(B1.shape)
A1 = np.dot(X, W1) + B1
print(A1)
Z1 = sigmoid(A1)
print(Z1)

W2 = np.array([[0.1, 0.4],
               [0.2, 0.5],
               [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
print(Z1.shape)
print(W2.shape)
print(B2.shape)
A2 = np.dot(Z1, W2) + B2
print(A2)
Z2 = sigmoid(A2)
print(Z2)

W3 = np.array([[0.1, 0.3],
              [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)
print(Y)