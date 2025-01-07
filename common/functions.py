import numpy as np

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

# sigmoid_function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# relu_function
def relu(x) :
    return np.maximum(0, x)

# identity_function (항등 함수)
def identity_function(x) :
    return x

# softmax_function
def softmax(a) :
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# softmax_function overloarding : large number problem solved version
def softmax(a) :
    c = np.exp(a)
    exp_a = np.exp(a-c) # counterplan against overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def sum_squares_error(y,t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return ((f(x+h) - f(x-h)) / (2*h))

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # x와 형상이 같은 배열 생성
    for idx in range(x.size):
        # f(x+h)
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    
    return grad

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x