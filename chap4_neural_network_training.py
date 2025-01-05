import numpy as np
def sum_squares_error(y,t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

# batch
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] - 1e-7)) / batch_size

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

print(numerical_gradient(lambda x : x[0]**2 + x[1]**2, np.array([3.0, 4.0])))