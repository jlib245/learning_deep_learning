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
    return y.astype(int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축 범위 지정
plt.show()