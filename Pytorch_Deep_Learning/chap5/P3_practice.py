from torch import nn
import torch
import pandas as pd
import numpy as np

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)
    
    def forward(self, x):
        x1 = self.l1(x)
        return x1
    
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+",
                     skiprows=22, header=None)
x_org = np.hstack([raw_df.values[::2, :],
                   raw_df.values[1::2, :2]])
yt = raw_df.values[1::2, :2]
feature_names = np.array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
                            'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
print('원본 데이터: ', x_org.shape, yt.shape)
print('항목명: ', feature_names)

x = x_org[:, feature_names=='RM']
print('추출 후', x.shape)
print(x[:5, :])
print('정답 데이터 ')
print(yt[:5])