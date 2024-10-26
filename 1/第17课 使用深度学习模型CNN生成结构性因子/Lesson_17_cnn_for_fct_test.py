import numpy as np
import pandas as pd
import talib as ta
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

start_time = time.time()

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 20, 1) # 41*20 + 20*1
        self.maxpool1 = nn.AdaptiveAvgPool1d(output_size=30)
        self.conv2 = nn.Conv1d(20, 30, 1) # 20*30 + 30*1
        self.maxpool2 = nn.AdaptiveAvgPool1d(output_size=2)
        self.fc = nn.Linear(30 * 2, output_dim) # 30*2 + 1
        # train 16000*41*50 = 3.2E 16000 / 1531 < 10
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = x.reshape(-1, x.shape[1] * x.shape[2])
        x = self.fc(x)
        return x

the_model = CNN(41, 1)
the_model = torch.load('D:/9_quant_course/cnn_for_fct_best_0506.pth')
# print(the_model)

total_params = 0
for name, param in the_model.named_parameters():
    if 'weight' or 'bias' in name:
        num_params = torch.prod(torch.tensor(param.size()))
        total_params += num_params
        
        layer_num = name.split('.')[1]  # 获取层数
        print(f"第{layer_num}层参数数量: {num_params}")

print(f"总参数数量: {total_params}")


data_array = np.load('D:/9_quant_course/data_array.npy') # 读取数据完毕
data_array_tensor = torch.from_numpy(data_array).to(torch.float32)


y_pred = the_model(data_array_tensor)
y_pred = y_pred.detach().numpy() # 把torch的数据类型，转换成为numpy
y_pred = [i[0] for i in y_pred] # （20000，）


data_50etf = pd.read_excel('D:/9_quant_course/510050.SH_15.xlsx')
data_50etf['return'] = data_50etf['close'].shift(-1) / data_50etf['close'] - 1
data_50etf = data_50etf[-20000:].reset_index(drop=True)
data_50etf['timestamp'] = pd.to_datetime(data_50etf['timestamp'])
data_50etf = data_50etf.replace([np.nan], 0.)

data_50etf['y_hat'] = y_pred
data_50etf = data_50etf[['return', 'y_hat']]
# 只截取了return和yhat作为验证数据

data_50etf.loc[data_50etf['return']*data_50etf['y_hat'] > 0, 'wins'] = 1
data_50etf.loc[data_50etf['return']*data_50etf['y_hat'] < 0, 'wins'] = 0

win_rate = data_50etf['wins'].sum() / data_50etf.shape[0] * 100
loss_rate = data_50etf[data_50etf['wins'] == 0].shape[0] / data_50etf.shape[0] * 100
#45%
# 因子数据输出，并使用轮子对他进行验证

print('该因子的获胜概率为：====', round(win_rate, 2), ' %===========')
print('该因子的亏损概率为：====', round(loss_rate, 2), ' %===========')


end_time = time.time()
print('time cost:---------', end_time-start_time, '   s----------')
