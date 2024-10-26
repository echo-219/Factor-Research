import numpy as np
import pandas as pd
import time
import os
import  matplotlib.image as img
import talib as ta
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

import numba
from numba import jit
from tqdm import tqdm

"""
内容提纲：
0、CNN，RNN（lstm gru） transformer
1、图片和CNN模型介绍；
2、数据前期准备及构造方式；
3、模型构建及训练；
4、模型保存和调用；
"""


# 1、数据前期筹备
start_time = time.time()
file_path = 'D:/9_quant_course/stock_50_data_all/'
dataframe = pd.DataFrame()  # 初始化汇总表格
dirs = os.listdir(file_path)  # 获取文件夹路径下的全部文件名称
dirs = sorted(dirs)  # 文件名称排序

stock_order = ['600519', '601857', '601398', '601288', '601628', '600036', '600028', '601318', '600900', '601088',
               '601166', '603288', '601888', '600276', '600809', '600309', '601668', '601012', '601899', '600030',
               '601601', '601225', '600887', '600438', '600406', '603259', '600048', '601633', '600104', '601066',
               '600436', '600690', '600905', '601919', '600031', '601669', '601995', '600893', '600346', '603501',
               '600585', '688599', '601688', '600111', '603260', '603799', '600010', '603986', '600196', '600745',
               ] # 1、按照市值权重进行排序；2、按照行业进行排序；3、按照近期动量进行排序；

stock_order_list = [(i + '.SH_15.csv') for i in stock_order]


# 20000*41*50

data_50etf = pd.read_excel('D:/9_quant_course/510050.SH_15.xlsx')
data_50etf['return'] = data_50etf['close'].shift(-1) / data_50etf['close'] - 1
data_50etf = data_50etf[-20000:].reset_index(drop=True)
data_50etf['timestamp'] = pd.to_datetime(data_50etf['timestamp'])
data_50etf = data_50etf.replace([np.nan], 0.)
time_period = list(data_50etf.timestamp)
return_series = data_50etf['return'].values
del data_50etf['return']

fct_length = 41
'''
def data_solution(name, start_time, end_time):
    data = pd.read_csv(file_path + name)
    data['stock_code'] = str(name)
    
    # 1、ma类
    data['ma5'] =  ta.MA(data['close'], timeperiod = 5 , matype = 0)
    data['ma10'] =  ta.MA(data['close'], timeperiod = 10 , matype = 0)
    data['ma20'] =  ta.MA(data['close'], timeperiod = 20 , matype = 0)
    data['ma5diff'] = data['ma5']/data['close'] - 1
    data['ma10diff'] = data['ma10']/data['close'] - 1
    data['ma20diff'] = data['ma20']/data['close'] - 1
    
    # 2、bollinger band类
    data['h_line'], data['m_line'], data['l_line'] = ta.BBANDS(data['close'], timeperiod=20, nbdevup=2,nbdevdn=2,matype=0)
    data['stdevrate'] = (data['h_line'] - data['l_line']) / (data['close']*4)

    # 3、sar因子
    data['sar_index'] = ta.SAR(data['high'], data['low'])
    data['sar_close'] = (data['sar_index'] - data['close']) / data['close']

    # 4、aroon
    data['aroon_index'] = ta.AROONOSC(data['high'], data['low'], timeperiod=14)

    # 5、CCI
    data['cci_14'] = ta.CCI(data['close'], data['high'], data['low'], timeperiod=14)
    data['cci_25'] = ta.CCI(data['close'], data['high'], data['low'], timeperiod=25)
    data['cci_55'] = ta.CCI(data['close'], data['high'], data['low'], timeperiod=55)

    # 6、CMO
    data['cmo_14'] = ta.CMO(data['close'], timeperiod=14)
    data['cmo_25'] = ta.CMO(data['close'], timeperiod=25)

    # 7、MFI
    data['mfi_index'] = ta.MFI(data['high'], data['low'], data['close'], data['volume'])

    # 8、MOM
    data['mom_14'] = ta.MOM(data['close'], timeperiod=14)
    data['mom_25'] = ta.MOM(data['close'], timeperiod=25)

    # 9、
    data['index'] = ta.PPO(data['close'], fastperiod=12, slowperiod=26, matype=0)

    # 10、AD
    data['ad_index'] = ta.AD(data['high'], data['low'], data['close'], data['volume'])
    data['ad_real'] = ta.ADOSC(data['high'], data['low'], data['close'], data['volume'], fastperiod=3, slowperiod=10)

    # 11、OBV
    data['obv_index'] = ta.OBV(data['close'],data['volume'])

    # 12、ATR
    data['atr_14'] = ta.ATR(data['high'], data['low'], data['close'], timeperiod=14)
    data['atr_25'] = ta.ATR(data['high'], data['low'], data['close'], timeperiod=25)
    data['atr_60'] = ta.ATR(data['high'], data['low'], data['close'], timeperiod=60)
    data['tr_index'] = ta.TRANGE(data['high'], data['low'], data['close'])
    data['tr_ma5'] = ta.MA(data['tr_index'], timeperiod=5, matype = 0)/data['close']
    data['tr_ma10'] = ta.MA(data['tr_index'], timeperiod=10, matype = 0)/data['close']
    data['tr_ma20'] = ta.MA(data['tr_index'], timeperiod=20, matype = 0)/data['close']

    # 13、KD
    data['kdj_k'], data['kdj_d'] = ta.STOCH(data['high'], data['low'], data['close'], fastk_period=9, slowk_period=5, slowk_matype=1,slowd_period=5, slowd_matype=1)
    data['kdj_j'] = data['kdj_k'] - data['kdj_d']

    # 14、MACD
    data['macd_dif'],  data['macd_dea'], data['macd_hist'] = ta.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # 15、RSI index
    data['rsi_6'] = ta.RSI(data['close'], timeperiod=6)
    data['rsi_12'] = ta.RSI(data['close'], timeperiod=12)
    data['rsi_25'] = ta.RSI(data['close'], timeperiod=25)

    data = data.replace([np.nan], 0.0)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data[(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)]
    data = data.drop(columns=['open', 'high', 'low', 'close', 'volume'], axis=1)
    data = data.reset_index(drop=True)
    # 需要增加一个滚动标准化# 作业
    
    return data

data_array = np.zeros((data_50etf.shape[0], fct_length, len(stock_order))) # 20000*41*50

for s, name in enumerate(stock_order_list):
    data = data_solution(name, time_period[0], time_period[-1])
    data_combo = pd.merge(data_50etf, data, on='timestamp', how='left')
    data_combo = data_combo.fillna(method='pad').fillna(method='bfill')
    data_combo = data_combo.drop(columns=['open', 'high', 'low', 'close', 'volume', 'amount', 'stock_code'], axis=1)
    data_combo = data_combo.set_index('timestamp')
    
    factors_mean = data_combo.cumsum() / np.arange(1, data_combo.shape[0] + 1)[:, np.newaxis]
    factors_std = data_combo.expanding().std()
    factor_value = (data_combo-factors_mean) / factors_std
    data_combo = factor_value.replace([np.nan], 0.0)
    data_combo = data_combo.clip(-6, 6)
    # print(data_combo)
    
    for i in range(0, data_combo.shape[0], 1): # data_combo.shape[0]
        for j in range(0, data_combo.shape[1]):
            data_array[i][j][s] = data_combo.iloc[i, j] # 
    # print(data_array) 

# print(data_array)
# print(data_array.shape)

data_array = np.array(data_array)
# np.save('D:/9_quant_course/data_array.npy', data_array)
'''

# 3、定义CNN模型必备参数
class Config():
    batch_size = 256  # 批次大小 sequence,
    output_dim = 1
    input_dim = 41 # zhongyao
    epochs = 50
    
    best_loss = 1000. 
    learning_rate = 0.001
    model_name = 'cnn_for_fct' # 模型名称
    best_save_path = 'D:/9_quant_course/{}_best.pth'.format(model_name) # 最优模型保存路径
    curr_save_path = 'D:/9_quant_course/{}_curr.pth'.format(model_name) # 当前模型保存路径
    
config = Config() # 调用模型


data_array = np.load('D:/9_quant_course/data_array.npy') # 加载所需数据
# print(data_array)
# print(data_array.shape)

x_train = torch.from_numpy(data_array[ : 16000]).to(torch.float32)
y_train = torch.from_numpy(return_series[ : 16000]).to(torch.float32)
x_test = torch.from_numpy(data_array[16000 : ]).to(torch.float32)
y_test = torch.from_numpy(return_series[16000 : ]).to(torch.float32)

# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)



train_data = TensorDataset(x_train, y_train) 
test_data = TensorDataset(x_test, y_test)


train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = config.batch_size, shuffle = True, drop_last = False)
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = config.batch_size, shuffle = False, drop_last = False)

# 5、开始搭建CNN模型  定义一维卷积模块
class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 20, 1) # 一维卷积-求均值 二维卷积-傅里叶变换
        self.maxpool1 = nn.AdaptiveAvgPool1d(output_size=30)
        self.conv2 = nn.Conv1d(20, 30, 1)
        self.maxpool2 = nn.AdaptiveAvgPool1d(output_size=2)
        self.fc = nn.Linear(30 * 2, output_dim)
    
    def forward(self, x): # 前向传播-后向传播
        # print(x.shape, 'inputting data')
        x = self.conv1(x)
        # print(x.shape, 'conv1 layer')
        x = self.maxpool1(x)
        # print(x.shape, 'maxpooling 1')
        x = self.conv2(x)
        # print(x.shape, 'conv2====')
        x = self.maxpool2(x)
        # print(x.shape, 'maxpooling  2')
        x = x.reshape(-1, x.shape[1] * x.shape[2])
        # print(x.shape, 'reshaped')
        x = self.fc(x)
        # DROPOUT(0.2)
        return x
    

# 初始化模型
model = CNN(input_dim=config.input_dim, output_dim=config.output_dim) # 模型初始化
loss_function = nn.MSELoss()  # 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)  # 定义优化器



# 6、训练模型过程

train_loss_all = []
test_loss_all = []

for epoch in range(config.epochs):
    model.train()
    train_loss = 0
    train_num = 0
    train_bar = tqdm(train_loader)  # 形成进度条
    for i, data in enumerate(train_bar): # 16000 / 256 
        x_train, y_train = data  # 解包迭代器中的X和Y
        optimizer.zero_grad() # 梯度清零
        y_train_pred = model(x_train) # 模型作用于xtrain，获得本次的预测结果
        loss = loss_function(y_train_pred, y_train.reshape(-1, 1))
        loss.backward() # 反向传播
        optimizer.step() # 优化器更新

        train_loss += loss.item()*x_train.size(0)
        train_num += x_train.size(0)
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 config.epochs,
                                                                 loss)
    train_loss_all.append(train_loss / train_num)

    # 7、模型验证
    model.eval()
    test_loss = 0
    test_num = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            x_test, y_test = data
            y_test_pred = model(x_test)
            loss = loss_function(y_test_pred, y_test.reshape(-1, 1))
            test_loss += loss.item() * x_test.size(0)
            test_num += x_test.size(0)
    test_loss_all.append(test_loss / test_num)
    
    torch.save(model, config.curr_save_path)

    if test_loss < config.best_loss:
        best_loss = test_loss
        torch.save(model, config.best_save_path)

print('Finished Training')

# 8.绘制结果
plt.figure(figsize=(12, 8))
plt.plot(train_loss_all, "b", label='train_loss')
plt.plot(test_loss_all, "r", label='test_loss')
plt.legend()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.yscale("log") # 更能体现出数据的起伏
plt.show()


end_time = time.time()
print('time cost:      ', end_time-start_time)
