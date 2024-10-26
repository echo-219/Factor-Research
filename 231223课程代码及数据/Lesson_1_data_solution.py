import os
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

"""
1、都有什么频率的行情数据——l1一档行情、l2多档行情、分钟频/日频/周频数据
2、都有什么格式的数据—— excel、csv、pickle、feather和h5

本节课内容：
第一部分 读取本地数据并处理成为feed_data
  # 1、读取csv文件
  # 2、替换列名
  # 3、设置timestamp为索引名称
  # 4、loc和iloc提取数据
  # 5、尝试csv,Excel,pickle、feather四种文件格式的读取速度(策略在1min策略上但是模型测试10s,有没有可能前10s已经走完了所有的行情)
  >>> 算子计算太慢,bar已经走完很多了就没有用了。

  # 6、将未来t期收益率计算出来并写入文件
  # 7、replace函数处理nan,inf,-inf
  
第二部分 本地数据get以后,怎样获得实时数据更新本地数据库？
  # 1、调用pytdx库
  # 2、处理好data数据,便于后续使用
  # 3、命名current_data作为新get到的数据作为dataframe,后续和data合并(concat)
  # 4、提取current_data的数据
  # 5、核心部分：把两个部分的数据合并在一起
  
# 第三部分：数据/因子标准化处理————体会滚动标准化的意义

# 第四部分：使用matplotlib画图显示
"""

# 第一部分 读取本地数据并处理成为feed_data

# 1、读取csv文件
data_15mins = pd.read_csv('510050.SH_15.csv')
print(data_15mins)
print(data_15mins.columns)
print(data_15mins.shape)


# 2、替换列名
data_15mins.rename(columns={'etime': 'timestamp'}, inplace=True)  # 把etime这一列的列名换成timestamp
# print(data_15mins)
# print(type(data_15mins['timestamp'][0]))  # 打印出type是str类型'


data_15mins['timestamp'] = pd.to_datetime(data_15mins['timestamp'])  # 将字符串、数字或其他日期时间格式的数据转换为datetime对象
# print(type(data_15mins['timestamp'][0]))  # 打印出来是timestamp类型
# print(data_15mins.timestamp.dt.minute)
# print(data_15mins[data_15mins.timestamp.dt.minute == 0]) # 处理整点时间类型



# 3、设置timestamp为索引名称
data_15mins = data_15mins.set_index('timestamp')  # 设置timestamp为索引
# print(data_15mins.index.name)
# print(data_15mins.columns)
# data_15mins = data_15mins.reset_index()
# print(data_15mins)


# 4、loc和iloc提取数据 loc一般是列名，iloc一般是索引选取
# print(data_15mins.loc['2022-11-29'])  # 提取出2022年11月29日的数据
# print(data_15mins.iloc[-1000:])  # 提取出starting 1000行的数据

# loc和iloc提取列数据
# print(data_15mins.loc[:, ['open', 'close', 'volume']])  # 提取open和close两列数据
# print(data_15mins.iloc[:10, 0:2])  # 一定要注意iloc在使用的时候,是不包括冒号后面的数据的,截止他前面的数字,这个在数据滚动标准化方面有大用！

# 5、尝试csv（投资研究使用）,Excel,pickle（对接模型交易用）、feather四种文件格式的读取速度
# data_15mins.to_excel('510050.SH_15.xlsx') # 将文件保存为Excel格式文件
# data_15mins.to_pickle('510050.SH_15.pkl') # 将文件保存为pickle格式文件

# data_15mins = data_15mins.reset_index() # 注意这里有一个小trick
# data_15mins.to_feather('510050.SH_15.feather') # 将文件保存为feather格式文件

# start_time = time.time()
# data_15mins_csv = pd.read_csv(file_path + '510050.SH_15.csv')
# end_time_0 = time.time()
# print('================================到此使用时间为： ', end_time_0 - start_time)
# data_15mins_excel = pd.read_excel(file_path + '510050.SH_15.xlsx')
# end_time_1 = time.time()
# print('================================到此使用时间为： ', end_time_1 - end_time_0)
# data_15mins_pkl = pd.read_pickle(file_path + '510050.SH_15.pkl')
# end_time_2 = time.time()
# print('================================到此使用时间为： ', end_time_2 - end_time_1)
# data_15mins_feather = pd.read_feather(file_path + '510050.SH_15.feather')
# end_time_3 = time.time()
# print('================================到此使用时间为： ', end_time_3 - end_time_2)


# 6、将未来t期收益率计算出来并写入文件
t_delay = [1, 3, 5, 7, 9, 12]
for t in t_delay:  # 分别获得未来1,3,5,7,9,12个周期的收益率,并将其shift,作为target
    data_15mins['y_{}'.format(t)] = data_15mins['close'].shift(-t) / data_15mins['close'] - 1 # 未来多少期的收益率
# 注意：需要理解format的用法(也可以用f'{t}'),注意shift的用法,需要注意数据中出现的nan,inf,-inf等怎么处理？
# print(data_15mins)



# 7、replace函数处理nan,inf,-inf
data_15mins = data_15mins.replace([np.nan, np.inf, -np.inf], 0.0)  # 替换掉所有的nan并撤销index
# print(data_15mins)
# print(data_15mins.info())
# print(data_15mins.describe()) # 专门方便与分析后续是否需要去极值

# 第二部分 本地数据get以后,怎样获得实时数据更新本地数据库？

# 1、调用pytdx库
import pytdx
from pytdx.hq import TdxHq_API
from pytdx.exhq import TdxExHq_API

# 2、处理好data数据,便于后续使用
data = pd.read_csv('510050.SH_15.csv')
data.rename(columns={'etime': 'timestamp'}, inplace=True)  # 对其中etime一列重命名为timestamp
data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]  # 截取timestamp、高开低收、收盘价这几列
data = data[:-100]
data['timestamp'] = pd.to_datetime(data['timestamp'])



# 3、命名current_data作为新get到的数据作为dataframe,后续和data合并
# https://gitee.com/better319/pytdx/

api = TdxHq_API()
if api.connect('119.147.212.81', 7709):  # 注意这里的IP地址和数据接口
    current_data = api.to_df(
        api.get_security_bars(1, 1, '510050', 0, 800))  # 第一个1表示是15分钟的数据,其中0为5分钟K线 1 15分钟K线 2 30分钟K线 3 1小时K线 4 日K线; 第二个代表沪深市场；第四个代表无延迟数据 1得到上一周期数据 800是数据量
    api.disconnect()  # 调用完以后一定要关闭接口

print(current_data)
 
# 4、提取current_data的数据
current_data = current_data[['datetime', 'open', 'high', 'low', 'close', 'vol']]
current_data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']  # 对current_data这列重命名
current_data['timestamp'] = pd.to_datetime(current_data['timestamp'])


# 5、核心部分：把两个部分的数据合并在一起
data = pd.concat([data, current_data], axis=0)  # axis = 0代表上下连接以行为基准，如果以列则为axis = 1 
data = data.sort_values(by='timestamp', ascending=True)  # 注意这一步是非常必要的,要以timestamp作为排序基准
data = data.drop_duplicates('timestamp')  # 注意这一步非常重要,把相同的timestamp去掉，只能以timestamo为基准进行去重
data = data.reset_index(drop=True)  # 重置索引,这里的drop=True和del data['index']作用一样
data = data.set_index('timestamp')

# 作业：把所有数据都填充好


# data.to_csv(file_path + '510050.SH_15.csv')  # 最终将数据保存


# 第三部分：数据/因子标准化处理————体会滚动标准化的意义
"""
标准化的方法主要有：

(1)线性标准化法：MMIN-MAX, 标准差法 ,Z-score 
(2)非线性标准化法： 对数法, 反正切函数法,L2范数法

这里讲解Z-score标准化 (x - x.mean) / x.std
"""

# 尝试1 性能灾难的滚动标准化！全局标准化会使用到未来信息
factor_in_use = data_15mins
start_1 = time.time()
factor_value = pd.DataFrame()
for i in range(factor_in_use.shape[0] + 1):  # factor_in_use.shape[0]
    tmp = factor_in_use.iloc[:i, :]
    factors_mean = factor_in_use.iloc[:i, :].mean(axis=0)  # 原因是这里加上了iloc之后,只能计算到i的前一个,
    factors_std = factor_in_use.iloc[:i, :].std(axis=0)
    factor_data = (factor_in_use.iloc[:i, :] - factors_mean) / factors_std
    if i > 0: factor_value =  pd.concat([factor_value, factor_data.iloc[[-1]]], ignore_index=True)
print("----------Method 1 ------for loop-----{}".format(time.time() - start_1))
factor_value.to_csv('factor_value.csv')

# 尝试2 (推荐的实现方式) - 向量化滚动标准化计算,用到了cumsum() 和 expanding() 函数
factor_value_2 = pd.DataFrame()
factor_in_use_2 = factor_in_use.copy()

start_2 = time.time()
t_np = np.arange(1, factor_in_use_2.shape[0] + 1)
test_np = np.arange(1, factor_in_use_2.shape[0] + 1)[:, np.newaxis]
test_cumsum = factor_in_use_2.cumsum()
factors_mean_2 = factor_in_use_2.cumsum() / np.arange(1, factor_in_use_2.shape[0] + 1)[:, np.newaxis]
factors_std_2 = factor_in_use_2.expanding().std()
factor_value_2 = (factor_in_use_2 - factors_mean_2) / factors_std_2
print("----------Method 2 ------尝试 cumsum() 和 expanding()-----{}".format(time.time() - start_2))
factor_value_2.to_csv('factor_value_2.csv')

# # 尝试3 -  指定的窗口rolling, 这个数值是固定窗口内的zscore的值,而不是累积的所有先前的值.
factor_value_3 = pd.DataFrame()
factor_in_use_3 = factor_in_use.copy()
start_3 = time.time()
window = 500
factors_mean_3 = factor_in_use_3.rolling(window=window, min_periods=1).mean()
factors_std_3 = factor_in_use_3.rolling(window=window, min_periods=1).std()
factor_value_3 = (factor_in_use_3 - factors_mean_3) / factors_std_3
print("----------Method 3 ------指定的窗口rolling-----{}".format(time.time() - start_3))
factor_value_3.to_csv('factor_value_3.csv')

factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
factor_value = factor_value.reset_index()
factor_value.rename(columns={'index': 'timestamp'}, inplace=True)
print(factor_value)
factor_value.to_csv('factor_value_file_510050.csv')



# 第四部分：使用matplotlib画图显示--pyecharts

data_15mins_plot = data_15mins[-1000:]
data_15mins_plot = data_15mins_plot.reset_index()
highs = data_15mins_plot['high']
lows = data_15mins_plot['low']

fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111, ylabel='stock price')
highs.plot(ax=ax1, color='c', lw=2.)
lows.plot(ax=ax1, color='y', lw=1.)
plt.hlines(highs.head(200).max(), lows.index.values[0], lows.index.values[-1], linewidth=1, color='g')
plt.hlines(lows.head(200).min(), lows.index.values[0], lows.index.values[-1], linewidth=1, color='r')
plt.axvline(linewidth=2, color='b', x=lows.index.values[200], linestyle=':')
plt.legend()
plt.grid()
plt.show()
