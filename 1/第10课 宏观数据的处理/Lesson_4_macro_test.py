import numpy as np
import pandas as pd
import time
import datetime
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

"""
 第八课目录：
 
 1、调用数据；
 2、处理因子数据；
 3、差值-比值设置；
 4、数据平滑处理；
 5、行情数据和因子数据交互；
 6、数据decay处理；
"""

# step_1 : 调用数据

# 调用行情数据
start_time = time.time()
target_path = 'D:/9_quant_course/510050_etf_renewed.csv'
target_price = pd.read_csv(target_path)
target_close = target_price[['timestamp', 'close']]
target_close['close'] = np.log(target_close['close']).diff(1) # OK,得到序列的收益率，如果加上shift(-1)，为上移一个单位
target_return = target_close

# 调用因子数据
file_add = 'D:/9_quant_course/0-1 国内生产总值(季-同比数据).csv'
fct_name = pd.read_csv(file_add)
fct_name = fct_name.replace([np.nan], 0.0)



# step_2 开始处理因子数据，引入renew_gap的概念
fct_name = fct_name.replace([np.nan], 0.0)
fct_name.reset_index(drop = True , inplace= True ) # 重新排列数据序列
renew_gap = 20
fct_name['timestamp'] = pd.to_datetime(fct_name['timestamp']) + datetime.timedelta(days=renew_gap)
fct_name = fct_name.set_index('timestamp')



# step_3 是否设置为差值或者比值，以bool函数控制
chsn_diff = False # chosn_diff

if chsn_diff == False:
    fct_name = (fct_name - fct_name.shift(1))/np.abs(fct_name.shift(1)) # 这一步是必须的，pct_change处理出现严重的bug
elif chsn_diff == True:
    fct_name = (fct_name - fct_name.shift(1)) # 将之前的比值修改为差值，看下效果怎么样

# step_4 进行平滑处理

smooth = True

def not_zero_quantile(df):
    ## 计算每一列非0的中位数
    ret_quantile=pd.Series(index=df.columns)
    for c in df.columns:
        ret_quantile[c]=(df[df[c]!=0].loc[:,[c]].quantile(0.5))
    return ret_quantile

if smooth == True: # 通过平滑处理，防止极值效应带来的数据不平稳，从而在回归过程中带来伪回归——
    fct_name_quantile=not_zero_quantile(fct_name)
    max_fct = fct_name_quantile + 6.0*(not_zero_quantile((fct_name*1.00 - fct_name_quantile).abs())) #确定上边界
    min_fct = fct_name_quantile - 6.0*(not_zero_quantile((fct_name*1.00 - fct_name_quantile).abs())) #确定下边界
    max_fct = max_fct.values*1.00000001
    min_fct = min_fct.values*1.00000001
    fct_name = fct_name.clip(min_fct, max_fct)

# plt.plot(fct_name, 'b', label='data')
# plt.title(file_add)
# plt.xlabel('time')
# plt.ylabel('height')
# plt.legend(loc='best')
# plt.show()


# stpe_5 开始和target对齐，让行情数据和因子数据之间产生交互
target_rtn = target_return.copy()

fct_name.reset_index(inplace=True) # 重新排列数据序列
target_rtn.loc[:,'timestamp'] = pd.to_datetime(target_rtn.loc[:,'timestamp']) # 把string转换为timestamp格式
fct_name.loc[:,'timestamp'] = pd.to_datetime(fct_name.loc[:,'timestamp'])

# 星期归一处理
for i in range(fct_name.shape[0]):
    if fct_name.loc[:,'timestamp'][i].weekday() == 5:
        fct_name.loc[:,'timestamp'][i] =  fct_name.loc[:,'timestamp'][i] + datetime.timedelta(days=2)# , hours=9, minutes=45
    elif fct_name.loc[:,'timestamp'][i].weekday() == 6:
        fct_name.loc[:,'timestamp'][i] =  fct_name.loc[:,'timestamp'][i] + datetime.timedelta(days=1) #, hours=9, minutes=45
    else:
        fct_name.loc[:,'timestamp'][i] = fct_name.loc[:,'timestamp'][i] # + datetime.timedelta(hours=9, minutes=45)

# print(fct_name.tail(30))


target_aim = 'close'


# 数据合并处理
fct_name = pd.merge(target_rtn, fct_name, on='timestamp', how='left')

fct_name = fct_name.dropna(subset=[target_aim])
fct_name = fct_name.fillna(method='pad')
fct_name = fct_name.fillna(method='bfill')
fct_name = fct_name.drop([target_aim], axis=1)
fct_name = fct_name.set_index('timestamp')

# fct_name.to_excel('D:/9_quant_course/fct_gdp_pro.xlsx')


# step_6 开始做decay处理

decay = True
if decay == True:
    fct_name['t_ondecay'] = 0
    for i in range(0, fct_name.shape[0], 1):
        if i == 0 :
            fct_name['t_ondecay'][i] = 0
        else:
            if fct_name[fct_name.columns[0]][i] == fct_name[fct_name.columns[0]][i-1]:
                fct_name['t_ondecay'][i] = fct_name['t_ondecay'][i-1] + 1
            else:
                fct_name['t_ondecay'][i] = 0

# fct_name.to_excel('D:/9_quant_course/fct_gdp_pro.xlsx') # 作业：看一下这里t_ondecay是怎么切换的

alpha = 0.2
for i in range(0,len(fct_name.columns),1):
    fct_name[fct_name.columns[i]] = fct_name[fct_name.columns[i]] * np.exp(-alpha * fct_name['t_ondecay'])

fct_name = fct_name.drop(['t_ondecay'], axis=1)

print(fct_name)

plt.plot(fct_name[-200:], 'b', label='data')
plt.title(file_add)
plt.xlabel('time')
plt.ylabel('height')
plt.legend(loc='best')
plt.show()

# fct_name.to_csv('D:/9_quant_course/fct_gdp.csv')

'''


'''

