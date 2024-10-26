import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import warnings
warnings.filterwarnings('ignore')

# step_1 调用数据

# 调用行情数据
target_path = 'D:/9_quant_course/commodities_data/期货收盘价(活跃合约)铁矿石.xlsx'
target_price = pd.read_excel(target_path)[2:-2]
target_price.columns = ['timestamp', 'close']
target_price = target_price.set_index('timestamp').reset_index()
target_price['timestamp'] = pd.to_datetime(target_price['timestamp'])
target_return = target_price


# 调用因子数据
file_add = 'D:/9_quant_course/commodities_data/国内钢材期货库存.xlsx' # 铁矿石夏普比率0.6886
# file_add = 'D:/9_quant_course/commodities_data/钢材产品产销量.xlsx' # -0.9
file_add = 'D:/9_quant_course/commodities_data/钢厂原料库存(周).xlsx' # -0.22
# file_add = 'D:/9_quant_course/commodities_data/中国出口数量热轧螺纹钢当月值.xlsx' # -0.24
# file_add = 'D:/9_quant_course/commodities_data/2-3 各项PPI按大类分(月).csv' # -0.34
# file_add = 'D:/9_quant_course/commodities_data/8-9 房地产行业数据_累计同比.csv' # 
# file_add = 'D:/9_quant_course/commodities_data/行业固定资产投资(月).xlsx' # 出口数量夏普-0.6

# 今天的作业：
# 1、1月的数据，我们给他往后延，data['m1'] = data['m1'].shift(n)————n，123456，-1
# 2、对数据进行简单组合——A+B，A/B，A*B - C，————gplearn的因子


fct_name = pd.read_excel(file_add)[2:-2]
fct_name.columns = ['timestamp', 'fct_product']
fct_name = fct_name.set_index('timestamp').reset_index()
fct_name['timestamp'] = pd.to_datetime(fct_name['timestamp'])
fct_name = fct_name.replace([np.nan, np.inf, -np.inf], 0.0)


# 2023-09-30
# step_2 开始处理因子数据，引入renew_gap的概念
fct_name.reset_index(drop=True, inplace=True)
renew_gap = 1
fct_name['timestamp'] = pd.to_datetime(fct_name['timestamp']) + datetime.timedelta(days=renew_gap)
fct_name = fct_name.set_index('timestamp')


# step_3 是否设置为差值或者比值，以bool函数来控制
chsn_diff = False
if chsn_diff == False:
    fct_name = (fct_name - fct_name.shift(1))/np.abs(fct_name.shift(1)) # pct_change
elif chsn_diff == True:
    fct_name = (fct_name - fct_name.shift(1))

# fct_name = fct_name.set_index('timestamp')

# step_4 进行平滑处理,我们在进行平滑处理的时候，要对极端值进行预处理，把数据压缩在他的6倍中位数的中间，也就是一个clip过程
smooth = True
def not_zero_quantile(df):
    ret_quantile = pd.Series(index=df.columns)
    for c in df.columns:
        ret_quantile[c] = (df[df[c]!=0].loc[: , [c]].quantile(0.5))
    return ret_quantile

if smooth == True:
    fct_name_quantile = not_zero_quantile(fct_name)
    max_fct = fct_name_quantile + 6.0 * (not_zero_quantile((fct_name*1.0 - fct_name_quantile).abs())) * 1.000000001 
    min_fct = fct_name_quantile - 6.0 * (not_zero_quantile((fct_name*1.0 - fct_name_quantile).abs())) * 1.000000001
    fct_name = fct_name.clip(min_fct, max_fct, axis=1)
elif smooth == False:
    fct_name = fct_name

# fct_name = fct_name[-200:]
# plt.plot(fct_name, 'b', label='data')
# plt.title('I')
# plt.xlabel('time')
# plt.ylabel('height')
# plt.legend(loc='best')
# plt.show()



# step_5 开始让他和target对齐，行情数据和因子数据之间的交互

target_rtn = target_return
fct_name.reset_index(inplace=True)
target_rtn.loc[:,'timestamp'] = pd.to_datetime(target_rtn.loc[:, 'timestamp'])
fct_name.loc[:, 'timestamp'] = pd.to_datetime(fct_name.loc[:, 'timestamp'])

for i in range(fct_name.shape[0]):
    if fct_name.loc[:, 'timestamp'][i].weekday() == 5:
        fct_name.loc[:, 'timestamp'][i] = fct_name.loc[:, 'timestamp'][i] + datetime.timedelta(days=2)
    elif fct_name.loc[:, 'timestamp'][i].weekday() == 6:
        fct_name.loc[:, 'timestamp'][i] = fct_name.loc[:, 'timestamp'][i] + datetime.timedelta(days=1)
    else:
        fct_name.loc[:, 'timestamp'][i] = fct_name.loc[:, 'timestamp'][i]
        
target_aim = 'close'


# 数据合并处理

fct_name = pd.merge(target_rtn, fct_name, on='timestamp', how='left')
fct_name = fct_name.dropna(subset=[target_aim])
fct_name = fct_name.fillna(method='pad')
fct_name = fct_name.fillna(method='bfill')
fct_name = fct_name.drop(['close'], axis=1)
fct_name = fct_name.set_index('timestamp')

# print(fct_name)
# fct_name = fct_name[2100:]
# plt.plot(fct_name, 'b', label='data')
# plt.title('I')
# plt.xlabel('time')
# plt.ylabel('height')
# plt.legend(loc='best')
# plt.show()




# step_6 开始做decay处理 0.8
# ACT(T) =  ACT(RAW)* NP.EXP( - ALPHA * T)

decay = True
if decay == True:
    fct_name['t_ondecay'] = 0
    for i in range(0, fct_name.shape[0], 1):
        if i == 0:
            fct_name['t_ondecay'] = 0
        else:
            if fct_name[fct_name.columns[0]][i] == fct_name[fct_name.columns[0]][i-1]: # 因为我们进行了复制填充
                fct_name['t_ondecay'][i] = fct_name['t_ondecay'][i-1] + 1
            else:
                fct_name['t_ondecay'][i] = 0
    alpha = 0.2
    for i in range(0, len(fct_name.columns), 1):
        fct_name[fct_name.columns[i]] = fct_name[fct_name.columns[i]] * np.exp(-alpha * fct_name['t_ondecay'])
    fct_name = fct_name.drop(['t_ondecay'], axis=1)

fct_name = fct_name.replace([np.nan, np.inf, -np.inf], 0.0)
# fct_name.to_excel('D:/9_quant_course/000_data_for_course_decay_effect.xlsx')
# fct_name_0 = fct_name[3000:]


fct_name_0 = fct_name[2000:]
plt.plot(fct_name_0, 'b', label='data')
plt.title('FACTOR FOR TEST')
plt.xlabel('time')
plt.ylabel('height')
plt.legend(loc='best')
plt.show()

print(fct_name)
fct_name.to_csv('D:/9_quant_course/commodities_data/rb_storage_1111.csv')
