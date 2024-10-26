import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime

"""
 第八课目录：
 
 1、调用数据；
 2、处理因子数据；
 3、差值-比值设置；
 4、数据平滑处理；
 5、行情数据和因子数据交互；
 6、数据decay处理；
"""

# stpe_1 调用数据
# 调用行情数据
target_path = 'D:/9_quant_course/510050_etf_renewed.csv'
target_price = pd.read_csv(target_path)
target_close = target_price[['timestamp', 'close']]
target_close['close'] = np.log(target_close['close']).diff(1)
target_return = target_close

# 调用因子数据
file_add = 'D:/9_quant_course/6-7 北水数据_当日资金净流入(人民币).csv'
file_add = 'D:/9_quant_course/0-1 国内生产总值(季-同比数据).csv'
fct_name = pd.read_csv(file_add)
fct_name = fct_name.replace([np.nan], 0.0)



# step_2 开始处理因子数据，引入renew_gap的概念
fct_name.reset_index(drop=True, inplace=True)
renew_gap = 0
fct_name['timestamp'] = pd.to_datetime(fct_name['timestamp']) + datetime.timedelta(days=renew_gap)
fct_name = fct_name.set_index('timestamp')


# step_3 是否设置为差值或者比值，以bool函数来控制
chsn_diff = False
if chsn_diff == False:
    fct_name = (fct_name - fct_name.shift(1))/np.abs(fct_name.shift(1)) # pct_change
elif chsn_diff == True:
    fct_name = (fct_name - fct_name.shift(1))

# step_4 进行平滑处理,我们在进行平滑处理的时候，要对极端值进行预处理，把数据压缩在他的6倍中位数的中间，也就是一个clip过程
smooth = False
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

print(fct_name)
fct_name = fct_name.dropna(subset=[target_aim])
fct_name = fct_name.fillna(method='pad')
fct_name = fct_name.fillna(method='bfill')
fct_name = fct_name.drop(['close'], axis=1)
fct_name = fct_name.set_index('timestamp')

'''
# step_6 开始做decay处理
# ACT(T) =  ACT(RAW)* NP.EXP( - ALPHA * T)
decay = True
if decay == True:
    fct_name['t_ondecay'] = 0
    for i in range(0, fct_name.shape[0], 1):
        if i == 0:
            fct_name['t_ondecay'] = 0
        else:
            if fct_name[fct_name.columns[0]][i] == fct_name[fct_name.columns[0]][i-1]:
                fct_name['t_ondecay'][i] = fct_name['t_ondecay'][i-1] + 1
            else:
                fct_name['t_ondecay'][i] = 0
                
alpha = 0.2
for i in range(0, len(fct_name.columns), 1):
    fct_name[fct_name.columns[i]] = fct_name[fct_name.columns[i]] * np.exp(-alpha * fct_name['t_ondecay'])
fct_name = fct_name.drop(['t_ondecay'], axis=1)

fct_name = fct_name.clip(-2, 2) # 有效数据附近是什么样的规律，40-40-30-40-100===80，           27--32--41--60--70-100
fct_name = fct_name.clip(-2, 2) # 有效数据附近是什么样的规律，0.7%-40-30-40-1.5%===80， 流出时：-0.5%--32--41--60--70---1.5%


# print(fct_name)
# print(fct_name.shape)
# fct_name.to_csv('D:/9_quant_course/fct_north_0401.csv')
file_add = 'D:/9_quant_course/0-1 国内生产总值(季-同比数据).csv'



'''
# fct_name.to_excel('D:/9_quant_course/000_data_for_course_decay_effect.xlsx')
# fct_name_0 = fct_name[3600:]
# plt.plot(fct_name_0, 'b', label='data')
# plt.title('GDP FOR FACTORS')
# plt.xlabel('time')
# plt.ylabel('height')
# plt.legend(loc='best')
# plt.show()
print(fct_name)

def data_wash_trading(file_add, renew_gap, target_rtn, target_aim, smooth=False, decay=False, chsn_diff=False):
    fct_name = pd.read_csv(file_add)
    fct_name.reset_index(drop = True , inplace= True ) # 重新排列数据序列
    fct_name['timestamp'] = pd.to_datetime(fct_name['timestamp']) + datetime.timedelta(days=renew_gap)
    fct_name = fct_name.fillna(method='pad').fillna(method='bfill').set_index('timestamp') + 0.0000001  
    if chsn_diff == False:
        fct_name = (fct_name - fct_name.shift(1))/np.abs(fct_name.shift(1))#这一步是必须的，pct_change处理
    elif chsn_diff == True:
        fct_name = (fct_name - fct_name.shift(1)) # 将之前的比值修改为差值，看下效果怎么样
    if smooth == True:
        fct_name_quantile=not_zero_quantile(fct_name)
        max_fct = fct_name_quantile + 6.0*(not_zero_quantile((fct_name*1.00 - fct_name_quantile).abs())) #确定上边界
        min_fct = fct_name_quantile - 6.0*(not_zero_quantile((fct_name*1.00 - fct_name_quantile).abs())) #确定下边界
        max_fct = max_fct.values*1.00000001
        min_fct = min_fct.values*1.00000001
        fct_name = fct_name.clip(min_fct, max_fct)
    #plt.plot(fct_name, 'b', label='data')
    #plt.title(file_add)
    #plt.xlabel('time')
    #plt.ylabel('height')
    #plt.legend(loc='best')
    #plt.show()
    fct_name.reset_index(inplace=True) # 重新排列数据序列
    target_rtn.loc[:,'timestamp'] = pd.to_datetime(target_rtn.loc[:,'timestamp']) # 把string转换为timestamp格式
    fct_name.loc[:,'timestamp'] = pd.to_datetime(fct_name.loc[:,'timestamp'])
    for i in range(fct_name.shape[0]):
        if fct_name.loc[:,'timestamp'][i].weekday() == 5:
            fct_name.loc[:,'timestamp'][i] =  fct_name.loc[:,'timestamp'][i] + datetime.timedelta(days=2)# , hours=9, minutes=45
        elif fct_name.loc[:,'timestamp'][i].weekday() == 6:
            fct_name.loc[:,'timestamp'][i] =  fct_name.loc[:,'timestamp'][i] + datetime.timedelta(days=1) #, hours=9, minutes=45
        else:
            fct_name.loc[:,'timestamp'][i] = fct_name.loc[:,'timestamp'][i] # + datetime.timedelta(hours=9, minutes=45)
    fct_name = pd.merge(target_rtn, fct_name, on='timestamp', how='left')
    fct_name = fct_name.dropna(subset=[target_aim]).fillna(method='pad').fillna(method='bfill').drop([target_aim], axis=1).set_index('timestamp')
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
        alpha = 0.2
        for i in range(0,len(fct_name.columns),1):
            fct_name[fct_name.columns[i]] = fct_name[fct_name.columns[i]] * np.exp(-alpha * fct_name['t_ondecay'])
        fct_name = fct_name.drop(['t_ondecay'], axis=1)
    return fct_name


# dataset_1 checking
# file_add = 'E:/Factor_Work_K/0 raw data/0 raw_data_in_use/0-1 国内生产总值(季-同比数据).csv'
# renew_gap = 20
# target_rtn = target_return
# target_aim = 'close'
# factor_gdp_yoy = data_wash_trading(file_add, renew_gap, target_rtn, target_aim, smooth=True, decay=True, chsn_diff=True)
# print(factor_gdp_yoy)
# factor_gdp_yoy.to_csv('D:/9_quant_course/fct_gdp_0325.csv')


'''