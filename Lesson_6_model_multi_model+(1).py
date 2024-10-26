"""
1、导入基础技术指标制作因子；
2、因子标准化； axis = 1
3、因子筛选：单因子成绩，corr；没有规律 - pj tucker
4、模型的coef和intercept检查；weights bias
5、模型部分，OLS RIDGE LASSO XGB LGB 模型及其使用；
6、模型的保存和调用，以及注意事项；
7、分析不同模型的成绩，发现其中的相同和不同；
8、其他的生成因子思路，因子过程中最重要的事儿；

skewness abs - 0.5以内， kurtosis在5以内

今天的作业：
# 作业：把上述5个模型的结果ensemble起来：
# 具体的方法：把各个模型输出的y_train_test_hat，
# 模式1：加起来除以5（等权），获得一个新的y_hat，再输入轮子
# 模式2：(y_hat_1 * sharpe_1 + yhat_2 * sharpe_2) / (sharpe1 + sharpe2) sharpe加权的测试方式

# 模型之前因子的筛选步骤：
  1、sharpe；2005-2016 2016年，滚动测试不能超过这个节点，必须是这个阶段之后；
  2、corr，skewness，kurtosis，不要超过Y的kurtosis，Y的尾部的数据，X不认识；  quantile regression
  3、因子择时，X-,PNL-ACF；
  4、择时区域的corr筛选；
  5、特色模型的运用——RF，CNN，GCN； EDGE VECTOR sample太少 Y
  6、滚动周期测试是基准；
  2020-1 2020-3
  2020-2 2020-4
         2024-2
  
  ATTENTION IS ALL YOU NEED   QKV

下节课预告：CNN模型构建50ETF指数的另类结构性因子

"""
# 标量，向量，矩阵
# Y-train X-train--定系数；X-TEST LOSS FUNCTION(Y-HAT Y-TEST)
# 数据，模型/范式，目标/损失函数，泛化能力，
# MSE RMSE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

import talib as ta
from lib_total import *

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb # 集成算法

# 机器学习

import joblib
import pickle


# 一、读取数据并根据talib生成各项基础因子
start_time = time.time()
file_path = 'D:/9_quant_course/510050.SH_15.pkl' # pickle feather
data_15mins_50etf = pd.read_pickle(file_path).reset_index() # 读取源文件
data_15mins_50etf['timestamp'] = pd.to_datetime(data_15mins_50etf['timestamp'])
data_15mins_50etf = data_15mins_50etf.sort_values(by='timestamp', ascending=True)
data_15mins_50etf = data_15mins_50etf.set_index('timestamp')

t = 1
data_15mins_50etf['return'] = data_15mins_50etf['close'].shift(-t)/data_15mins_50etf['close'] - 1
data_15mins_50etf = data_15mins_50etf.replace([np.nan], 0.0)

fct_value = pd.DataFrame() # 新建dataframe，命名为fct_value

# 1、ma类
fct_value['ma5'] =  ta.MA(data_15mins_50etf['close'], timeperiod = 5 , matype = 0)
fct_value['ma10'] =  ta.MA(data_15mins_50etf['close'], timeperiod = 10 , matype = 0)
fct_value['ma20'] =  ta.MA(data_15mins_50etf['close'], timeperiod = 20 , matype = 0)
fct_value['ma5diff'] = fct_value['ma5']/data_15mins_50etf['close'] - 1
fct_value['ma10diff'] = fct_value['ma10']/data_15mins_50etf['close'] - 1
fct_value['ma20diff'] = fct_value['ma20']/data_15mins_50etf['close'] - 1 # log1p()

# 2、bollinger band类
fct_value['h_line'], fct_value['m_line'], fct_value['l_line'] = ta.BBANDS(data_15mins_50etf['close'], timeperiod=20, nbdevup=2,nbdevdn=2,matype=0)
fct_value['stdevrate'] = (fct_value['h_line'] - fct_value['l_line']) / (data_15mins_50etf['close']*4)

# 3、sar因子
fct_value['sar_index'] = ta.SAR(data_15mins_50etf['high'], data_15mins_50etf['low'])
fct_value['sar_close'] = (fct_value['sar_index'] - data_15mins_50etf['close']) / data_15mins_50etf['close']

# 4、aroon
fct_value['aroon_index'] = ta.AROONOSC(data_15mins_50etf['high'], data_15mins_50etf['low'], timeperiod=14)

# 5、CCI
fct_value['cci_14'] = ta.CCI(data_15mins_50etf['close'], data_15mins_50etf['high'], data_15mins_50etf['low'], timeperiod=14)
fct_value['cci_25'] = ta.CCI(data_15mins_50etf['close'], data_15mins_50etf['high'], data_15mins_50etf['low'], timeperiod=25)
fct_value['cci_55'] = ta.CCI(data_15mins_50etf['close'], data_15mins_50etf['high'], data_15mins_50etf['low'], timeperiod=55)

# 6、CMO
fct_value['cmo_14'] = ta.CMO(data_15mins_50etf['close'], timeperiod=14)
fct_value['cmo_25'] = ta.CMO(data_15mins_50etf['close'], timeperiod=25)

# 7、MFI
fct_value['mfi_index'] = ta.MFI(data_15mins_50etf['high'], data_15mins_50etf['low'], data_15mins_50etf['close'], data_15mins_50etf['volume'])

# 8、MOM
fct_value['mom_14'] = ta.MOM(data_15mins_50etf['close'], timeperiod=14)
fct_value['mom_25'] = ta.MOM(data_15mins_50etf['close'], timeperiod=25)

# 9、
fct_value['index'] = ta.PPO(data_15mins_50etf['close'], fastperiod=12, slowperiod=26, matype=0)

# 10、AD
fct_value['ad_index'] = ta.AD(data_15mins_50etf['high'], data_15mins_50etf['low'], data_15mins_50etf['close'], data_15mins_50etf['volume'])
fct_value['ad_real'] = ta.ADOSC(data_15mins_50etf['high'], data_15mins_50etf['low'], data_15mins_50etf['close'], data_15mins_50etf['volume'], fastperiod=3, slowperiod=10)

# 11、OBV
fct_value['obv_index'] = ta.OBV(data_15mins_50etf['close'],data_15mins_50etf['volume'])
# 量纲，不能给他一上来就做标准化

# 12、ATR
fct_value['atr_14'] = ta.ATR(data_15mins_50etf['high'], data_15mins_50etf['low'], data_15mins_50etf['close'], timeperiod=14)
fct_value['atr_25'] = ta.ATR(data_15mins_50etf['high'], data_15mins_50etf['low'], data_15mins_50etf['close'], timeperiod=25)
fct_value['atr_60'] = ta.ATR(data_15mins_50etf['high'], data_15mins_50etf['low'], data_15mins_50etf['close'], timeperiod=60)
fct_value['tr_index'] = ta.TRANGE(data_15mins_50etf['high'], data_15mins_50etf['low'], data_15mins_50etf['close'])
fct_value['tr_ma5'] = ta.MA(fct_value['tr_index'], timeperiod=5, matype = 0)/data_15mins_50etf['close']
fct_value['tr_ma10'] = ta.MA(fct_value['tr_index'], timeperiod=10, matype = 0)/data_15mins_50etf['close']
fct_value['tr_ma20'] = ta.MA(fct_value['tr_index'], timeperiod=20, matype = 0)/data_15mins_50etf['close']

# 13、KD
fct_value['kdj_k'], fct_value['kdj_d'] = ta.STOCH(data_15mins_50etf['high'], data_15mins_50etf['low'], data_15mins_50etf['close'], fastk_period=9, slowk_period=5, slowk_matype=1,slowd_period=5, slowd_matype=1)
fct_value['kdj_j'] = fct_value['kdj_k'] - fct_value['kdj_d']

# 14、MACD
fct_value['macd_dif'],  fct_value['macd_dea'], fct_value['macd_hist'] = ta.MACD(data_15mins_50etf['close'], fastperiod=12, slowperiod=26, signalperiod=9)

# 15、RSI index
fct_value['rsi_6'] = ta.RSI(data_15mins_50etf['close'], timeperiod=6)
fct_value['rsi_12'] = ta.RSI(data_15mins_50etf['close'], timeperiod=12)
fct_value['rsi_25'] = ta.RSI(data_15mins_50etf['close'], timeperiod=25)

fct_value = fct_value.replace([np.nan], 0.0)

# 二、因子处理环节：标准化，筛选，最终形成因子名单和数据

# 1、因子标准化
factors_mean_2 = fct_value.cumsum() / np.arange(1, fct_value.shape[0] + 1)[:, np.newaxis]
factors_std_2 = fct_value.expanding().std()
factor_value = (fct_value-factors_mean_2) / factors_std_2
factor_value = factor_value.replace([np.nan], 0.0)
factor_value = factor_value.clip(-6, 6) # 条件概率进行分析，

# factor_value.to_csv('D:/9_quant_course/factor_base_0419.csv')
# 1 covariance drift X P(X-train) != P(X-test) 60 监控前15名因子，
# 2 label drift P(Y-TRAIN) != P(Y-TEST) X-234 Y 0.2 0.3 0.4 y -0.1 -0.7 -1.3
# 3 concept drift P(Y | X )_train != P(Y | X )_test

# 2、载入因子，观察其corr
fct_file = pd.read_csv('D:/9_quant_course/factor_base_0419.csv').set_index('timestamp')
fct_corr = fct_file.corr()
fct_file['return'] = (data_15mins_50etf['close'].shift(-1)/data_15mins_50etf['close'] - 1).values # 注意这里需要加上.values
fct_file = fct_file.replace([np.nan], 0.0)

column_list = list(set(fct_file.columns) - set(['return']))
fct_corr_05 = factor_selection_by_correlation(fct_file, column_list, 0.5) #

# print(fct_corr_05)
# print(len(fct_corr_05))

# 2、单因子成绩排序
fct_result = pd.read_csv('D:/9_quant_course/result_base_0419.csv').set_index('年份')
fct_result = fct_result.loc['样本内', :]
fct_result = fct_result.sort_values(by='夏普比率', ascending=False)
fct_selected = fct_result[fct_result['夏普比率'] > 0.2]
fct_selected_list = list(fct_selected['params'])

fct_selected_list = sorted(fct_selected_list) # string

# 因子的sharpe必须高于当前所做的品种的sharpe 

# print(fct_selected_list)
# print(len(fct_selected_list))

# 3、准备好feed_data，划分好训练集和测试集，准备喂给模型

feed_data = factor_value[fct_corr_05]
feed_data = factor_value[fct_selected_list]

fct_in_use = fct_selected_list
feed_data['y'] = fct_file['return'].values
feed_data = feed_data.reset_index()

# 4、准备训练集和测试集

train_set_end_index = feed_data[(feed_data['timestamp'].dt.year == 2016) & (feed_data['timestamp'].dt.month == 12) & (feed_data['timestamp'].dt.day == 30)  & (feed_data['timestamp'].dt.hour == 15) ].index.values[0]

X_train = feed_data[fct_in_use][ : train_set_end_index].values.reshape(-1, len(fct_in_use)) # X_train
y_train = feed_data['y'][ : train_set_end_index].values.reshape(-1, 1)
X_test = feed_data[fct_in_use][train_set_end_index : ].values.reshape(-1, len(fct_in_use))
y_test = feed_data['y'][train_set_end_index : ].values.reshape(-1, 1)

etime_train = feed_data['timestamp'][ : train_set_end_index].values # 这里使用这种写法和使用loc写法的结果不一样，loc的话最后一个数字是包含的
etime_test = feed_data['timestamp'][train_set_end_index : ].values
etime_train_test = feed_data['timestamp'].values


# 模型准备和学习过程
# 0、最简单的线性回归
model = LinearRegression(fit_intercept=True) # 简单OLS模型， 模型初始化，先把他call出来，也就是初始化
model.fit(X_train, y_train)
# print(model.coef_) # X是多少维，那么coef一定也是多少维，X_train, fct_name =  sorted(list)
# y_test_hat = model.predict(X_test)

# 0-1：模型保存和模型调用
joblib.dump(model, 'D:/9_quant_course/LRmodel.pkl') # 序列化保存模型的一个工具
estimator = joblib.load('D:/9_quant_course/LRmodel.pkl') # 经过check，保存和调用模型没问题
# y_test_hat = estimator.predict(X_test)


# 1、LassoCV和Ridge回归
model_0 = Ridge(alpha=0.2, fit_intercept=True) # ridge回归，超参数的概念
model_0.fit(X_train, y_train)


model_1 = Lasso(fit_intercept=True) # lasso回归,L1正则化的特征所决定的，那么这个lasso算法可以帮助我们去筛选因子
model_1 = LassoCV(fit_intercept=True)
model_1.fit(np.array(X_train), np.array(y_train))
# print(model_1.coef_) # lasso是筛选因子的一种重要方法


# 2、xgboost 模型阶段
# 2-1：数据准备
X_train_xgb = feed_data[fct_in_use][ : train_set_end_index]
y_train_xgb = feed_data['y'][ : train_set_end_index]
X_test_xgb = feed_data[fct_in_use][train_set_end_index : ]
y_test_xgb = feed_data['y'][train_set_end_index : ]

# 2-2 调整为xgboost独有的数据格式
xgb_train = xgb.DMatrix(X_train_xgb,label = y_train_xgb)
xgb_test = xgb.DMatrix(X_test_xgb,label = y_test_xgb)
#xgboost模型训练

# 2-3 模型调用的两种状态

model_xgb = xgb.XGBRegressor(max_depth=3,          # 每一棵树最大深度，默认6；
                        learning_rate=0.1,      # 学习率，每棵树的预测结果都要乘以这个学习率，默认0.3；# 梯度下降
                        n_estimators=100,        # 使用多少棵树来拟合，也可以理解为多少次迭代。默认100；
                        objective='reg:linear',   # 此默认参数与 XGBClassifier 不同
                        booster='gbtree',         # 有两种模型可以选择gbtree和gblinear。gbtree使用基于树的模型进行提升计算，gblinear使用线性模型进行提升计算。默认为gbtree
                        gamma=0,                 # 叶节点上进行进一步分裂所需的最小"损失减少"。默认0；
                        min_child_weight=1,      # 可以理解为叶子节点最小样本数，默认1；
                        subsample=1,              # 训练集抽样比例，每次拟合一棵树之前，都会进行该抽样步骤。默认1，取值范围(0, 1]
                        colsample_bytree=1,       # 每次拟合一棵树之前，决定使用多少个特征，参数默认1，取值范围(0, 1]。
                        reg_alpha=0,             # 默认为0，控制模型复杂程度的权重值的 L1 正则项参数，参数值越大，模型越不容易过拟合。
                        reg_lambda=1,            # 默认为1，控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                        random_state=0)           # 随机种子
# 超参数优化_——hyperopt optuna

# model_xgb = XGBRegressor() # 所有参数都为默认的时候的模型状态
# 2-4 模型训练、保存和加载
model_xgb.fit(X_train_xgb, y_train_xgb, early_stopping_rounds=20, eval_set=[(X_test_xgb, y_test_xgb)], verbose=False)
y_test_xgb = model_xgb.predict(X_test_xgb)
# print(y_test_xgb)

# model_xgb.save_model('D:/9_quant_course/xgb_model.pkl')

# 2-5 两种保存和调用模型的方式
joblib.dump(model_xgb, 'D:/9_quant_course/xgb_model.pkl')
estimator = joblib.load('D:/9_quant_course/xgb_model.pkl') # 经过check，保存和调用模型没问题
y_test_xgb = estimator.predict(X_test_xgb) # checked 


# print(y_test_xgb)

# 3、lightGBM 模型阶段 HIST

best_parameters=dict(silent=True, 
                     verbose=-1, 
                     n_jobs=16, 
                     objective='regression', 
                     metric='rmse',
                     boosting='dart', # rf,gbdt
                     learning_rate=0.054,
                     n_estimators=100, # 不要让他超过因子数
                     max_depth=5, # 不要让他超过sqrt（因字数）
                     num_leaves=32,
                     min_data_in_leaf=50,
                     min_child_weight=0.4108,
                     feature_fraction=0.5000,
                     bagging_fraction=0.5000,
                     device='gpu',
                     subsample_freq=1,
                     lambda_l1=0.05, 
                     lambda_l2=120) 

# metric, lr, maxdepth, l1, l2

lgb_train = lgb.Dataset(X_train_xgb, y_train_xgb)
lgb_val = lgb.Dataset(X_test_xgb, y_test_xgb, reference=lgb_train) # 根据Lightgbm API转化数据集格式

model_lgb = lgb.train(best_parameters,  # 设置最优超参数，注意：首先需要运行parameter_optimizaiton得到最优超参数
                    lgb_train, 
                    num_boost_round=2000, # boost round-10000 verbose val-500 early stopping rounds 200
                    valid_sets=lgb_val, 
                    verbose_eval=20, 
                    early_stopping_rounds=100)
# ===================上述为模型训练过程====================================================================
model_lgb.save_model('D:/9_quant_course/lgb_model.txt')
y_hat_lgb = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) # 评估模型：MSE




# feature_importance = model_lgb.feature_importance()




# 因子正交的代码部分：

# fct_standarized as the dataframe of fct data
# M = (fct_standarized.shape[0]-1)* np.cov(fct_standarized.T.astype(float))   # metrixM
# D,U = np.linalg.eig(M)  # get eigenvalue and eigenvector
# U = np.mat(U)           # transfered as a np metrix
# d = np.mat(np.diag(D**(-0.5)))  # get square squared of eigenvalue
# S = U * d * U.T         # get transition metrix as S
# factors_orthogonal_mat = np.mat(fct_standarized) * S   # get the Symmetric orthognal matrices
# factors_orthogonal= pd.DataFrame(factors_orthogonal_mat,columns = fct_standarized.columns,index=fct_standarized.index)   # metrix as a dataframe. task complete
# 因子聚类-800个因子，KNN KMEARNS

# 1、等权，20%-20%-20%
# 2、adaboost，


# =============注意，这里是模型切换的部分=================
# 本次一共需要输入的模型：
# model, model_0, model_1，model_xgb，model_lgb
# 分别对应的模型：OLS，ridge，lassoCV，xgboost，lightGBM
# 训练集预测
y_train_hat = model_lgb.predict(X_train)
# y_train_hat = [i[0] for i in y_train_hat] # 注意这里如果是lasso模型，不需要再使用i[0] for i in y_train_hat
print(y_train_hat)

# 测试集预测
y_test_hat = model_lgb.predict(X_test) # 计算出来预测的Y值
# y_test_hat = [i[0] for i in y_test_hat]  # 注意：此处不要试图通过归一化进行仓位映射
print(y_test_hat)

# =============注意，这里是模型切换的部分，下面再把ytrain和ytest输入到统计模型中=================


position_size = 1.0
clip_num = 2.0
#=====================================
# 把下面的写成一个函数，专门用于处理数据
#=====================================
# 其中上面的train_test_split也都需要函数化

#====================  2：测算持仓净值（训练集）
begin_date_train = pd.to_datetime(str(etime_train[0])).strftime('%Y-%m-%d %H:%M:%S')
end_date_train = pd.to_datetime(str(etime_train[-1])).strftime('%Y-%m-%d %H:%M:%S')
ret_frame_train_total = generate_etime_close_data_divd_time(begin_date_train, end_date_train, '510050', '15')

start_index = ret_frame_train_total[ret_frame_train_total['etime'] == etime_train[0]].index.values[0]
end_index = ret_frame_train_total[ret_frame_train_total['etime'] == etime_train[-1]].index.values[0]
ret_frame_train_total = ret_frame_train_total.loc[start_index: end_index, :].reset_index(drop=True)  # 进一步根据起止时刻筛选数据

#=========================开始映射仓位======================
ret_frame_train_total['position'] = [(i / 0.0005) * position_size for i in y_train_hat]  # 训练值每间隔0.0005对应仓位变化1%
ret_frame_train_total['position'] = ret_frame_train_total['position'].clip(-1*clip_num, clip_num) # 如果仓位大于1或者小于-1，不再加仓
ret_frame_train = ret_frame_train_total
ret_frame_train.loc[0, '持仓净值'] = 1 # 持仓净值的第一个数据是1，其余都是nan

for i in range(0, len(ret_frame_train), 1):
    # 计算持仓净值
    if i == 0 or ret_frame_train.loc[i - 1, 'position'] == 0:  # 如果是第一个时间步或前一个区间的结束时刻为空仓状态
        ret_frame_train.loc[i, '持仓净值'] = 1
    else:
        close_2 = ret_frame_train.loc[i, 'close']
        close_1 = ret_frame_train.loc[i - 1, 'close']
        position = abs(ret_frame_train.loc[i - 1, 'position'])  # 获取仓位大小（上一周期）
        
        if ret_frame_train.loc[i - 1, 'position'] > 0:  # 如果上一周期开的是多仓 之前是i-1，暂时删除试试
            ret_frame_train.loc[i, '持仓净值'] = 1 * (close_2 / close_1) * position + 1 * (1 - position)
        elif ret_frame_train.loc[i - 1, 'position'] < 0:  # 如果上一周期开的是空仓
            ret_frame_train.loc[i, '持仓净值'] = 1 * (1 - (close_2 / close_1 - 1)) * position + 1 * (1 - position)
            
#=========================== 3：滚动测算累计持仓净值==================================
ret_frame_train.loc[0, '持仓净值（累计）'] = 1
for i in range(1, len(ret_frame_train), 1):
    ret_frame_train.loc[i, '持仓净值（累计）'] = ret_frame_train.loc[i - 1, '持仓净值（累计）'] * ret_frame_train.loc[i, '持仓净值']

# =================  3：测算持仓净值（测试集）=========================

# 测算周期的起始日期和结束日期
begin_date_test = pd.to_datetime(str(etime_test[0])).strftime('%Y-%m-%d %H:%M:%S')
end_date_test = pd.to_datetime(str(etime_test[-1])).strftime('%Y-%m-%d %H:%M:%S')
ret_frame_test_total = generate_etime_close_data_divd_time(begin_date_test, end_date_test, '510050', '15')

# 初始化测算持仓净值的预备表格
start_index = ret_frame_test_total[ret_frame_test_total['etime'] == etime_test[0]].index.values[0]
end_index = ret_frame_test_total[ret_frame_test_total['etime'] == etime_test[-1]].index.values[0]


ret_frame_test_total = ret_frame_test_total.loc[start_index: end_index, :].reset_index(drop=True)  # 进一步根据起止时刻筛选数据
ret_frame_test_total['position'] = [(i / 0.0005) * position_size for i in y_test_hat]  # 预测值每间隔0.0005对应仓位变化1%
ret_frame_test_total['position'] = ret_frame_test_total['position'].clip(-1*clip_num, clip_num)
ret_frame_test = ret_frame_test_total
ret_frame_test = ret_frame_test.dropna(axis=0).reset_index(drop=True)  # 去除空值并重置索引
#===================== 1：初始化持仓净值 ==============================
ret_frame_test.loc[0, '持仓净值'] = 1

# 2：分周期测算持仓净值
for i in range(0, len(ret_frame_test), 1):
    # 计算持仓净值
    if i == 0 or ret_frame_test.loc[i - 1, 'position'] == 0:  # 如果是第一个时间步或前一个区间的结束时刻为空仓状态
        ret_frame_test.loc[i, '持仓净值'] = 1
    else:
        close_2 = ret_frame_test.loc[i, 'close']
        close_1 = ret_frame_test.loc[i - 1, 'close']
        position = abs(ret_frame_test.loc[i - 1, 'position'])  # 获取仓位大小（上一周期）

        if ret_frame_test.loc[i - 1, 'position'] > 0:  # 如果上一周期开的是多仓
            ret_frame_test.loc[i, '持仓净值'] = 1 * (close_2 / close_1) * position + 1 * (1 - position)
        elif ret_frame_test.loc[i - 1, 'position'] < 0:  # 如果上一周期开的是空仓
            ret_frame_test.loc[i, '持仓净值'] = 1 * (1 - (close_2 / close_1 - 1)) * position + 1 * (1 - position)

# 3：滚动测算累计持仓净值
ret_frame_test.loc[0, '持仓净值（累计）'] = 1
for i in range(1, len(ret_frame_test), 1):
    ret_frame_test.loc[i, '持仓净值（累计）'] = ret_frame_test.loc[i - 1, '持仓净值（累计）'] * ret_frame_test.loc[i, '持仓净值']


# -============  4：测算持仓净值（训练集 + 测试集）===================

# 测算周期的起始日期和结束日期
begin_date_train_test = pd.to_datetime(str(etime_train_test[0])).strftime('%Y-%m-%d %H:%M:%S')
end_date_train_test = pd.to_datetime(str(etime_train_test[-1])).strftime('%Y-%m-%d %H:%M:%S')
ret_frame_train_test_total = generate_etime_close_data_divd_time(begin_date_train_test, end_date_train_test, '510050', '15')

# 初始化测算持仓净值的预备表格
start_index = ret_frame_train_test_total[ret_frame_train_test_total['etime'] == etime_train_test[0]].index.values[0]
end_index = ret_frame_train_test_total[ret_frame_train_test_total['etime'] == etime_train_test[-1]].index.values[0]
ret_frame_train_test_total = ret_frame_train_test_total.loc[start_index: end_index, :].reset_index(drop=True)  # 进一步根据起止时刻筛选数据
ret_frame_train_test_total['position'] = [(i / 0.0005) * position_size for i in y_train_hat] + [(i / 0.0005) * position_size for i in y_test_hat]  # 训练值每间隔0.0005对应仓位变化1% + 预测值每间隔0.0005对应仓位变化1%
ret_frame_train_test_total['position'] = ret_frame_train_test_total['position'].clip(-1*clip_num, clip_num)
ret_frame_train_test = ret_frame_train_test_total
ret_frame_train_test = ret_frame_train_test.dropna(axis=0).reset_index(drop=True)  # 去除空值并重置索引

#================== 1：初始化持仓净值 =============================
ret_frame_train_test.loc[0, '持仓净值'] = 1

#================== 2：分周期测算持仓净值==========================
for i in range(0, len(ret_frame_train_test), 1):
    # 计算持仓净值
    if i == 0 or ret_frame_train_test.loc[i - 1, 'position'] == 0:  # 如果是第一个时间步或前一个区间的结束时刻为空仓状态
        ret_frame_train_test.loc[i, '持仓净值'] = 1
    else:
        close_2 = ret_frame_train_test.loc[i, 'close']
        close_1 = ret_frame_train_test.loc[i - 1, 'close']
        position = abs(ret_frame_train_test.loc[i - 1, 'position'])  # 获取仓位大小（上一周期）
        
        if ret_frame_train_test.loc[i - 1, 'position'] > 0:  # 如果上一周期开的是多仓
            ret_frame_train_test.loc[i, '持仓净值'] = 1 * (close_2 / close_1) * position + 1 * (1 - position)
        elif ret_frame_train_test.loc[i - 1, 'position'] < 0:  # 如果上一周期开的是空仓
            ret_frame_train_test.loc[i, '持仓净值'] = 1 * (1 - (close_2 / close_1 - 1)) * position + 1 * (1 - position)

# =======================3：滚动测算累计持仓净值===============================
ret_frame_train_test.loc[0, '持仓净值（累计）'] = 1
for i in range(1, len(ret_frame_train_test), 1):
    ret_frame_train_test.loc[i, '持仓净值（累计）'] = ret_frame_train_test.loc[i - 1, '持仓净值（累计）'] * ret_frame_train_test.loc[i, '持仓净值']
# ========================训练集验证集测试集数据统计完毕========================
#===================================================================================================================



#    ========================================================================================================================
#    PART 2：单因子风险指标测算
#    ========================================================================================================================

# 0：设置无风险利率和费用
fixed_return = 0.0
fees_rate = 0.004

# 1：初始化
indicators_frame = pd.DataFrame()
year_list = [i for i in ret_frame_train_test['etime'].dt.year.unique()]  # 获取年份列表
indicators_frame['年份'] = year_list + ['样本内', '样本外', '总体']
indicators_frame = indicators_frame.set_index('年份')  # 将年份列表设置为表格索引

# 2：计算风险指标（总体）
start_index = ret_frame_train_test.index[0]  # 获取总体的起始索引
end_index = ret_frame_train_test.index[-1]  # 获取总体的结束索引

# 1：总收益
net_value_2 = ret_frame_train_test.loc[end_index, '持仓净值（累计）']
net_value_1 = ret_frame_train_test.loc[start_index, '持仓净值（累计）']
total_return = net_value_2 / net_value_1 - 1
indicators_frame.loc['总体', '总收益'] = total_return

# 2：年化收益率
date_list = [i for i in ret_frame_train_test['etime'].dt.date.unique()]
run_day_length = len(date_list)  # 计算策略运行天数
annual_return = math.pow(1 + total_return, 252 / run_day_length) - 1

indicators_frame.loc['总体', '年化收益'] = annual_return

# 3：夏普比率、年化波动率
net_asset_value_list = []  # 初始化累计持仓净值列表（日度）
net_asset_value_index = [i for i in ret_frame_train_test.groupby(['tdate']).tail(1).index]  # 获取每日的结束索引

for date_index in net_asset_value_index:
    net_asset_value = ret_frame_train_test.loc[date_index, '持仓净值（累计）']
    net_asset_value_list.append(net_asset_value)  # 附加每日结束时对应的累计持仓净值

net_asset_value_frame = pd.DataFrame({'tdate': date_list, 'nav': net_asset_value_list})  # 构建日度累计持仓净值表格
net_asset_value_frame.loc[0, 'daily_log_return'] = 0  # 初始化对数收益率（日度）
for i in range(1, len(net_asset_value_frame), 1):
    net_asset_value_frame.loc[i, 'daily_log_return'] = math.log(net_asset_value_frame.loc[i, 'nav']) - math.log(net_asset_value_frame.loc[i - 1, 'nav'])  # 计算对数收益率（日度）
annual_volatility = math.sqrt(252) * net_asset_value_frame['daily_log_return'].std()  # 计算年化波动率

sharpe_ratio = (annual_return - fixed_return) / annual_volatility  # 计算夏普比率

indicators_frame.loc['总体', '年化波动率'] = annual_volatility
indicators_frame.loc['总体', '夏普比率'] = sharpe_ratio

# 4：最大回撤率及其对应的起止日（需要利用计算夏普比率过程中构建的日度累计持仓净值表格）
mdd_end_index = np.argmax((np.maximum.accumulate(net_asset_value_list) - net_asset_value_list) / (np.maximum.accumulate(net_asset_value_list)))
# if mdd_end_index == 0:return 0
mdd_end_date = net_asset_value_frame.loc[mdd_end_index, 'tdate']  # 最大回撤起始日
mdd_start_index = np.argmax(net_asset_value_list[: mdd_end_index])
mdd_start_date = net_asset_value_frame.loc[mdd_start_index, 'tdate']  # 最大回撤结束日
maximum_drawdown = (net_asset_value_list[mdd_start_index] - net_asset_value_list[mdd_end_index]) / (net_asset_value_list[mdd_start_index])  # 计算最大回撤率

indicators_frame.loc['总体', '最大回撤率'] = maximum_drawdown
indicators_frame.loc['总体', '最大回撤起始日'] = mdd_start_date
indicators_frame.loc['总体', '最大回撤结束日'] = mdd_end_date

# 5：卡尔玛比率（基于夏普比率以及最大回撤率）
calmar_ratio = (annual_return - fixed_return) / maximum_drawdown  # 计算卡尔玛比率

indicators_frame.loc['总体', '卡尔玛比率'] = calmar_ratio

# 6：总交易次数、交易胜率、交易盈亏比
total_trading_times = len(ret_frame_train_test)  # 计算总交易次数
win_times = 0  # 初始化盈利次数
win_lose_frame = pd.DataFrame()  # 初始化盈亏表格

for i in range(1, len(ret_frame_train_test), 1):
    delta_value = ret_frame_train_test.loc[i, '持仓净值（累计）'] - ret_frame_train_test.loc[i - 1, '持仓净值（累计）']  # 计算每次交易过程中累计持仓净值的变化量
    win_lose_frame.loc[i, 'delta_value'] = delta_value
    if delta_value > 0:
        win_times = win_times + 1

gain_amount = abs(win_lose_frame[win_lose_frame['delta_value'] > 0]['delta_value'].sum())  # 计算总盈利额
loss_amount = abs(win_lose_frame[win_lose_frame['delta_value'] < 0]['delta_value'].sum())  # 计算总亏损额
winning_rate = win_times / total_trading_times  # 计算胜率
gain_loss_ratio = gain_amount / loss_amount  # 计算盈亏比

indicators_frame.loc['总体', '总交易次数'] = total_trading_times
indicators_frame.loc['总体', '胜率'] = winning_rate
indicators_frame.loc['总体', '盈亏比'] = gain_loss_ratio


# 3：计算风险指标（分年度）
for year in year_list:
    data_demo = ret_frame_train_test[ret_frame_train_test['etime'].dt.year == year]  # 提取数据
    data_demo = data_demo.reset_index(drop=True)  # 重置索引
    data_demo['持仓净值（累计）'] = data_demo['持仓净值（累计）'] / data_demo.loc[0, '持仓净值（累计）']  # 缩放区间内部累计持仓净值

    start_index = data_demo.index[0]  # 获取当年的起始索引
    end_index = data_demo.index[-1]  # 获取当年的结束索引
    # 1：总收益
    net_value_2 = data_demo.loc[end_index, '持仓净值（累计）']
    net_value_1 = data_demo.loc[start_index, '持仓净值（累计）']
    total_return = net_value_2 / net_value_1 - 1

    indicators_frame.loc[year, '总收益'] = total_return

    # 2：年化收益率
    date_list = [i for i in data_demo['etime'].dt.date.unique()]
    run_day_length = len(date_list)  # 计算策略运行天数
    annual_return = math.pow(1 + total_return, 252 / run_day_length) - 1

    indicators_frame.loc[year, '年化收益'] = annual_return

    # 3：夏普比率、年化波动率
    net_asset_value_list = []  # 初始化累计持仓净值列表（日度）
    net_asset_value_index = [i for i in data_demo.groupby(['tdate']).tail(1).index]  # 获取每日的结束索引

    for date_index in net_asset_value_index:
        net_asset_value = data_demo.loc[date_index, '持仓净值（累计）']
        net_asset_value_list.append(net_asset_value)  # 附加每日结束时对应的累计持仓净值
    
    net_asset_value_frame = pd.DataFrame({'tdate': date_list, 'nav': net_asset_value_list})  # 构建日度累计持仓净值表格
    net_asset_value_frame.loc[0, 'daily_log_return'] = 0  # 初始化对数收益率（日度）

    for i in range(1, len(net_asset_value_frame), 1):
        net_asset_value_frame.loc[i, 'daily_log_return'] = math.log(net_asset_value_frame.loc[i, 'nav']) - math.log(net_asset_value_frame.loc[i - 1, 'nav'])  # 计算对数收益率（日度）
    
    annual_volatility = math.sqrt(252) * net_asset_value_frame['daily_log_return'].std()  # 计算年化波动率
    sharpe_ratio = (annual_return - fixed_return) / annual_volatility  # 计算夏普比率

    indicators_frame.loc[year, '年化波动率'] = annual_volatility
    indicators_frame.loc[year, '夏普比率'] = sharpe_ratio

    # 4：最大回撤率及其对应的起止日（需要利用计算夏普比率过程中构建的日度累计持仓净值表格）
    mdd_end_index = np.argmax((np.maximum.accumulate(net_asset_value_list) - net_asset_value_list) / (np.maximum.accumulate(net_asset_value_list)))
    # if mdd_end_index == 0:return 0
    mdd_end_date = net_asset_value_frame.loc[mdd_end_index, 'tdate']  # 最大回撤起始日
    mdd_start_index = np.argmax(net_asset_value_list[: mdd_end_index])
    mdd_start_date = net_asset_value_frame.loc[mdd_start_index, 'tdate']  # 最大回撤结束日
    maximum_drawdown = (net_asset_value_list[mdd_start_index] - net_asset_value_list[mdd_end_index]) / (net_asset_value_list[mdd_start_index])  # 计算最大回撤率

    indicators_frame.loc[year, '最大回撤率'] = maximum_drawdown
    indicators_frame.loc[year, '最大回撤起始日'] = mdd_start_date
    indicators_frame.loc[year, '最大回撤结束日'] = mdd_end_date

    # 5：卡尔玛比率（基于夏普比率以及最大回撤率）
    calmar_ratio = (annual_return - fixed_return) / maximum_drawdown  # 计算卡尔玛比率

    indicators_frame.loc[year, '卡尔玛比率'] = calmar_ratio

    # 6：总交易次数、交易胜率、交易盈亏比
    total_trading_times = len(data_demo)  # 计算总交易次数
    
    win_times = 0  # 初始化盈利次数
    win_lose_frame = pd.DataFrame()  # 初始化盈亏表格
    
    for i in range(1, len(data_demo), 1):
        delta_value =  data_demo.loc[i, '持仓净值（累计）'] - data_demo.loc[i - 1, '持仓净值（累计）']  # 计算每次交易过程中累计持仓净值的变化量
        win_lose_frame.loc[i, 'delta_value'] = delta_value
        if delta_value > 0:
            win_times = win_times + 1
    
    gain_amount = abs(win_lose_frame[win_lose_frame['delta_value'] > 0]['delta_value'].sum())  # 计算总盈利额
    loss_amount = abs(win_lose_frame[win_lose_frame['delta_value'] < 0]['delta_value'].sum())  # 计算总亏损额

    winning_rate = win_times / total_trading_times  # 计算胜率
    gain_loss_ratio = gain_amount / loss_amount  # 计算盈亏比

    indicators_frame.loc[year, '总交易次数'] = total_trading_times
    indicators_frame.loc[year, '胜率'] = winning_rate
    indicators_frame.loc[year, '盈亏比'] = gain_loss_ratio

# -=====================4：计算风险指标（样本内）=======================================
start_index = ret_frame_train.index[0]  # 获取训练集的起始索引
end_index = ret_frame_train.index[-1]  # 获取训练集的结束索引

# 1：总收益
net_value_2 = ret_frame_train.loc[end_index, '持仓净值（累计）']
net_value_1 = ret_frame_train.loc[start_index, '持仓净值（累计）']
total_return = net_value_2 / net_value_1 - 1

indicators_frame.loc['样本内', '总收益'] = total_return

# 2：年化收益率
date_list = [i for i in ret_frame_train['etime'].dt.date.unique()]
run_day_length = len(date_list)  # 计算策略运行天数
annual_return = math.pow(1 + total_return, 252 / run_day_length) - 1

indicators_frame.loc['样本内', '年化收益'] = annual_return

# 3：夏普比率、年化波动率
net_asset_value_list = []  # 初始化累计持仓净值列表（日度）
net_asset_value_index = [i for i in ret_frame_train.groupby(['tdate']).tail(1).index]  # 获取每日的结束索引

for date_index in net_asset_value_index:
    net_asset_value = ret_frame_train.loc[date_index, '持仓净值（累计）']
    net_asset_value_list.append(net_asset_value)  # 附加每日结束时对应的累计持仓净值

net_asset_value_frame = pd.DataFrame({'tdate': date_list, 'nav': net_asset_value_list})  # 构建日度累计持仓净值表格
net_asset_value_frame.loc[0, 'daily_log_return'] = 0  # 初始化对数收益率（日度）

for i in range(1, len(net_asset_value_frame), 1):
    net_asset_value_frame.loc[i, 'daily_log_return'] = math.log(net_asset_value_frame.loc[i, 'nav']) - math.log(net_asset_value_frame.loc[i - 1, 'nav'])  # 计算对数收益率（日度）

annual_volatility = math.sqrt(252) * net_asset_value_frame['daily_log_return'].std()  # 计算年化波动率
sharpe_ratio = (annual_return - fixed_return) / annual_volatility  # 计算夏普比率

indicators_frame.loc['样本内', '年化波动率'] = annual_volatility
indicators_frame.loc['样本内', '夏普比率'] = sharpe_ratio

# 4：最大回撤率及其对应的起止日（需要利用计算夏普比率过程中构建的日度累计持仓净值表格）
mdd_end_index = np.argmax((np.maximum.accumulate(net_asset_value_list) - net_asset_value_list) / (np.maximum.accumulate(net_asset_value_list)))
# if mdd_end_index == 0:return 0
mdd_end_date = net_asset_value_frame.loc[mdd_end_index, 'tdate']  # 最大回撤起始日
mdd_start_index = np.argmax(net_asset_value_list[: mdd_end_index])
mdd_start_date = net_asset_value_frame.loc[mdd_start_index, 'tdate']  # 最大回撤结束日
maximum_drawdown = (net_asset_value_list[mdd_start_index] - net_asset_value_list[mdd_end_index]) / (net_asset_value_list[mdd_start_index])  # 计算最大回撤率

indicators_frame.loc['样本内', '最大回撤率'] = maximum_drawdown
indicators_frame.loc['样本内', '最大回撤起始日'] = mdd_start_date
indicators_frame.loc['样本内', '最大回撤结束日'] = mdd_end_date

# 5：卡尔玛比率（基于夏普比率以及最大回撤率）
calmar_ratio = (annual_return - fixed_return) / maximum_drawdown  # 计算卡尔玛比率

indicators_frame.loc['样本内', '卡尔玛比率'] = calmar_ratio

# 6：总交易次数、交易胜率、交易盈亏比
total_trading_times = len(ret_frame_train)  # 计算总交易次数
win_times = 0  # 初始化盈利次数
win_lose_frame = pd.DataFrame()  # 初始化盈亏表格

for i in range(1, len(ret_frame_train), 1):
    delta_value = ret_frame_train.loc[i, '持仓净值（累计）'] - ret_frame_train.loc[i - 1, '持仓净值（累计）']  # 计算每次交易过程中累计持仓净值的变化量
    win_lose_frame.loc[i, 'delta_value'] = delta_value
    if delta_value > 0:
        win_times = win_times + 1

gain_amount = abs(win_lose_frame[win_lose_frame['delta_value'] > 0]['delta_value'].sum())  # 计算总盈利额
loss_amount = abs(win_lose_frame[win_lose_frame['delta_value'] < 0]['delta_value'].sum())  # 计算总亏损额
winning_rate = win_times / total_trading_times  # 计算胜率
gain_loss_ratio = gain_amount / loss_amount  # 计算盈亏比

indicators_frame.loc['样本内', '总交易次数'] = total_trading_times
indicators_frame.loc['样本内', '胜率'] = winning_rate
indicators_frame.loc['样本内', '盈亏比'] = gain_loss_ratio

#==========================5：计算风险指标（样本外）===========================

start_index = ret_frame_test.index[0]  # 获取测试集的起始索引
end_index = ret_frame_test.index[-1]  # 获取测试集的结束索引

# 1：总收益
net_value_2 = ret_frame_test.loc[end_index, '持仓净值（累计）']
net_value_1 = ret_frame_test.loc[start_index, '持仓净值（累计）']
total_return = net_value_2 / net_value_1 - 1

indicators_frame.loc['样本外', '总收益'] = total_return

# 2：年化收益率
date_list = [i for i in ret_frame_test['etime'].dt.date.unique()]
run_day_length = len(date_list)  # 计算策略运行天数
annual_return = math.pow(1 + total_return, 252 / run_day_length) - 1

indicators_frame.loc['样本外', '年化收益'] = annual_return

# 3：夏普比率、年化波动率
net_asset_value_list = []  # 初始化累计持仓净值列表（日度）
net_asset_value_index = [i for i in ret_frame_test.groupby(['tdate']).tail(1).index]  # 获取每日的结束索引

for date_index in net_asset_value_index:
    net_asset_value = ret_frame_test.loc[date_index, '持仓净值（累计）']
    net_asset_value_list.append(net_asset_value)  # 附加每日结束时对应的累计持仓净值

net_asset_value_frame = pd.DataFrame({'tdate': date_list, 'nav': net_asset_value_list})  # 构建日度累计持仓净值表格
net_asset_value_frame.loc[0, 'daily_log_return'] = 0  # 初始化对数收益率（日度）

for i in range(1, len(net_asset_value_frame), 1):
    net_asset_value_frame.loc[i, 'daily_log_return'] = math.log(net_asset_value_frame.loc[i, 'nav']) - math.log(net_asset_value_frame.loc[i - 1, 'nav'])  # 计算对数收益率（日度）

annual_volatility = math.sqrt(252) * net_asset_value_frame['daily_log_return'].std()  # 计算年化波动率
sharpe_ratio = (annual_return - fixed_return) / annual_volatility  # 计算夏普比率

indicators_frame.loc['样本外', '年化波动率'] = annual_volatility
indicators_frame.loc['样本外', '夏普比率'] = sharpe_ratio

# 4：最大回撤率及其对应的起止日（需要利用计算夏普比率过程中构建的日度累计持仓净值表格）
mdd_end_index = np.argmax((np.maximum.accumulate(net_asset_value_list) - net_asset_value_list) / (np.maximum.accumulate(net_asset_value_list)))
# if mdd_end_index == 0:return 0
mdd_end_date = net_asset_value_frame.loc[mdd_end_index, 'tdate']  # 最大回撤起始日
mdd_start_index = np.argmax(net_asset_value_list[: mdd_end_index])
mdd_start_date = net_asset_value_frame.loc[mdd_start_index, 'tdate']  # 最大回撤结束日
maximum_drawdown = (net_asset_value_list[mdd_start_index] - net_asset_value_list[mdd_end_index]) / (net_asset_value_list[mdd_start_index])  # 计算最大回撤率

indicators_frame.loc['样本外', '最大回撤率'] = maximum_drawdown
indicators_frame.loc['样本外', '最大回撤起始日'] = mdd_start_date
indicators_frame.loc['样本外', '最大回撤结束日'] = mdd_end_date

# 5：卡尔玛比率（基于夏普比率以及最大回撤率）
calmar_ratio = (annual_return - fixed_return) / maximum_drawdown  # 计算卡尔玛比率

indicators_frame.loc['样本外', '卡尔玛比率'] = calmar_ratio

# 6：总交易次数、交易胜率、交易盈亏比
total_trading_times = len(ret_frame_test)  # 计算总交易次数

win_times = 0  # 初始化盈利次数
win_lose_frame = pd.DataFrame()  # 初始化盈亏表格

for i in range(1, len(ret_frame_test), 1):
    delta_value = ret_frame_test.loc[i, '持仓净值（累计）'] - ret_frame_test.loc[i - 1, '持仓净值（累计）']  # 计算每次交易过程中累计持仓净值的变化量
    win_lose_frame.loc[i, 'delta_value'] = delta_value
    if delta_value > 0:
        win_times = win_times + 1

gain_amount = abs(win_lose_frame[win_lose_frame['delta_value'] > 0]['delta_value'].sum())  # 计算总盈利额
loss_amount = abs(win_lose_frame[win_lose_frame['delta_value'] < 0]['delta_value'].sum())  # 计算总亏损额
winning_rate = win_times / total_trading_times  # 计算胜率
gain_loss_ratio = gain_amount / loss_amount  # 计算盈亏比

indicators_frame.loc['样本外', '总交易次数'] = total_trading_times
indicators_frame.loc['样本外', '胜率'] = winning_rate
indicators_frame.loc['样本外', '盈亏比'] = gain_loss_ratio

print(indicators_frame)


plot_output = ret_frame_test['持仓净值（累计）']
plot_output.index = etime_test
plt.figure(figsize=(8,6))
plt.plot(plot_output, 'b-', label='Test curve')
plt.legend()
plt.grid()
plt.xlabel('Model_output')
plt.ylabel('Return on lvg')
plt.show()

end_time = time.time()
print('time cost:      ', end_time-start_time)
