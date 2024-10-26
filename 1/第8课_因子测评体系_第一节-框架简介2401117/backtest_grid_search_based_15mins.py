# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 10:40:41 2022

@author: P15
"""
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression


'''
========================================================================================================================
生成数据表格（列名：etime、tdate、close）
========================================================================================================================
'''

def generate_etime_close_data_divd_time(bgn_date, end_date, index_code, frequency): # 2022-11-25 edition last edited on 2023-01-17
    # 读取数据
    # read_file_path = 'D:/9_quant_course/' + index_code + '_' + frequency + '.xlsx'
    read_file_path = 'D:/9_quant_course/' + index_code + '_' + frequency + '.xlsx'
    kbars = pd.read_excel(read_file_path)
    kbars['tdate'] = pd.to_datetime(kbars['etime']).dt.date # 这一段需要优化
    dt = pd.to_datetime(kbars['etime'], format='%Y-%m-%d %H:%M:%S.%f')
    kbars['etime'] = pd.Series([pd.Timestamp(x).round('s').to_pydatetime() for x in dt])
    kbars['label'] = '-1'
    
    # 根据区间开始和结束日期截取数据
    bgn_date = pd.to_datetime(bgn_date)
    end_date = pd.to_datetime(end_date)
    for i in range(0, len(kbars), 1): # .strftime('%Y-%m-%d %H:%M:%S')
        if (bgn_date <= kbars.loc[i, 'etime']) and (kbars.loc[i, 'etime'] <= end_date):
            kbars.loc[i, 'label'] = '1'

    # 筛选数据并重置索引
    kbars = kbars[kbars['label'] == '1']
    kbars = kbars.reset_index(drop=True)
    etime_close_data = kbars[['etime', 'tdate', 'close']]
    etime_close_data = etime_close_data.reset_index(drop=True)

    return etime_close_data


# lib_feeddata.py
# lib_nn.py

'''
========================================================================================================================
单因子分析框架
========================================================================================================================
'''
def backtest(original_data, index_code, frequency, n_days):
    '''
    INPUT：
        original_data               ：因子计算结果表格
        index_code                  ：指数代码（e.g., '000016.SH'）
        frequency                   ：数据频率（e.g., '5', '15', '30', '60', '120', 'd'）
        n_days                      ：收益率计算时间间隔（e.g., 0.25, 0.5, 1, 1.5, 单位：天）
    RETURN：
        ret_frame_train_test（etime、tdate、close、position、fct、持仓净值、持仓净值（累计））
        indicators_frame（总收益、年化收益率、夏普比率、年化波动率、最大回撤率、最大回撤起止日、卡尔玛比率、总交易次数、交易胜率、交易盈亏比）
    '''

    '''
    ========================================================================================================================
    PART 0：提取数据 + 收益率测算
    ========================================================================================================================
    '''
    
    position_size = 1 
    # 提取tdate、etime、close、fct四列，去除缺失值并重置索引
    final_frame = original_data[['tdate', 'etime', 'close', 'fct']].dropna(axis=0).reset_index(drop=True)
    # 在item_func函数中，'fct'列传入了所有因子数据：data['fct'] = fct_series.values， col_name是因子名称,每一个因子一轮迭代对应一个fct

    # 计算实际时间步的索引间隔
    if frequency == 'd': # 修改为d
        t_delta = int(1 * n_days) # n_days在这里首次出现
    else:
        t_delta = int(int(240 / int(frequency)) * n_days) 
    
    # 遍历计算未来n天的收益率（注意行索引的上限）
    for i in range(0, len(final_frame) - t_delta, 1):
        final_frame.loc[i, 'ret'] = final_frame.loc[i + t_delta, 'close'] / final_frame.loc[i, 'close'] - 1
        # 可优化
    # print(final_frame['ret']) # 这里输出的len(final_frame['ret'])和原序列长度一致
    # 这里的处理非常巧妙，这种写法让return自动向上shift一列，保证对齐
    # 去除收益率为空的行并重置索引
    final_frame = final_frame.dropna(axis=0).reset_index(drop=True)
    # print(final_frame) # 这里的final_frame是tdate，etime，close，fct值，return一共五列

    '''
    ========================================================================================================================
    PART 1：单因子回测
    ========================================================================================================================
    '''
    
    '''1：训练线性回归模型'''

    # 提取数据
    data_for_model = final_frame[['etime', 'close', 'fct', 'ret']]

    # 查询2016-12-31 15:00:00对应的索引（训练集和测试集的划分界限）
    train_set_end_index = data_for_model[(data_for_model['etime'].dt.year == 2019) & (data_for_model['etime'].dt.month == 12) & (data_for_model['etime'].dt.day == 31)].index.values[0] #  & (data_for_model['etime'].dt.hour == 15) # 先省去
    # 这里取的是.index.values[0]是因为在main中读取的时候已经设置index_col=0，index为timestamp
    # print(train_set_end_index)
    # 初始化线性回归模型
    model = LinearRegression(fit_intercept=True) # 极简主义，保证因子的原汁原味的成绩 weight,bias

    # 划分训练集、测试集
    X_train = data_for_model.loc[: train_set_end_index, 'fct'].values.reshape(-1, 1) # reshape成为一列 [1,2,3,4,5,6]
    y_train = data_for_model.loc[: train_set_end_index, 'ret'].values.reshape(-1, 1)
    X_test = data_for_model.loc[train_set_end_index + 1: , 'fct'].values.reshape(-1, 1)

    #print(X_test) 经过reshape(-1,1)的处理，成为一个二维数组，其中每一个数字有一个[]为了方便后续回归运算

    # 获取etime列表
    etime_train = data_for_model.loc[: train_set_end_index, 'etime'].values  # train set
    etime_test = data_for_model.loc[train_set_end_index + 1: , 'etime'].values  #test set
    etime_train_test = data_for_model.loc[:, 'etime'].values  # train set + test set

    # 模型训练————从这里开始，可以把模型变化为lightgbm，xgboost等
    model.fit(X_train, y_train)
    # print(model.coef_) # weights
    # print(model.intercept_) # bias

    # 测试集预测
    y_test_hat = model.predict(X_test) # 计算出来预测的Y值
    y_test_hat = [i[0] for i in y_test_hat]  # 注意：此处不要试图通过归一化进行仓位映射 [[1], [2], [3]]
    # 注意这里因为是linear regression，故选取了i[0]作为输出结果，如果是lgb，可能不需要这样做
    #print(y_test_hat)
    # i[0]的原因是reshape(-1,1)以后，是一列[[]……[]]，竖着的
    # 这里的问题是test数据集也全都是正数，这说明要么因子有问题，要么linearregression算法有问题

    # 训练集回归
    y_train_hat = model.predict(X_train)
    y_train_hat = [i[0] for i in y_train_hat]
    #print(y_train_hat) # 数据格式为正常的list
    # 把train部分也用模型计算一下，得出来的数据存在拟合，但是是模型输出的真是结果
    # 这里出现的问题是y_train_hat全部都是正数，这明显不对，明天用其他的代码试试

    '''2：测算持仓净值（训练集）'''
    
    # 测算周期的起始日期和结束日期
    begin_date_train = pd.to_datetime(str(etime_train[0])).strftime('%Y-%m-%d %H:%M:%S')
    end_date_train = pd.to_datetime(str(etime_train[-1])).strftime('%Y-%m-%d %H:%M:%S') # etime_train[::-1]
    # print(begin_date_train)
    # print(end_date_train)

    # 初始化测算持仓净值的预备表格
    ret_frame_train_total = generate_etime_close_data_divd_time(begin_date_train, end_date_train, index_code, frequency)
    #print(ret_frame_train_total) # 1464*3，从开始到2019年底作为训练集，etime,tdate,close为输出数据，用来计算return

    dt = pd.to_datetime(ret_frame_train_total['etime'], format='%Y-%m-%d %H:%M:%S.%f')
    ret_frame_train_total['etime'] = pd.Series([pd.Timestamp(x).round('s').to_pydatetime() for x in dt]) # 15:00:00:000000000000000001

    start_index = ret_frame_train_total[ret_frame_train_total['etime'] == etime_train[0]].index.values[0]
    end_index = ret_frame_train_total[ret_frame_train_total['etime'] == etime_train[-1]].index.values[0]




    ret_frame_train_total = ret_frame_train_total.loc[start_index: end_index, :].reset_index(drop=True)  # 进一步根据起止时刻筛选数据
    print(ret_frame_train_total)
    # print(ret_frame_train_total) # 这一步是check一下起始时刻的数据，和前面的数据完全相同

    #ret_frame_train_total['position'] = [i for i in y_train_hat] # check一下i for i in y_train_hat的结果
    
    ret_frame_train_total['position'] = [(i / 0.0005) * position_size for i in y_train_hat] # 开始映射仓位了 
    
    #作业1：  这里，可以给他滚动标准化一下（）不理解的话，可以暂时不做
    
    # print(ret_frame_train_total['position'])
    # print(ret_frame_train_total)
    #ret_frame_train_total['position'] = [1 for i in y_train_hat]
    # 让仓位的映射直接是1，而不是预测出来的数字

    for i in range(0, len(ret_frame_train_total), 1):
        if ret_frame_train_total.loc[i, 'position'] > 1:
            ret_frame_train_total.loc[i, 'position'] = 1
        elif ret_frame_train_total.loc[i, 'position'] < -1:
            ret_frame_train_total.loc[i, 'position'] = -1
            
    #作业2：  clip() 可以优化的： 作业2

    ret_frame_train = ret_frame_train_total # 不再使用上方的针对30mins的映射，改成1:1映射
    #print(ret_frame_train[:50]) # OK，到这一步，position全部都是1，没问题

    # 1：初始化持仓净值
    ret_frame_train.loc[0, '持仓净值'] = 1 # 持仓净值的第一个数据是1，其余都是nan
    
    # 2：分周期测算持仓净值
    for i in range(0, len(ret_frame_train), 1):

        # 计算持仓净值
        if i == 0 or ret_frame_train.loc[i - 1, 'position'] == 0:  # 如果是第一个时间步或前一个区间的结束时刻为空仓状态
            ret_frame_train.loc[i, '持仓净值'] = 1
        else:
            close_2 = ret_frame_train.loc[i, 'close']
            close_1 = ret_frame_train.loc[i - 1, 'close']
            position = abs(ret_frame_train.loc[i - 1, 'position'])  # 获取仓位大小（上一周期）
            

            if ret_frame_train.loc[i - 1, 'position'] > 0:  # 如果上一周期开的是多仓 之前是i-1，暂时删除试试
                # ret_frame_train.loc[i, '持仓净值'] = 1.0 * (close_2 / close_1)
                ret_frame_train.loc[i, '持仓净值'] = 1 * (close_2 / close_1) * position + 1 * (1 - position)
            elif ret_frame_train.loc[i - 1, 'position'] < 0:  # 如果上一周期开的是空仓
                # ret_frame_train.loc[i, '持仓净值'] = 1.0 * (1 - (close_2 / close_1 - 1))
                ret_frame_train.loc[i, '持仓净值'] = 1 * (1 - (close_2 / close_1 - 1)) * position + 1 * (1 - position) 
                
                # 作业3： 这个地方这样写，可以加入手续费===
                
                
                # 这些程序全都是调整仓位，而不是long or short，比较有意思
                # 可以在这个基础上：if ret_frame_train.loc[i - 1, 'position'] > 0 进行修改，如果预测的是开多，仓位映射，
                #*********注意，这里应该是i，而不是i-1，否则会出很大的错
                

    # 3：滚动测算累计持仓净值
    ret_frame_train.loc[0, '持仓净值（累计）'] = 1
    for i in range(1, len(ret_frame_train), 1):
        ret_frame_train.loc[i, '持仓净值（累计）'] = ret_frame_train.loc[i - 1, '持仓净值（累计）'] * ret_frame_train.loc[i, '持仓净值']
        # 作业4：用cumprod（）优化， 然后使用cumsum（）优化
        # cumsum（）
        
    
    '''3：测算持仓净值（测试集）'''

    # 测算周期的起始日期和结束日期
    begin_date_test = pd.to_datetime(str(etime_test[0])).strftime('%Y-%m-%d %H:%M:%S')
    end_date_test = pd.to_datetime(str(etime_test[-1])).strftime('%Y-%m-%d %H:%M:%S')
    # print(begin_date_test)

    # 初始化测算持仓净值的预备表格
    ret_frame_test_total = generate_etime_close_data_divd_time(begin_date_test, end_date_test, index_code, frequency)

    start_index = ret_frame_test_total[ret_frame_test_total['etime'] == etime_test[0]].index.values[0]
    end_index = ret_frame_test_total[ret_frame_test_total['etime'] == etime_test[-1]].index.values[0]
    # print(start_index)
    # print(end_index)

    ret_frame_test_total = ret_frame_test_total.loc[start_index: end_index, :].reset_index(drop=True)  # 进一步根据起止时刻筛选数据
    ret_frame_test_total['position'] = [(i / 0.0005) * position_size for i in y_test_hat]  # 预测值每间隔0.0005对应仓位变化1%

    
    # 根据实际交易中仓位的上下限调整position数值
    for i in range(0, len(ret_frame_test_total), 1):
        if ret_frame_test_total.loc[i, 'position'] > 1:
            ret_frame_test_total.loc[i, 'position'] = 1
        elif ret_frame_test_total.loc[i, 'position'] < -1:
            ret_frame_test_total.loc[i, 'position'] = -1
    
    ret_frame_test = ret_frame_test_total
    ret_frame_test = ret_frame_test.dropna(axis=0).reset_index(drop=True)  # 去除空值并重置索引
    #print(ret_frame_test)
    # 1：初始化持仓净值
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
                # ret_frame_test.loc[i, '持仓净值'] = 1.0 * (close_2 / close_1)
            elif ret_frame_test.loc[i - 1, 'position'] < 0:  # 如果上一周期开的是空仓
                ret_frame_test.loc[i, '持仓净值'] = 1 * (1 - (close_2 / close_1 - 1)) * position + 1 * (1 - position)
                # ret_frame_test.loc[i, '持仓净值'] = 1.0 * (1 - (close_2 / close_1 - 1))
                #*********注意，这里应该是i，而不是i-1，否则会出很大的错
                
                # 作业5： 这里使用100% -100%仓位进出，和调整仓位，哪个更好？

    # 3：滚动测算累计持仓净值
    ret_frame_test.loc[0, '持仓净值（累计）'] = 1
    for i in range(1, len(ret_frame_test), 1):
        ret_frame_test.loc[i, '持仓净值（累计）'] = ret_frame_test.loc[i - 1, '持仓净值（累计）'] * ret_frame_test.loc[i, '持仓净值']
    
    out_data_test = pd.DataFrame()
    out_data_test['etime'] = etime_test
    out_data_test['fct_value'] = X_test
    out_data_test['y_hat'] = y_test_hat
    out_data_test['return_real'] = data_for_model.loc[train_set_end_index + 1:, 'ret'].values.reshape(-1, 1)
    out_data_test['position'] = ret_frame_test['position']
    out_data_test['net'] = ret_frame_test['持仓净值']
    out_data_test['accum'] = ret_frame_test['持仓净值（累计）']
    # out_data_test.to_csv('C:/Users/P15/Desktop/single_factor_test/returns_1013_check.csv', encoding='utf-8-sig')
    # print(out_data_test.tail(20))



    '''4：测算持仓净值（训练集 + 测试集）'''

    # 测算周期的起始日期和结束日期
    begin_date_train_test = pd.to_datetime(str(etime_train_test[0])).strftime('%Y-%m-%d %H:%M:%S')
    end_date_train_test = pd.to_datetime(str(etime_train_test[-1])).strftime('%Y-%m-%d %H:%M:%S')
    # print(begin_date_train_test)
    # print(end_date_train_test) # checked at 202211261916

    # 初始化测算持仓净值的预备表格
    ret_frame_train_test_total = generate_etime_close_data_divd_time(begin_date_train_test, end_date_train_test, index_code, frequency)
    # print(ret_frame_train_test_total)

    start_index = ret_frame_train_test_total[ret_frame_train_test_total['etime'] == etime_train_test[0]].index.values[0]
    end_index = ret_frame_train_test_total[ret_frame_train_test_total['etime'] == etime_train_test[-1]].index.values[0]
    # print(start_index)
    # print(end_index) # 202211261924 checked

    ret_frame_train_test_total = ret_frame_train_test_total.loc[start_index: end_index, :].reset_index(drop=True)  # 进一步根据起止时刻筛选数据
    ret_frame_train_test_total['position'] = [(i / 0.0005) * position_size for i in y_train_hat] + [(i / 0.0005) * position_size for i in y_test_hat]  # 训练值每间隔0.0005对应仓位变化1% + 预测值每间隔0.0005对应仓位变化1%
    #ret_frame_train_test_total['position'] = [1 for i in y_train_hat] + [1 for i in y_test_hat]
    # print(ret_frame_train_test_total['position'])
    ret_frame_train_test_total['fct'] = [i[0] for i in X_train] + [i[0] for i in X_test]  # 添加因子值列

    # 根据实际交易中仓位的上下限调整position数值
    for i in range(0, len(ret_frame_train_test_total), 1):
        if ret_frame_train_test_total.loc[i, 'position'] > 1:
            ret_frame_train_test_total.loc[i, 'position'] = 1
        elif ret_frame_train_test_total.loc[i, 'position'] < -1:
            ret_frame_train_test_total.loc[i, 'position'] = -1

    ret_frame_train_test = ret_frame_train_test_total
    ret_frame_train_test = ret_frame_train_test.dropna(axis=0).reset_index(drop=True)  # 去除空值并重置索引
    
    # 1：初始化持仓净值
    ret_frame_train_test.loc[0, '持仓净值'] = 1
    
    # 2：分周期测算持仓净值
    for i in range(0, len(ret_frame_train_test), 1):

        # 计算持仓净值
        if i == 0 or ret_frame_train_test.loc[i - 1, 'position'] == 0:  # 如果是第一个时间步或前一个区间的结束时刻为空仓状态
            ret_frame_train_test.loc[i, '持仓净值'] = 1
        else:
            close_2 = ret_frame_train_test.loc[i, 'close']
            close_1 = ret_frame_train_test.loc[i - 1, 'close']
            position = abs(ret_frame_train_test.loc[i - 1, 'position'])  # 获取仓位大小（上一周期）
            
            if ret_frame_train_test.loc[i - 1, 'position'] > 0:  # 如果上一周期开的是多仓
                # ret_frame_train_test.loc[i, '持仓净值'] = 1.0 * (close_2 / close_1)
                ret_frame_train_test.loc[i, '持仓净值'] = 1 * (close_2 / close_1) * position + 1 * (1 - position)
            elif ret_frame_train_test.loc[i - 1, 'position'] < 0:  # 如果上一周期开的是空仓
                # ret_frame_train_test.loc[i, '持仓净值'] = 1.0 * (1 - (close_2 / close_1 - 1))
                ret_frame_train_test.loc[i, '持仓净值'] = 1 * (1 - (close_2 / close_1 - 1)) * position + 1 * (1 - position)
                # 下面这种模式不行，是因为如果这么做，后续计算业绩报酬的时候回出错
                #*********注意，这里应该是i，而不是i-1，否则会出很大的错

    # print((ret_frame_train_test['持仓净值'] - 1).tail(50))
    # 3：滚动测算累计持仓净值
    ret_frame_train_test.loc[0, '持仓净值（累计）'] = 1
    for i in range(1, len(ret_frame_train_test), 1):
        ret_frame_train_test.loc[i, '持仓净值（累计）'] = ret_frame_train_test.loc[i - 1, '持仓净值（累计）'] * ret_frame_train_test.loc[i, '持仓净值']
    
    '''
    ========================================================================================================================
    PART 2：单因子风险指标测算
    ========================================================================================================================
    '''

    '''0：设置无风险利率'''
    fixed_return = 0.0

    '''1：初始化'''
    indicators_frame = pd.DataFrame()
    year_list = [i for i in ret_frame_train_test['etime'].dt.year.unique()]  # 获取年份列表
    indicators_frame['年份'] = year_list + ['样本内', '样本外', '总体']
    indicators_frame = indicators_frame.set_index('年份')  # 将年份列表设置为表格索引

    '''2：计算风险指标（总体）'''
    start_index = ret_frame_train_test.index[0]  # 获取总体的起始索引
    end_index = ret_frame_train_test.index[-1]  # 获取总体的结束索引

    # 1：总收益
    net_value_2 = ret_frame_train_test.loc[end_index, '持仓净值（累计）']
    net_value_1 = ret_frame_train_test.loc[start_index, '持仓净值（累计）']
    total_return = net_value_2 / net_value_1 - 1

    indicators_frame.loc['总体', '总收益'] = total_return
    # print(indicators_frame) # checked at 202211261942 all arrays must be of the same length unchecked

    # 2：年化收益率
    date_list = [i for i in ret_frame_train_test['etime'].dt.date.unique()]
    run_day_length = len(date_list)  # 计算策略运行天数
    annual_return = math.pow(1 + total_return, 252 / run_day_length) - 1

    indicators_frame.loc['总体', '年化收益'] = annual_return
    # print(indicators_frame, '--------to line475')

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
    # print(indicators_frame, 'to line 497') # OK-- checked at 202211262303

    # 4：最大回撤率及其对应的起止日（需要利用计算夏普比率过程中构建的日度累计持仓净值表格）
    mdd_end_index = np.argmax((np.maximum.accumulate(net_asset_value_list) - net_asset_value_list) / (np.maximum.accumulate(net_asset_value_list)))
    if mdd_end_index == 0:
        return 0
    mdd_end_date = net_asset_value_frame.loc[mdd_end_index, 'tdate']  # 最大回撤起始日

    mdd_start_index = np.argmax(net_asset_value_list[: mdd_end_index])
    mdd_start_date = net_asset_value_frame.loc[mdd_start_index, 'tdate']  # 最大回撤结束日

    maximum_drawdown = (net_asset_value_list[mdd_start_index] - net_asset_value_list[mdd_end_index]) / (net_asset_value_list[mdd_start_index])  # 计算最大回撤率

    indicators_frame.loc['总体', '最大回撤率'] = maximum_drawdown
    indicators_frame.loc['总体', '最大回撤起始日'] = mdd_start_date
    indicators_frame.loc['总体', '最大回撤结束日'] = mdd_end_date
    # print(indicators_frame, ' -------checked line 512') # checked at 202211262310

    # 5：卡尔玛比率（基于夏普比率以及最大回撤率）
    calmar_ratio = (annual_return - fixed_return) / maximum_drawdown  # 计算卡尔玛比率

    indicators_frame.loc['总体', '卡尔玛比率'] = calmar_ratio

    # 6：总交易次数、交易胜率、交易盈亏比
    total_trading_times = len(ret_frame_train_test)  # 计算总交易次数

    win_times = 0  # 初始化盈利次数
    win_lose_frame = pd.DataFrame()  # 初始化盈亏表格
    
    for i in range(1, len(ret_frame_train_test), 1): # 作业5：这个地方也可以优化
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
    # print(indicators_frame, '=----checked line 540') # checked at 202211262310
    
    '''3：计算风险指标（分年度）'''
    for year in year_list: # n_batch = len(data) // batchsize + 1 
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
        if mdd_end_index == 0:
            return 0
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
    
    '''4：计算风险指标（样本内）'''
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
    if mdd_end_index == 0:
        return 0
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

    '''5：计算风险指标（样本外）'''
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
    if mdd_end_index == 0:
        return 0
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
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8,6))
    plt.plot(ret_frame_test['持仓净值（累计）'], 'b-', label='Test curve')
    plt.legend()
    plt.grid()
    plt.xlabel('Factor')
    plt.ylabel('Return')
    plt.show()

    
    return indicators_frame


