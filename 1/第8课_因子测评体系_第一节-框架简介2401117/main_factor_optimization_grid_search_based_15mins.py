# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 10:40:38 2022
E:\spyder_code_fold\factor_optimization_grid_search_based.py
@author: P15

# 大纲_主要用于时间序列的测试

# 1、单因子测试的主架构设计；
# 2、行情数据和因子数据处理；
# 3、多进程并行运算处理方式；

# 4、核心回测部件及其函数构造；
# 5、绩效统计模块及其构造；

"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from tracemalloc import start
import warnings
from backtest_grid_search_based_15mins import backtest
import time
import numpy as np # 1.18.0
import pandas as pd # 1.4.2  python 3.8.3


'''
============================================================
根据不同频率的close（收盘价）生成原始数据表格
============================================================
'''

def generate_etime_close_data_divd_time(bgn_date, end_date, index_code, frequency): # 2022-11-25 edition last edited on 2023-01-17
    # 读取数据
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
            kbars.loc[i, 'label'] = '1' # 注意，这里之后会经常用到

    # 筛选数据并重置索引
    kbars = kbars[kbars['label'] == '1']
    kbars = kbars.reset_index(drop=True)
    etime_close_data = kbars[['etime', 'tdate', 'close']]
    etime_close_data = etime_close_data.reset_index(drop=True)

    return etime_close_data


  # 注意这里只传输进去close就可以，在backtest函数里面会计算return
'''
============================================================
网格搜索核心函数
============================================================
'''
def iter_func(params):
    
    # 元组传参
    freq, col_name, fct_series, data = params
    # 表格构建，这里的data就是行情数据
    data['fct'] = fct_series.values # data包含的：etime，tdate，close, fct_value
    ind_frame = backtest(original_data=data, index_code='510050', frequency=freq, n_days = 1) # n_days非常重要
    print('frequency: {}\nfct_name: {}\n'.format(freq, col_name))
    print(ind_frame)
    print('\n')
    print('夏普比率（样本外）：{}\n\n'.format(ind_frame.loc['样本外', '夏普比率']))  # 输出样本外夏普比率
    #print('frequency: {} fct_name: {} 夏普比率（样本外）：{}\n\n'.format(freq, col_name, ind_frame.loc['样本外', '夏普比率']))
    # 最终输出因子成绩的部分在这儿，可以输出一个只有sharpe的集合，然后对因子进行排序
    # 向表格中添加参数列
    #param_str = freq + '-' + col_name # 这一行不让他再加freq，保留最原始的因子名称
    ind_frame['params'] = col_name

    return ind_frame

'''
============================================================
主程序
============================================================
'''
if __name__ == '__main__': 

    # 初始化
    start_time = time.time()
    final_frame = pd.DataFrame()
    # file_path = 'D:/9_quant_course/commodities_data/rb_storage_1111.csv' # 因子的路径
    file_path = 'E:/Factor_Work_K/12_fct_reinforcement/fct_compare_0702.csv'
    
    # 因子数据和文件路径
    warnings.filterwarnings('ignore')
    job_num = 8  # 设置并行核数 设置为-1，将会调用电脑所有线程
    freq_val = 'd'
    
    # 引入行情数据，注意日期
    original_frame = generate_etime_close_data_divd_time(bgn_date='2005-02-23', end_date='2022-11-30', index_code='510050', frequency=freq_val)  # 生成原始数据
    print(original_frame, '==============original frame')
    #print(original_frame.shape) 
    fct_file = pd.read_csv(file_path, index_col=0)  # 读取因子表格
    print(len(fct_file), '============length of fct file')

    inputs = []
    for fct_name in fct_file.columns:
        print(fct_name)
        inputs.append((freq_val, fct_name, fct_file[fct_name], original_frame)) # 15， factor_name，fct_value，etime，tdate，close
        # freqval factorname factorvalue etime tdate close
        # 注意这里fct_file的columns就需要全部都是fct，不能出现timestamp和return
    #print(inputs)
    
    # 周期频率，因子名称，对应的因子数据，etime，tdate，close
    with ProcessPoolExecutor(max_workers=job_num) as executor:
        results = {executor.submit(iter_func, param): param for param in inputs} # 通过这种方式，把我们的行情数据和因子数据进行一一对应
        for r in as_completed(results):
            try:
                final_frame = pd.concat([final_frame, r.result()])
            except Exception as exception:
                print(exception)
    
    final_frame.to_csv('E:/Factor_Work_K/12_fct_reinforcement/result_reinforcement_0702.csv', encoding='utf-8-sig') # change
    end_time = time.time()
    print('Time cost:====', end_time - start_time)
    
