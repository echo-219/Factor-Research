import numpy as np
import pandas as pd
import pytdx
import schedule
from pytdx.hq import TdxHq_API
from pytdx.exhq import TdxExHq_API
import datetime
import time 

start_time = time.time()
raw_data50_15mins_path = 'E:/Factor_Work_K/9_15_and_30_mins_models/0_data_raw/50ETF_15mins_rawdata.pkl'

data = pd.read_pickle(raw_data50_15mins_path) 



def get_rawdata_15min_50etf(local_path, k_name):
    print("开始抓取"+k_name+"数据")
    data_50etf = pd.read_pickle(local_path)
    data_50etf = data_50etf.reset_index()
    data_50etf['timestamp'] = pd.to_datetime(data_50etf['timestamp'])

    api = TdxHq_API()
    if api.connect('119.147.212.81', 7709):
        current_data = api.to_df(api.get_security_bars(1, 1, '510050', 0, 50))
        api.disconnect()
    
    current_data = current_data[['datetime', 'open', 'high', 'low', 'close', 'vol']]
    current_data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    current_data['timestamp'] = pd.to_datetime(current_data['timestamp'])
    
    data_50etf = pd.concat([data_50etf, current_data], axis=0)
    data_50etf = data_50etf.sort_values(by='timestamp', ascending=True)
    data_50etf = data_50etf.drop_duplicates('timestamp').reset_index() # .set_index('timestamp')
    del data_50etf['index']
    data_50etf = data_50etf.set_index('timestamp')
    data_50etf.to_pickle(local_path)
    data_50etf_0 = data_50etf.reset_index()
    last_bar_timestamp = data_50etf_0['timestamp'][data_50etf_0.shape[0] - 1]
    print(last_bar_timestamp)
    print(data_50etf_0.iloc[data.shape[0]-1, :])
    print('Task_1 : get data time cost:---------', time.time() - start_time, '   s----------')

def job(k_name):
    get_rawdata_15min_50etf(raw_data50_15mins_path) # 模块一：获得实时数据
    # fct_calculate_standardized() # 模块二：因子数据计算和标准化
    # model_predict() # 模块三：模型预测
    # get_mail() # 模块四：信号发送----check
    # trade() # 模块五：交易
    
    

if __name__ == '__main__':
    
    minutes_15_k=["09:45","10:00","10:15","10:30","10:45","11:00","11:15","11:30","13:15","13:30","13:45","14:00","14:15","14:30","14:45","15:00"]
    # minutes_30_k=["10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30" ,"15:00"]
    # minutes_60_k=["10:30", "11:30", "14:00", "15:00"]

    for t in minutes_15_k:
        schedule.every().day.at(t).do(job, t)
    while True:
        schedule.run_pending()
        time.sleep(100)

