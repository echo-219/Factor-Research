import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


def indicator_cal(indicators_frame: pd.DataFrame, df: pd.DataFrame, name: str):
    """
    input
    indicators_frame：初始化后的业绩统计表格
    df：总样本集、训练集、测试集、分年度统计集
    name:是总样本集、训练集、测试集，还是分年度的统计？
    """
    # 0：设置无风险利率
    fixed_return = 0.0

    # 1：总收益
    total_return = df['持仓净值(累计)'][-1] / df['持仓净值(累计)'][0] - 1
    indicators_frame.loc[name, '总收益'] = total_return

    # 2：年化收益率
    date_list = [i for i in pd.Series(df.index).dt.date.unique()]
    run_day_length = len(date_list)  # 计算策略运行天数
    annual_return = math.pow(1 + total_return, 252 / run_day_length) - 1

    indicators_frame.loc[name, '年化收益'] = annual_return

    # 3：夏普比率、年化波动率
    net_asset_value_list = []  # 初始化累计持仓净值列表（日度）
    net_asset_value_index = [i[-1] for i in df.index.groupby(pd.Series(df.index.date)).values()]

    for date_index in net_asset_value_index:
        net_asset_value = df.loc[date_index, '持仓净值(累计)']
        net_asset_value_list.append(net_asset_value)  # 附加每日结束时对应的累计持仓净值

    net = pd.DataFrame({'date': date_list, 'nav': net_asset_value_list})  # 构建日度累计持仓净值表格

    net['daily_log_return'] = np.log(net['nav']).diff()

    annual_volatility = math.sqrt(252) * net['daily_log_return'].std()  # 计算年化波动率
    sharpe_ratio = (annual_return - fixed_return) / annual_volatility  # 计算夏普比率

    indicators_frame.loc[name, '年化波动率'] = annual_volatility
    indicators_frame.loc[name, '夏普比率'] = sharpe_ratio

    # 4：最大回撤率及其对应的起止日（需要利用计算夏普比率过程中构建的日度累计持仓净值表格）
    mdd_end_index = np.argmax((np.maximum.accumulate(net_asset_value_list) - net_asset_value_list) / (
        np.maximum.accumulate(net_asset_value_list)))
    if mdd_end_index == 0:
        return 0
    mdd_end_date = net.loc[mdd_end_index, 'date']  # 最大回撤起始日
    mdd_start_index = np.argmax(net_asset_value_list[: mdd_end_index])
    mdd_start_date = net.loc[mdd_start_index, 'date']  # 最大回撤结束日

    maximum_drawdown = (net_asset_value_list[mdd_start_index] - net_asset_value_list[mdd_end_index]) / (
        net_asset_value_list[mdd_start_index])  # 计算最大回撤率

    indicators_frame.loc[name, '最大回撤率'] = maximum_drawdown
    indicators_frame.loc[name, '最大回撤起始日'] = mdd_start_date
    indicators_frame.loc[name, '最大回撤结束日'] = mdd_end_date

    # 5：卡尔玛比率（基于夏普比率以及最大回撤率）
    calmar_ratio = (annual_return - fixed_return) / maximum_drawdown  # 计算卡尔玛比率

    indicators_frame.loc[name, '卡尔玛比率'] = calmar_ratio

    # 6：总交易次数、交易胜率、交易盈亏比
    total_trading_times = len(df)  # 计算总交易次数

    win_times = 0  # 初始化盈利次数
    win_lose_frame = pd.DataFrame()  # 初始化盈亏表格
    win_lose_frame['delta_value'] = df['持仓净值(累计)'].diff()
    win_times = (win_lose_frame['delta_value'] > 0).sum()

    gain_amount = abs(win_lose_frame[win_lose_frame['delta_value'] > 0]['delta_value'].sum())  # 计算总盈利额
    loss_amount = abs(win_lose_frame[win_lose_frame['delta_value'] < 0]['delta_value'].sum())  # 计算总亏损额

    winning_rate = win_times / total_trading_times  # 计算胜率
    gain_loss_ratio = gain_amount / loss_amount  # 计算盈亏比

    indicators_frame.loc[name, '总交易次数'] = total_trading_times
    indicators_frame.loc[name, '胜率'] = winning_rate
    indicators_frame.loc[name, '盈亏比'] = gain_loss_ratio

    return indicators_frame
