import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

"""

期权课程计划安排：

  第一节：期权及其五个希腊字母的定价原理；
      五个希腊字母：delta，gamma，vega，theta，rho
      
  第二节：常用期权组合的收益曲线以及实战代码；
  
  第三节：实战中怎样发现当前市场状态下期权组合的非对称性交易机会？

"""


'''
目录
1、平价定价公式
2、欧式期权定价公式
3、delta算法
4、gamma算法
5、theta算法
6、vega算法
7、rho算法

8、牛顿法求解隐含波动率  iv
9、二分法求解隐含波动率 iv

备注：newton法的计算比binary法求解所需内存更大

期权定价最重要的几个因素：

1、c-认购/看涨期权价格，p-认沽/看跌期权价格（适用于parity）

2、S-spot_price，即股票现价； foward，
3、K-strike；
4、sigma-波动率；
5、r-无风险利率，一般上选择1年期国债利率；
6、T-存续期限，注意为trading day，决不能用calendar day

期权的类型：
  vanilla option：欧式期权，美式期权；
  exotic option：亚式期权，障碍期权等等；

'''


# 1、平价定价公式-主要用于套利，发现双边期权的定价错误

def call_parity(p, S, K, r, T):
    return p + S - K * np.exp(-r * T / 252)

def put_parity(c, S, K, r, T):
    return c + K * np.exp(-r * T / 252) - S


# print('call price calculated by parity', call_parity(0.5, 3, 2.5, 0.03, 20))
# print('put price calculated by parity', put_parity(0.5, 2.5, 3., 0.03, 20))
# PS：put-call parity在实战中经常会遇到kurtosis漂移带来的分布变化从而导致定价错误
# 那么在这种情况下，我们需要做的并非纠结于定价错误，而是把定价错误本身当做交易机会


# 2、欧式期权定价公式

def call_BS(S, K , sigma, r, T):
    """
    S-spot_price，即股票现价；
    K-strike；
    sigma-波动率；
    r-无风险利率，一般上选择1年期国债利率；
    T-存续期限，注意为trading day，决不能用calendar day
    """
    d1 = (np.log(S/K) + (r + pow(sigma,2)/2) * (T/252)) / (sigma * np.sqrt(T/252))
    d2 = d1 - sigma*np.sqrt(T/252)
    return S * norm.cdf(d1) - K * np.exp(-r*T/252) * norm.cdf(d2)

call_price = call_BS(2.9, 3., 0.2, 0.03, 20)

# print(call_price)
# 引入两个概念：实值期权和虚值期权，平值期权
# spot_price - strike > 0 : 实值期权
# spot_price - strike == 0 
# e.g. σ为20%且还有20个trading day到期的call，他的价格为什么不是3.2-3.0 = 0.2，而是0.21716338？
# 那么这0.0176代表了什么？
# 引入一个非常重要的概念：时间价值


def put_BS(S,K,sigma,r,T):
    """
    S-spot_price，即股票现价；
    K-strike；
    sigma-波动率；
    r-无风险利率，一般上选择1年期国债利率；
    T-存续期限，注意为trading day，决不能用calendar day
    """
    d1 = (np.log(S/K) + (r + pow(sigma,2) / 2) * (T / 252)) / (sigma * np.sqrt(T / 252))
    d2 = d1 - sigma * np.sqrt(T / 252)
    
    return K * np.exp(-r * (T / 252)) * norm.cdf( - d2) - S * norm.cdf( - d1)


# print('call price calculated by BSM', call_BS(3.2, 3., 0.2, 0.03, 20))
# print('put price calculated by BSM', put_BS(3., 3.2, 0.2, 0.03, 20))

# 各位有没有发现，同样的条件（间隔空间）下，put和call的价格存在价差？
# 那么在这种情况下我们会发掘什么样的套利方式呢？
# 非常负责的说，所有的套利机会都是来自于基础定价，而非宏大叙事或者天马行空。2023-09-02



# 3、delta算法
def delta_option(S,K,sigma,r,T,optype,positype):
    """
    S-spot_price，即股票现价；
    K-strike；
    sigma-波动率；
    r-无风险利率，一般上选择1年期国债利率；
    T-存续期限，注意为trading day，决不能用calendar day

    optype期权类型：'call'看涨期权，'put'看跌期权
    positype 头寸类型，'long'多头 'short'空头
    """
    d1 = (np.log(S / K) + (r + pow(sigma, 2) / 2) * (T / 252)) / (sigma * np.sqrt(T / 252))
    
    if optype == 'call':
        if positype == 'long':
            delta = norm.cdf(d1)
        else:
            delta = -norm.cdf(d1)
    else:
        if positype == 'long':
            delta = norm.cdf(d1) - 1
        else:
            delta = 1 - norm.cdf(d1)
    return delta

# print(delta_option(2.61, 2.6, 0.1646, 0.021, 17, 'call', 'long'))
# print(delta_option(2.61, 2.6, 0.1646, 0.021, 17, 'call', 'short'))
# print(delta_option(2.61, 2.6, 0.1646, 0.021, 17, 'put', 'long'))
# print(delta_option(2.61, 2.6, 0.1646, 0.021, 17, 'put', 'short'))

price_curr = 2.6

# 1、当价格在-10%至10%变动时，期权的delta变动状况
# vol_index = np.arange(-0.1, 0.1, 0.01) 

# plt.plot(price_curr * (1 + vol_index), delta_option(price_curr * (1 + vol_index), 2.6, 0.1646, 0.021, 17, 'call', 'long'), 
#          color='green', marker='o', linestyle='solid' ) # 不同strike下，delta的变动
# plt.xlabel(u'Spot_Price', fontsize=12)
# plt.ylabel(u'Delta', fontsize=12, rotation=0)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'Delta on different prices', fontsize=12)
# plt.legend(fontsize=12)
# plt.axhline(0.5)
# plt.grid('True')
# plt.show()

# 2、当波动率在10%至40%变动时，期权的delta变动状况
# vol_index = np.arange(0.1, 0.4, 0.01) 
# plt.plot(vol_index, delta_option(2.61, 2.6, vol_index, 0.021, 17, 'call', 'long'), 
#          color='red', marker='o', linestyle='solid' ) # 不同strike下，delta的变动
# plt.xlabel(u'volatility', fontsize=12)
# plt.ylabel(u'Delta', fontsize=12, rotation=0)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'Delta on different volatilities', fontsize=12)
# plt.legend(fontsize=12)
# plt.axhline(0.5)
# plt.grid('True')
# plt.show()

# 3、当到期时间在50至1天变动时，期权的delta变动状况
# vol_index = np.arange(1, 51, 1) 
# plt.plot(vol_index, delta_option(2.61, 2.6, 0.1646, 0.021, vol_index, 'call', 'long'), 
#          color='blue', marker='o', linestyle='solid' ) # 不同strike下，delta的变动
# plt.xlabel(u'term to maturity', fontsize=12)
# plt.ylabel(u'Delta', fontsize=12, rotation=0)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'Delta on different terms', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()


#4、gamma算法——加速度
def gamma_option(S, K, sigma, r, T):
    """
    S-spot_price，即股票现价；
    K-strike；
    sigma-波动率；
    r-无风险利率，一般上选择1年期国债利率；
    T-存续期限，注意为trading day，决不能用calendar day
    """
    d1 = (np.log(S / K) + (r + pow(sigma, 2) / 2)*(T / 252)) / (sigma * np.sqrt(T / 252))
    return np.exp( -pow(d1, 2) / 2)/(S * sigma * np.sqrt(2 * np.pi * (T / 252)))


gamma_option_1 = gamma_option(S=2.61,K=2.6,sigma=0.1646,r=0.021,T=17)
# print(gamma_option_1, 'gammmmmmmmmmmmmmmmmmmmma')

# 1、当价格在-10%至10%变动时，期权的gamma变动状况
# vol_index = np.arange(-0.1, 0.1, 0.01) 
# plt.plot(price_curr * (1 + vol_index), gamma_option(price_curr * (1 + vol_index), 2.6, 0.1646, 0.021, 17), 
#          color='green', marker='o', linestyle='solid' ) # 不同strike下，delta的变动
# plt.xlabel(u'Spot_Price', fontsize=12)
# plt.ylabel(u'Gamma', fontsize=12, rotation=0)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'Gamma on different prices', fontsize=12)
# plt.legend(fontsize=12)
# plt.axhline(0.5)
# plt.grid('True')
# plt.show()

# 2、当波动率在10%至40%变动时，期权的Gamma变动状况
# vol_index = np.arange(0.1, 0.4, 0.01) 
# plt.plot(vol_index, gamma_option(price_curr, 2.6, vol_index, 0.021, 17), 
#          color='red', marker='o', linestyle='solid' ) # 不同strike下，delta的变动
# plt.xlabel(u'volatility', fontsize=12)
# plt.ylabel(u'Gamma', fontsize=12, rotation=0)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'Gamma on different volatilities', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()

# 3、当到期时间在50至1天变动时，期权的Gamma变动状况
# vol_index = np.arange(1, 51, 1) 
# plt.plot(vol_index, gamma_option(price_curr, 2.6, 0.2, 0.021, vol_index), 
#          color='red', marker='o', linestyle='solid' ) # 不同strike下，delta的变动
# plt.xlabel(u'term to maturity', fontsize=12)
# plt.ylabel(u'Gamma', fontsize=12, rotation=0)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'Gamma on different terms', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()


# 5、theta算法
def theta_option(S,K,sigma,r,T,optype):
    """
    S-spot_price，即股票现价；
    K-strike；
    sigma-波动率；
    r-无风险利率，一般上选择1年期国债利率；
    T-存续期限，注意为trading day，决不能用calendar day
    """
    d1 = (np.log(S / K) + (r + pow(sigma, 2) / 2) * (T / 252)) / (sigma * np.sqrt(T / 252))
    d2 = d1 - sigma * np.sqrt(T / 252)
    theta_call = -(S * sigma * np.exp( -pow(d1, 2) / 2))/(2 * np.sqrt(2 * np.pi * (T / 252)))-r * K * np.exp( -r * T / 252) * norm.cdf(d2)
    if optype == 'call':
        theta = theta_call
    else:
        theta = theta_call + r * K * np.exp( -r * T / 252)
    return theta / 252 # 注意这里除以252得到的是每天的daily bleeding

# 这个theta需要除以252，得到的是每天的theta，思考为什么不能不除以252？（今后各位一定会遇到不除以252的算法，忽视即可。）

theta_option_1 = theta_option(S=2.61, K=2.6, sigma=0.1646, r=0.021, T=17, optype='call')
# print(theta_option_1)

# 注意：theta这个指标对于cdf并没有那么的敏感，思考究竟是为什么？从norm.cdf(d2)考虑问题。

# 1、当价格在-10%至10%变动时，期权的theta变动状况
# vol_index = np.arange(-0.1, 0.1, 0.01) 
# plt.plot(price_curr * (1 + vol_index), theta_option(price_curr * (1 + vol_index), 2.6, 0.1646, 0.021, 17, optype='call'), 
#          color='green', marker='o', linestyle='solid' ) # 不同strike下，delta的变动
# plt.xlabel(u'Spot_Price', fontsize=12)
# plt.ylabel(u'Theta', fontsize=12, rotation=0)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'Theta on different prices', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()

# 2、当波动率在10%至40%变动时，期权的Theta变动状况
# vol_index = np.arange(0.1, 0.4, 0.01) 
# plt.plot(vol_index, theta_option(price_curr, 2.6, vol_index, 0.021, 17, optype='call'), 
#          color='red', marker='o', linestyle='solid' ) # 不同strike下，delta的变动
# plt.xlabel(u'volatility', fontsize=12)
# plt.ylabel(u'Theta', fontsize=12, rotation=0)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'Theta on different volatilities', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()

# 3、当到期时间在50至1天变动时，期权的Theta变动状况
# vol_index = np.arange(1, 51, 1) 
# plt.plot(vol_index, theta_option(price_curr, 2.6, 0.2, 0.021, vol_index, optype='call'), 
#          color='red', marker='o', linestyle='solid' ) # 不同strike下，delta的变动
# plt.xlabel(u'term to maturity', fontsize=12)
# plt.ylabel(u'Theta', fontsize=12, rotation=0)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'Theta on different terms', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()

# 注意，在这里就有问题了，我们short theta的时候，最好选取什么样的到期日？——30-12


# 6、vega算法
def vega_option(S, K, sigma, r, T):
    """
    S-spot_price，即股票现价；
    K-strike；
    sigma-波动率；
    r-无风险利率，一般上选择1年期国债利率；
    T-存续期限，注意为trading day，决不能用calendar day
    """
    d1 = (np.log(S / K) + (r + pow(sigma, 2) / 2) * T / 252) / (sigma * np.sqrt(T / 252))
    return S * np.sqrt(T / 252) * np.exp( -pow(d1, 2) / 2)/np.sqrt(2 * np.pi)/100

# print(vega_option(2.61, 2.6, 0.1646, 0.021, 18))

# 思考：为什么vega并不区分call或者put？
# 注意，vega的套利方式和其他几个希腊字母完全不一样

# 1、当价格在-10%至10%变动时，期权的Vega变动状况
# vol_index = np.arange(-0.1, 0.1, 0.01) 
# plt.plot(price_curr * (1 + vol_index), vega_option(price_curr * (1 + vol_index), 2.6, 0.1646, 0.021, 17), 
#          color='green', marker='o', linestyle='solid' ) # 不同strike下，delta的变动
# plt.xlabel(u'Spot_Price', fontsize=12)
# plt.ylabel(u'Vega', fontsize=12, rotation=0)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'Vega on different prices', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()

# 2、当波动率在10%至40%变动时，期权的Vega变动状况
# vol_index = np.arange(0.1, 0.4, 0.01) 
# plt.plot(vol_index, vega_option(price_curr, 2.6, vol_index, 0.021, 17), 
#          color='red', marker='o', linestyle='solid' ) # 不同strike下，delta的变动
# plt.xlabel(u'volatility', fontsize=12)
# plt.ylabel(u'Vega', fontsize=12, rotation=0)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'Vega on different volatilities', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()

# 3、当到期时间在50至1天变动时，期权的Vega变动状况
# vol_index = np.arange(1, 51, 1) 
# plt.plot(vol_index, vega_option(price_curr, 2.6, 0.2, 0.021, vol_index), 
#          color='red', marker='o', linestyle='solid' ) # 不同strike下，delta的变动
# plt.xlabel(u'term to maturity', fontsize=12)
# plt.ylabel(u'Vega', fontsize=12, rotation=0)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'Vega on different terms', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()

# 思考：我们想要赚Vega的钱，应该找近期合约还是远期合约？
# 思考：那么到这里，对临近到期日的期权，和远期的期权有什么策略和想法么？


# 6、rho算法

def rho_option(S,K,sigma,r,T,optype):
    """
    S-spot_price，即股票现价；
    K-strike；
    sigma-波动率；
    r-无风险利率，一般上选择1年期国债利率；
    T-存续期限，注意为trading day，决不能用calendar day
    """
    d1 = (np.log(S / K) + (r + pow(sigma,2) / 2) * T / 252) / (sigma * np.sqrt(T / 252))
    d2 = d1 - sigma * np.sqrt(T / 252)
    if optype == 'call':
        rho = K * T * np.exp(-r * T / 252) * norm.cdf(d2)
    else:
        rho = -K * T * np.exp(-r * T / 252) * norm.cdf(-d2)
    return rho / (10000)


# 本节课程内容：
# 1、牛顿法和二分法反向求解implied volatility
# 2、指数的多空和净值；
# 3、call和put的理论到期价值以及价格路径；
# 4、call spread的long-short以及理论价值-价格路径；
# 5、put spread的long-short以及理论价值-价格路径；
# 6、straddle的long-short以及理论价值-价格路径；


# 7、牛顿法求解隐含波动率  iv
def impvol_call_Newton(C,S,K,r,T):# forward
    """
    S-股票现价，K-strike
    sigma-波动率
    r-无风险利率，T-存续期限，注意为交易日
    """
    def call_BS(S,K,sigma,r,T):
        d1=(np.log(S/K)+(r+pow(sigma,2)/2)*T/252)/(sigma*np.sqrt(T/252))
        d2=d1-sigma*np.sqrt(T/252)
        return S*norm.cdf(d1)-K*np.exp(-r*T/252)*norm.cdf(d2)
    sigma0=0.2
    diff = C-call_BS(S,K,sigma0,r,T)
    i=0.0001
    while abs(diff)>0.0001:
        diff=C-call_BS(S,K,sigma0,r,T)
        if diff>0:
            sigma0 +=i
        else:
            sigma0 -=i
    return sigma0

def impvol_put_Newton(P,S,K,r,T):
    """
    S-股票现价，K-strike
    sigma-波动率
    r-无风险利率，T-存续期限，注意为交易日
    """
    def put_BS(S,K,sigma,r,T):
        d1=(np.log(S/K)+(r+pow(sigma,2)/2)*T/252)/(sigma*np.sqrt(T/252))
        d2=d1-sigma*np.sqrt(T/252)
        return K*np.exp(-r*T/252)*norm.cdf(-d2)-S*norm.cdf(-d1)
    sigma0=0.2
    diff = P - put_BS(S,K,sigma0,r,T)
    i=0.0001
    while abs(diff)>0.0001:
        diff=P-put_BS(S,K,sigma0,r,T)
        if diff>0:
            sigma0 +=i
        else:
            sigma0 -=i
    return sigma0


# 8、二分法求解隐含波动率 iv
def impvol_call_Binary(C,S,K,r,T):
    """
    S-股票现价，K-strike
    sigma-波动率
    r-无风险利率，T-存续期限，注意为交易日
    """
    def call_BS(S,K,sigma,r,T):
        d1=(np.log(S/K)+(r+pow(sigma,2)/2)*T/252)/(sigma*np.sqrt(T/252))
        d2=d1-sigma*np.sqrt(T/252)
        return S*norm.cdf(d1)-K*np.exp(-r*T/252)*norm.cdf(d2)
    sigma_min=0.001
    sigma_max=1.0
    sigma_mid=(sigma_min+sigma_max)/2
    call_min=call_BS(S,K,sigma_min,r,T)
    call_max=call_BS(S,K,sigma_max,r,T)
    call_mid=call_BS(S,K,sigma_mid,r,T)
    diff=C-call_mid
    if C<call_min or C>call_max:
        print('Error!')
    while abs(diff)>1e-6:
        diff=C-call_BS(S,K,sigma_mid,r,T)
        sigma_mid=(sigma_min+sigma_max)/2
        call_mid=call_BS(S,K,sigma_mid,r,T)
        if C>call_mid:
            sigma_min=sigma_mid
        else:
            sigma_max=sigma_mid
    return sigma_mid

def impvol_put_Binary(P,S,K,r,T):
    """
    S-股票现价，K-strike
    sigma-波动率
    r-无风险利率，T-存续期限，注意为交易日
    """
    def put_BS(S,K,sigma,r,T):
        d1=(np.log(S/K)+(r+pow(sigma,2)/2)*T/252)/(sigma*np.sqrt(T/252))
        d2=d1-sigma*np.sqrt(T/252)
        return K*np.exp(-r*T/252)*norm.cdf(-d2)-S*norm.cdf(-d1)
    sigma_min=0.001
    sigma_max=1.000
    sigma_mid=(sigma_min+sigma_max)/2
    put_min=put_BS(S,K,sigma_min,r,T)
    put_max=put_BS(S,K,sigma_max,r,T)
    put_mid=put_BS(S,K,sigma_mid,r,T)
    diff=P-put_mid
    if P<put_min or P>put_max:
        print('Error!')
    while abs(diff)>1e-6:
        diff=P-put_BS(S,K,sigma_mid,r,T)
        sigma_mid=(sigma_min+sigma_max)/2
        put_mid=put_BS(S,K,sigma_mid,r,T)
        if P>put_mid:
            sigma_min=sigma_mid
        else:
            sigma_max=sigma_mid
    return sigma_mid
# ======================上述为期权各项公式================
import time
start_time = time.time()

# print('本次牛顿法测试的28天到期的120元的call的隐含波动率为：===={}'.format(impvol_call_Newton(0.012, 2.474, 2.6, 0.026, 28)))
# print('本次牛顿法测试的28天到期的1302元的put的隐含波动率为：===={}'.format(impvol_put_Newton(0.1302, 2.474, 2.6, 0.026, 28)))
# print('本次二分法测试的28天到期的120元的call的隐含波动率为：===={}'.format(impvol_call_Binary(0.012, 2.474, 2.6, 0.026, 28)))
# print('本次二分法测试的28天到期的1302元的put的隐含波动率为：===={}'.format(impvol_put_Binary(0.1302, 2.474, 2.6, 0.026, 28)))
# print('time cost should be =={}==seconds'.format(time.time() - start_time))

C = 0.08
P = 0.065
K = 2.5
P0_etf = 2.5
P0_index = 2500. # 假设当前价格为2500点/元
Pt_index = np.linspace(500, 4500, 500) #

Pt_etf = P0_etf * Pt_index/P0_index
N_etf = 10000
N_Call = 1
N_put = 1
N_underlying = 10000
# print(Pt_etf)

# 1、单纯的多空ETF基金的收益曲线
return_etf_short = -N_etf * (Pt_etf - P0_etf) # 做空ETF基金
return_etf_long = N_etf * (Pt_etf - P0_etf) # 做多ETF基金
# plt.figure(figsize=(8, 6))
# plt.plot(Pt_index, return_etf_short, 'b-', label=u'上证50etf空头', lw=2.5 )
# plt.plot(Pt_index, return_etf_long, 'g-', label=u'上证50etf多头', lw=2.5 )
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'LONG AND SHORT STOCK OF 50ETF', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()


# 2-1、到期价格基础上认购期权的多空
return_call_long = N_Call * N_underlying * (np.maximum(Pt_etf-K, 0) - C )
return_call_short = N_Call * N_underlying * (C - np.maximum(Pt_etf-K, 0))
# plt.plot(Pt_index, return_call_long, 'g-', label=u'认购期权多头', lw=1.6 )
# plt.plot(Pt_index, return_call_short, 'y-', label=u'认购期权空头', lw=1.6 )
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'LONG AND SHORT A CALL', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()


# 2-2、到期价格基础上认沽期权的多空
return_put_long = N_put * N_underlying * (np.maximum(-Pt_etf+K, 0) - P )
return_put_short = N_put * N_underlying * (P - np.maximum(-Pt_etf+K, 0))
# plt.plot(Pt_index, return_put_long, 'g--', label=u'认沽期权多头', lw=1.6 ) # long一张put的权益曲线
# plt.plot(Pt_index, return_put_short, 'b--', label=u'认沽期权空头', lw=1.6 ) # long一张put的权益曲线
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'LONG AND SHORT A PUT', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()

# 3-1 现实场景中认购期权的多头
return_call_1 = call_BS(Pt_index/N_underlying*10, K, 0.2, 0.03, 200) * N_underlying
return_call_2 = call_BS(Pt_index/N_underlying*10, K, 0.4, 0.03, 200) * N_underlying
return_call_3 = call_BS(Pt_index/N_underlying*10, K, 0.6, 0.03, 200) * N_underlying
# plt.plot(Pt_index, return_call_1, 'r--', label=u'volatility==20%的call价格', lw=1.6 )
# plt.plot(Pt_index, return_call_2, 'b--', label=u'volatility==40%的call价格', lw=1.6 )
# plt.plot(Pt_index, return_call_3, 'y--', label=u'volatility==60%的call价格', lw=1.6 )
# plt.plot(Pt_index, return_call_long, 'g-', label=u'认购期权多头', lw=2 ) # long一张put的权益曲线
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'LONG TWO CALLS IN REAL TRADING', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()

# 3-2 现实场景中认沽期权的多头
return_put_1 = put_BS(Pt_index/N_underlying*10, K, 0.2, 0.03, 200) * N_underlying
return_put_2 = put_BS(Pt_index/N_underlying*10, K, 0.4, 0.03, 200) * N_underlying
# plt.plot(Pt_index, return_put_1, 'r--', label=u'不同volatility的PUT价格', lw=2 )
# plt.plot(Pt_index, return_put_2, 'b--', label=u'不同volatility的PUT价格', lw=1.5 )
# plt.plot(Pt_index, return_put_long, 'g-', label=u'认沽期权多头', lw=2.5 ) # long一张put的权益曲线
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'LONG TWO PUTS IN REAL TRADING', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()


# 4-1 call spread 多头的理论价格和实际价格
return_call_long_0 = N_Call * N_underlying * (np.maximum(Pt_etf-K*0.92, 0) - C )
return_call_1 = call_BS(Pt_index/N_underlying*10, K, 0.2, 0.03, 200) * N_underlying
return_call_2 = call_BS(Pt_index/N_underlying*10, K*0.92, 0.2, 0.03, 200) * N_underlying
# plt.plot(Pt_index, return_call_2 - return_call_1, 'r--', label=u'long call spread的实际价格', lw=1.6 )
# plt.plot(Pt_index, return_call_long_0 - return_call_long, 'g-', label=u'long call spread的理论价格', lw=2.0 ) # long一张put的权益曲线
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'LONG CALL SPREAD CURVE', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()
# debit stratergy breakeven：2500 + 300

# 4-2 call spread 空头的理论价格和实际价格
return_call_long_0 = N_Call * N_underlying * (np.maximum(Pt_etf-K*0.92, 0) - C )
return_call_1 = call_BS(Pt_index/N_underlying*10, K, 0.2, 0.03, 200) * N_underlying
return_call_2 = call_BS(Pt_index/N_underlying*10, K*0.92, 0.2, 0.03, 200) * N_underlying
# plt.plot(Pt_index, -return_call_2 + return_call_1, 'b--', label=u'short call spread的实际价格', lw=2.1 )
# plt.plot(Pt_index, -return_call_long_0 + return_call_long, 'y--', label=u'short call spread的理论价格', lw=1.6 ) # long一张put的权益曲线
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'SHORT CALL SPREAD CURVE', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()
# credit stratergy 2500-300 2200

# 5-1 put spread 多头的理论价格和实际价格
return_put_long_0 = N_put * N_underlying * (np.maximum(-Pt_etf + K*0.92, 0) - P )
return_put_1 = put_BS(Pt_index/N_underlying*10, K, 0.2, 0.03, 200) * N_underlying
return_put_2 = put_BS(Pt_index/N_underlying*10, K*0.92, 0.2, 0.03, 200) * N_underlying
# plt.plot(Pt_index, return_put_1 - return_put_2, 'r--', label=u'long put spread的实际价格', lw=2 )
# plt.plot(Pt_index, return_put_long - return_put_long_0, 'g-', label=u'long put spread的理论价格', lw=2.5 )
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'LONG PUT SPREAD CURVE', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()
# debit stratergy breakeven 2500 - 300 = 2200

# 5-2 put spread 空头的理论价格和实际价格
return_put_long_0 = N_put * N_underlying * (np.maximum(-Pt_etf + K*0.92, 0) - P )
return_put_1 = put_BS(Pt_index/N_underlying*10, K, 0.2, 0.03, 200) * N_underlying
return_put_2 = put_BS(Pt_index/N_underlying*10, K*0.92, 0.2, 0.03, 200) * N_underlying
# plt.plot(Pt_index, - return_put_1 + return_put_2, 'r--', label=u'short put spread的实际价格', lw=2 )
# plt.plot(Pt_index, - return_put_long + return_put_long_0, 'g-', label=u'short put spread的理论价格', lw=2.5 )
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'SHORT PUT SPREAD CURVE', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()
# credit stratergy breakeven ： 2500 + 300 



# 6-1 现实中双买的权益图——long straddle
return_long_straddle_end = return_call_long + return_put_long
return_long_straddle_dynamic = return_call_1 + return_put_1
# plt.plot(Pt_index, return_long_straddle_end, 'r-', label=u'long straddle 最终价格', lw=1.1 )
# plt.plot(Pt_index, return_long_straddle_dynamic, 'g--', label=u'long straddle 未到期价格', lw=1.1 ) 
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.title(u'LONG STRADDLE CURVE', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()


# 6-2 现实中双卖的权益图——short straddle
return_short_straddle_end = return_call_short + return_put_short
return_short_straddle_dynamic = - return_long_straddle_dynamic
# plt.plot(Pt_index, return_short_straddle_end, 'r-', label=u'short straddle 最终价格', lw=2.1 )
# plt.plot(Pt_index, return_short_straddle_dynamic, 'g--', label=u'short straddle 未到期价格', lw=1.1 ) 
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.title(u'SHORT STRADDLE CURVE', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()



"""
本节安排：
  1、strange的long和short；
  2、butterfly and iron condor的long和short；
  3、组合的delta，gamma，vega，theta计算；
  4、非对称性机会的挖掘；
"""

# 7-1 现实中的宽跨双买 ——— long strange
theshold = 0.1
return_call_3 = call_BS(Pt_index/N_underlying*10, K*(1+theshold), 0.2, 0.03, 200) * N_underlying
return_put_3 = put_BS(Pt_index/N_underlying*10, K*(1-theshold), 0.2, 0.03, 200) * N_underlying
return_long_strangle_dynamic = return_call_3 + return_put_3

return_call_4 = call_BS(Pt_index/N_underlying*10, K*(1+theshold), 0.4, 0.03, 200) * N_underlying
return_put_4 = put_BS(Pt_index/N_underlying*10, K*(1-theshold), 0.4, 0.03, 200) * N_underlying
return_long_strangle_dynamic_0 = return_call_4 + return_put_4

return_call_long_3 = N_Call * N_underlying * (np.maximum(Pt_etf-K*(1+theshold), 0) - C )
return_put_long_3 = N_put * N_underlying * (np.maximum(-Pt_etf+K*(1-theshold), 0) - P )
return_long_strangle_end = return_call_long_3 + return_put_long_3
# plt.plot(Pt_index, return_long_strangle_dynamic, 'r--', label=u'long strangle 最终价格-vol-20%', lw=1.6 )
# plt.plot(Pt_index, return_long_strangle_dynamic_0, 'b--', label=u'long strangle 最终价格-vol-40%', lw=1.6 )
# plt.plot(Pt_index, return_long_strangle_end, 'g-', label=u'long straddle 未到期价格', lw=2.1 ) 
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.title(u'LONG STRANGLE CURVE', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()


# 7-2 现实中的宽跨双卖 ——— short strange
theshold = 0.1
return_call_3 = call_BS(Pt_index/N_underlying*10, K*(1+theshold), 0.2, 0.03, 200) * N_underlying
return_put_3 = put_BS(Pt_index/N_underlying*10, K*(1-theshold), 0.2, 0.03, 200) * N_underlying
return_long_strangle_dynamic = return_call_3 + return_put_3

return_call_4 = call_BS(Pt_index/N_underlying*10, K*(1+theshold), 0.4, 0.03, 200) * N_underlying
return_put_4 = put_BS(Pt_index/N_underlying*10, K*(1-theshold), 0.4, 0.03, 200) * N_underlying
return_long_strangle_dynamic_0 = return_call_4 + return_put_4

return_call_long_3 = N_Call * N_underlying * (np.maximum(Pt_etf-K*(1+theshold), 0) - C )
return_put_long_3 = N_put * N_underlying * (np.maximum(-Pt_etf+K*(1-theshold), 0) - P )
return_long_strangle_end = return_call_long_3 + return_put_long_3
# plt.plot(Pt_index, -return_long_strangle_dynamic, 'r--', label=u'short strangle 最终价格-vol-20%', lw=1.6 )
# plt.plot(Pt_index, -return_long_strangle_dynamic_0, 'b--', label=u'short strangle 最终价格-vol-40%', lw=1.6 )
# plt.plot(Pt_index, -return_long_strangle_end, 'g-', label=u'short straddle 未到期价格', lw=2.1 ) 
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.title(u'SHORT STRANGLE CURVE', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()

# 8-1 现实中的买入蝶式组合——long butterfly 同时引入 pin_risk的概念，组合的难度开始提高

theshold = 0.1
return_call_1 = call_BS(Pt_index/N_underlying*10, K*(1+theshold), 0.2, 0.03, 200) * N_underlying
return_call_2 = call_BS(Pt_index/N_underlying*10, K, 0.2, 0.03, 200) * N_underlying
return_call_3 = call_BS(Pt_index/N_underlying*10, K*(1-theshold), 0.2, 0.03, 200) * N_underlying

return_call_1_0 = call_BS(Pt_index/N_underlying*10, K*(1+theshold), 0.1, 0.03, 200) * N_underlying
return_call_2_0 = call_BS(Pt_index/N_underlying*10, K, 0.1, 0.03, 200) * N_underlying
return_call_3_0 = call_BS(Pt_index/N_underlying*10, K*(1-theshold), 0.1, 0.03, 200) * N_underlying

return_call_1_1 = call_BS(Pt_index/N_underlying*10, K*(1+theshold), 0.3, 0.03, 200) * N_underlying
return_call_2_1 = call_BS(Pt_index/N_underlying*10, K, 0.3, 0.03, 200) * N_underlying
return_call_3_1 = call_BS(Pt_index/N_underlying*10, K*(1-theshold), 0.3, 0.03, 200) * N_underlying

return_butterfly_long_dynamic = -2 * return_call_2 + return_call_1 + return_call_3
return_butterfly_long_dynamic_0 = -2 * return_call_2_0 + return_call_1_0 + return_call_3_0
return_butterfly_long_dynamic_1 = -2 * return_call_2_1 + return_call_1_1 + return_call_3_1

return_butterfly_long_end = -2 * N_Call * N_underlying * (np.maximum(Pt_etf-K, 0) - C ) + N_Call * N_underlying * (np.maximum(Pt_etf-K*(1+theshold), 0) - C ) + N_Call * N_underlying * (np.maximum(Pt_etf-K*(1-theshold), 0) - C )

# plt.plot(Pt_index, return_butterfly_long_dynamic_0, 'r--', label=u'long butterfly 最终价格-vol-10%', lw=1.6 )
# plt.plot(Pt_index, return_butterfly_long_dynamic, 'b--', label=u'long butterfly 最终价格-vol-20%', lw=1.6 )
# plt.plot(Pt_index, return_butterfly_long_dynamic_1, 'k--', label=u'long butterfly 最终价格-vol-30%', lw=1.6 )
# plt.plot(Pt_index, return_butterfly_long_end, 'g-', label=u'long butterfly 未到期价格', lw=2.1 ) 
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.title(u'LONG BUTTERFLY CURVE', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()

# 8-2 现实中的卖出蝶式组合——short butterfly 
theshold = 0.1
return_call_1 = call_BS(Pt_index/N_underlying*10, K*(1+theshold), 0.2, 0.03, 200) * N_underlying
return_call_2 = call_BS(Pt_index/N_underlying*10, K, 0.2, 0.03, 200) * N_underlying
return_call_3 = call_BS(Pt_index/N_underlying*10, K*(1-theshold), 0.2, 0.03, 200) * N_underlying

return_call_1_0 = call_BS(Pt_index/N_underlying*10, K*(1+theshold), 0.1, 0.03, 200) * N_underlying
return_call_2_0 = call_BS(Pt_index/N_underlying*10, K, 0.1, 0.03, 200) * N_underlying
return_call_3_0 = call_BS(Pt_index/N_underlying*10, K*(1-theshold), 0.1, 0.03, 200) * N_underlying

return_call_1_1 = call_BS(Pt_index/N_underlying*10, K*(1+theshold), 0.3, 0.03, 200) * N_underlying
return_call_2_1 = call_BS(Pt_index/N_underlying*10, K, 0.3, 0.03, 200) * N_underlying
return_call_3_1 = call_BS(Pt_index/N_underlying*10, K*(1-theshold), 0.3, 0.03, 200) * N_underlying

return_butterfly_long_dynamic = -2 * return_call_2 + return_call_1 + return_call_3
return_butterfly_long_dynamic_0 = -2 * return_call_2_0 + return_call_1_0 + return_call_3_0
return_butterfly_long_dynamic_1 = -2 * return_call_2_1 + return_call_1_1 + return_call_3_1

return_butterfly_long_end = -2 * N_Call * N_underlying * (np.maximum(Pt_etf-K, 0) - C ) + N_Call * N_underlying * (np.maximum(Pt_etf-K*(1+theshold), 0) - C ) + N_Call * N_underlying * (np.maximum(Pt_etf-K*(1-theshold), 0) - C )

# plt.plot(Pt_index, -return_butterfly_long_dynamic_0, 'r--', label=u'short butterfly 最终价格-vol-10%', lw=1.6 )
# plt.plot(Pt_index, -return_butterfly_long_dynamic, 'b--', label=u'short butterfly 最终价格-vol-20%', lw=1.6 )
# plt.plot(Pt_index, -return_butterfly_long_dynamic_1, 'k--', label=u'short butterfly 最终价格-vol-30%', lw=1.6 )
# plt.plot(Pt_index, -return_butterfly_long_end, 'g-', label=u'short butterfly 未到期价格', lw=2.1 ) 
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.title(u'SHORT BUTTERFLY CURVE', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()


# 9-1 现实中的买入铁鹰组合——long iron condor 
# 四条腿：
theshold = 0.1
return_call_1_0 = call_BS(Pt_index/N_underlying*10, K*(1+theshold), 0.1, 0.03, 200) * N_underlying
return_call_2_0 = call_BS(Pt_index/N_underlying*10, K*(1+2*theshold), 0.1, 0.03, 200) * N_underlying
return_put_1_0 = put_BS(Pt_index/N_underlying*10, K, 0.1, 0.03, 200) * N_underlying
return_put_2_0 = put_BS(Pt_index/N_underlying*10, K*(1-theshold), 0.1, 0.03, 200) * N_underlying

return_call_1_1 = call_BS(Pt_index/N_underlying*10, K*(1+theshold), 0.2, 0.03, 200) * N_underlying
return_call_2_1 = call_BS(Pt_index/N_underlying*10, K*(1+2*theshold), 0.2, 0.03, 200) * N_underlying
return_put_1_1 = put_BS(Pt_index/N_underlying*10, K, 0.2, 0.03, 200) * N_underlying
return_put_2_1 = put_BS(Pt_index/N_underlying*10, K*(1-theshold), 0.2, 0.03, 200) * N_underlying

return_call_1_2 = call_BS(Pt_index/N_underlying*10, K*(1+theshold), 0.3, 0.03, 200) * N_underlying
return_call_2_2 = call_BS(Pt_index/N_underlying*10, K*(1+2*theshold), 0.3, 0.03, 200) * N_underlying
return_put_1_2 = put_BS(Pt_index/N_underlying*10, K, 0.3, 0.03, 200) * N_underlying
return_put_2_2 = put_BS(Pt_index/N_underlying*10, K*(1-theshold), 0.3, 0.03, 200) * N_underlying

return_ironcondor_long_dynamic_0 = return_call_1_0 - return_call_2_0 + return_put_1_0 - return_put_2_0
return_ironcondor_long_dynamic_1 = return_call_1_1 - return_call_2_1 + return_put_1_1 - return_put_2_1
return_ironcondor_long_dynamic_2 = return_call_1_2 - return_call_2_2 + return_put_1_2 - return_put_2_2

return_ironcondor_long_end = N_Call * N_underlying * (np.maximum(Pt_etf-K*(1+theshold), 0) - C ) - N_Call * N_underlying * (np.maximum(Pt_etf-K*(1+2*theshold), 0) - C ) +\
    N_put * N_underlying * (np.maximum(-Pt_etf+K, 0) - P ) - N_put * N_underlying * (np.maximum(-Pt_etf+K*(1-theshold), 0) - P )

# plt.plot(Pt_index, return_ironcondor_long_dynamic_0, 'r--', label=u'long iron condor 最终价格-vol-10%', lw=1.6 )
# plt.plot(Pt_index, return_ironcondor_long_dynamic_1, 'b--', label=u'long iron condor 最终价格-vol-20%', lw=1.6 )
# plt.plot(Pt_index, return_ironcondor_long_dynamic_2, 'k--', label=u'long iron condor 最终价格-vol-30%', lw=1.6 )
# plt.plot(Pt_index, return_ironcondor_long_end, 'g-', label=u'long iron condor 未到期价格', lw=2.1 ) 
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.title(u'LONG IRON CONDOR CURVE', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()


# 9-2 现实中的卖出铁鹰组合——short iron condor 

theshold = 0.1
return_call_1_0 = call_BS(Pt_index/N_underlying*10, K*(1+theshold), 0.1, 0.03, 200) * N_underlying
return_call_2_0 = call_BS(Pt_index/N_underlying*10, K*(1+2*theshold), 0.1, 0.03, 200) * N_underlying
return_put_1_0 = put_BS(Pt_index/N_underlying*10, K, 0.1, 0.03, 200) * N_underlying
return_put_2_0 = put_BS(Pt_index/N_underlying*10, K*(1-theshold), 0.1, 0.03, 200) * N_underlying

return_call_1_1 = call_BS(Pt_index/N_underlying*10, K*(1+theshold), 0.2, 0.03, 200) * N_underlying
return_call_2_1 = call_BS(Pt_index/N_underlying*10, K*(1+2*theshold), 0.2, 0.03, 200) * N_underlying
return_put_1_1 = put_BS(Pt_index/N_underlying*10, K, 0.2, 0.03, 200) * N_underlying
return_put_2_1 = put_BS(Pt_index/N_underlying*10, K*(1-theshold), 0.2, 0.03, 200) * N_underlying

return_call_1_2 = call_BS(Pt_index/N_underlying*10, K*(1+theshold), 0.3, 0.03, 200) * N_underlying
return_call_2_2 = call_BS(Pt_index/N_underlying*10, K*(1+2*theshold), 0.3, 0.03, 200) * N_underlying
return_put_1_2 = put_BS(Pt_index/N_underlying*10, K, 0.3, 0.03, 200) * N_underlying
return_put_2_2 = put_BS(Pt_index/N_underlying*10, K*(1-theshold), 0.3, 0.03, 200) * N_underlying

return_ironcondor_long_dynamic_0 = return_call_1_0 - return_call_2_0 + return_put_1_0 - return_put_2_0
return_ironcondor_long_dynamic_1 = return_call_1_1 - return_call_2_1 + return_put_1_1 - return_put_2_1
return_ironcondor_long_dynamic_2 = return_call_1_2 - return_call_2_2 + return_put_1_2 - return_put_2_2

return_ironcondor_long_end = N_Call * N_underlying * (np.maximum(Pt_etf-K*(1+theshold), 0) - C ) - N_Call * N_underlying * (np.maximum(Pt_etf-K*(1+2*theshold), 0) - C ) +\
    N_put * N_underlying * (np.maximum(-Pt_etf+K, 0) - P ) - N_put * N_underlying * (np.maximum(-Pt_etf+K*(1-theshold), 0) - P )

# plt.plot(Pt_index, -return_ironcondor_long_dynamic_0, 'r--', label=u'short iron condor 最终价格-vol-10%', lw=1.6 )
# plt.plot(Pt_index, -return_ironcondor_long_dynamic_1, 'b--', label=u'short iron condor 最终价格-vol-20%', lw=1.6 )
# plt.plot(Pt_index, -return_ironcondor_long_dynamic_2, 'k--', label=u'short iron condor 最终价格-vol-30%', lw=1.6 )
# plt.plot(Pt_index, -return_ironcondor_long_end, 'g-', label=u'short iron condor 未到期价格', lw=2.1 ) 
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.title(u'SHORT IRON CONDOR CURVE', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid('True')
# plt.show()


# 10-实战篇：组合的delta，gamma，vega，theta计算————call spread，put spread，strangle，iron condor分别计算出他们的delta，gamma，vega和theta；
"""
  场景：
    假设当前价格为2.6元，IV处于20%的状态，我们对组合进行希腊字母运算：
    1、call spread，分别有三种场景：
      （1）非常ITM的：2.2的long和2.3的short；
      （2）完全处于ATM的：2.55的long和2.65的short；
      （3）完全处于OTM的：2.8的long和2.9的short；

"""
# 10-1 call spread的PnL和Greeks计算

T = np.arange(0, 30, 1)[::1]
S_dynamic = 1 + np.arange(-10, 11, 1)/100
# print(T)
# print(S_dynamic)

call_long_itm_init = call_BS(2.6, 2.2, 0.2, 0.03, 29) - call_BS(2.6, 2.3, 0.2, 0.03, 29)
call_long_atm_init = call_BS(2.6, 2.55, 0.2, 0.03, 29) - call_BS(2.6, 2.65, 0.2, 0.03, 29)
call_long_otm_init = call_BS(2.6, 2.8, 0.2, 0.03, 29) - call_BS(2.6, 2.9, 0.2, 0.03, 29)

# 时间可变刻度
call_long_itm = call_BS(2.6, 2.2, 0.2, 0.03, T) - call_BS(2.6, 2.3, 0.2, 0.03, T)
call_long_atm = call_BS(2.6, 2.55, 0.2, 0.03, T) - call_BS(2.6, 2.65, 0.2, 0.03, T)
call_long_otm = call_BS(2.6, 2.8, 0.2, 0.03, T) - call_BS(2.6, 2.9, 0.2, 0.03, T)

# 价格可变刻度
call_long_itm = call_BS(2.6*S_dynamic, 2.2, 0.2, 0.03, 29) - call_BS(2.6*S_dynamic, 2.3, 0.2, 0.03, 29)
call_long_atm = call_BS(2.6*S_dynamic, 2.55, 0.2, 0.03, 29) - call_BS(2.6*S_dynamic, 2.65, 0.2, 0.03, 29)
call_long_otm = call_BS(2.6*S_dynamic, 2.8, 0.2, 0.03, 29) - call_BS(2.6*S_dynamic, 2.9, 0.2, 0.03, 29)

# print('当前ITM组合的价格为：{}'.format(call_long_itm_init))
# print('当前ATM组合的价格为：{}'.format(call_long_atm_init))
# print('当前OTM组合的价格为：{}'.format(call_long_otm_init))


# import matplotlib.pyplot as plt

# plt.plot(call_long_itm / call_long_itm_init, 'g--', label=u'long call spread of ITM', lw=1.2 ) # 各位思考一下这里绿色线向上的原因？？？BSM的SMILE
# plt.plot(call_long_atm / call_long_atm_init, 'r--', label=u'long call spread of ATM', lw=1.4 ) 
# plt.plot(call_long_otm / call_long_otm_init, 'b--', label=u'long call spread of OTM', lw=1.8 ) 
# # 上述三行是表示显示的是三个曲线，也就是df中的三个series
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.autoscale(enable=True, axis='x') 
# plt.autoscale(enable=True, axis='y') # 上述两行是选择x和y的刻度自适应
# # plt.xlim(30,0) # 这里是选择x的刻度是从30到0的倒序
# plt.title(u'LONG CALL SPREAD OF ALL SORTS', fontsize=12) # 给这个图生成一个title
# plt.legend(fontsize=12)
# plt.grid('True') # 网格加上
# plt.show() # 一定要有plt.show()，否则不会显示


# 时间和价格刻度同时可变，需要3D图像进行分析
# fig = plt.figure(figsize=(24, 16))
# ax = Axes3D(fig)
# ax = fig.gca(projection='3d')
# x = S_dynamic
# y = T
# X, Y = np.meshgrid(x, y) # 对x、y数据执行网格化

# call_long_itm = call_BS(2.6*X, 2.2, 0.2, 0.03, Y) - call_BS(2.6*X, 2.3, 0.2, 0.03, Y)
# Z = call_long_itm / call_long_itm_init

# 绘制3D图形
# surf = ax.plot_surface(X, Y, Z,
#                        rstride=1,  # rstride（row）指定行的跨度
#                        cstride=1,  # cstride(column)指定列的跨度
#                        cmap=plt.get_cmap('rainbow'))  # 设置颜色映射
# plt.title("LONG CALL SPREAD 3D SHOW") # 设置标题
# ax.set_xlabel('spot price space')
# ax.set_ylabel('time to maturity')
# ax.set_zlabel('portfolio')
# fig.colorbar(surf, shrink=0.5, aspect=5) # 非常重要，必须有
# plt.show()

# 10-1 call spread的Greeks计算
call_long_itm_init = call_BS(2.6, 2.2, 0.2, 0.03, 29) - call_BS(2.6, 2.3, 0.2, 0.03, 29)
call_long_atm_init = call_BS(2.6, 2.55, 0.2, 0.03, 29) - call_BS(2.6, 2.65, 0.2, 0.03, 29)
call_long_otm_init = call_BS(2.6, 2.7, 0.2, 0.03, 29) - call_BS(2.6, 2.8, 0.2, 0.03, 29)

# print('当前ITM组合的价格为：{}'.format(call_long_itm_init))
# print('当前ATM组合的价格为：{}'.format(call_long_atm_init))
# print('当前OTM组合的价格为：{}'.format(call_long_otm_init))


# 10-1-1:组合的delta计算方法
cs_delta_itm = delta_option(2.6, 2.2, 0.2, 0.03, 29, 'call', 'long') - delta_option(2.6, 2.3, 0.2, 0.03, 29, 'call', 'long')
cs_delta_atm = delta_option(2.6, 2.55, 0.2, 0.03, 29, 'call', 'long') - delta_option(2.6, 2.65, 0.2, 0.03, 29, 'call', 'long')
cs_delta_otm = delta_option(2.6, 2.7, 0.2, 0.03, 29, 'call', 'long') - delta_option(2.6, 2.8, 0.2, 0.03, 29, 'call', 'long')
# print('当前ITM组合的delta为：{}'.format(cs_delta_itm))
# print('当前ATM组合的delta为：{}'.format(cs_delta_atm))
# print('当前OTM组合的delta为：{}'.format(cs_delta_otm))


# 10-1-2:组合的gamma计算方法
cs_gamma_itm = gamma_option(2.6, 2.2, 0.2, 0.03, 29) - gamma_option(2.6, 2.3, 0.2, 0.03, 29)
cs_gamma_atm = gamma_option(2.6, 2.55, 0.2, 0.03, 29) - gamma_option(2.6, 2.65, 0.2, 0.03, 29)
cs_gamma_otm = gamma_option(2.6, 2.7, 0.2, 0.03, 29) - gamma_option(2.6, 2.8, 0.2, 0.03, 29)
# print('当前ITM组合的gamma为：{}'.format(cs_gamma_itm))
# print('当前ATM组合的gamma为：{}'.format(cs_gamma_atm))
# print('当前OTM组合的gamma为：{}'.format(cs_gamma_otm))

# 10-1-3:组合的theta计算方法
cs_theta_itm = theta_option(2.6, 2.2, 0.2, 0.03, 29, 'call') - theta_option(2.6, 2.3, 0.2, 0.03, 29, 'call')
cs_theta_atm = theta_option(2.6, 2.55, 0.2, 0.03, 29, 'call') - theta_option(2.6, 2.65, 0.2, 0.03, 29, 'call')
cs_theta_otm = theta_option(2.6, 2.7, 0.2, 0.03, 29, 'call') - theta_option(2.6, 2.8, 0.2, 0.03, 29, 'call')
# print('当前ITM组合的theta为：{}'.format(cs_theta_itm))
# print('当前ATM组合的theta为：{}'.format(cs_theta_atm))
# print('当前OTM组合的theta为：{}'.format(cs_theta_otm))

# 10-1-4:组合的vega计算方法
cs_vega_itm = vega_option(2.6, 2.2, 0.2, 0.03, 29) - vega_option(2.6, 2.3, 0.2, 0.03, 29)
cs_vega_atm = vega_option(2.6, 2.55, 0.2, 0.03, 29) - vega_option(2.6, 2.65, 0.2, 0.03, 29)
cs_vega_otm = vega_option(2.6, 2.7, 0.2, 0.03, 29) - vega_option(2.6, 2.8, 0.2, 0.03, 29)
# print('当前ITM组合的vega为：{}'.format(cs_vega_itm))
# print('当前ATM组合的vega为：{}'.format(cs_vega_atm))
# print('当前OTM组合的vega为：{}'.format(cs_vega_otm))


# 12 案例 iron condor 为什么是风险巨大的一个组合？
# 一个典型的iron condor short：
# 当前价格为2.6的时候，博弈未来价格波动处于2.5-2.7的区间
# 那么这里建立的头寸应该是：long 2.8的call，long2.4的put，同时short2.7的call，short2。5的put

iron_condor_short = call_BS(2.6, 2.7, 0.2, 0.03, 29) - call_BS(2.6, 2.8, 0.2, 0.03, 29) +\
    put_BS(2.6, 2.5, 0.2, 0.03, 29) - put_BS(2.6, 2.4, 0.2, 0.03, 29)

delta_condor_short = delta_option(2.6, 2.7, 0.2, 0.03, 29, 'call', 'long') - delta_option(2.6, 2.8, 0.2, 0.03, 29, 'call', 'long') +\
    delta_option(2.6, 2.5, 0.2, 0.03, 29, 'put', 'long') - delta_option(2.6, 2.4, 0.2, 0.03, 29, 'put', 'long') # 近乎于中性，但是需要check gamma

gamma_condor_short = gamma_option(2.6, 2.7, 0.2, 0.03, 29) - gamma_option(2.6, 2.8, 0.2, 0.03, 29) +\
    gamma_option(2.6, 2.5, 0.2, 0.03, 29) - gamma_option(2.6, 2.4, 0.2, 0.03, 29)

# print(-iron_condor_short)
# print(-delta_condor_short)
# print(gamma_condor_short)


# 动态portfolio
T = np.arange(0, 30, 1)[::1]
S_dynamic = 1 + np.arange(-10, 11, 1)/100

fig = plt.figure(figsize=(24, 16))
ax = Axes3D(fig)
ax = fig.gca(projection='3d')
x = S_dynamic * 2.6
y = T
X, Y = np.meshgrid(x, y) # 对x、y数据执行网格化

iron_condor_short_dynamic = (call_BS(X, 2.7, 0.2, 0.03, Y) - call_BS(X, 2.8, 0.2, 0.03, Y) +\
    put_BS(X, 2.5, 0.2, 0.03, Y) - put_BS(X, 2.4, 0.2, 0.03, Y))

delta_condor_short_dynamic = (delta_option(X, 2.7, 0.2, 0.03, Y, 'call', 'long') - delta_option(X, 2.8, 0.2, 0.03, Y, 'call', 'long') +\
    delta_option(X, 2.5, 0.2, 0.03, Y, 'put', 'long') - delta_option(X, 2.4, 0.2, 0.03, Y, 'put', 'long')) # 近乎于中性，但是需要check gamma

gamma_condor_short_dynamic = gamma_option(X, 2.7, 0.2, 0.03, Y) - gamma_option(X, 2.8, 0.2, 0.03, Y) + gamma_option(X, 2.5, 0.2, 0.03, Y) - gamma_option(X, 2.4, 0.2, 0.03, Y) 

# Z = (iron_condor_short - iron_condor_short_dynamic) / 0.2
# print('THE MAX LOSS IS:=={}=='.format((np.max(Z) - iron_condor_short)/0.2))

# 400 2000 - 400 = 1600 1600/2000 = 80%
Z = gamma_condor_short_dynamic


# 绘制3D图形
# surf = ax.plot_surface(X, Y, Z,
#                        rstride=1,  # rstride（row）指定行的跨度
#                        cstride=1,  # cstride(column)指定列的跨度
#                        cmap=plt.get_cmap('rainbow'))  # 设置颜色映射
# plt.title("IRON CONDOR SHORT SPREAD 3D SHOW") # 设置标题
# ax.set_xlabel('spot price space')
# ax.set_ylabel('time to maturity')
# ax.set_zlabel('portfolio')
# fig.colorbar(surf, shrink=0.5, aspect=5) # 非常重要，必须有
# plt.show()