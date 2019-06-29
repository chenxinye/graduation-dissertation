#@author:chenxinye
#@2019.06.09

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot

df = pd.read_csv("data/DJIA_table.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by = 'Date')
df.index = df['Date']
df['Adj Close'].plot()

def draw_acf_pacf(ts,lags):
    f = plt.figure(facecolor='white',figsize = (20,20))
    ax1 = f.add_subplot(211)
    plot_acf(ts,ax=ax1,lags=lags)
    ax2 = f.add_subplot(212)
    plot_pacf(ts,ax=ax2,lags=lags)
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    
#自相关检验,初步
draw_acf_pacf(df['Adj Close'],100)
draw_acf_pacf(df['Adj Close'].diff().dropna(),100)

adfuller(df['Adj Close'])

t=adfuller(df['Adj Close'],maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)
output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
output['value']['Test Statistic Value'] = t[0]
output['value']['p-value'] = t[1]
output['value']['Lags Used'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print(output)

#help(adfuller)
#p=  0.6619805931301748 < 0.05 因此不是平稳序列

t=adfuller(df['Adj Close'].diff().dropna(),maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)
output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
output['value']['Test Statistic Value'] = t[0]
output['value']['p-value'] = t[1]
output['value']['Lags Used'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print(output)
#p=  0 > 0.05 因此是平稳序列
"""
Returns
    -------
    adf : float
        Test statistic
    pvalue : float
        MacKinnon's approximate p-value based on MacKinnon (1994, 2010)
    usedlag : int
        Number of lags used
    nobs : int
        Number of observations used for the ADF regression and calculation of
        the critical values
    critical values : dict
        Critical values for the test statistic at the 1 %, 5 %, and 10 %
        levels. Based on MacKinnon (2010)
    icbest : float
        The maximized information criterion if autolag is not None.
    resstore : ResultStore, optional
        A dummy class with results attached as attributes
"""

from statsmodels.stats.diagnostic import acorr_ljungbox
print (u'序列的纯随机性检测结果为：',acorr_ljungbox(df['Adj Close'].diff().dropna(),lags = 1))
#P=0.00085535，统计量的P值小于显著性水平0.05，则可以以95%的置信水平拒绝原假设，认为序列为非白噪声序列（否则，接受原假设，认为序列为纯随机序列。）

#确定ARIMA阶数
import statsmodels.api as sm
sm.tsa.arma_order_select_ic(df['Adj Close'].diff().dropna(),max_ar=50,max_ma=50,ic='aic')['aic_min_order']  # AIC(5,5)

sm.tsa.arma_order_select_ic(df['Adj Close'].diff().dropna(),max_ar=5,max_ma=5,ic='bic')['bic_min_order']  # BIC

sm.tsa.arma_order_select_ic(df['Adj Close'].diff().dropna(),max_ar=3,max_ma=3,ic='hqic')['hqic_min_order'] # HQIC

#以AIC准则为准
order = (0,1)
series = df['Adj Close'].diff().dropna()
length = len(series)
train = series[:-int(0.5*length)]
test = series[-int(0.5*length):]
tempModel = sm.tsa.ARMA(np.array(train),order).fit()
print(tempModel.summary2())

tempModel.forecast(int(0.5*length))

delta = tempModel.fittedvalues - train
score = 1 - delta.var()/train.var()
print (score)#远小于1
"""
总结
导致拟合效果欠佳的原因可能有：

使用数据为原始数据，未加任何预处理（主要原因）。原始数据中存在着异常值、不一致、缺失值，严重影响了建模的执行效率，造成较大偏差。；
在模型定阶过程中，为了控制计算量，限制AR最大阶不超过6，MA最大阶不超过4，从了影响了参数的确定，导致局部最优。
"""