#@author:chenxinye
#@2019.06.09

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import stockstats
from matplotlib import pyplot

df = pd.read_csv("data/DJIA_table.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Date'], ascending=True)
df.plot(figsize = (20,10),grid = True, use_index = True,x = 'Date',y = "Adj Close")
plt.show()

print("show_index distribution!\n")
sns.distplot(df['Adj Close'])
plt.show()

sns.distplot(df['Open'])
plt.show()

sns.distplot(df['High'])
plt.show()

sns.distplot(df['Low'])
plt.show()

sns.distplot(df['Close'])
plt.show()

sns.distplot(df['Volume'])
plt.show()

df['diff'] = df['Adj Close'].diff()

df['rate_of_return'] = df['diff']/df['Adj Close']

def return_int(df,sr):
    df[sr][df[sr] > 0] = 1
    df[sr][df[sr] <= 0] = 0
    return (df[sr])

df['rate_of_return_change'] = return_int(df.copy(),sr = 'rate_of_return')
df[['rate_of_return','rate_of_return_change']]
df['rate_of_return_change_shift'] = df['rate_of_return_change'].shift(-1)

sns.boxplot(x='rate_of_return_change_shift', y= 'Volume', data=df)

sns.pairplot(df, vars=['Open', 'High', 'Low', 'Close', 'Volume'])

df = df.rename(columns = {"Date":"date","Adj Close":"close","High":"high","Open":"open","Low":"low","Volume":"volume"})
del df['Close']
df.index = df["date"]
#CR指标, MACD,KDJ指标,SMA指标,BOLL指标,RSI指标,MWR指标,CCI指标,TR、ATR指标
st = stockstats.StockDataFrame.retype(df.copy())
print("init finish .")
"""
CR指标
http://wiki.mbalib.com/wiki/CR%E6%8C%87%E6%A0%87 

KDJ指标
http://wiki.mbalib.com/wiki/%E9%9A%8F%E6%9C%BA%E6%8C%87%E6%A0%87

SMA指标
http://wiki.mbalib.com/wiki/Sma 

MACD指标
http://wiki.mbalib.com/wiki/MACD 
平滑异同移动平均线(Moving Average Convergence Divergence，简称MACD指标)，也称移动平均聚散指标

BOLL指标
http://wiki.mbalib.com/wiki/BOLL 
布林线指标(Bollinger Bands)

RSI指标
http://wiki.mbalib.com/wiki/RSI 
相对强弱指标（Relative Strength Index，简称RSI），也称相对强弱指数、相对力度指数

WR指标
http://wiki.mbalib.com/wiki/%E5%A8%81%E5%BB%89%E6%8C%87%E6%A0%87
威廉指数（Williams%Rate）该指数是利用摆动点来度量市场的超买超卖现象。 

CCI指标
http://wiki.mbalib.com/wiki/%E9%A1%BA%E5%8A%BF%E6%8C%87%E6%A0%87
顺势指标又叫CCI指标，其英文全称为“Commodity Channel Index”， 

TR、ATR指标
http://wiki.mbalib.com/wiki/%E5%9D%87%E5%B9%85%E6%8C%87%E6%A0%87 
均幅指标（Average True Ranger,ATR）
均幅指标（ATR）是取一定时间周期内的股价波动幅度的移动平均值，主要用于研判买卖时机。

DMA指标
http://wiki.mbalib.com/wiki/DMA
DMA指标（Different of Moving Average）又叫平行线差指标，是目前股市分析技术指标中的一种中短期指标，它常用于大盘指数和个股的研判。

TRIX，MATRIX指标
http://wiki.mbalib.com/wiki/TRIX 
TRIX指标又叫三重指数平滑移动平均指标（Triple Exponentially Smoothed Average）

VR，MAVR指标
http://wiki.mbalib.com/wiki/%E6%88%90%E4%BA%A4%E9%87%8F%E6%AF%94%E7%8E%87
成交量比率（Volumn Ratio，VR）（简称VR），是一项通过分析股价上升日成交额（或成交量，下同）与股价下降日成交额比值， 
从而掌握市场买卖气势的中期技术指标。
https://blog.csdn.net/freewebsys/article/details/78578548
"""


index_ = ['close',
    'cr','cr-ma1','cr-ma2','cr-ma3',
    'kdjk','kdjd','kdjj',
    'close_5_sma','close_10_sma',
    'macd','macds','macdh',
    'boll','boll_ub','boll_lb',
    'rsi_6','rsi_12',
    'wr_10','wr_6',
    'cci','cci_20',
    'tr','atr',
    'dma',
    'trix','trix_9_sma',
    'vr','vr_6_sma'
    ]

#st[index_].plot(subplots=True, figsize=(40,120), grid=True)

for i in index_:
    st[[i]].plot(subplots=True, figsize=(10,10), grid=True)
    plt.savefig("index/"+str(i)+".png")

INDEX = ['close','cr','kdjk','macd','boll','rsi_6','wr_10','cci','tr','atr','dma','trix','vr']
f,ax = plt.subplots(figsize=(30,30))
corr = st[INDEX].corr()
sns.heatmap(corr,annot=True)

colormap = pyplot.cm.afmhot
pyplot.figure(figsize=(16,12))
pyplot.title('Pearson correlation of continuous features', y=1.05, size=15)
sns.heatmap(st[INDEX].corr(),linewidths=0.1,vmax=1.0, square=True, 
            cmap=colormap, linecolor='white', annot=True)
pyplot.show()

origin_index = ['close','open','high','low','close','volume']
colormap = pyplot.cm.afmhot
pyplot.figure(figsize=(16,12))
pyplot.title('Pearson correlation of continuous features', y=1.05, size=15)
sns.heatmap(df[origin_index].corr(),linewidths=0.1,vmax=1.0, square=True, 
            cmap=colormap, linecolor='white', annot=True)
pyplot.show()

st['date'] = df["date"]
st = st.drop(['diff'],axis = 1)
st.fillna(0,inplace = True)
st.to_csv("text/df_numeric.csv",index = 0)