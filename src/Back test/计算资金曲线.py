#@author:chenxinye
#@2019.06.09

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_hdf('output/pos.h5', key='df')

# =====选取相关时间。
# 截取上市一年之后的数据
#df = df.iloc[250-1:]  # 股市一年交易日大约250天

#df = df[df['date'] >= pd.to_datetime('20070101')]

# =====找出开仓、平仓条件
condition1 = df['pos'] != 0
condition2 = df['pos'] != df['pos'].shift(1)
open_pos_condition = condition1 & condition2

condition1 = df['pos'] != 0
condition2 = df['pos'] != df['pos'].shift(-1)
close_pos_condition = condition1 & condition2


# =====对每次交易进行分组
df.loc[open_pos_condition, 'start_time'] = df['date']
df['start_time'].fillna(method='ffill', inplace=True)
df.loc[df['pos'] == 0, 'start_time'] = pd.NaT


# =====开始计算资金曲线
# ===基本参数
initial_cash = 100000000  # 初始资金，默认为1000000元
slippage = 0.01  # 滑点，股票默认为0.01元，etf为0.001元
c_rate = 2.5 / 10000  # 手续费，commission fees，默认为万分之2.5
t_rate = 1.0 / 1000  # 印花税，tax，默认为千分之1

# ===在买入的K线
# 在发出信号的当根K线以收盘价买入
df.loc[open_pos_condition, 'stock_num'] = initial_cash * (1 - c_rate) / (df['前收盘价'] + slippage)

# 实际买入股票数量
df['stock_num'] = np.floor(df['stock_num'] / 100.0) * 100

# 买入股票之后剩余的钱，扣除了手续费
df['cash'] = initial_cash - df['stock_num'] * (df['前收盘价'] + slippage) * (1 + c_rate)

# 收盘时的股票净值
df['stock_value'] = df['stock_num'] * df['收盘价']

# ===在买入之后的K线
# 买入之后现金不再发生变动
df['cash'].fillna(method='ffill', inplace=True)
df.loc[df['pos'] == 0, ['cash']] = None

# 股票净值随着涨跌幅波动
group_num = len(df.groupby('start_time'))
if group_num > 1:
    t = df.groupby('start_time').apply(lambda x: x['收盘价_复权'] / x.iloc[0]['收盘价_复权'] * x.iloc[0]['stock_value'])
    t = t.reset_index(level=[0])
    df['stock_value'] = t['收盘价_复权']
elif group_num == 1:
    t = df.groupby('start_time')[['收盘价_复权', 'stock_value']].apply(lambda x: x['收盘价_复权'] / x.iloc[0]['收盘价_复权'] * x.iloc[0]['stock_value'])
    df['stock_value'] = t.T.iloc[:, 0]

# ===在卖出的K线
# 股票数量变动
df.loc[close_pos_condition, 'stock_num'] = df['stock_value'] / df['收盘价']

# 现金变动
df.loc[close_pos_condition, 'cash'] += df.loc[close_pos_condition, 'stock_num'] * (df['收盘价'] - slippage) * (1 - c_rate - t_rate)

# 股票价值变动
df.loc[close_pos_condition, 'stock_value'] = 0

# ===账户净值
df['net_value'] = df['stock_value'] + df['cash']

# ===计算资金曲线
df['equity_change'] = df['net_value'].pct_change(fill_method=None)
df.loc[open_pos_condition, 'equity_change'] = df.loc[open_pos_condition, 'net_value'] / initial_cash - 1  # 开仓日的收益率
df['equity_change'].fillna(value=0, inplace=True)
df['equity_curve'] = (1 + df['equity_change']).cumprod()

## ===删除无关数据
#df.drop(['start_time', 'stock_num', 'cash', 'stock_value', 'net_value'], axis=1, inplace=True)

print(df['equity_curve'].head(5))
print(df['date'].head(5))

#import seaborn as sns
#sns.set(style="darkgrid")
#
## Load an example dataset with long-form data
## Plot the responses for different events and regions
#sns.lineplot(x="date", y="equity_curve",data=df)
df.plot(figsize = (8,8),grid = True, use_index = True,x = 'date',y = "stock_num",color = 'black')
df.plot(figsize = (8,8),grid = True, use_index = True,x = 'date',y = "net_value",color = 'r')
df.plot(figsize = (8,8),grid = True, use_index = True,x = 'date',y = "equity_curve",color = 'g')
df.plot(figsize = (8,8),grid = True, use_index = True,x = 'date',y = "equity_change")
plt.show()

print("the frequency of buy or sell:",len(df.loc[open_pos_condition + close_pos_condition, 'stock_num']))
max_net = df['net_value'].max()

remains = df['equity_curve']/df['equity_curve'].expanding().max()

print("最大回撤率为：",round((1 - remains.min()),2))
print("收益率均值为：",df['equity_change'].mean())
print("收益率标准差为：",df['equity_change'].std())
print("交易频率为：",len(df.loc[open_pos_condition + close_pos_condition, 'stock_num'])/len(df))

