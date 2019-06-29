#@author:chenxinye
#@2019.06.09

import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 5000) 

# =====读入股票数据output
df = pd.read_csv("output/signal.csv")
# 任何原始数据读入都进行一下排序、去重，以防万一
df.sort_values(by=['date'], inplace=True)
df.drop_duplicates(subset=['date'], inplace=True)
df.reset_index(inplace=True, drop=True)

# =====计算后复权价
df['最高价'] = df['high']
df['最低价'] = df['low']
df['收盘价'] = df['close']
df['开盘价'] = df['open']
df['前收盘价'] = df['收盘价'].shift(-1)
df['涨跌幅'] = df['收盘价'] / df['前收盘价'] - 1
df['复权因子'] = (1 + df['涨跌幅']).cumprod()
df['收盘价_复权'] = df['复权因子'] * (df.iloc[0]['收盘价'] / df.iloc[0]['复权因子'])
df['开盘价_复权'] = df['开盘价'] / df['收盘价'] * df['收盘价_复权']
df['最高价_复权'] = df['最高价'] / df['收盘价'] * df['收盘价_复权']
df['最低价_复权'] = df['最低价'] / df['收盘价'] * df['收盘价_复权']
#df.drop(['复权因子'], axis=1, inplace=True)


# =====计算涨跌停价格
df['涨停价'] = df['前收盘价'] * 1.1
df['跌停价'] = df['前收盘价'] * 0.9
# 四舍五入
# print(round(3.5), round(4.5))  # 银行家舍入法：四舍六进，五，奇进偶不进
df['涨停价'] = df['涨停价'].apply(lambda x: float(Decimal(x*100).quantize(Decimal('1'), rounding=ROUND_HALF_UP) / 100))
df['跌停价'] = df['跌停价'].apply(lambda x: float(Decimal(x*100).quantize(Decimal('1'), rounding=ROUND_HALF_UP) / 100))

df['signal'] = df['finalsig']
df.loc[df.signal > 0.5,'signal'] = 1
df.loc[df.signal < 0.5,'signal'] = 0
df.to_hdf('output/signals.h5', key='df', mode='w')
