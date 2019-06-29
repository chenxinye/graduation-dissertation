# -*- coding: utf-8 -*-
#@author:chenxinye
#@2019.06.09

"""
Created on Sat Mar 16 06:32:49 2019
@author: Chenxinye
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("output/signal.csv")

def test(df,Commission = 0):
    #交易佣金一般是买卖金额的0.1%－0.3%,这里设置初始资金为100000000,交易佣金Commission默认为0
    #bid = 0说明买入，否则卖出，account是入账金额
    capital_set = 100000000
    account = pd.Series(np.zeros((len(df))))
    capital_set_d = pd.Series(np.zeros((len(df))))
    capital_set_d[0] = 100000000
    bid = 1
    num = 0
    for i in [j + 1 for j in range(len(df)-1)]:
        
        change = (df['close'][i] - df['close'][i - 1])/df['close'][i - 1]
        
        #print(change)
        if (df['finalsig'][i] < 0.5):
            #buy
            if bid == 1:
                account[i] = capital_set - Commission*capital_set
                capital_set = 0
                capital_set_d[i] = 0
                bid = 0
                num = num + 1
            else:
                account[i] = account[i - 1]*(1 + change)
            
        elif (df['finalsig'][i] > 0.5):
            #sell
            if bid == 0:
                account[i] = account[i - 1]*(1 + change)
                capital_set  = account[i] - Commission*account[i]
                capital_set_d[i+1] = account[i] - Commission*account[i]
                account[i + 1] = 0
                bid = 1
                num = num + 1
            else:capital_set_d[i+1] = capital_set_d[i]
        else:
            account[i] = account[i - 1]*(1 + change)
            capital_set_d[i+1] = capital_set_d[i]
            
        if (i == len(df) - 1) and (bid == 0):
            #when the last day come, sell all of the capital in your account
            account[i] = account[i - 1]*(1 + change)
            capital_set  = account[i] - Commission*account[i]
            capital_set_d[i+1] = account[i] - Commission*account[i]
            bid = 1
            num = num + 1
#        if capital_set != 0:
#            print(capital_set)
        
    df['投入资金'] = account
    df['资金账户'] = capital_set_d
    df['net_value'] = account + capital_set_d
    df['equity_change'] = df['net_value'].pct_change(fill_method=None)
    df['equity_curve'] = (1 + df['equity_change']).cumprod()
    profit = (capital_set - 100000000)/100000000
    print("佣金率:%s"%str(Commission) + " 这4年年化收益率为：%s"%str(profit/4))
    return(df,capital_set,capital_set_d,num)

df,capital_set,capital_set_d,num = test(df,Commission = 0)
df.to_csv("output/result_1.csv",index = False)
            

#plot
df.plot(figsize = (8,8),grid = True, use_index = True,x = 'date',y = "equity_curve",color = 'red')
plt.show()

remains = df['equity_curve']/df['equity_curve'].expanding().max()

print("最大回撤率为：",round((1 - remains.min()),2))
print("收益率均值为：",df['equity_change'].mean())
print("收益率标准差为：",df['equity_change'].std())
print("交易频率为：",num/len(df))