import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as scs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import dateutil
from dateutil.parser import parse
import os,pymssql
import re
import scipy.optimize as sco


def get_data(sql1):
    server="X"
    user="X"
    password="X"
    conn=pymssql.connect(server,user,password,database="X",charset='utf8')
    cursor=conn.cursor()
    cursor.execute(sql1)
    row=cursor.fetchall()
    conn.close()
    data =pd.DataFrame(row,columns=zip(*cursor.description)[0])
    data = l2gbyR(data)
    return data

def latin2gbk(s):
    if type(s)==unicode:
        s = s.encode('latin1').decode('gbk')
    elif s is None:
        s = np.nan
    return s

def l2gbyR(data):
    for i in data.columns:    
        try:
            data[i] = data[i].apply(lambda s: latin2gbk(s))
        except:
            continue
    return data
def re_s(s,strr):
    pattern = re.compile(strr)
    m = pattern.search(s)
    if m is not None:
        return m.group()
    else:
        return np.nan
    
def decimal_to_float(df,s):
    for i in s:
        df[i] = df[i].apply(lambda s:float(s))
    return df


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
funds_list=['159920','518880','513500']
index_list=['000012','000013','000300','000905']
asset_num=len(funds_list)+len(index_list)

data=pd.DataFrame()
for fund in range(len(funds_list)):
    i="'"+str(funds_list[fund])+"'"
    if fund==0:
        
        a=get_data("SELECT b.TradingDay, b.ClosePrice as "+i+" FROM SecuMain a, QT_DailyQuote b\
                   WHERE a.InnerCode=b.InnerCode AND a.SecuCode="+ i +" \
                   AND a.SecuCategory=8 AND b.TradingDay>='2016-07-01'\
                   AND b.TradingDay<='2018-06-30' ORDER BY b.TradingDay ASC")
        data=a
    else:
        a=get_data("SELECT b.TradingDay, b.ClosePrice as "+i+" FROM SecuMain a, QT_DailyQuote b\
                   WHERE a.InnerCode=b.InnerCode AND a.SecuCode="+ i +" \
                   AND a.SecuCategory=8 AND b.TradingDay>='2016-07-01'\
                   AND b.TradingDay<='2018-06-30' ORDER BY b.TradingDay ASC")
        data=pd.merge(data,a,how='outer')

for index in range(len(index_list)):
    i="'"+str(index_list[index])+"'"
    a=get_data("SELECT b.TradingDay, b.ClosePrice  as"+i+" FROM SecuMain a, QT_DailyQuote b\
               WHERE a.InnerCode=b.InnerCode AND a.SecuCode="+ i +" \
               AND a.SecuCategory=4 AND b.TradingDay>='2016-07-01'\
               AND b.TradingDay<='2018-06-30' ORDER BY b.TradingDay ASC")
    data=pd.merge(data,a,how='outer')

data=data.set_index('TradingDay',drop=True)


data=data.astype(float)
data=data.rename(columns={'159920':'华夏恒指ETF','518880':'华安黄金ETF',
                          '513500':'博时标普500ETF','000012':'上证国债',
                          '000013':'上证企债','000300':'沪深300','000905':'中证500'})
(data/data.iloc[0]*100).plot(figsize=(15,10))
plt.grid(True)


log_returns=np.log(data.pct_change()+1)


port_returns = []
port_variance = []
for p in range(10000):
    weights = np.random.random(asset_num)
    weights /=np.sum(weights)
    port_returns.append(np.sum(log_returns.mean()*252*weights))
    port_variance.append(np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*252, weights))))

port_returns = np.array(port_returns)
port_variance = np.array(port_variance)

#无风险利率设定为4%
risk_free = 0.04
plt.figure(figsize = (12,8))
plt.scatter(port_variance, port_returns, c=(port_returns-risk_free)/port_variance, marker = 'o')
plt.grid(True)
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.colorbar(label = 'Sharpe Ratio')



def statistics(weights):
    weights=np.array(weights)
    pret=np.sum(log_returns.mean()*weights*252)
    pvol=np.sqrt(np.dot(weights.T,np.dot(log_returns.cov()*252,weights)))
    diverpara=np.sum(log_returns.std()*weights)/pvol
    return np.array([pret,pvol,((pret-0.04)/pvol),diverpara])


#1.最大化夏普比率
def min_sharpe(weights):
    return -statistics(weights)[2]

#约束是所有参数(权重)的总和为1。这可以用minimize函数的约定表达如下
cons = ({'type':'eq', 'fun':lambda x: np.sum(x)-1})
#我们还将参数值(权重)限制在0和1之间。这些值以多个元组组成的一个元组形式提供给最小化函数
bnds = tuple((0,1) for x in range(asset_num))
opts = sco.minimize(min_sharpe, asset_num*[1./asset_num,], method = 'SLSQP',
                    bounds = bnds, constraints = cons)
sharpe_weights=opts['x'].round(3)#获取满足条件的各资产比重
sharpe_stats=statistics(opts['x']).round(3)#获取预期收益率，预期波动率和夏普比率

#输出结果
def get_maxsharpe_portfolio():
    for i in range(asset_num):
        print (str(data.columns[i])+ ':' + str(sharpe_weights[i]))
    tag=['Return','Volatility','SharpeRatio']
    for i in range(len(sharpe_stats)):
        print (tag[i]+ ':' + str(sharpe_stats[i]))
        
        
        
#2.最小方差
def min_vol(weights):
    return statistics(weights)[1]

optv = sco.minimize(min_vol, asset_num*[1./asset_num,],method = 'SLSQP',
                    bounds = bnds, constraints = cons)
minvol_weights=optv['x'].round(3)
minvol_stats=statistics(optv['x']).round(3)

def get_minvol_portfolio():
    for i in range(asset_num):
        print (str(data.columns[i])+ ':' + str(minvol_weights[i]))
        
    tag=['Return','Volatility','SharpeRatio']
    
    for i in range(len(minvol_stats)):
        print (tag[i]+ ':' + str(minvol_stats[i]))
        
        
        
#3.给定目标风险，寻找最优组合
def tar_vol(weights):
    return -1*statistics(weights)[0]

def get_tarvol_portfolio(VolTar):
    cons1 = ({'type':'ineq','fun':lambda x: VolTar - statistics(x)[1]},
             {'type':'eq','fun':lambda x:np.sum(x)-1})    
    opttarv=sco.minimize(tar_vol,asset_num*[1./asset_num,],
                         method = 'SLSQP',bounds = bnds, constraints = cons1)
    tarvol_weights=opttarv['x'].round(3)
    tarvol_stats=statistics(opttarv['x']).round(3)
    
    for i in range(asset_num):
        print (str(data.columns[i])+ ':' + str(tarvol_weights[i]))
        
    tag=['Return','Volatility','SharpeRatio']
    
    for i in range(len(tarvol_stats)):
        print (tag[i]+ ':' + str(tarvol_stats[i]))

        
        
        
#4.风险分散最优化
def opt_diverse(weights):
    return -1*statistics(weights)[3]

def get_optdiver_portfolio():
    cons2 = ({'type':'eq','fun':lambda x:np.sum(x)-1})    
    optdiver=sco.minimize(opt_diverse,asset_num*[1./asset_num,],
                          method = 'SLSQP',bounds = bnds, constraints = cons2)
    optdiver_weights=optdiver['x'].round(3)
    optdiver_stats=statistics(optdiver['x']).round(3)
    
    for i in range(asset_num):
        print (str(data.columns[i])+ ':' + str(optdiver_weights[i]))
        
    tag=['Return','Volatility','SharpeRatio']
    
    for i in range(len(optdiver_stats)):
        print (tag[i]+ ':' + str(optdiver_stats[i]))
        
        
get_optdiver_portfolio()
get_tarvol_portfolio('VolTar')
get_minvol_portfolio()
get_maxsharpe_portfolio()
