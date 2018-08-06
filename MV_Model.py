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
