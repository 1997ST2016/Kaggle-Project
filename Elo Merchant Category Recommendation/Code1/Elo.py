import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
# plt.style.use("fivethirtyeight")
plt.style.use('ggplot')#选择绘图风格，就是能好看一点！
import seaborn as sns#类似matplotlib的画图包
import gc#gc.collect()，显式回收内存，见②

sns.set(style="ticks", color_codes=True)#设置画图空间为 Seaborn 默认风格
import matplotlib.pyplot as plt
from tqdm._tqdm_notebook import tqdm_notebook as tqdm#③
tqdm.pandas()#③
import datetime
#关于plotly库④
#import plotly.offline as ply
import ply
#ply.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')#不显示warning


#functions

#pictures
# Read in the dataframes
import pandas as pd
import gc
def load_data():
    train = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/train.csv',parse_dates=["first_active_month"])
    test = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/test.csv',parse_dates=["first_active_month"])
    merchant = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/merchants.csv')
    hist_trans = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/historical_transactions.csv')
    print('train shape', train.shape)
    print('test shape', train.shape)
    print('merchants shape', merchant.shape)
    print('historical_transactions', hist_trans.shape)
    return (train,test,merchant,hist_trans)


train = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/train.csv')
#train = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/train.csv',parse_dates=["first_active_month"])
train.head()
'''
train.target[train.target<-30].value_counts()
len(train.target[train.target<-30])/len(train)
'''


def mis_value_graph(data, name = ""):
    data = [
    go.Bar(
        x = data.columns,
        y = data.isnull().sum(),
        name = name,
        textfont=dict(size=20),
        marker=dict(
        color= generate_color(),
        line=dict(
            color='#000000',
            width=1,
        ), opacity = 0.85
    )
    ),
    ]
    layout= go.Layout(
        title= 'Total Missing Value of'+ str(name),
        xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),
        yaxis=dict(title='Value Count', ticklen=5, gridwidth=2),
        showlegend=True
    )
    fig= go.Figure(data=data, layout=layout)
    ply.iplot(fig, filename='skin')
    
def datatypes_pie(data, title = ""):
    # Create a trace
    colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']
    trace1 = go.Pie(
        labels = ['float64','Int64'],
        values = data.dtypes.value_counts(),
        textfont=dict(size=20),
        marker=dict(colors=colors,line=dict(color='#000000', width=2)), hole = 0.45)
    layout = dict(title = "Data Types Count Percentage of "+ str(title))
    data = [trace1]
    ply.iplot(dict(data=data, layout=layout), filename='basic-line')
    

def mis_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data


import random

def generate_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
    return color




import pandas as pd
#mis_impute函数就是封装好的用于处理缺失值的函数
def mis_impute(data):
    for i in data.columns:
    #对data的列循环，如果该列类型为object，填充为other。。。
    #暂时不用管有没有用，意思是这样的
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
            #入错该列为数值，就用这列的均值填
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data
#time#显示运行时间
train = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/train.csv',parse_dates=["first_active_month"])
test = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/test.csv',parse_dates=["first_active_month"])
merchant = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/merchants.csv')
hist_trans = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/historical_transactions.csv')
for i in [train,test,merchant, hist_trans]:
    #print("Impute the Missing value of ", i.name)
    mis_impute(i)
    # print("Done Imputation on", i.name)



#主要看就是有没有离群值outlier。顺便看一下数据分布，这部分代码哦我们可以直接用的。
#%%time
import pandas as pd
import plotly.offline
import plotly.offline as ply
import ply
import plotly.graph_objs as go
train = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/train.csv',parse_dates=["first_active_month"])
x = train.target
data = [go.Histogram(x=x,
                     histnorm='probability')]
layout = go.Layout(
    title='Target Distribution',
    xaxis=dict(title='Value'),yaxis=dict(title='Count'),
    bargap=0.2,
    bargroupgap=0.1
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename='normalized histogram')


#这个图做起来很容易，就是把x排序然后画图。
x = train['first_active_month'].dt.date.value_counts()
x = x.sort_index()
data0 = [go.Histogram(x=x.index,y = x.values,histnorm='probability', marker=dict(color = generate_color()))]  #abc
layout = go.Layout(
    title='First active month count Train Data',
    xaxis=dict(title='First active month',ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Number of cards',ticklen=5, gridwidth=2),
    bargap=0.1,
    bargroupgap=0.2
)
fig = go.Figure(data=data0, layout=layout)
plotly.offline.iplot(fig, filename='normalized histogram')


train.target[train.target<-30].value_counts()
len(train.target[train.target<-30])/len(train)




#from time import date
train = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/train.csv',parse_dates=["first_active_month"])
test = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/test.csv',parse_dates=["first_active_month"])

##---------------Time based Feature
train['day']= train['first_active_month'].dt.day 
train['dayofweek']= train['first_active_month'].dt.dayofweek
train['dayofyear']= train['first_active_month'].dt.dayofyear
train['days_in_month']= train['first_active_month'].dt.days_in_month
train['daysinmonth']= train['first_active_month'].dt.daysinmonth 
train['month']= train['first_active_month'].dt.month
train['week']= train['first_active_month'].dt.week 
train['weekday']= train['first_active_month'].dt.weekday
train['weekofyear']= train['first_active_month'].dt.weekofyear
train['year']= train['first_active_month'].dt.year
#train['elapsed_time'] = (datetime.date(2018, 2, 1) - train['first_active_month'].dt.date).dt.days
##---------------Time based Test Feature      
test['day']= test['first_active_month'].dt.day 
test['dayofweek']= test['first_active_month'].dt.dayofweek
test['dayofyear']= test['first_active_month'].dt.dayofyear
test['days_in_month']= test['first_active_month'].dt.days_in_month
test['daysinmonth']= test['first_active_month'].dt.daysinmonth 
test['month']= test['first_active_month'].dt.month
test['week']= test['first_active_month'].dt.week 
test['weekday']= test['first_active_month'].dt.weekday
test['weekofyear']= test['first_active_month'].dt.weekofyear
test['year']= test['first_active_month'].dt.year
#test['elapsed_time'] = (datetime.date(2018, 2, 1) - test['first_active_month'].dt.date).dt.days
print('train shape', train.shape)
print('test shape', test.shape)



feat1 = pd.get_dummies(train['feature_1'], prefix='f1_')
feat2 = pd.get_dummies(train['feature_2'], prefix='f2_')
feat3 = pd.get_dummies(train['feature_3'], prefix='f3_')
feat4 = pd.get_dummies(test['feature_1'], prefix='f1_')
feat5 = pd.get_dummies(test['feature_2'], prefix='f2_')
feat6 = pd.get_dummies(test['feature_3'], prefix='f3_')

##---------------Numerical representation of the first active month
train = pd.concat([train,feat1, feat2, feat3], axis=1)
test = pd.concat([test,feat4, feat5, feat6], axis=1)

#shape of data
print('train shape', train.shape)
print('test shape', test.shape)




correlation = train_df.corr()
plt.figure(figsize=(20,15))
# mask = np.zeros_like(correlation)
# mask[np.triu_indices_from(mask)] = True
sns.heatmap(correlation, annot=True)


def aggregate_transactions(trans, prefix):  
    trans.loc[:, 'purchase_date'] = pd.DatetimeIndex(trans['purchase_date']).\
                                      astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['mean'],
        'category_2_1.0': ['mean'],
        'category_2_2.0': ['mean'],
        'category_2_3.0': ['mean'],
        'category_2_4.0': ['mean'],
        'category_2_5.0': ['mean'],
        'category_3_A': ['mean'],
        'category_3_B': ['mean'],
        'category_3_C': ['mean'],
        'merchant_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_date': [np.ptp],
        'month_lag': ['min', 'max']
    }
    agg_trans = trans.groupby(['card_id']).agg(agg_func)
    agg_trans.columns = [prefix + '_'.join(col).strip() 
                           for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)
    
    df = (trans.groupby('card_id')
          .size()
          .reset_index(name='{}transactions_count'.format(prefix)))
    
    '''agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')
    agg_trans.columns = [prefix + '_'.join(col).strip() 
                           for col in agg_trans.columns.values]
    merch_hist = aggregate_transactions(hist_trans, prefix='hist_')
    merch_hist.head()
    '''
    agg_trans.reset_index(inplace=True)
    '''
    修改一个对象时：
          inplace=True：不创建新的对象，直接对原始对象进行修改；
          inplace=False：对数据进行修改，创建并返回新的对象承载其修改结果。
      '''
    
    agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')
    #merge就是融合了;left就是根据左边的，也就是df的卡号card_id来融合;要比普通合并conact好一些.
    return agg_trans

#shape of data
print('train shape', train.shape)
print('test shape', test.shape)



#train 和 test 的合并
hist_trans = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/historical_transactions.csv')
merch_hist = aggregate_transactions(hist_trans, prefix='hist_')
train = pd.merge(train, merch_hist, on='card_id',how='left')
test = pd.merge(test, merch_hist, on='card_id',how='left')
#shape of data
print('train shape', train.shape)
print('test shape', test.shape)


###  New Merchant Feature
new_trans_df = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/new_merchant_transactions.csv')
display(new_trans_df.head())
new_trans_df.hist(figsize = (17,12))

train[["target","feature_1"]].hist()   #画直方图


new_trans_df = pd.get_dummies(new_trans_df, columns=['category_2', 'category_3'])
new_trans_df['authorized_flag'] = new_trans_df['authorized_flag'].map({'Y': 1, 'N': 0})
new_trans_df['category_1'] = new_trans_df['category_1'].map({'Y': 1, 'N': 0})
new_trans_df.head()
#shape of data
print('train shape', train.shape)
print('test shape', test.shape)

merch_hist = aggregate_transactions(hist_trans, prefix='hist_')
train = pd.merge(train, merch_hist, on='card_id',how='left')
test = pd.merge(test, merch_hist, on='card_id',how='left')
#shape of data
print('train shape', train.shape)
print('test shape', test.shape)



target = train['target']
drops = ['card_id', 'first_active_month', 'target', 'date']
use_cols = [c for c in train.columns if c not in drops]
features = list(train[use_cols].columns)
train[features].head()

print('train shape', train.shape)
print('test shape', test.shape)
train_df = train.copy()
test_df = test.copy()
print('train shape', train_df.shape)
print('test shape', test_df.shape)

train_df = train.copy()
correlation = train_df.corr()
plt.figure(figsize=(20,15))
# mask = np.zeros_like(correlation)
# mask[np.triu_indices_from(mask)] = True
sns.heatmap(correlation, annot=True)
    

