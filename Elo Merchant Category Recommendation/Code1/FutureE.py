import pandas as pd
import numpy as np
import copy
from IPython.display import display
import matplotlib.pylab as plt
import matplotlib.pyplot as plt

#from time import date
train = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/train.csv',parse_dates=["first_active_month"])
test = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/test.csv',parse_dates=["first_active_month"])
merchant = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/merchants.csv')
hist_trans = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/historical_transactions.csv')


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


hist_trans = pd.get_dummies(hist_trans, columns=['category_2', 'category_3'])
hist_trans['authorized_flag'] = hist_trans['authorized_flag'].map({'Y': 1, 'N': 0})
hist_trans['category_1'] = hist_trans['category_1'].map({'Y': 1, 'N': 0})
hist_trans.head()
#shape of data
print('train shape', train.shape)
print('test shape', test.shape)


correlation = train_df.corr()
plt.figure(figsize=(20,15))
#mask = np.zeros_like(correlation)
# mask[np.triu_indices_from(mask)] = True
sns.heatmap(correlation, annot=True)


'''
correlation = train_df.corr()
plt.figure(figsize=(20,15))
# mask = np.zeros_like(correlation)
# mask[np.triu_indices_from(mask)] = True
sns.heatmap(correlation, annot=True)
'''

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
    '''
    修改一个对象时：
          inplace=True：不创建新的对象，直接对原始对象进行修改；
          inplace=False：对数据进行修改，创建并返回新的对象承载其修改结果。
    '''
    df = (trans.groupby('card_id')
          .size()
          .reset_index(name='{}transactions_count'.format(prefix)))
    
    agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')
    #merge就是融合了;left就是根据左边的，也就是df的卡号card_id来融合;要比普通合并conact好一些.    
    return agg_trans
#shape of data
print('train shape', train.shape)
print('test shape', test.shape)
'''
agg_trans = trans.groupby(['card_id']).agg(agg_func)
agg_trans.columns = [prefix + '_'.join(col).strip() 
                           for col in agg_trans.columns.values]
merch_hist = aggregate_transactions(hist_trans, prefix='hist_')
merch_hist.head()
'''

#train 和 test 的合并
#hist_trains = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/historical_transactions.csv')
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


correlation = train_df.corr()
plt.figure(figsize=(20,15))
# mask = np.zeros_like(correlation)
# mask[np.triu_indices_from(mask)] = True
sns.heatmap(correlation, annot=True)
