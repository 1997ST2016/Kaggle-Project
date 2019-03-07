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

#gc.collect()
train = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/train.csv')
train = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/train.csv',parse_dates=["first_active_month"])
train.head()
train.target[train.target<-30].value_counts()
len(train.target[train.target<-30])/len(train)
