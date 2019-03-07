# -*- coding: utf8 -*-
from pylab import *
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
data_train = pd.read_csv("D:/My University/2019WinterVacation/Project/Titanic/train.csv")

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()

df=pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
df
