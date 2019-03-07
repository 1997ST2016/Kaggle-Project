
# -*- coding: utf8 -*-
from pylab import *
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
import pandas as pd
import re
import numpy as np
from pandas import Series,DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model

data_train = pd.read_csv("D:/My University/2019WinterVacation/Project/Titanic/train.csv")

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()

df=pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()
 
# 用正则取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()
 
# y即Survival结果
y = train_np[:,0]
 
# X即特征属性值
X = train_np[:,1:]
 
# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
#penalty 一个字符串，制定了正则化策略
#tol 一个浮点数，制定判断迭代收敛与否的阈值
#C 
clf.fit(X, y)

 
clf
