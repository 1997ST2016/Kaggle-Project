#Cabin分布,cabin只有204个乘客有值，我们先看看它的一个分布
# -*- coding: utf8 -*-
from pylab import *
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
data_train = pd.read_csv("D:/My University/2019WinterVacation/Project/Titanic/train.csv")
#cabin只有204个乘客有值，我们先看看它的一个分布
data_train.Cabin.value_counts()

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df=pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title(u"按Cabin有无看获救情况")
plt.xlabel(u"Cabin有无") 
plt.ylabel(u"人数")
plt.show()
