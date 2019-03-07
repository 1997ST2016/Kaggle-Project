
#在船上的父母/孩子个数影响情况
# -*- coding: utf8 -*-
from pylab import *
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
data_train = pd.read_csv("D:/My University/2019WinterVacation/Project/Titanic/train.csv")
#ticket是船票编号，应该是unique的，和最后的结果没有太大的关系，先不纳入考虑的特征范畴把
#cabin只有204个乘客有值，我们先看看它的一个分布
data_train.Cabin.value_counts()
