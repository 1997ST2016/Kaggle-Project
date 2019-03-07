#在船上的堂兄/弟、姐/妹个数的获救情况
# -*- coding: utf8 -*-
from pylab import *
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
data_train = pd.read_csv("D:/My University/2019WinterVacation/Project/Titanic/train.csv")
g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print df

