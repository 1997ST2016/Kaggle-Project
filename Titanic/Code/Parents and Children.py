
#�ڴ��ϵĸ�ĸ/���Ӹ���Ӱ�����
# -*- coding: utf8 -*-
from pylab import *
mpl.rcParams['font.sans-serif'] = ['FangSong'] # ָ��Ĭ������
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
data_train = pd.read_csv("D:/My University/2019WinterVacation/Project/Titanic/train.csv")
#ticket�Ǵ�Ʊ��ţ�Ӧ����unique�ģ������Ľ��û��̫��Ĺ�ϵ���Ȳ����뿼�ǵ����������
#cabinֻ��204���˿���ֵ�������ȿ�������һ���ֲ�
data_train.Cabin.value_counts()
