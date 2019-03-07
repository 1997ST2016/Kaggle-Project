分析建模

数据载入：

```python
import pandas as pd
import numpy as np
ht = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/historical_transactions.csv', dtype={'city_id': np.int16, 'installments': np.int8, 'merchant_category_id': np.int16, 'month_lag': np.int8, 'purchase_amount': np.float32, 'category_2': np.float16, 'state_id': np.int8, 'subsector_id':np.int8})
nmt = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/new_merchant_transactions.csv', dtype={'city_id': np.int16, 'installments': np.int8, 'merchant_category_id': np.int16, 'month_lag': np.int8, 'purchase_amount': np.float32, 'category_2': np.float16, 'state_id': np.int8, 'subsector_id':np.int8})
train = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/train.csv')
test = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/test.csv')
#ht.info()
train.head()
train.info()
```

train.info()结果：

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 201917 entries, 0 to 201916
Data columns (total 6 columns):
first_active_month    201917 non-null object
card_id               201917 non-null object
feature_1             201917 non-null int64
feature_2             201917 non-null int64
feature_3             201917 non-null int64
target                201917 non-null float64
dtypes: float64(1), int64(3), object(2)
memory usage: 9.2+ MB
```

 train.head()结果：

```
  first_active_month          card_id  feature_1  feature_2  feature_3  \
0         2017-06-01  C_ID_92a2005557          5          2          1   
1         2017-01-01  C_ID_3d0044924f          4          1          0   
2         2016-08-01  C_ID_d639edf6cd          2          2          0   
3         2017-09-01  C_ID_186d6a6901          4          3          0   
4         2017-11-01  C_ID_cdbd2c0db2          1          3          0 
```

在交互式界面shell输入语句查看数据信息：

查看这些数据有哪些类型： ‘data’.dtypes.value_counts 或者 ‘train’.dtypes.unique 或者 train.dtypes.nunique ;

```
<bound method Series.nunique of first_active_month     object
card_id                object
feature_1               int64
feature_2               int64
feature_3               int64
target                float64
```

查看缺失值：

```
输入‘train’.isnull().sum()
```

结果：

```
first_active_month    0
card_id               0
feature_1             0
feature_2             0
feature_3             0
target                0
dtype: int64
```

```python 
在Python Shell输入：ht.isnull().sum()
```

结果：

```
authorized_flag               0
card_id                       0
city_id                       0
category_1                    0
installments                  0
category_3               178159
merchant_category_id          0
merchant_id              138481
month_lag                     0
purchase_amount               0
purchase_date                 0
category_2              2652864
state_id                      0
subsector_id                  0
dtype: int64
```

```
输入：train.target[train.target<-30].value_counts()
```


​	离群值 结果：

```
-33.219281    2207
Name: target, dtype: int64
```

离群值占比：

```
输入：len(train.target[train.target<-30])/len(train)
```

输出：

```
0.010930233709890698
```

特征工程：

```python
train = pd.concat([train,feat1, feat2, feat3], axis=1, sort=False)
```

报错：TypeError: concat() got an unexpected keyword argument 'sort'

解决：去掉sort 这一项，然后运行。

##### 但是结果不一样！！！！！

sort：默认为True，将合并的数据进行排序。在大多数情况下设置为False可以提高性能

# **time用法和datetime**???

##### test['elapsed_time'] = (datetime.date(2018, 2, 1) - test['first_active_month'].dt.date).dt.days



#### DistributionD.py ：   NameError: name 'generate_color' is not defined