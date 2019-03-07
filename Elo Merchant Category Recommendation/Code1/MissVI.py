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
    print("Impute the Missing value of ", i.name)
    mis_impute(i)
    print("Done Imputation on", i.name)
#gc.collect()
