import pandas as pd
#mis_impute�������Ƿ�װ�õ����ڴ���ȱʧֵ�ĺ���
def mis_impute(data):
    for i in data.columns:
    #��data����ѭ���������������Ϊobject�����Ϊother������
    #��ʱ���ù���û���ã���˼��������
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
            #������Ϊ��ֵ���������еľ�ֵ��
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data
#time#��ʾ����ʱ��
train = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/train.csv',parse_dates=["first_active_month"])
test = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/test.csv',parse_dates=["first_active_month"])
merchant = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/merchants.csv')
hist_trans = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/historical_transactions.csv')
for i in [train,test,merchant, hist_trans]:
    print("Impute the Missing value of ", i.name)
    mis_impute(i)
    print("Done Imputation on", i.name)
#gc.collect()
