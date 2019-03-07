#��Ҫ��������û����Ⱥֵoutlier��˳�㿴һ�����ݷֲ����ⲿ�ִ���Ŷ���ǿ���ֱ���õġ�
#%%time
import pandas as pd
import plotly.offline
import plotly.offline as ply
import ply
import plotly.graph_objs as go
train = pd.read_csv('D:/MyUniversity/2019WV/Project/Elo/Data/train.csv',parse_dates=["first_active_month"])
x = train.target
data = [go.Histogram(x=x,
                     histnorm='probability')]
layout = go.Layout(
    title='Target Distribution',
    xaxis=dict(title='Value'),yaxis=dict(title='Count'),
    bargap=0.2,
    bargroupgap=0.1
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename='normalized histogram')

#���ͼ�����������ף����ǰ�x����Ȼ��ͼ��
x = train['first_active_month'].dt.date.value_counts()
x = x.sort_index()
data0 = [go.Histogram(x=x.index,y = x.values,histnorm='probability', marker=dict(color = generate_color()))]  #abc
layout = go.Layout(
    title='First active month count Train Data',
    xaxis=dict(title='First active month',ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Number of cards',ticklen=5, gridwidth=2),
    bargap=0.1,
    bargroupgap=0.2
)
fig = go.Figure(data=data0, layout=layout)
plotly.offline.iplot(fig, filename='normalized histogram')
