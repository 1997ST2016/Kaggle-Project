def mis_value_graph(data, name = ""):
    data = [
    go.Bar(
        x = data.columns,
        y = data.isnull().sum(),
        name = name,
        textfont=dict(size=20),
        marker=dict(
        color= generate_color(),
        line=dict(
            color='#000000',
            width=1,
        ), opacity = 0.85
    )
    ),
    ]
    layout= go.Layout(
        title= 'Total Missing Value of'+ str(name),
        xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),
        yaxis=dict(title='Value Count', ticklen=5, gridwidth=2),
        showlegend=True
    )
    fig= go.Figure(data=data, layout=layout)
    ply.iplot(fig, filename='skin')
    
def datatypes_pie(data, title = ""):
    # Create a trace
    colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']
    trace1 = go.Pie(
        labels = ['float64','Int64'],
        values = data.dtypes.value_counts(),
        textfont=dict(size=20),
        marker=dict(colors=colors,line=dict(color='#000000', width=2)), hole = 0.45)
    layout = dict(title = "Data Types Count Percentage of "+ str(title))
    data = [trace1]
    ply.iplot(dict(data=data, layout=layout), filename='basic-line')
    

def mis_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data


import random

def generate_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
    return color
