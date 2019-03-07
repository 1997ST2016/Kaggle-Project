Kaggle：Titanic人员是否生还预测

#### 题目来源：https://www.kaggle.com/c/titanic

题目概述如下图：

![](D:\My University\2019WinterVacation\Project\Titanic\Pictures\Description.png)

1. ##### 收集数据：

   ​	从Kaggle下载，如下图所示位置。在Data下给了两个数据文件：train.csv和test.csv，分别为训练数据和测试数据。

   ![](D:\My University\2019WinterVacation\Project\Titanic\Pictures\Data.png)

2. ##### 准备数据：

   ​	我们来看看数据，用pandas数据库处理包将csv文件读入成dataframe格式。

   ```python
   import pandas as pd
   import numpy as np
   from pandas import Series,DataFrame
   
   
   data_train = pd.read_csv("D:/My University/2019WinterVacation/Project/Titanic/train.csv")
   data_train
   ```

   ​	处理后的数据总共有12列，各字段含义如下：

   | 字段属性 | 含义                  | 字段属性    | 含义           |
   | -------- | --------------------- | ----------- | -------------- |
   | Survived | 是否获救（1是0否）    | PassengerId | 乘客ID         |
   | Pclass   | 乘客等级(1/2/3等舱位) | Name        | 乘客姓名       |
   | Sex      | 性别                  | Age         | 年龄           |
   | SibSp    | 船上堂兄弟/姐妹个数   | Parch       | 父母与小孩个数 |
   | Ticket   | 船票信息              | Fare        | 票价           |
   | Cabin    | 客舱号                | Embarked    | 登船港口       |

   ​	用data_train.info()语句来统计一下数据信息，如下图所示：

   ![](D:\My University\2019WinterVacation\Project\Titanic\Pictures\DataFrame格式的数据.png)

   ​	从上面的数据我们可以看到训练数据文件中的有些属性的数据不全：

   （1）Age（年龄）属性只有714名乘客有数据；

   （2）Cabin（客舱）属性只有204名乘客有数据；

   ​	进行**数据缺失值处理**：

   ​	代码如下：

   ```python
   #缺失值处理
   # -*- coding: utf8 -*-
   from pylab import *
   mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
   import pandas as pd
   import numpy as np
   from pandas import Series,DataFrame
   from sklearn.ensemble import RandomForestRegressor
   
   data_train = pd.read_csv("D:/My University/2019WinterVacation/Project/Titanic/train.csv")
   
   ### 使用 RandomForestClassifier 填补缺失的年龄属性
   def set_missing _ages(df):
   
       # 把已有的数值型特征取出来丢进Random Forest Regressor中
       age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
   
       # 乘客分成已知年龄和未知年龄两部分
       known_age = age_df[age_df.Age.notnull()].values
       unknown_age = age_df[age_df.Age.isnull()].values
   
       # y即目标年龄
       y = known_age[:, 0]
   
       # X即特征属性值
       X = known_age[:, 1:]
   
       # fit到RandomForestRegressor之中
       rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
       rfr.fit(X, y)
   
       # 用得到的模型进行未知年龄结果预测
       predictedAges = rfr.predict(unknown_age[:, 1::])
   
       # 用得到的预测结果填补原缺失数据
       df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
   
       return df, rfr
   
   def set_Cabin_type(df):
       df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
       df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
       return df
   
   data_train, rfr = set_missing_ages(data_train)
   data_train = set_Cabin_type(data_train)
   ```

   **RandomForestRegressor：**

   ​	随机森林是一个集成工具，它使用观测数据的子集来建立一个决策树。它建立多个这样的决策树，然后将它们合并在一起以获得更准确更准确和稳定的预测。这样做最直接的事实是，在这一组独立的预测中，用投票方式得到最高投票结果，这个比单独使用最好模型预测的结果要好。

   ​	我们通过调整以下特征来**改善模型的预测能力**：

   （1）max_features：

   ​	随机森林允许单个决策树使用特征的最大数量。 Python为最大特征数提供了多个可选项。 下面是其中的几个：

   ​	*Auto/None ：简单地选取所有特征，每颗树都可以利用他们。这种情况下，每颗树都没有任何的限制。

   ​	*sqrt ：此选项是每颗子树可以利用总特征数的平方根个。 例如，如果变量（特征）的总数是100，所以每颗子树只能取其中的10个。“log2”是另一种相似类型的选项。

   ​	*0.2：此选项允许每个随机森林的子树可以利用变量（特征）数的20％。如果想考察的特征x％的作用， 我们可以使用“0.X”的格式。

   ​	**增加max_features一般能提高模型的性能**，因为在每个节点上，我们有更多的选择可以考虑。 然而，这未必完全是对的，因为它降低了单个树的多样性，而这正是随机森林独特的优点。 但是，可以肯定，你通过增加max_features会降低算法的速度。 因此，你需要适当的平衡和选择最佳max_features。

   （2） n_estimators：

   ​	在利用最大投票数或平均值来预测之前，你想要建立子树的数量。 较多的子树可以让模型有更好的性能，但同时让你的代码变慢。 你应该选择尽可能高的值，只要你的处理器能够承受的住，因为这使你的预测更好更稳定。

   （3）min_sample_leaf：

   ​	最小样本叶片， 叶是决策树的末端节点。 较小的叶子使模型更容易捕捉训练数据中的噪声。 一般来说，我更偏向于将最小叶子节点数目设置为大于50。在你自己的情况中，你应该尽量尝试多种叶子大小种类，以找到最优的那个。

   ​	调整下面三个特征可以**改善模型的训练速度**：

   （1）n_jobs：

   ​	这个参数告诉引擎有多少处理器是它可以使用。 “-1”意味着没有限制，而“1”值意味着它只能使用一个处理器。 

   （2）random_state：

   ​	此参数让结果容易复现。 一个确定的随机值将会产生相同的结果，在参数和训练数据不变的情况下。 我曾亲自尝试过将不同的随机状态的最优参数模型集成，有时候这种方法比单独的随机状态更好。

   （3）oob_score：

   ​	这是一个随机森林交叉验证方法。 它和留一验证方法非常相似，但这快很多。 这种方法只是简单的标记在每颗子树中用的观察数据。 然后对每一个观察样本找出一个最大投票得分，是由那些没有使用该观察样本进行训练的子树投票得到。

   **特征因子化：**

   ​	因为逻辑回归建模时，需要输入的特征都是数值型特征，我们通常会先对类目型的特征因子化。

   什么叫做因子化呢？举个例子：

   ​	以Cabin为例，原本一个属性维度，因为其取值可以是[‘yes’,’no’]，而将其平展开为’Cabin_yes’,’Cabin_no’两个属性

   ​	**原本Cabin取值为yes的，在此处的”Cabin_yes”下取值为1，在”Cabin_no”下取值为0；

   ​	**原本Cabin取值为no的，在此处的”Cabin_yes”下取值为0，在”Cabin_no”下取值为1；

   ​	我们使用pandas的”get_dummies”来完成这个工作，并拼接在原来的”data_train”之上，如下所示。

   代码如下：

   ```python
   #特征因子化
   # -*- coding: utf8 -*-
   from pylab import *
   mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
   import pandas as pd
   import numpy as np
   from pandas import Series,DataFrame
   from sklearn.ensemble import RandomForestRegressor
   
   data_train = pd.read_csv("D:/My University/2019WinterVacation/Project/Titanic/train.csv")
   
   dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
   
   dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
   
   dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
   
   dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')
   
   df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
   df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
   df
   ```

   ​	离散因子化后把类目型属性全都转换成0,1的数值属性了。

   **归一化（scaling）：**

   ​	将特征值缩放到相同区间可以使得获取性能更好的模型。归一化是利用特征的最大最小值，将特征的值缩放到[0,1]区间，对于每一列的特征使用min-max函数进行缩放。例如此数据集中的Age和Fare两个属性，乘客的数值幅度变化很大，对收敛速度造成了巨大影响，甚至不收敛！！！所以，我们先用scikit-learn里面的preprocessing模块进行scaling，即归一化处理。

   ​	代码如下：

   ```python
   import sklearn.preprocessing as preprocessing
   scaler = preprocessing.StandardScaler()
   age_scale_param = scaler.fit(df['Age'])
   df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
   fare_scale_param = scaler.fit(df['Fare'])
   df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
   df
   ```

3. ##### 分析数据：

   ​        	下面我们来分析**各属性数据的与是否获救之间的关系**，即每个/多个属性和Survived之间有着什么样的关系。

   ```python
   import matplotlib.pyplot as plt
   fig = plt.figure()
   fig.set(alpha=0.2)  # 设定图表颜色alpha参数
   
   plt.subplot2grid((2,3),(0,0))       #在一张大图里分列几个小图
   data_train.Survived.value_counts().plot(kind='bar')# 柱状图 
   plt.ylabel(u"人数")  
   plt.title(u'获救情况，（1为获救）') # 标题
   
   plt.subplot2grid((2,3),(0,1))
   data_train.Pclass.value_counts().plot(kind="bar")
   plt.ylabel(u"人数")
   plt.title(u"乘客等级分布")
   
   plt.subplot2grid((2,3),(0,2))
   plt.scatter(data_train.Survived, data_train.Age)
   plt.ylabel(u"年龄")                         # 设定纵坐标名称
   plt.grid(b=True, which='major', axis='y') 
   plt.title(u"按年龄看获救分布 (1为获救)")
   
   
   plt.subplot2grid((2,3),(1,0), colspan=2)
   data_train.Age[data_train.Pclass == 1].plot(kind='kde')   
   data_train.Age[data_train.Pclass == 2].plot(kind='kde')
   data_train.Age[data_train.Pclass == 3].plot(kind='kde')
   plt.xlabel(u"年龄")# plots an axis lable
   plt.ylabel(u"密度") 
   plt.title(u"各等级的乘客年龄分布")
   plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.
   
   
   plt.subplot2grid((2,3),(1,2))
   data_train.Embarked.value_counts().plot(kind='bar')
   plt.title(u"各登船口岸上船人数")
   plt.ylabel(u"人数")  
   plt.show()
   ```

   数据基本图示：

   ![](D:\My University\2019WinterVacation\Project\Titanic\Pictures\乘客各属性分布.png)

   ​	图像将数据 可视化，从上图中可以看到如下信息：

   （1）获救人数为三百多人，远少于未获救的五百多人，占比不到一半；

   （2）三等舱乘客人数最多，远超过一、二等舱总人数；

   （3）遇难和获救人员年龄年龄跨度大，其中遇难人数遍布了(0,75)整个年龄段,而获救人数年龄段不包括六七十多岁；

   （4）不同年龄人员的客舱分布总体趋势基本一致，中间年龄段人数最多。其中头等舱大多分布在40岁左右，而二、三等舱二十多岁的人占比很大。

   （5）从登船口岸人数可以看出登船人数地区差异巨大，其中Southampton口岸登录人数最多，远多于另外两个口岸。

   下面我们进行各属性与获救结果之间的**关联统计**。

   ***各乘客等级的获救情况：**

   ​	代码如下：

   ```python
   #各乘客等级的获救情况
   from pylab import *
   import pandas as pd
   import numpy as np
   from pandas import Series,DataFrame
   mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
   data_train = pd.read_csv("D:/My University/2019WinterVacation/Project/Titanic/train.csv")
   
   fig = plt.figure()
   fig.set(alpha=0.2)  # 设定图表颜色alpha参数
   
   Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
   Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
   df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
   df.plot(kind='bar', stacked=True)
   plt.title(u"各乘客等级的获救情况")
   plt.xlabel(u"乘客等级") 
   plt.ylabel(u"人数") 
   plt.show()
   ```

   运行结果如图：

   ![](D:\My University\2019WinterVacation\Project\Titanic\Pictures\各乘客等级的获救情况.png)

   分析：

   ​	头等舱（等级为1）的乘客大多数获救，而三等舱获救人数差不多只是总人数的四分之一。显然，乘客等级对于是否获救有很大影响，将其作为一个判断是否获救的特征。

   ***各性别的乘客的获救情况：**

   ​	代码如下：

   ```
   #各性别的获救情况
   fig = plt.figure()
   fig.set(alpha=0.2)  # 设定图表颜色alpha参数
   
   Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
   Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
   df=pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
   df.plot(kind='bar', stacked=True)
   plt.title(u"按性别看获救情况")
   plt.xlabel(u"性别") 
   plt.ylabel(u"人数")
   plt.show()
   ```

   运行结果如图：

   ![](D:\My University\2019WinterVacation\Project\Titanic\Pictures\各乘客性别的获救情况.png)

   分析：

   ​	女性获救比例远大于男性，符合背景知识中所说的女士优先，性别也将作为一个判断是否获救的特征。

   ​	下面我们将上面两个属性数据综合起来看一下对于是否获救的影响。	

   ***各种舱级别情况下各性别的获救情况：**

   代码如下：

   ```python
   #各种舱级别情况下各性别的获救情况
   fig=plt.figure()
   fig.set(alpha=0.65) # 设置图像透明度，无所谓
   plt.title(u"根据舱等级和性别的获救情况")
   
   ax1=fig.add_subplot(141)
   data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
   ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
   ax1.legend([u"女性/高级舱"], loc='best')
   
   ax2=fig.add_subplot(142, sharey=ax1)
   data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
   ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
   plt.legend([u"女性/低级舱"], loc='best')
   
   ax3=fig.add_subplot(143, sharey=ax1)
   data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
   ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
   plt.legend([u"男性/高级舱"], loc='best')
   
   ax4=fig.add_subplot(144, sharey=ax1)
   data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
   ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
   plt.legend([u"男性/低级舱"], loc='best')
   
   plt.show()
   ```

   运行结果如下图：

   ![](D:\My University\2019WinterVacation\Project\Titanic\Pictures\舱等级和性别的获救情况.png)

   证明之前的判断有据可依，有理可寻。

   ***各登船口岸的获救情况：**

   代码如下：

   ```python
   #各登船口岸的获救情况
   # -*- coding: utf8 -*-
   from pylab import *
   mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
   import pandas as pd
   import numpy as np
   from pandas import Series,DataFrame
   data_train = pd.read_csv("D:/My University/2019WinterVacation/Project/Titanic/train.csv")
   
   fig = plt.figure()
   fig.set(alpha=0.2)  # 设定图表颜色alpha参数
   
   Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
   Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
   df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
   df.plot(kind='bar', stacked=True)
   plt.title(u"各登录港口乘客的获救情况")
   plt.xlabel(u"登录港口") 
   plt.ylabel(u"人数")
   
   plt.show()
   ```

   运行结果如下：

   ![](D:\My University\2019WinterVacation\Project\Titanic\Pictures\各登船港口的获救情况.png)

   ***船上堂兄弟/姐妹个数的获救情况：**

   代码如下：

   ```python
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
   ```

   运行结果如下图：

   ![#在船上的堂兄/弟、姐/妹个数的获救情况](D:\My University\2019WinterVacation\Project\Titanic\Pictures\堂兄弟妹个数的获救情况.png)

   ***Cabin分布与获救情况：**

   代码如下：

   ```python
   #Cabin分布与获救情况,cabin只有204个乘客有值，我们先看看它的一个分布
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
   
   ```

   运行结果如下：

   ![](D:\My University\2019WinterVacation\Project\Titanic\Pictures\有无Cabin记录影响.png)

   分析：

   ​	有Cabin的乘客的获救比例较大。

4. ##### 逻辑回归建模：

   代码如下：

   ```
   
   ```

5. ##### 测试数据：

6. ##### 使用算法：

7. 对数据进行了类型分析和缺失值处理；

8. 用随机森林构建决策树，使预测结果更加准确和稳定；

9. 对数据进行了特征因子化处理；

10. 对数据进行归一化处理来获得更好的模型；

11. 对个特征量和结果数据进行了关联分析；

- 对kaggle给定的train和test数据进行了类型分析和缺失值处理；
- 通过用随机森林构建决策树使得预测结果更加准确和稳定；
- 对Cabin等数据进行了特征因子化处理，将类目型特征处理为数值型的；
- 对差值较大的数据进行归一化处理来获得更好的模型；
- 对各属性数据和是否获救进行了关联分析；