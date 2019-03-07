## NameError: name   “      ”  is not defined

问题分析：



#### 问题一：name ‘*name*’ is not defined

​	"__name__"两端是双下划线"__"，不是只有一个"_"。

#### 问题二：name 'messagebox' is not defined

​	“     ”  内为某个数据库的子module。

​	在代码中加上如下语句：

```python
from tkinter import messagebox
```

默认情况下子module不会自动import。



#### 问题三：name 'reload' is not defined.

##### 	对于 Python 2.X

```python
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
```

##### 	对于 Python 3.3

```python
import imp
imp.reload(sys)
```

**注意：** 

1. **Python 3** 与 **Python 2** 有**很大的区别**，其中**Python 3** 系统默认使用的就是`utf-8`编码。 

2. 所以，对于使用的是**Python 3** 的情况，就不需要`sys.setdefaultencoding("utf-8")`这段代码。 

3. **最重要的是**，**Python 3** 的 **sys** 库里面已经**没有** `setdefaultencoding()` 函数了。

   ##### 对于  **Python 3.4**：

```
import importlib
importlib.reload(sys)
```

#### 问题四：name 'file' is not defined

```python
  f = file('poem.txt', 'w') # open for 'w'riting
NameError: name 'file' is not defined
```

解决办法：file()改为open()

#### 问题五：name 'array' is not defined

区分array和list，弄清楚道题想要啥类型；加载array模块。

```python
from array import array
```

#### 问题六： name 'xx' is not defined

IndentationError:expected an indented block

问题在于tab和空格混用导致出现了问题。

#### 问题七：name 'math' is not defined

将

```py
from math import *
```

改为

```py
import math
```

#### 问题八：python2中input出现的name“      ”  is notdefined.

Python 2.X中对于input函数来说，它所希望读取到的是一个**合法的Python表达式，即你在输入字符串的时候必须要用""将其扩起来**；而**在Python 3中，input默认接受的是str类型。**

解决办法：1、在控制台进行输入参数时，将其变为一个合法的Python表达式，用"  "将其扩起来

​    2、使用raw_input，因为raw_input将所有的输入看作字符串，并且返回一个字符串类型。



