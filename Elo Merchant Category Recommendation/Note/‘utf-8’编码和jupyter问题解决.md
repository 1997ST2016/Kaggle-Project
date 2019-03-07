#### 问题一：解决UnicodeDecodeError: 'utf-8' codec can't decode byte..问题

- 问题描述：

- 问题分析：

  ​	该情况是由于出现了无法进行转换的 二进制数据 造成的，可以写一个小的脚本来判断下，是整体的字符集参数选择上出现了问题，还是出现了部分的无法转换的二进制块：

- 问题解决

  法一：**将gbk编码数据decode('gbk')得到utf8编码的数据**

  法二：修改字符集参数，一般这种情况出现得较多是在国标码(GBK)和utf8之间选择出现了问题。
  ​	出现异常报错是由于设置了decode()方法的第二个参数errors为严格（strict）形式造成的，因为默认就是这个参数，将其更改为ignore等即可。例如:

  ```python
  line.decode("utf8","ignore")
  ```

  法三：在网上搜了很多人的处理方法是在程序的顶部加

  ```python
  import sys
  reload( sys )
  sys.setdefaultencoding('gbk')
  ```


  可是我自己试一了，程序错是不报了，但是实际没有运行

  最后，我在程度的顶部加了下面两行就好了，中文也可以正常显示了

  ```python
  # !/usr/bin/env Python
  # coding=utf-8
  ```

  ​	还有一种情况就是如果你用pyhton IO读取一个文件，那么要求将文件的编码方式转换成UTF-8。

#### 问题二：（jupyter安装出现问题：安装后无法打开）

traitlets.traitlets.TraitError: Could not decode 'C:\\Users\\\xce\xa2\xcc\xf0\xd0\xc4\xd3\xef\\.jupyter' for unicode trait 'config_dir' of a NotebookApp instance.

解决办法：将下面一段代码保存为py文件，文件名为“python jupyter_nootbook_start.py”，以后运行该文件即可打开jupyter。

```python
import os
import subprocess

#base = 'D:\\Anaconda'
base = 'D:\ProgrammingTool\python\Ana\Anaconda'
jupyter_dir = os.path.join(base,'.jupyter')
if not os.path.exists(jupyter_dir):
    os.mkdir(jupyter_dir)

dirs = {'JUPYTER_CONFIG_DIR' : jupyter_dir, 'JUPYTER_RUNTIME_DIR' : os.path.join(jupyter_dir,'runtime'),'JUPYTER_DATA_DIR' : os.path.join(jupyter_dir,'data')}

for k,v in dirs.iteritems():   
    if not os.path.exists(v):
        os.mkdir(v)
    os.environ[k] = v

ipython_dir = os.path.join(base,'.ipython')

os.environ['IPYTHONDIR'] = ipython_dir

subprocess.call(['D:\ProgrammingTool\python\Ana\Anaconda\Scripts\jupyter-notebook.exe'])

```

