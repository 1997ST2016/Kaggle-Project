import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
# plt.style.use("fivethirtyeight")
plt.style.use('ggplot')#选择绘图风格，就是能好看一点！
import seaborn as sns#类似matplotlib的画图包
import gc#gc.collect()，显式回收内存，见②

sns.set(style="ticks", color_codes=True)#设置画图空间为 Seaborn 默认风格
import matplotlib.pyplot as plt
from tqdm._tqdm_notebook import tqdm_notebook as tqdm#③
tqdm.pandas()#③
import datetime
#关于plotly库④
#import plotly.offline as ply
import ply
ply.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')#不显示warning
