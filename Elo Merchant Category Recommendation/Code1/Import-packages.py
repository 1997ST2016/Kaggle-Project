import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
# plt.style.use("fivethirtyeight")
plt.style.use('ggplot')#ѡ���ͼ��񣬾����ܺÿ�һ�㣡
import seaborn as sns#����matplotlib�Ļ�ͼ��
import gc#gc.collect()����ʽ�����ڴ棬����

sns.set(style="ticks", color_codes=True)#���û�ͼ�ռ�Ϊ Seaborn Ĭ�Ϸ��
import matplotlib.pyplot as plt
from tqdm._tqdm_notebook import tqdm_notebook as tqdm#��
tqdm.pandas()#��
import datetime
#����plotly���
#import plotly.offline as ply
import ply
ply.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')#����ʾwarning
