import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Parameter import num,url,v_trans,Q,c_d,f,h1,h2,rawdata,D,T,time_windows,location,demands

sns.set()#切换到seaborn的默认运行配置
sns.set_style("whitegrid")

def plot(ind,url=url,num=num):
  """
  :data 配送批次顺序, eg：data=[0,10,4,14,12,0,7,8,19,3,0,9,16,0,15,5,6,0,13,11,18,20,2,0,17,1,0]
  :url：客户数据文件路径, eg:url = "D://Onedrive/SEU/Paper_chen/Paper_2/data_figure_table/cus_data_origin.csv"
  :num 客户数
  """
  rawdata = pd.read_csv(url,nrows =num+1, header=None)
  # 坐标
  X = list(rawdata.iloc[:, 1])
  Y = list(rawdata.iloc[:, 2])
  Xorder = [X[i] for i in ind]
  Yorder = [Y[i] for i in ind]
  plt.plot(Xorder, Yorder, c='black', lw=1,zorder=1)
  plt.scatter(X, Y, c='black',marker='*',zorder=2)
  plt.scatter([X[0]], [Y[0]], c='black',marker='o', zorder=3)
  # plt.scatter(X[,m:], Y[,m:], marker='^', zorder=3)
  plt.xticks(range(11))
  plt.yticks(range(11))
  # plt.title(self.name)
  plt.show()

#url = "D://Onedrive/SEU/Paper_chen/Paper_2/data_figure_table/cus_data_origin.csv"

# data=[0,3,8,0,12,14,4,0,9,16,0,2,18,0,15,5,0,10,6,0,13,11,0,17,1,7,0]
#num=20
params = {
  'font.family': 'serif',
  'figure.dpi': 300,
  # 'savefig.dpi': 300,
  'font.size': 10,
  # 'text.usetex': True, #用latex渲染
  'legend.fontsize': 'small'}

data=[0,5,23,4,2,0,14,9,3,7,0,22,16,0,6,15,19,11,0,10,24,8,21,20,0,25,17,12,0,1,13,18,0]
plot(data,url,num)
