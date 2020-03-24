# -*- coding: utf-8 -*-

from time import time
from gurobipy import *
import numpy as np
import pandas as pd
from collections import Counter
import operator
from Parameter import num,url,v_trans,Q,c_d,c_od_d,f,h1,h2,rawdata,D,T,time_windows,location,demands,G,lm

st=time()
st_time=time()
params = {
    'font.family': 'serif',
    'figure.dpi': 300,
    # 'savefig.dpi': 300,
    'font.size': 10,
    # 'text.usetex': True, #用latex渲染
    'legend.fontsize': 'small'
# 创建模型
num=5 #订单数
D=D[[0,8,10,20,21,24],:]
D=D[:,[0,8,10,20,21,24]]
T=T[[0,8,10,20,21,24],:]
T=T[:,[0,8,10,20,21,24]]
q=[demands[i] for i in [8,10,20,21,24]]
s=[0]+[1]*num #服务时间
l=20.9
N_max=num #订单量
N=range(1,N_max+1) #订单集合 [1,...,N_max]
N0=range(N_max+1)  #包含前置仓的配送点集合 [0,...,N_max]

MODEL = Model("TSPTW")

# 创建变量
a=MODEL.addVars(range(N_max+1), lb=0,vtype=GRB.CONTINUOUS,name="a") #a_i
w= MODEL.addVars(range(N_max+1),range(N_max+1),vtype=GRB.BINARY,name="w") #w_ig 若车辆从i到g则为1
z=MODEL.addVars(range(1,N_max+1), lb=0,vtype=GRB.CONTINUOUS,name="z") #z_i
u=MODEL.addVars(range(1,N_max+1), vtype=GRB.CONTINUOUS,name="u") #u_i

# 更新变量环境
MODEL.update()

# 创建目标函数CONTINUOUS
# MODEL.setObjective(
#     c_d*(quicksum(D[i,g]*w[i,g] for i in N0 for g in N0))
#     +f+c_od_d*(quicksum(z[i] for i in N)), 
#     sense=GRB.MINIMIZE)
MODEL.setObjective(
    c_d*(quicksum(D[i,g]*w[i,g] for i in N0 for g in N0))
    +f, 
    sense=GRB.MINIMIZE)

# 创建约束条件

MODEL.addConstrs((quicksum(w[i,g] for i in N0 )==1 for g in N0 ),name="out") 
MODEL.addConstrs((quicksum(w[i,g] for g in N0 )==1 for i in N0 ),name="in") 
MODEL.addConstrs((u[i]-u[g]+(num+1-1)*w[i,g]<=num+1-2 for i in N for g in N ),name="route") 


MODEL.addConstr(a[0]==0,name="init")
MODEL.addConstrs((a[i]+s[i]+T[i,g]-a[g]<=G*(1-w[i,g]) for i in N0 for g in N if i!=g),name="arrival time")

# MODEL.addConstrs((z[i]==max_((a[i]-25),0)  for i in N),name="tw linearization")
MODEL.addConstrs((a[i]<=l for i in N),name="s.t.12")

#运行参数调整
MODEL.Params.TimeLimit = 600 #设置终止的最长求解时间
# MODEL.Params.SolutionLimit=200 #设置终止最大可行解数量
# MODEL.Params.MIPGap=0.1 #设置终止的gap即上下界误差
MODEL.Params.ImproveStartTime=150 #设置启动提升策略时间
# MODEL.Params.FeasibilityTol=0.01 #设置求解精度
# MODEL.Params.Heuristics=0.6 #设置启发式算法比重
MODEL.Params.LogToConsole=1 #log记录是否不在控制台显示
# MODEL.Params.Method=4 #设定根节点求解方法

# 执行最优化
MODEL.optimize()
#查看模型不可解的原因
# MODEL.computeIIS()
#输出模型
# MODEL.write("TSPTW.lp")
#查看最优目标函数值 
C_d=round(MODEL.objVal,2)
print("Current C_d Value:", round(MODEL.objVal,2))

# 查看变量取值
for var in MODEL.getVars():
    if var.X !=0:
        print(f"{var.varName}: {round(var.X, 2)}")



