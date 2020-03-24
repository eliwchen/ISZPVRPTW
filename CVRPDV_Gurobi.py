from time import time
from gurobipy import *
import numpy as np
import pandas as pd
from collections import Counter
import operator
from Parameter import num,url,v_trans,Q,c_d,f,h1,h2,rawdata,D,T,time_windows,location,demands,G,lm

st=time()
st_time=time()
# 创建模型
q=demands
s=[0]+[1]*num #服务时间
N_max=num #订单量
N=range(1,N_max+1) #订单集合 [1,...,N_max]
N0=range(N_max+1)  #包含前置仓的配送点集合 [0,...,N_max]
V_max=50 #配送车数量
V=range(1,V_max+1) #配送车集合[1,...,V_max]
MODEL = Model("VRPTWVS")

# 创建变量
a=MODEL.addVars(range(N_max+1), lb=0,vtype=GRB.CONTINUOUS,name="a") #a_i
# Q_leave=MODEL.addVars(range(N_max+1),vtype=GRB.CONTINUOUS,lb=0,ub=Q,name="Q_leave") #Q_leave_i
# v_leave=MODEL.addVars(range(N_max+1),vtype=GRB.CONTINUOUS,lb=v_trans*(1-h1),ub=v_trans, name="v_leave") #v_leave_i
t_depart=MODEL.addVars(range(V_max+1),lb=0,ub=0,vtype=GRB.CONTINUOUS,name="t_depart") #t_depart_v
w= MODEL.addVars(range(N_max+1),range(N_max+1),range(1,V_max+1),vtype=GRB.BINARY,name="w") #w_igv 若车辆v从i到g则为1
z= MODEL.addVars(range(1,N_max+1),range(1,V_max+1),vtype=GRB.BINARY,name="z") #z_iv 若订单i交由车辆v配送则为1
v_num=MODEL.addVar(lb=0,vtype=GRB.CONTINUOUS,name="v_num") #启用的车辆数

# 更新变量环境
MODEL.update()

# 创建目标函数CONTINUOUS
MODEL.setObjective(
    c_d*(quicksum(D[i,g]*w[i,g,v] for i in N0 for g in N0 for v in V))+f*quicksum(w[0,i,v] for i in N for v in V), 
    sense=GRB.MINIMIZE)


# 创建约束条件
MODEL.addConstrs((quicksum(w[i,g,v] for g in N0 if g!=i) <=z[i,v]  for i in N for v in V),name="s.t.3")
MODEL.addConstrs((quicksum(w[i,g,v] for i in N0 if i!=g for v in V) == 1 for g in N),name="s.t.4")
MODEL.addConstrs((quicksum(w[i,g,v] for g in N0 if g!=i for v in V) == 1 for i in N),name="s.t.5")
MODEL.addConstrs((quicksum(w[i,g,v] for i in N0 if i!=g)-quicksum(w[g,i,v] for i in N0 if i!=g) == 0 for g in N for v in V),name="s.t.6")
MODEL.addConstrs((quicksum(w[0,i,v] for i in N )<=1 for v in V),name="s.t.7") #车辆闭环约束
MODEL.addConstrs((quicksum(w[i,0,v] for i in N )<=1 for v in V),name="s.t.8") #车辆闭环约束
MODEL.addConstrs((quicksum(z[i,v]*q[i] for i in N) <= Q for v in V),name="s.t.9")
MODEL.addConstr(a[0]==0,name="s.t.10")
# MODEL.addConstrs((t_depart[v]==0 for v in V),name="s.t.8")
MODEL.addConstrs((a[i]+s[i]+T[i,g]-a[g]<=G*(1-quicksum(w[i,g,v] for v in V)) for i in N0 for g in N if i!=g),name="s.t.11")
# MODEL.addConstrs((quicksum(z[i,v]*t_depart[v] for v in V) <=a[i] for i in N),name="s.t.10")
MODEL.addConstrs((a[i]<=lm for i in N),name="s.t.12")
MODEL.addConstr((v_num==quicksum(w[0,i,v] for i in N for v in V)),name="s.t.13") #计算使用的车辆数

#运行参数调整
MODEL.Params.TimeLimit = 600 #设置终止的最长求解时间
# MODEL.Params.SolutionLimit=200 #设置终止最大可行解数量
# MODEL.Params.MIPGap=0.1 #设置终止的gap即上下界误差
MODEL.Params.ImproveStartTime=150 #设置启动提升策略时间
MODEL.Params.FeasibilityTol=0.01 #设置求解精度
# MODEL.Params.Heuristics=0.6 #设置启发式算法比重
MODEL.Params.LogToConsole=1 #log记录是否不在控制台显示
# MODEL.Params.Method=4 #设定根节点求解方法

# 执行最优化
MODEL.optimize()
#查看模型不可解的原因
# MODEL.computeIIS()
#输出模型
# MODEL.write("VRPTWVS.lp")
#查看最优目标函数值 
C_d=round(MODEL.objVal,2)
print("Current C_d Value:", round(MODEL.objVal,2))
print("Current Optimal distance:", round(MODEL.objVal,2)/c_d)

#查看模型解状态
# print("Model solution status:", MODEL.status) #解释：https://www.gurobi.com/documentation/9.0/refman/optimization_status_codes.html?tdsourcetag=s_pctim_aiomsg


# 查看变量取值
# for var in MODEL.getVars():
#     if var.X !=0:
#         print(f"{var.varName}: {round(var.X, 2)}")

def getvar_w():
    Var_name=[]
    Var_value=[]
    for var in MODEL.getVars():
        if var.X !=0:
            Var_name.append(var.varName)
            Var_value.append(round(var.X, 2))
    vartup=list(zip(Var_name,Var_value))
    var_w=[i for i in vartup if i[0].startswith("w")]
    var_w=[eval(i[0].replace("w","")) for i in var_w]
    var_w.sort(key=lambda x:x[2])
    return var_w

def getvar_a():
    Var_name=[]
    Var_value=[]
    for var in MODEL.getVars():
        if var.X !=0:
            Var_name.append(var.varName)
            Var_value.append(round(var.X, 2))
    vartup=list(zip(Var_name,Var_value))
    var_a=[i[1] for i in vartup if i[0].startswith("a")] #获取1~num客户的订单交付时间
    # var_a=[eval(i[0].replace("a","")) for i in var_a]
    return var_a

def chainsort(ls): 
    """将复合列表按照链表进行排序"""
    ls_len=len(ls)
    ls_chain=[]
    ls_chain.append(ls[0])
    value_look=ls[0][1]
    while len(ls_chain) != ls_len:
        for i in ls:
            if i[0]==value_look:
                ls_chain.append(i)
                value_look=i[1]
                ls.remove(i)
    ls_chain=[i[1] for i in ls_chain]
    ls_chain.insert(0,0)
    return ls_chain
def getroute(var_w):
    """ 通过决策变量获得路径"""
    vehicle_used=[i[2] for i in var_w] 
    # vehicle_used=list(set(vehicle_used))
    vehi_count=dict(Counter(vehicle_used)) #统计每个车包含的订单数
    vehi_count=sorted(vehi_count.items(),key=operator.itemgetter(0))#按照item中的第一个字符进行排序，即按照key排序
    cut_num=[i[1] for i in vehi_count]
    cut_num_sum=[]
    count_sum=0
    for i in cut_num:
        count_sum+=i
        cut_num_sum.append(count_sum)
    cut_num_sum_left=cut_num_sum.copy()
    del cut_num_sum_left[-1] #删除列表尾部最后一个元素
    cut_num_sum_left.insert(0,0)  #头部追加0
    cut_sum_index=list(zip(cut_num_sum_left,cut_num_sum))   #切片的索引组合值 
    # print(cut_sum_index)
    var_w_cut=[]
    for i in cut_sum_index:
        var_w_cut.append(var_w[i[0]:i[1]])
    routes=[chainsort(i) for i in var_w_cut ]
    route_tem=[i[:-1] for i in routes]
    route=[]
    for i in route_tem:
        route+=i
    route.append(0)
    return route

#查看程序运行时间
print("CVRPDV_Gurobi used time:",time()-st_time)

#绘制路径图
from Routing_plot import plot
var_w=getvar_w()
route=getroute(var_w)
var_a=getvar_a()
# print(var_w)
# print("route:",route)
# print("route_num:",dict(Counter(route))[0]-2)
plot(route,url,num)
# print(getvar_a())
    

