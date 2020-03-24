
from time import time
from gurobipy import *
from numpy import *
from Order_batching import getBatch
from pick_time_S_shape import getBatchPicktime
from Parameter import t_convey,c_op,c_od,c_d,delivery_cost_perMin,lm,t_pack
st=time()
ind_from=0  #0--from ortools;1--from GA ；2--gurobi
""" ind ：例如[0,1,2,5,6,0,4,7,8,0,9,11,12,0]"""

if ind_from==2:
    from CVRPDV_Gurobi import route,var_a,C_d
    ind=route
    batch,batch_item=getBatch(ind)
    last_order_perBatch=[i[-2] for i in batch]
    ODtime=[var_a[i-1] for i in last_order_perBatch]
    C_d_Bat=0 #待确认

elif ind_from==0:
    from CVRPDV_ortools import vehicle_routing,batch_delivery_time,batch_route_dis
    ind=vehicle_routing
    batch,batch_item=getBatch(ind)
    ODtime=batch_delivery_time #每个批次订单最后一个订单的交付时间
    C_d=sum(batch_route_dis)*c_d
    C_d_Bat=array(batch_route_dis)*c_d
else:
    from CVRPDV_GA import bestInd,ODtime,C_d,C_d_Bat
    ind=bestInd
    batch,batch_item=getBatch(ind)

    
batch_Quantity=[len(i) for i in batch_item]
data=getBatchPicktime(batch_item)
batch_packingtime=[t_pack*i for i in batch_Quantity]
op_duetime=[lm-i for i in ODtime]

st_time=time()
# 创建模型
M_max=4
B_max=data.shape[1]
B=range(B_max) #批次订单b集合
M=range(M_max) #分区m集合
G=999
Pick_time=data[1:,:]

MODEL = Model("Order picking Schedule")

# 创建变量
y= MODEL.addVars(range(B_max),range(B_max),vtype=GRB.BINARY,name="y") #y_bk
c= MODEL.addVars(range(B_max),range(M_max),vtype=GRB.CONTINUOUS,name="c") #c_km
Z=MODEL.addVar(vtype=GRB.CONTINUOUS,name="Z")
z=MODEL.addVars(range(B_max),vtype=GRB.CONTINUOUS,name="z") #各批次的违约时间z_k

# 更新变量环境
MODEL.update()

# 创建目标函数CONTINUOUS
MODEL.setObjective(
    c_op/10*(quicksum(c[k,M_max-1] for k in B)+B_max*t_convey+sum(batch_packingtime))
    +Z, sense=GRB.MINIMIZE)


# 创建约束条件
MODEL.addConstrs((quicksum(y[b,k] for k in B) == 1 for b in B),name="s.t.4")
MODEL.addConstrs((quicksum(y[b,k] for b in B) == 1 for k in B),name="s.t.5")                 
MODEL.addConstr((c[0,0]>=quicksum(y[b,0]*Pick_time[0][b] for b in B)),name="s.t.6")                 
MODEL.addConstrs((c[0,m]>=quicksum(y[b,0]*Pick_time[m][b] for b in B)+c[0,m-1]+t_convey for m in range(1,M_max)))                 
MODEL.addConstrs((c[k,0]>=quicksum(y[b,k]*Pick_time[0][b] for b in B)+c[k-1,0] for k in range(1,B_max)))                 

MODEL.addConstrs((c[k,m]>=c[k-1,m]+quicksum(y[b,k]*Pick_time[m][b] for b in B) for k in range(1,B_max) for m in range(1,M_max)))                 
MODEL.addConstrs((c[k,m]>=c[k,m-1]+t_convey+quicksum(y[b,k]*Pick_time[m][b] for b in B) for k in range(1,B_max) for m in range(1,M_max)))                 
MODEL.addConstrs((c[k,m]>=0 for k in B for m in M),name="s.t.22") 
                
MODEL.addConstrs((z[k]>=0 for k in B),name="s.t.24")                 
MODEL.addConstrs((z[k]>=c_od/10 *quicksum(y[b,k]*(c[k,M_max-1]+t_convey+batch_packingtime[b]-op_duetime[b]) for b in B) for k in B),name="s.t.25")                 
MODEL.addConstr(Z==quicksum(z[k] for k in B)*c_od/10,name="s.t.26")   

MODEL.Params.LogToConsole=0 #log记录不在控制台显示
# 执行最优化
MODEL.optimize()
#输出模型
# MODEL.write("fsp.lp")

# 输出约束
# print(MODEL.getConstrs())

#查看变量取值
def getSeq():
    Var_name=[]
    Var_value=[]
    for var in MODEL.getVars():
        if var.X !=0:
            Var_name.append(var.varName)
            Var_value.append(round(var.X, 2))
    vartup=list(zip(Var_name,Var_value))
    var_y=[i for i in vartup if i[0].startswith("y")]
    var_y=[eval(i[0].replace("y","")) for i in var_y]
    var_y.sort(key=lambda x:x[1])
    Seq=[i[0]+1 for i in var_y]
    return Seq
Seq=getSeq()
#查看最优目标函数值
print("C_d ：", C_d)
print("C_op+C_od ：", round(MODEL.objVal,2))
print("C_d+C_op+C_od ：", round(MODEL.objVal+C_d,2)) #订单总履行成本
print("BSP_Gurobi used time:",time()-st_time)
print("H-2 time used:",time()-st)

