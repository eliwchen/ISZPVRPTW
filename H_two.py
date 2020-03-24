# -*- coding: utf-8 -*-
from time import time
from numpy import *
import matplotlib.pyplot as plt
import GA
from tool import makespan
from tool import makespan_left
from Parameter import num,url,v_trans,Q,c_d,c_od,c_op,f,h1,h2,rawdata,D,T,time_windows,location,demands,t_convey
from tool import gatt

st=time()

Bsp_Method_choose=3 #0--GA;1--Gurobi;2--SPT;3--LDT;4--SST

if Bsp_Method_choose==1:
    from BSP_Gurobi import data,Seq,batch_packingtime,op_duetime,ODtime,C_d_Bat
    Seq=array(Seq)
    print("Best Seq：",Seq)

elif Bsp_Method_choose==0:
    from BSP_GA import Seq,data,batch_packingtime,op_duetime,ODtime,C_d_Bat
    print("Best Seq:",Seq)
    
elif Bsp_Method_choose==2:
    from BSP_SPT import Seq,data,batch_packingtime,op_duetime,ODtime,C_d_Bat
    print("Best Seq:",Seq)
    
elif Bsp_Method_choose==3:
    from BSP_LDT import Seq,data,batch_packingtime,op_duetime,ODtime,C_d_Bat,C_d
    print("Best Seq:",Seq)
    
elif Bsp_Method_choose==4:
    from BSP_SSL import Seq,data,batch_packingtime,op_duetime,ODtime,C_d_Bat
    print("Best Seq:",Seq)
    

data_best_all=data[:,Seq-1]
order=array(data_best_all[0,:])-1
"""绘制甘特图"""
Draw=0
if Draw==1:
    plt.figure()
    gatt(data_best_all,t_convey)
    plt.show()

"""计算各指标"""
"""时间类"""
#拣货开始时间
op_start_time=makespan_left(data_best_all)
#拣货完成时间
makespan_time=makespan(data_best_all)
#各分区订单抵达后台时间
batch_arrive_time=array([i+t_convey for i in (makespan(data_best_all)[-1, :])])
#批次订单开始配送时间=抵达后台时间+打包分类时间
batch_delivery_start_time = batch_arrive_time + array([batch_packingtime[i] for i in order])

#批次订单违约时间
batch_over_duetime = []
for j in list(array(batch_delivery_start_time) - array([op_duetime[i] for i in order])):
    if j < 0:
        batch_over_duetime.append(0)
    else:
        batch_over_duetime.append(j)
#批次订单完成时间
batch_of_time=array(batch_delivery_start_time)+array([ODtime[i] for i in order]) 

"""成本类：将时间/10"""
#各批次订单拣选成本
C_op_Bat=array(batch_delivery_start_time)/10*c_op
#各批次订单违约成本
C_od_Bat=array(batch_over_duetime)/10*c_od

if Bsp_Method_choose!=1:
    print("TC:",sum(C_od_Bat)+sum(C_op_Bat)+C_d)
    
print("H-2 time used:",time()-st)
    



