from NEH import neh
import GA
from time import time
from numpy import *
from Order_batching import getBatch
from pick_time_S_shape import getBatchPicktime
from Parameter import t_convey,c_op,c_od,c_d,delivery_cost_perMin,lm,t_pack

H_two_stime=time()

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


"""bsp"""
BSP_GA_start_time = time()
Seq_neh,Value_neh=neh(data,t_convey,draw=0)
data_best_all = data[:, Seq_neh - 1]
Seq=GA.ga_fsp_new(data_best_all,c_op=c_op,
                     op_duetime=op_duetime,
                     c_od=c_od,
                     batch_packingtime=batch_packingtime,
                     t_convey=t_convey,
                     draw=222) #draw分别甘特图、适应度图、动态适应度图，1表示绘制，2表示不绘制




