#基于通道相似度聚类的订单分批
import numpy as np
from Parameter import srtime,num,url,v_trans,Q,c_d,c_op,c_od,f,h1,h2,rawdata,D,T,time_windows,location,demands,t_convey,t_pack,lm
Item_OBP=[[77, 655, 756, 1193, 299],
[796, 1169],
[168, 338],
[1189],
[1170, 369],
[893],
[96, 303],
[921, 459, 592, 1108, 695],
[96],
[552, 946],
[199, 1157, 881],
[317, 1049, 39],
[208, 1110, 48, 774],
[53, 533, 140],
[1156, 449, 872],
[226, 687, 872],
[351, 26, 1185, 504, 266],
[1104, 780, 1149],
[932, 885],
[1089],
[720],
[673, 495, 990, 851],
[836],
[287, 1100, 1132],
[1176, 326, 509, 617]]

def aisle_index(Item_OBP):
    import math
    aisle_index_OBP=[]
    for i in Item_OBP:
        temp=[]
        for j in i:
            if j<=300:
                temp.append(math.ceil(j / 60))
            elif 300<j<=600:
                temp.append(math.ceil((j - 300) / 60)+5)
            elif 600<j<=900:
                temp.append(math.ceil((j - 600) / 60)+10)
            else:
                temp.append(math.ceil((j - 900) / 60)+15)
        aisle_index_OBP.append(set(temp))
    return aisle_index_OBP
#计算订单相似度
def simidegree(A,B):
     intersection= A & B
     union=A| B
     simi=round(len(intersection)/len(union),3)
     return simi
def  simidegrees(aisle_index_OBP,aisle_index_OBP_copied):  
    simiset=[]
    for i in aisle_index_OBP:
        temp=[]
        for j in aisle_index_OBP_copied:
            simi_value=simidegree(i,j)
            if simi_value==1:
                simi_value=0
            temp.append(simi_value)
        simiset.append(temp)
    return np.triu(np.array(simiset),1) #转化为上三角矩阵 

def OBP(Item_OBP,Q):
    Batch=[]
    count=0
    aisle_index_OBP=aisle_index(Item_OBP) #计算订单包含的通道编号
    aisle_index_OBP_copied=aisle_index_OBP.copy()
    simiset=simidegrees(aisle_index_OBP,aisle_index_OBP_copied) #计算订单相似度
    
    raw, column = simiset.shape# get the matrix of a raw and column
    simiset=simiset.flatten()
    simiset=list(enumerate(simiset))
    simiset=sorted(simiset,key=lambda x:x[1],reverse=True) #相似度排序
    
    m,n=divmod(simiset[count][0], column)
    new_Item=Item_OBP[m]+Item_OBP[n]
    while len(new_Item)>Q:
        count+=1
        m,n=divmod(simiset[count][0], column)
        new_Item=Item_OBP[m]+Item_OBP[n]
        
    if len(new_Item)<=Q:
        Item_OBP= [Item_OBP[i] for i in range(0, len(Item_OBP), 1) if i not in [m,n]]
        Item_OBP.append(new_Item)
    elif len(new_Item)==Q:
        Item_OBP= [Item_OBP[i] for i in range(0, len(Item_OBP), 1) if i not in [m,n]]
        Batch.append(new_Item)
        
    return Item_OBP,Batch


n=0
while n<20:
    Item_OBP,Batch=OBP(Item_OBP,Q)
    n+=1


    

    
    
    


