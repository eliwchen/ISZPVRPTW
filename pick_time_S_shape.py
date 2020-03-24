import random
import pandas as pd
from numpy import *
from Parameter import L,W,Vpick,Vtravel,t_set_up

def getBatchPicktime(batch_item,L=L,W=W,Vpick=Vpick,Vtravel=Vtravel,t_set_up=t_set_up):
    """
    imput：
    batch_item:包含品项的批次
    L=15：通道长度
    W=2：两通道之间的距离
    Vpick=15：拣选速度
    Vtravel=80：行走速度
    t_set_up：拣货准备时间

    output：
    data：m行n列，第1行批次订单编号，值:订单分区处理时间=拣货时间+准备时间（时间有放大10倍）

    """

    """生成各批次订单的分区子订单"""
    def generator_batch_item_block(batch_item):
        batch_item_block = []
        for i in batch_item:
            block = []
            temlistA = []
            temlistB = []
            temlistC = []
            temlistD = []
            for j in i:
                if j <= 300:
                    temlistA.append(j)
                elif j <= 600:
                    temlistB.append(j)
                elif j <= 900:
                    temlistC.append(j)
                else:
                    temlistD.append(j)
            block.append(temlistA)
            block.append(temlistB)
            block.append(temlistC)
            block.append(temlistD)
            batch_item_block.append(block)
        return batch_item_block

    """各批次订单的分区子订单品项所属通道编号"""
    def generator_block_aisle(batch_item_block):
        blockA = []
        blockB = []
        blockC = []
        blockD = []
        for i in range(len(batch_item_block)):
            blockA.append(batch_item_block[i][0])
            blockB.append(batch_item_block[i][1])
            blockC.append(batch_item_block[i][2])
            blockD.append(batch_item_block[i][3])
        import math
        blockA_aisle = []
        for i in blockA:
            temp = []
            for j in i:
                temp.append(math.ceil(j / 60))
            blockA_aisle.append(temp)

        blockB_aisle = []
        for i in blockB:
            temp = []
            for j in i:
                temp.append(math.ceil((j - 300) / 60))
            blockB_aisle.append(temp)

        blockC_aisle = []
        for i in blockC:
            temp = []
            for j in i:
                temp.append(math.ceil((j - 600) / 60))
            blockC_aisle.append(temp)

        blockD_aisle = []
        for i in blockD:
            temp = []
            for j in i:
                temp.append(math.ceil((j - 900) / 60))
            blockD_aisle.append(temp)
        return blockA,blockB,blockC,blockD,blockA_aisle,blockB_aisle,blockC_aisle,blockD_aisle

    """各批次订单的分区子订单通道数"""
    def generator_block_aisle_num(blockA_aisle,blockB_aisle,blockC_aisle,blockD_aisle):
        blockA_aisle_num = []
        for i in blockA_aisle:
            blockA_aisle_num.append(len(set(i)))
        blockB_aisle_num = []
        for i in blockB_aisle:
            blockB_aisle_num.append(len(set(i)))
        blockC_aisle_num = []
        for i in blockC_aisle:
            blockC_aisle_num.append(len(set(i)))
        blockD_aisle_num = []
        for i in blockD_aisle:
            blockD_aisle_num.append(len(set(i)))
        return blockA_aisle_num,blockB_aisle_num,blockC_aisle_num,blockD_aisle_num


    batch_item_block = generator_batch_item_block(batch_item)
    blockA,blockB,blockC,blockD,blockA_aisle,blockB_aisle,blockC_aisle,blockD_aisle=generator_block_aisle(batch_item_block)
    blockA_aisle_num,blockB_aisle_num,blockC_aisle_num,blockD_aisle_num=generator_block_aisle_num(blockA_aisle,blockB_aisle,blockC_aisle,blockD_aisle)


    """等奇偶判定"""
    def generator_block_isEOE(blockA_aisle_num,blockB_aisle_num,blockC_aisle_num,blockD_aisle_num):
        blockA_isEOE = []
        for i in blockA_aisle_num:
            if i == 1 or i==0:
                blockA_isEOE.append(0)
            elif i % 2 == 0:
                blockA_isEOE.append(2)
            else:
                blockA_isEOE.append(1)
        blockB_isEOE = []
        for i in blockB_aisle_num:
            if i == 1:
                blockB_isEOE.append(0)
            elif i % 2 == 0:
                blockB_isEOE.append(2)
            else:
                blockB_isEOE.append(1)
        blockC_isEOE = []
        for i in blockC_aisle_num:
            if i == 1:
                blockC_isEOE.append(0)
            elif i % 2 == 0:
                blockC_isEOE.append(2)
            else:
                blockC_isEOE.append(1)
        blockD_isEOE = []
        for i in blockA_aisle_num:
            if i == 1:
                blockD_isEOE.append(0)
            elif i % 2 == 0:
                blockD_isEOE.append(2)
            else:
                blockD_isEOE.append(1)
        return blockA_isEOE,blockB_isEOE,blockC_isEOE,blockD_isEOE

    blockA_isEOE,blockB_isEOE,blockC_isEOE,blockD_isEOE=generator_block_isEOE(blockA_aisle_num,blockB_aisle_num,blockC_aisle_num,blockD_aisle_num)

    """最左通道"""
    def generator_block_left(blockA_aisle,blockB_aisle,blockC_aisle,blockD_aisle):
        blockA_left = []
        for i in blockA_aisle:
            if i==[]:
                blockA_left.append(0)
            else:
                blockA_left.append(min(i))
        blockB_left = []
        for i in blockB_aisle:
            if i==[]:
                blockB_left.append(0)
            else:
                blockB_left.append(min(i))
        blockC_left = []
        for i in blockC_aisle:
            if i==[]:
                blockC_left.append(0)
            else:
                blockC_left.append(min(i))
        blockD_left = []
        for i in blockD_aisle:
            if i==[]:
                blockD_left.append(0)
            else:
                blockD_left.append(min(i))
        return blockA_left,blockB_left,blockC_left,blockD_left
    blockA_left,blockB_left,blockC_left,blockD_left=generator_block_left(blockA_aisle,blockB_aisle,blockC_aisle,blockD_aisle)

    """最右通道"""
    def generator_block_right(blockA_aisle,blockB_aisle,blockC_aisle,blockD_aisle):
        blockA_right = []
        for i in blockA_aisle:
            if i==[]:
                blockA_right.append(0)
            else:
                blockA_right.append(max(i))
        blockB_right = []
        for i in blockB_aisle:
            if i==[]:
                blockB_right.append(0)
            else:
                blockB_right.append(max(i))
        blockC_right = []
        for i in blockC_aisle:
            if i==[]:
                blockC_right.append(0)
            else:
                blockC_right.append(max(i))
        blockD_right = []
        for i in blockD_aisle:
            if i==[]:
                blockD_right.append(0)
            else:
                blockD_right.append(max(i))
        return blockA_right,blockB_right,blockC_right,blockD_right
    blockA_right,blockB_right,blockC_right,blockD_right=generator_block_right(blockA_aisle,blockB_aisle,blockC_aisle,blockD_aisle)

    """最右通道最后一个品项的拣货距离"""
    def generator_block_far(blockA,blockB,blockC,blockD):
        import math
        blockA_far = []
        for i in blockA:
            if i == []:
                blockA_far.append(0)
            else:
                blockA_far.append(round(math.ceil((max(i) % 60) / 2 - 1) * L / 29, 2))
        blockB_far = []
        for i in blockB:
            if i == []:
                blockB_far.append(0)
            else:
                blockB_far.append(round(math.ceil((max(i) % 60) / 2 - 1) * L / 29, 2))
        blockC_far = []
        for i in blockC:
            if i == []:
                blockC_far.append(0)
            else:
                blockC_far.append(round(math.ceil((max(i) % 60) / 2 - 1) * L / 29, 2))
        blockD_far = []
        for i in blockD:
            if i == []:
                blockD_far.append(0)
            else:
                blockD_far.append(round(math.ceil((max(i) % 60) / 2 - 1) * L / 29, 2))
        return blockA_far,blockB_far,blockC_far,blockD_far
    blockA_far,blockB_far,blockC_far,blockD_far=generator_block_far(blockA,blockB,blockC,blockD)

    """各分区子订单拣货距离计算"""
    def generator_block_dis(blockA,blockB,blockC,blockD):
        blockA_dis = []
        for i in range(len(blockA)):
            if blockA_isEOE[i] == 0:
                blockA_dis.append((blockA_left[i] - 1) * W + 2 * blockA_far[i] + (blockA_right[i] - 1) * W)
            elif blockA_isEOE[i] == 2:
                blockA_dis.append((blockA_left[i] - 1) * W + blockA_aisle_num[i] * L + (blockA_aisle_num[i] - 1) * W + (
                            blockA_right[i] - 1) * W)
            else:
                blockA_dis.append(
                    (blockA_left[i] - 1) * W + 2 * blockA_far[i] + (blockA_right[i] - 1) * W + (blockA_aisle_num[i] - 1) * (
                                L + W))

        blockB_dis = []
        for i in range(len(blockB)):
            if blockB_isEOE[i] == 0:
                blockB_dis.append((blockB_left[i] - 1) * W + 2 * blockB_far[i] + (blockB_right[i] - 1) * W)
            elif blockB_isEOE[i] == 2:
                blockB_dis.append((blockB_left[i] - 1) * W + blockB_aisle_num[i] * L + (blockB_aisle_num[i] - 1) * W + (
                            blockB_right[i] - 1) * W)
            else:
                blockB_dis.append(
                    (blockB_left[i] - 1) * W + 2 * blockB_far[i] + (blockB_right[i] - 1) * W + (blockB_aisle_num[i] - 1) * (
                                L + W))

        blockC_dis = []
        for i in range(len(blockC)):
            if blockC_isEOE[i] == 0:
                blockC_dis.append((blockC_left[i] - 1) * W + 2 * blockC_far[i] + (blockC_right[i] - 1) * W)
            elif blockC_isEOE[i] == 2:
                blockC_dis.append((blockC_left[i] - 1) * W + blockC_aisle_num[i] * L + (blockC_aisle_num[i] - 1) * W + (
                            blockC_right[i] - 1) * W)
            else:
                blockC_dis.append(
                    (blockC_left[i] - 1) * W + 2 * blockC_far[i] + (blockC_right[i] - 1) * W + (blockC_aisle_num[i] - 1) * (
                                L + W))

        blockD_dis = []
        for i in range(len(blockD)):
            if blockD_isEOE[i] == 0:
                blockD_dis.append((blockD_left[i] - 1) * W + 2 * blockD_far[i] + (blockD_right[i] - 1) * W)
            elif blockD_isEOE[i] == 2:
                blockD_dis.append((blockD_left[i] - 1) * W + blockD_aisle_num[i] * L + (blockD_aisle_num[i] - 1) * W + (
                            blockD_right[i] - 1) * W)
            else:
                blockD_dis.append(
                    (blockD_left[i] - 1) * W + 2 * blockD_far[i] + (blockD_right[i] - 1) * W + (blockD_aisle_num[i] - 1) * (
                                L + W))
        return blockA_dis,blockB_dis,blockC_dis,blockD_dis

    blockA_dis,blockB_dis,blockC_dis,blockD_dis=generator_block_dis(blockA,blockB,blockC,blockD)


    """分区子订单拣货量"""
    def generator_block_Q(blockA,blockB,blockC,blockD):
        blockA_Q = []
        for i in blockA:
            if i == []:
                blockA_Q.append(0)
            else:
                blockA_Q.append(len(i))
        blockB_Q = []
        for i in blockB:
            if i == []:
                blockB_Q.append(0)
            else:
                blockB_Q.append(len(i))

        blockC_Q = []
        for i in blockC:
            if i == []:
                blockC_Q.append(0)
            else:
                blockC_Q.append(len(i))

        blockD_Q = []
        for i in blockD:
            if i == []:
                blockD_Q.append(0)
            else:
                blockD_Q.append(len(i))
        return blockA_Q,blockB_Q,blockC_Q,blockD_Q

    blockA_Q,blockB_Q,blockC_Q,blockD_Q=generator_block_Q(blockA,blockB,blockC,blockD)

    """分区子订单拣选时间计算"""

    def generator_block_picktime(blockA,blockB,blockC,blockD):
        import math
        blockA_picktime = []
        for i in range(len(blockA)):
            a=math.floor((blockA_dis[i] / Vtravel + blockA_Q[i] / Vpick)*10)/10
            if a<0:
                blockA_picktime.append(0)
            else:
                blockA_picktime.append(a)
        blockB_picktime = []
        for i in range(len(blockB)):
            b=math.floor((blockB_dis[i] / Vtravel + blockB_Q[i] / Vpick)*10)/10
            if b<0:
                blockB_picktime.append(0)
            else:
                blockB_picktime.append(b)
        blockC_picktime = []
        for i in range(len(blockC)):
            c=math.floor((blockC_dis[i] / Vtravel + blockC_Q[i] / Vpick)*10)/10
            if c<0:
                blockC_picktime.append(0)
            else:
                blockC_picktime.append(c)
        blockD_picktime = []
        for i in range(len(blockD)):
            d=math.floor((blockD_dis[i] / Vtravel + blockD_Q[i] / Vpick)*10)/10
            if d<0:
                blockD_picktime.append(0)
            else:
                blockD_picktime.append(d)
        return blockA_picktime,blockB_picktime,blockC_picktime,blockD_picktime

    blockA_picktime,blockB_picktime,blockC_picktime,blockD_picktime=generator_block_picktime(blockA,blockB,blockC,blockD)
    #订单分区处理时间=拣货时间+准备时间
    data1=array([blockA_picktime,blockB_picktime,blockC_picktime,blockD_picktime])
    data=[]
    for i in data1:
        tem = []
        for j in i:
            if j>0:
                tem.append(int((j*10+t_set_up)))
            else:
                tem.append(0)
        data.append(tem)
    data=insert(data,0,array(list(range(1,len(blockA_picktime)+1))),axis=0)
    #data: m行n列，第1行工序编号，值:订单分区处理时间=拣货时间+准备时间
    
    
    return data

