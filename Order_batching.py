import random
import pandas as pd
from Order_generate import Item
import numpy as np

def getBatch(vehicle_routing):
    """
    vehicle_routing：例如[0,3,19,8,7,0,10,6,4,14,12,9,16,0,13,11,18,20,2,0,17,1,15,5,0]
    """
    """生成批次订单"""
    def generator_batch(ind):

        '''从染色体解码回各批次，每条路径都是以0为开头与结尾'''
        indCopy = np.array(ind.copy()) # 复制ind，防止直接对染色体进行改动
        idxList = list(range(len(indCopy)))
        zeroIdx = np.asarray(idxList)[indCopy == 0]
        batch = []
        for i,j in zip(zeroIdx[0::], zeroIdx[1::]):
            batch.append(ind[i:j]+[0])
        return batch


    """订单批次品项合并"""
    def generator_batch_item(batch):
        batch_item = []
        for i in batch:
            temlist = []
            for j in range(len(i)):
                temlist += Item[i[j]]
                temlist.sort()
            batch_item.append(temlist)
        return batch_item

    batch=generator_batch(vehicle_routing)
    batch_item=generator_batch_item(batch)
    return batch,batch_item

