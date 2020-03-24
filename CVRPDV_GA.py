## 环境设定
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, tools, creator, algorithms
import geatpy as ea
import random
import multiprocessing
from time import time
from Routing_plot import plot
from Order_batching import getBatch
from pick_time_S_shape import getBatchPicktime
from gurobipy import *
from Parameter import srtime,G,num,url,v_trans,Q,c_d,f,h1,h2,rawdata,D,T,time_windows,location,demands

params = {
    'font.family': 'serif',
    # 'figure.dpi': 300,
    # 'savefig.dpi': 300,
    'font.size': 12,
    # 'text.usetex': True, #用latex渲染
    'legend.fontsize': 'small'
    
}
plt.rcParams.update(params)

from copy import deepcopy
#-----------------------------------
## 问题定义
creator.create('FitnessMin', base.Fitness, weights=(-1.0,)) # 最小化问题
# 给个体一个routes属性用来记录其表示的路线
creator.create('Individual', list, fitness=creator.FitnessMin) 

#-----------------------------------
## 个体编码
# 用字典存储所有参数 -- 配送中心坐标、顾客坐标、顾客需求、到达时间窗口、服务时间、车型载重量
dataDict = {}
# 节点坐标，节点0是配送中心的坐标
dataDict['NodeCoor'] = [(l[0], l[1]) for l in location]
# 将配送中心的需求设置为0
dataDict['Demand'] = demands
# 将配送中心的服务时间设置为0
dataDict['Timewindow'] = time_windows
dataDict['MaxLoad'] = Q
dataDict['ServiceTime'] = srtime
dataDict['Velocity'] = v_trans # 车辆的平均行驶速度


"""生成初始染色体"""

def getrandomind(dataDict = dataDict): 
    """获取随机排列的染色体，随机分为2~5条线路"""
    nCustomer = len(dataDict['NodeCoor']) - 1 # 顾客数量
    ind = (np.random.permutation(nCustomer) + 1).tolist()
    index_0=[]
    while len(index_0)==0:
        index_0=sorted(random.sample(list(range(2,nCustomer-1)), random.randint(2,5)))
        for i in range(1,len(index_0)):
            if abs(index_0[i]-index_0[i-1])<=2:
                index_0=[]
                break            
    for i in index_0:
        ind.insert(i,0)
    ind.insert(0,0)
    ind.append(0)
    return ind

def getcind(dataDict = dataDict): 
    '''生成满足载重约束的不确定车辆数的染色体'''
    nCustomer = len(dataDict['NodeCoor']) - 1 # 顾客数量
    perm = np.random.permutation(nCustomer) + 1 # 生成顾客的随机排列,注意顾客编号为1--n
    pointer = 0 # 迭代指针
    lowPointer = 0 # 指针指向下界
    permSlice = []
    # 当指针不指向序列末尾时
    while pointer < nCustomer -1:
        vehicleLoad = 0
        # 当不超载时，继续装载
        while (vehicleLoad < dataDict['MaxLoad']) and (pointer < nCustomer -1):
            vehicleLoad += dataDict['Demand'][perm[pointer]]
            pointer += 1
        if lowPointer+1 < pointer:
            tempPointer = np.random.randint(lowPointer+1, pointer)
            permSlice.append(perm[lowPointer:tempPointer].tolist())
            lowPointer = tempPointer
            pointer = tempPointer
        else:
            permSlice.append(perm[lowPointer::].tolist())
            break
    # 将路线片段合并为染色体
    ind = [0]
    for eachRoute in permSlice:
        ind = ind + eachRoute + [0]
    return ind

def genInd(dataDict = dataDict):
    
    choose_prob=random.random()
    
    if choose_prob>=0.75: #质量最差 占25%
        return getrandomind(dataDict = dataDict)
   
    elif 0.75>choose_prob>=0.3:
        return getcind(dataDict = dataDict) #质量中等 占45%
    
    elif 0.3>choose_prob>=0.15:
        from CVRPV_CW import ind #质量中等 占15%
        return ind

    else:
        from CVRPDV_ortools import vehicle_routing #质量最优 占15%
        return vehicle_routing
    

#-----------------------------------
## 评价函数
# 染色体解码
def decodeInd(ind):
    '''从染色体解码回路线片段，每条路径都是以0为开头与结尾'''
    indCopy = np.array(deepcopy(ind)) # 复制ind，防止直接对染色体进行改动
    idxList = list(range(len(indCopy)))
    zeroIdx = np.asarray(idxList)[indCopy == 0]
    routes = []
    for i,j in zip(zeroIdx[0::], zeroIdx[1::]):
        routes.append(ind[i:j]+[0])
    return routes

def calDist(pos1, pos2):
    '''计算距离的辅助函数，根据给出的坐标pos1和pos2，返回两点之间的距离
    输入： 
    pos1, pos2 -- (x,y)元组 ;
    输出： 曼哈顿距离'''
    return (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))*300

#
def loadPenalty(routes):
    '''辅助函数，因为在交叉和突变中可能会产生不符合负载约束的个体，需要对不合要求的个体进行惩罚'''
    penalty = 0
    # 计算每条路径的负载，取max(0, routeLoad - maxLoad)计入惩罚项
    for eachRoute in routes:
        routeLoad = np.sum([dataDict['Demand'][i] for i in eachRoute])
        penalty += max(0, routeLoad - dataDict['MaxLoad'])
    return penalty

def calcRouteServiceTime(route, dataDict = dataDict,h1=h1,h2=h2):
    '''辅助函数，根据给定路径，计算到达该路径上各顾客的时间'''
    # 初始化serviceTime数组，其长度应该比eachRoute小2
    serviceTime = [0] * (len(route) - 2)
    # 从仓库到第一个客户时不需要服务时间
    arrivalTime = calDist(dataDict['NodeCoor'][0], dataDict['NodeCoor'][route[1]])/(dataDict['Velocity']*(1-h1))
    arrivalTime = max(arrivalTime, dataDict['Timewindow'][route[1]][0])
    serviceTime[0] = arrivalTime
    arrivalTime += dataDict['ServiceTime'][route[0]] # 在出发前往下个节点前完成服务
    for i in range(1, len(route)-2):
        # 计算从路径上当前节点[i]到下一个节点[i+1]的花费的时间
        if route[i+1]==0:
            arrivalTime += calDist(dataDict['NodeCoor'][route[i]], dataDict['NodeCoor'][route[i+1]])/(dataDict['Velocity']*(1-0))
        else: 
            arrivalTime += calDist(dataDict['NodeCoor'][route[i]], dataDict['NodeCoor'][route[i+1]])/(dataDict['Velocity']*(1-h2))
        arrivalTime = max(arrivalTime, dataDict['Timewindow'][route[i+1]][0])
        serviceTime[i] = arrivalTime
        arrivalTime += dataDict['ServiceTime'][route[i]] # 在出发前往下个节点前完成服务
    return serviceTime

def timeTable(distributionPlan, dataDict = dataDict):
    '''辅助函数，依照给定配送计划，返回每个顾客受到服务的时间'''
    # 对于每辆车的配送路线，第i个客户受到服务的时间serviceTime[i]是min(TimeWindow[i][0], arrivalTime[i])
    # arrivalTime[i] = serviceTime[i-1] + 服务时间 + distance(i,j)/averageVelocity
    timeArrangement = [] #容器，用于存储每个顾客受到服务的时间
    for eachRoute in distributionPlan:
        serviceTime = calcRouteServiceTime(eachRoute)
        timeArrangement.append(serviceTime)
    # 将数组重新组织为与基因编码一致的排列方式
    realignedTimeArrangement = [0]
    for routeTime in timeArrangement:
        realignedTimeArrangement = realignedTimeArrangement + routeTime + [0]
    return realignedTimeArrangement

def timePenalty(ind, routes):
    '''辅助函数，对不能按服务时间到达顾客的情况进行惩罚'''
    timeArrangement = timeTable(routes) # 对给定路线，计算到达每个客户的时间
    # 索引给定的最迟到达时间
    desiredTime = [dataDict['Timewindow'][ind[i]][1] for i in range(len(ind))]
    # 如果最迟到达时间大于实际到达客户的时间，则延迟为0，否则延迟设为实际到达时间与最迟到达时间之差
    timeDelay = [max(timeArrangement[i]-desiredTime[i],0) for i in range(len(ind))]
    return np.sum(timeDelay)

def calRouteLen(routes,dataDict=dataDict):
    '''辅助函数，返回给定路径的总长度'''
    totalDistance = 0 # 记录各条路线的总长度
    for eachRoute in routes:
        # 从每条路径中抽取相邻两个节点，计算节点距离并进行累加
        for i,j in zip(eachRoute[0::], eachRoute[1::]):
            totalDistance += calDist(dataDict['NodeCoor'][i], dataDict['NodeCoor'][j])    
    return totalDistance

def evaluate(ind, f=f,c_d=c_d,c2=8, c3=5,c4=1):
    '''评价函数，返回解码后路径的总长度，
    f,c_d,c2,c3,c4分别为为单位派车成本、单位行驶成本、超载惩罚、时间窗惩罚、单线惩罚因子'''
    routes = decodeInd(ind) # 将个体解码为路线
    totalDistance = calRouteLen(routes)
    cost_num_node=0
    for i in routes:
        if len(i)<=3:
            cost_num_node=+10 #惩罚一条线路只有1个订单的情况
    return (f*len(routes)+c_d*totalDistance + c2*loadPenalty(routes) + c3*timePenalty(ind,routes)+c4*cost_num_node),
#-----------------------------------
## 交叉操作
def genChild(ind1, ind2, nTrail=5): #前置交叉
    '''参考《基于电动汽车的带时间窗的路径优化问题研究》中给出的交叉操作，生成一个子代'''
    # 在ind1中随机选择一段子路径subroute1，将其前置
    routes1 = decodeInd(ind1) # 将ind1解码成路径
    numSubroute1 = len(routes1) # 子路径数量
    subroute1 = routes1[np.random.randint(0, numSubroute1)]
    # 将subroute1中没有出现的顾客按照其在ind2中的顺序排列成一个序列
    unvisited = set(ind1) - set(subroute1) # 在subroute1中没有出现访问的顾客
    unvisitedPerm = [digit for digit in ind2 if digit in unvisited] # 按照在ind2中的顺序排列
    # 多次重复随机打断，选取适应度最好的个体
    bestRoute = None # 容器
    bestFit = np.inf
    for _ in range(nTrail):
        # 将该序列随机打断为numSubroute1-1条子路径
        breakPos = [0]+random.sample(range(1,len(unvisitedPerm)),numSubroute1-2) # 产生numSubroute1-2个断点
        breakPos.sort()
        breakSubroute = []
        for i,j in zip(breakPos[0::], breakPos[1::]):
            breakSubroute.append([0]+unvisitedPerm[i:j]+[0])
        breakSubroute.append([0]+unvisitedPerm[j:]+[0])
        # 更新适应度最佳的打断方式
        # 将先前取出的subroute1添加入打断结果，得到完整的配送方案
        breakSubroute.append(subroute1)
        # 评价生成的子路径
        routesFit = calRouteLen(breakSubroute) + loadPenalty(breakSubroute)
        if routesFit < bestFit:
            bestRoute = breakSubroute
            bestFit = routesFit
    # 将得到的适应度最佳路径bestRoute合并为一个染色体
    child = []
    for eachRoute in bestRoute:
        child += eachRoute[:-1]
    return child+[0]

def cro_xovpmx(ind1,ind2): #部分匹配交叉
    index_0 = [i for i, x in enumerate(ind1) if x == 0]
    ind1_r0=[i for i in ind1 if i !=0]
    ind2_r0=[i for i in ind2 if i !=0]
    cho=np.array([ind1_r0,ind2_r0])
    ind_new = ea.xovpmx(cho, XOVR=1, Half=True).tolist()[0]
    for i in index_0:
        ind_new.insert(i,0)
    return ind_new

def cro_xovox(ind1,ind2): #顺序交叉
    index_0 = [i for i, x in enumerate(ind1) if x == 0]
    ind1_r0=[i for i in ind1 if i !=0]
    ind2_r0=[i for i in ind2 if i !=0]
    cho=np.array([ind1_r0,ind2_r0])
    ind_new = ea.xovpmx(cho, XOVR=1, Half=True).tolist()[0]
    for i in index_0:
        ind_new.insert(i,0)
    return ind_new

def crossover(ind1, ind2):
    '''交叉操作'''
    choose_prob=random.random()
    if choose_prob>=0.9:
        ind1[:], ind2[:] = genChild(ind1, ind2), genChild(ind2, ind1)
    
    elif 0.7>choose_prob>=0.3:
        ind1[:], ind2[:] = cro_xovox(ind1, ind2), cro_xovox(ind2, ind1)
    else:
        ind1[:], ind2[:] = cro_xovpmx(ind1, ind2), cro_xovpmx(ind2, ind1)
    return ind1, ind2

#-----------------------------------
## 突变操作
def opt(route,dataDict=dataDict, k=3, c1=1.0, c2=500.0):
    # 用2-opt算法优化路径
    # 输入：
    # route -- sequence，记录路径
    # k -- k-opt，这里用2opt
    # c1, c2 -- 寻求最短路径长度和满足时间窗口的相对重要程度
    # 输出： 优化后的路径optimizedRoute及其路径长度
    nCities = len(route) # 城市数
    optimizedRoute = route # 最优路径
    desiredTime = [dataDict['Timewindow'][route[i]][1] for i in range(len(route))]
    serviceTime = calcRouteServiceTime(route)
    timewindowCost = [max(serviceTime[i]-desiredTime[1:-1][i],0) for i in range(len(serviceTime))]
    timewindowCost = np.sum(timewindowCost)/len(timewindowCost)
    minCost = c1*calRouteLen([route]) +  c2*timewindowCost # 最优路径代价
    for i in range(1,nCities-2):
        for j in range(i+k, nCities):
            if j-i == 1:
                continue
            reversedRoute = route[:i]+route[i:j][::-1]+route[j:]# 翻转后的路径
            # 代价函数中需要同时兼顾到达时间和路径长度
            desiredTime = [dataDict['Timewindow'][reversedRoute[i]][1] for i in range(len(reversedRoute))]
            serviceTime = calcRouteServiceTime(reversedRoute)
            timewindowCost = [max(serviceTime[i]-desiredTime[1:-1][i],0) for i in range(len(serviceTime))]
            timewindowCost = np.sum(timewindowCost)/len(timewindowCost)
            reversedRouteCost = c1*calRouteLen([reversedRoute]) + c2*timewindowCost
            # 如果翻转后路径更优，则更新最优解
            if  reversedRouteCost < minCost:
                minCost = reversedRouteCost
                optimizedRoute = reversedRoute
    return optimizedRoute

def mu_mutswap(route): #两点变异算子
    ind_r0=[i for i in route if i !=0]
    Dim=len(ind_r0)
    lb = [1] * Dim # 决策变量下界
    ub = [max(ind_r0)] * Dim # 决策变量上界
    varTypes = [1] * Dim
    FieldDR=np.array([lb,ub,varTypes])
    cho=np.array([ind_r0])
    route_new = ea.mutswap(Encoding='P', OldChrom=cho, FieldDR=FieldDR, Pm=1).tolist()[0]
    route_new.append(0)
    route_new.insert(0,0)
    return route_new


def mutate(ind):
    '''用2-opt算法对各条子路径进行局部优化'''
    routes = decodeInd(ind)
    optimizedAssembly = []
    for eachRoute in routes:
        choose_prob=random.random()
        if choose_prob>=0.9:
            optimizedRoute = opt(eachRoute)
        else:
            optimizedRoute = mu_mutswap(eachRoute)
        optimizedAssembly.append(optimizedRoute)
    # 将路径重新组装为染色体
    child = []
    for eachRoute in optimizedAssembly:
        child += eachRoute[:-1]
    ind[:] = child+[0]
    return ind,

#-----------------------------------
## 注册遗传算法操作

toolbox = base.Toolbox()
toolbox.register('individual', tools.initIterate, creator.Individual, genInd)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('evaluate', evaluate)
toolbox.register('select', tools.selTournament, tournsize=4) #锦标赛
toolbox.register('mate', crossover)
toolbox.register('mutate', mutate)

## 生成初始族群
toolbox.popSize = 100
pop = toolbox.population(toolbox.popSize)

## 记录迭代数据
# pool = multiprocessing.Pool(processes=4)
# toolbox.register("map", pool.map)
stats=tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('min', np.min)
stats.register('avg', np.mean)
stats.register('std', np.std)
hallOfFame = tools.HallOfFame(maxsize=1)

## 遗传算法参数
toolbox.ngen = 100
toolbox.cxpb = 0.85
toolbox.mutpb = 0.15

## 遗传算法主程序
start_tiem=time()
pop,logbook=algorithms.eaMuPlusLambda(pop, toolbox, mu=toolbox.popSize, 
                                      lambda_=toolbox.popSize,cxpb=toolbox.cxpb, mutpb=toolbox.mutpb,
                   ngen=toolbox.ngen ,stats=stats, halloffame=hallOfFame, verbose=True)

# pool.close()
from pprint import pprint

def calLoad(routes):
    loads = []
    for eachRoute in routes:
        routeLoad = np.sum([dataDict['Demand'][i] for i in eachRoute])
        loads.append(routeLoad)
    return loads

bestInd = hallOfFame.items[0]
distributionPlan = decodeInd(bestInd)
C_d_Bat=[]
for eachRoute in distributionPlan:
    # 从每条路径中抽取相邻两个节点，计算节点距离并进行累加
    totalDistance = 0 # 记录各条路线的总长度
    for i,j in zip(eachRoute[0::], eachRoute[1::]):
        totalDistance += calDist(dataDict['NodeCoor'][i], dataDict['NodeCoor'][j])
    C_d_Bat.append(totalDistance*c_d+f)
print("The delivery cost of each vehicle：",C_d_Bat)
bestFit = bestInd.fitness.values
print('Best distribution plan：')
pprint(distributionPlan)
print('The load of each vehicle：')
print(calLoad(distributionPlan))
print('Total distance：')
print(evaluate(bestInd,f=0,c_d=1,c2=0, c3=0,c4=0)[0]) #
print('Total cost：')
C_d=evaluate(bestInd)[0]
print(C_d)  #f,c_d,c2,c3,c4分别为为单位派车成本、单位行驶成本、超载惩罚、时间窗惩罚、单线惩罚因子


timeArrangement = timeTable(distributionPlan) # 对给定路线，计算到达每个客户的时间
ODtime=[i[-2]*10 for i in decodeInd(timeArrangement)] #计算每条线路最后一个订单交付时间
# 索引给定的最迟到达时间
desiredTime = [dataDict['Timewindow'][bestInd[i]][1] for i in range(len(bestInd))]
# 如果最迟到达时间大于实际到达客户的时间，则延迟为0，否则延迟设为实际到达时间与最迟到达时间之差
timeDelay = [max(timeArrangement[i]-desiredTime[i],0) for i in range(len(bestInd))]
print('Delay time：')
print(timeDelay)
print("CVRPDV_GA time used:",time()-start_tiem)
# 画出迭代图
Draw=0# 0-不绘制；1-绘制
if Draw==1:
    minFit = logbook.select('min')
    avgFit = logbook.select('avg')
    plt.figure()
    # plt.plot(minFit, 'b-', label='Minimum Function Value')
    plt.plot(avgFit, 'r-', label='Average Function Value')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.show()

#画出路径优化图
Draw_route=0
if Draw_route==1:
    plot(bestInd,url,num)

