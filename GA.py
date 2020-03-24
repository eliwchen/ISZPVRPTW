# -*- coding:utf-8 -*-
import numpy as np
import time
from tool import makespan_value
from tool import makespan
from ga_crossover import baker, xpm


class GA_FSP_NEW:
    def __init__(self, data, pop_size=80, max_gen=300, Pc=0.65, Pm=0.2):
        self.data = data
        self.pop_size = pop_size
        self.NDA = data.shape[1]
        self.max_gen = max_gen
        self.Pc = Pc
        self.Pm = Pm
        self.n = 1

    def crtp(self):
        pop = np.zeros([self.pop_size, self.NDA], dtype=int)
        pop[0] = np.random.permutation(self.NDA) #返回顺序打乱的数组/list
        for i in range(1, self.pop_size):
            data_pop = np.random.permutation(self.NDA)
            for k in range(i):
                if (data_pop == pop[k]).all():
                    data_pop = np.random.permutation(self.NDA)
            pop[i] = data_pop
        return pop

    def fitness(self, pop,c_op,op_duetime,c_od,batch_packingtime,t_convey):
        fitness = np.zeros([self.pop_size, 1])
        for i in range(self.pop_size):
            # fitness[i] = 1 / makespan_value(self.data[:, pop[i]]) #最大完成时间

            # fitness[i] = 1 / sum(makespan(self.data[:, pop[i]])[-1, :]) #合计完成时间
            # fitness[i] = 1 / (sum(makespan(self.data[:, pop[i]])[-1, :])*c_op)  # 合计完成时间*单位成本


            ######目标：合计订单拣选成本+违约成本##########
            #各批次订单抵达后台时间
            batch_arrive_time=np.array([i+t_convey for i in (makespan(self.data[:, pop[i]],t_convey)[-1, :])])
            order=pop[i]
            # 各批次订单开始配送时间
            batch_delivery_start_time=batch_arrive_time+np.array([batch_packingtime[i] for i in order])
             #违约时间
            batch_over_duetime=[]
            for j in list(np.array(batch_delivery_start_time)-np.array([op_duetime[i] for i in order])):
                if j<0:
                    batch_over_duetime.append(0)
                else:
                    batch_over_duetime.append(j)
            # 总订单拣选成本+违约成本=合计订单开始配送时间*单位时间成本+合计违约时间*单位违约时间成本
            TC=((sum(batch_delivery_start_time)/10 * c_op)+sum(batch_over_duetime)/10*c_od)
            fitness[i] = 1 /(TC)


            ######合计加权时间##########
            # W=np.array([1,1,1,1]) #各工件权重
            # Value_all=np.array(makespan(self.data[:, pop[i]])[-1, :])
            # Seq_temp=makespan(self.data[:, pop[i]])[0,:]
            # W_new=W[Seq_temp-1] #工件权重调整
            # fitness[i] = 1 / sum(np.multiply(Value_all, W_new)) #加权合计完成时间
            ################################
        return fitness

    def select(self, pop, fitness):
        index = np.argsort(-fitness[:, 0])
        p = np.zeros([self.pop_size, 3, self.NDA])
        select = pop[index, :]
        M = self.pop_size
        a, b = baker(M)
        for i in range(self.pop_size):
            p[i, 0] = pop[i]
            p[i, 1] = (a - b * (i + 1)) / (M * (M + 1))  # select posibility
            p[i, 2] = np.sum(p[:i + 1, 1, 0])
        for i in range(self.n, self.pop_size - 1):
            Pi = np.random.rand()
            if p[i, 2, 0] > Pi and p[i + 1, 2, 0] < Pi:
                select[i + 1] = np.array(p[i + 1, 0], dtype=int)
        return select

    def crossover(self, select):
        n = int((self.pop_size - self.n) * self.Pc)
        index = np.random.choice(np.arange(self.n + 1, self.pop_size), n, replace=False)
        crossover = np.copy(select)
        crossover[index] = xpm(select[index])
        return crossover

    def mutation(self, crossover):
        n = int((self.pop_size - self.n) * self.Pm)
        index = np.random.choice(np.arange(self.n, crossover.shape[0]), n, replace=False)
        for i in range(n):
            p = np.random.choice(self.NDA, 2, replace=False)
            temp = np.copy(crossover[index[i], p[0]]), np.copy(crossover[index[i], p[1]])
            crossover[index[i], p[0]], crossover[index[i], p[1]] = temp[1], temp[0]
        return crossover


def ga_fsp_new(data,c_op,op_duetime,c_od,batch_packingtime,t_convey,pop_size=80, max_gen=300, Pc=0.65, Pm=0.35, draw=222):
    """
    流水车间作业调度的改进遗传算法。
    新增精英保留机制，即将每次迭代及其之前的最优个体保留下来
    轮盘赌选、部分匹配交叉混合顺序交叉
    :param data: m行n列，第1行工序编号，值加工时间
    :param pop_size: 种群大小
    :param max_gen: 最大进化代数
    :param Pc: 交叉概率
    :param Pm: 变异概率
    :param draw:甘特图、适应度图、动态适应度图
    :return:
    """
    data = data[:, np.argsort(data[0])]
    # pop_size=max(data.shape[1]*12,80)
    pop_size=80
    # max_gen=max(data.shape[1]*40,200)
    max_gen=800
    new = GA_FSP_NEW(data, pop_size, max_gen, Pc, Pm)
    pop = new.crtp()
    pop_trace = np.zeros([max_gen, 3])
    genetic_trace = np.zeros([max_gen, data.shape[1]], dtype=int)
    start_time = time.time()
    for g in range(max_gen):
        fitness= new.fitness(pop,c_op,op_duetime,c_od,batch_packingtime,t_convey)
        pop_trace[g] = [g, np.mean(fitness), np.max(fitness)]
        genetic_trace[g] = pop[np.argmax(fitness)]

        select = new.select(pop, fitness)
        crossover = new.crossover(select)
        pop = new.mutation(crossover)
    end_time = time.time()
    best_genetic = genetic_trace[np.argmax(pop_trace[:, 2])]
    total_best = np.where(pop_trace[:, 2] == np.max(pop_trace[:, 2]))[0]
    print("GA Time used:" ,(end_time - start_time))
    # print("The first best generation：%s" % np.argmax(pop_trace[:, 2]))
    # print("Best generations：%s" % total_best)
    # print("Numbers of best generation：%s" % total_best.shape[0])
    print("The minimum makespan: %s" % makespan_value(data[:, best_genetic],t_convey)) #输出最大完成时间
    if draw != 222:
        import matplotlib.pyplot as plt
        from tool import gatt
        if int(str(draw)[0]) != 2:
            plt.figure(1)
            gatt(data[:, best_genetic],t_convey)
            plt.show()
        if int(str(draw)[1]) != 2:
            plt.figure(2)
            plt.plot(pop_trace[:, 0], pop_trace[:, 2], "r-", label=r"$Best$ $fitness$")
            # plt.plot(pop_trace[:, 0], pop_trace[:, 1], "b-", label=r"$Pop$ $fitness$")
            plt.xlabel(r"$Generation_i$")
            plt.ylabel(r"$Fitness$")
            plt.legend()
            plt.show()
        if int(str(draw)[2]) != 2:
            plt.ioff()
            for i in range(1, max_gen):
                plt.figure(2)
                plt.plot([pop_trace[i - 1, 0], pop_trace[i, 0]], [pop_trace[i - 1, 2], pop_trace[i, 2]], "r-",
                         label=r"$Best$ $fitness$")
                plt.plot([pop_trace[i - 1, 0], pop_trace[i, 0]], [pop_trace[i - 1, 1], pop_trace[i, 1]], "b-",
                         label=r"$Pop$ $fitness$")
                plt.xlabel(r"$Generation$")
                plt.ylabel(r"$Fitness$")
                plt.pause(0.01)
            plt.show()
    best_genetic += 1
    return best_genetic

