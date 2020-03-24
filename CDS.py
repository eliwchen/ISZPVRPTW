# -*- coding:utf-8 -*-
import numpy as np
import time
from tool import makespan_value
from tool import makespan
from Johnson import johnson


class CDS:
    def group(self, data):
        data_group = np.zeros([data.shape[0] - 2, 3, data.shape[1]])
        for i in range(data_group.shape[0]):
            data_group[i, 0] = data[0]
            for j in range(data.shape[1]):
                data_group[i, 1, j] = np.sum(data[1:i + 2, j])
                data_group[i, 2, j] = np.sum(data[-i - 1:, j])
        return data_group

    def johnson(self, data_group):
        data_johnson = np.zeros([data_group.shape[0], data_group.shape[2]])
        for i in range(data_group.shape[0]):
            data_johnson[i] = johnson(data_group[i])
        return data_johnson

    def select(self, data,transfer_time, data_johnson):
        data_johnson = np.array(data_johnson, dtype=int) - 1
        data_best = data_johnson[0]
        for i in range(1, data_johnson.shape[0]):
            if makespan_value(data[:, data_best],transfer_time,) > makespan_value(data[:, data_johnson[i]],transfer_time,):
                data_best = data_johnson[i]
        data_best += 1
        return data_best


def cds(data, transfer_time,draw=0):
    """
    :param data: n行m列，第一行工序编号，其他是加工时间
    :return:
    """
    data = data[:, np.argsort(data[0])]
    new = CDS()
    start_time = time.time()
    data_group = new.group(data)
    data_johnson = new.johnson(data_group)
    data_best = new.select(data, transfer_time,data_johnson)
    end_time = time.time()
    # print("CDS// Time used: %s" % (end_time - start_time))
    # print("CDS// The minimum makespan: %s" %makespan_value(data[:, data_best - 1]))
    if draw:
        import matplotlib.pyplot as plt
        from tool import gatt
        gatt(data[:, data_best - 1])
        plt.show()
    return data_best,makespan_value(data[:, data_best - 1],transfer_time)


