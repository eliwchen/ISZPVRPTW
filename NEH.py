# -*- coding:utf-8 -*-
import numpy as np
import time
from tool import makespan_value


class NEH:
    def sort(self, data):
        total_time = np.zeros([2, data.shape[1]])
        total_time[0] = data[0]
        for i in range(data.shape[1]):
            total_time[1, i] = np.sum(data[:, i])
        total_time = total_time[:, np.argsort(-total_time[1])]
        return total_time

    def insert(self, data,t_convey, total_time):
        neh_data = np.zeros([data.shape[0], 2], dtype=data.dtype)
        neh_data[:, :2] = data[:, np.array(total_time[0, :2], dtype=int) - 1]
        for k in range(2, data.shape[1]):
            temp = data[:, int(total_time[0, k] - 1)]
            neh_data = np.insert(neh_data, k, temp, axis=1)
            for i in range(k):
                if makespan_value(neh_data,t_convey) > makespan_value(np.insert(neh_data, i, temp, axis=1),t_convey):
                    neh_data = np.delete(neh_data, k, axis=1)
                    neh_data = np.insert(neh_data, i, temp, axis=1)
        data_neh = neh_data[0]
        return data_neh


def neh(data,t_convey, draw=0):
    """
    :param data: n行m列，第一行工序编号，其他是加工时间
    :return:
    """
    data = data[:, np.argsort(data[0])]
    new = NEH()
    start_time = time.time()
    total_time = new.sort(data)
    data_neh = new.insert(data,t_convey, total_time)
    end_time = time.time()
    # print("Time used: %s" % (end_time - start_time))
    # print("The minimum makespan: %s" % makespan_value(data[:, data_neh - 1]))
    if draw:
        import matplotlib.pyplot as plt
        from tool import gatt
        gatt(data[:, data_neh - 1])
        plt.show()
    return data_neh,makespan_value(data[:, data_neh - 1],t_convey)


