# -*- coding:utf-8 -*-
import numpy as np
import time
from tool import makespan_value


class Palmer:
    def slope_index(self, data):
        slope_index = np.zeros([2, data.shape[1]])
        slope_index[0] = data[0]
        m = data.shape[0] - 1
        for job in range(data.shape[1]):
            for machine in range(1, m + 1):
                slope_index[1, job] += (machine - (m + 1) / 2) * data[machine, job]
        return slope_index

    def sort_slope(self, slope_index):
        sort_slope = slope_index[:, np.argsort(-slope_index[1])]
        return sort_slope

    def palmer_data(self, data, sort_slope):
        index = np.array(sort_slope[0], dtype=int) - 1
        palmer_data = data[:, index][0]
        return palmer_data


def palmer(data,transfer_time, draw=0):
    """
    :param data: n行m列，第一行工序编号，其他是加工时间
    :return:
    """
    data = data[:, np.argsort(data[0])]
    new = Palmer()
    start_time = time.time()
    slope_index = new.slope_index(data)
    sort_slope = new.sort_slope(slope_index)
    palmer_data = new.palmer_data(data, sort_slope)
    end_time = time.time()
    # print("Palmer// Time used: %s" % (end_time - start_time))
    # print("Palmer// The minimum makespan: %s" % makespan_value(data[:, palmer_data - 1]))
    if draw:
        import matplotlib.pyplot as plt
        from tool import gatt
        gatt(data[:, palmer_data - 1])
        plt.show()
    return palmer_data,makespan_value(data[:, palmer_data - 1],transfer_time)


