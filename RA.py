# -*- coding:utf-8 -*-
import numpy as np
import time
from tool import makespan_value
from Johnson import johnson


class RA:
    def group_machine(self, data):
        group_data = np.zeros([3, data.shape[1]], dtype=data.dtype)
        group_data[0] = data[0]
        for i in range(1, data.shape[0]):
            for j in range(data.shape[1]):
                group_data[1, j] += (data.shape[0] - i) * data[i, j]
                group_data[2, j] += i * data[i, j]
        return group_data

    def apply_johnson(self, group_data):
        ra_data = np.array(johnson(group_data), dtype=int)
        return ra_data


def ra(data, transfer_time,draw=0):
    """
    :param data: n行m列，第一行工序编号，其他是加工时间
    :return:
    """
    data = data[:, np.argsort(data[0])]
    new = RA()
    start_time = time.time()
    group_data = new.group_machine(data)
    ra_data = new.apply_johnson(group_data)
    end_time = time.time()
    # print("Time used: %s" % (end_time - start_time))
    # print("The minimum makespan: %s" % makespan_value(data[:, ra_data - 1]))
    if draw:
        import matplotlib.pyplot as plt
        from tool import gatt
        gatt(data[:, ra_data - 1])
        plt.show()
    return ra_data,makespan_value(data[:, ra_data - 1],transfer_time)


