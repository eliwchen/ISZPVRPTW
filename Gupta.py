# -*- coding:utf-8 -*-
import numpy as np
import time
from tool import makespan_value


class Gupta:
    def cals(self, data):
        s = np.zeros([2, data.shape[1]])
        s[0] = data[0]
        for j in range(data.shape[1]):
            temp = []
            c = 1
            if data[-1, j] > data[1, j]:
                c = -1
            for m in range(data.shape[0] - 1):
                temp.append(np.sum([data[m, j], data[m + 1, j]]))
            s[1, j] = c / np.min(temp)
        return s

    def sort(self, data, s):
        col = np.array(s[:, np.argsort(s[1])][0], dtype=int) - 1
        data_gupta = data[:, col][0]
        return data_gupta


def gupta(data,transfer_time, draw=0):
    """
    :param data:3行，工序编号，机器1加工时间，机器2加工时间
    :return:
    """
    data = data[:, np.argsort(data[0])]
    new = Gupta()
    start_time = time.time()
    s = new.cals(data)
    data_gupta = new.sort(data, s)
    end_time = time.time()
    # print("Gupta// Time used: %s" % (end_time - start_time))
    # print("Gupta// The minimum makespan: %s" % makespan_value(data[:, data_gupta - 1]))
    if draw:
        import matplotlib.pyplot as plt
        from tool import gatt
        gatt(data[:, data_gupta - 1])
        plt.show()
    return data_gupta,makespan_value(data[:, data_gupta - 1],transfer_time)


