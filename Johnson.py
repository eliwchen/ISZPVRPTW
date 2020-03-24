# -*- coding:utf-8 -*-
import numpy as np


class Johnson:
    def group(self, data):
        P = data[:, np.where(data[1] < data[2])[0]]
        Q = data[:, np.where(data[1] >= data[2])[0]]
        return P, Q

    def sort(self, P, Q):
        P = P[:, np.argsort(P[1])]
        Q = Q[:, np.argsort(-Q[2])]
        return P, Q

    def combine(self, P, Q):
        try:
            data = np.hstack([P, Q])
        except ValueError:
            data = P
        data_johnson = data[0]
        return data_johnson


def johnson(data, draw=0):
    """
    :param data:3行，工序编号，机器1加工时间，机器2加工时间
    :return:
    """
    data = data[:, np.argsort(data[0])]
    new = Johnson()
    P, Q = new.group(data)
    P, Q = new.sort(P, Q)
    data_johnson = new.combine(P, Q)
    if draw:
        import matplotlib.pyplot as plt
        from .tool import gatt
        gatt(data[:, data_johnson - 1])
        plt.show()
    return data_johnson

