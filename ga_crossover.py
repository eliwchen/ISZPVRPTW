# -*- coding:utf-8 -*-
import numpy as np


def baker(M):
    """
    线性排序参数
    :param M: 正整数
    :return:a,b
    """
    b = 2
    a = (M + 1) * (1 + b / 2)
    return a, b


def get_different(a, b):
    """
    在b的元素中获取与a中元素不同的元素
    :param a:1维
    :param b:1维
    :return:different
    """
    different = set()
    a, b = np.array(a), np.array(b)
    for i in range(b.shape[0]):
        if b[i] not in a:
            different.add((b[i]))
    different = list(different)
    return different


def same_index(a, b):
    """
    获取a中元素包含于b中元素的在a中的索引
    :param a:1维
    :param b:1维
    :return:c
    """
    a, b = np.array(a), np.array(b)
    c = []
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            if a[i] == b[j]:
                c.append(i)
    c = np.array(c)
    return c


def xcx(pop):
    """
    循环交叉
    :param pop: 若干个排列组合
    :return:crossover
    """
    crossover = np.zeros_like(pop)
    for c in range(pop.shape[0]):
        p = np.random.choice(pop.shape[1], 2, replace=False)
        p = p[np.argsort(p)]
        genetic = pop[c, :p[0]], pop[c, p[0]:p[1] + 1], pop[c, p[1] + 1:]
        crossover[c] = np.hstack([genetic[2], genetic[0], genetic[1]])
    return crossover


def xpm(pop):
    """
    部分匹配交叉、顺序交叉的混合交叉
    :param pop: 若干个排列组合
    :return:crossover
    """
    crossover = np.copy(pop)
    cn = crossover.shape[0]
    if cn % 2 != 0:
        cn -= 1
    for c in range(int(cn / 2)):
        p = np.random.choice(np.arange(1, pop.shape[1] - 1), 2, replace=False)
        p = p[np.argsort(p)]
        genetic_a = pop[c, :p[0]], pop[c, p[0]:p[1] + 1], pop[c, p[1] + 1:]
        genetic_b = pop[-c - 1, :p[0]], pop[-c - 1, p[0]:p[1] + 1], pop[-c - 1, p[1] + 1:]
        beside_a = np.hstack([genetic_a[0], genetic_a[2]])
        beside_b = np.hstack([genetic_b[0], genetic_b[2]])
        substitute_a = get_different(genetic_b[1], genetic_a[1])
        substitute_b = get_different(genetic_a[1], genetic_b[1])
        same_index_a = same_index(beside_a, genetic_b[1])
        same_index_b = same_index(beside_b, genetic_a[1])
        for i in range(same_index_a.shape[0]):
            beside_a[same_index_a[i]] = substitute_a[i]
        for j in range(same_index_b.shape[0]):
            beside_b[same_index_b[j]] = substitute_b[j]

        crossover[c, :p[0]] = beside_a[:p[0]]
        crossover[c, p[0]:p[1] + 1] = genetic_b[1]
        crossover[c, p[1] + 1:] = beside_a[p[0]:]
        crossover[-c - 1, :p[0]] = beside_b[:p[0]]
        crossover[-c - 1, p[0]:p[1] + 1] = genetic_a[1]
        crossover[-c - 1, p[1] + 1:] = beside_b[p[0]:]
    return crossover

