#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(42)


class PSO:
    def __init__(self, c1, c2, pop_size, v_max):
        """
        :param c1: learning factor
        :param c2: learning factor
        :param pop_size: population size
        :param v_max: maximun velocity
        """
        self.c1 = c1
        self.c2 = c2
        self.pop_size = pop_size
        self.v_max = v_max

    def fit(self, d, left, right, w, max_iter):
        """
        :param d: dimension of the problem
        :param left: left margin of x
        :param right: right margin of x
        :max_iter: maximun iteration

        :return: best solution and fitness
        """
        x = np.random.uniform(left, right, size=(self.pop_size, d))
        v = np.random.rand(self.pop_size, d) * self.v_max

        l_best_fitness = np.apply_along_axis(f, 1, x)
        g_best_fitness = l_best_fitness.min()
        l_best_point = x.copy()
        g_best_point = x[np.argmin(l_best_fitness)].copy()

        for i in range(max_iter):
            xi, eta = np.random.rand(2)
            for j in range(self.pop_size):
                v[j] = w * v[j] + self.c1 * xi * (
                    l_best_point[j] - x[j]) + self.c2 * eta * (g_best_point -
                                                               x[j])
                v[j][v[j] > self.v_max] = self.v_max

                x[j] += v[j]
                x[j][x[j] < left] = left
                x[j][x[j] > right] = right

                fitness = f(x[j])
                if fitness < l_best_fitness[j]:
                    l_best_fitness[j] = fitness
                    l_best_point[j] = x[j].copy()

                if fitness < g_best_fitness:
                    g_best_fitness = fitness
                    g_best_point = x[j].copy()

        return g_best_point, g_best_fitness


def f(x):
    """Function to be optimized."""
    return np.sum(np.power(x, 2) - 10 * np.cos(2*np.pi*x) + 10)


def main():
    pso = PSO(c1=2, c2=2, pop_size=60, v_max=2)
    p = pso.fit(d=30, left=-5.12, right=5.12, w=0.5, max_iter=100)
    print('Best solution:\n', p[0], sep='')
    print('\n')
    print('Fitness:\n', p[1], sep='')


if __name__ == '__main__':
    main()
