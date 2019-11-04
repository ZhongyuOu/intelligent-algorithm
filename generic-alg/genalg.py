#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Johnny Au
# Mail: zhongyu.ou@gmail.com
# Created Time: Thu 12 Sep 2019 10:24:42 AM CST


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=3)
sns.set()


def init_population(pop_size, start, end, precision):
    """Generate a initial population.

    :param pop_size: size of the population.
    :param start: left endpoint of the interval.
    :param end: right endpoint of the interval.
    :param precision: precision of encoding.
    :return: the initialize population.
    """
    length = np.ceil(
        np.log2((end-start)*np.power(10, precision))).astype('int')
    return np.random.randint(2, size=(pop_size, length))


def decoder(population, start, end):
    """Binary to decimal."""
    dna_length = population.shape[1]
    scale = population.dot(2**np.arange(dna_length)[::-1])/(2**dna_length-1)
    return start + (end-start) * scale


def selector(population, pop_decoded, fitness):
    pop_size = len(pop_decoded)
    idx = np.random.choice(range(pop_size), size=pop_size, replace=True,
                           p=fitness/fitness.sum())
    return population[idx]


def crossover(parent, population, cro_rate):
    if np.random.rand() < cro_rate:
        cro_length = np.ceil(len(parent)/3).astype('int')
        cro_point = np.random.choice(range(1, len(parent)-cro_length-1))
        another_parent = population[np.random.choice(len(population))]
        parent[cro_point:cro_point +
               cro_length] = another_parent[cro_point:cro_point+cro_length]
    return parent


def mutation(child, mut_rate):
    for point in range(len(child)):
        if np.random.rand() < mut_rate:
            child[point] = 1 - child[point]
    return child


def F(x):
    """Function to be optimized."""
    return x*np.sin(10*np.pi*x)+2
    # return x**3 - 60*x**2 + 900*x+100


def get_fitness(pop_decoded):
    return F(pop_decoded) - F(pop_decoded).min() + np.exp(-3)


def main():
    POP_SIZE = 100
    START = -1
    END = 2
    PRECISION = 2
    MAX_ITER = 200
    CRO_RATE = 0.9
    MUT_RATE = 0.003

    population = init_population(POP_SIZE, START, END, PRECISION)
    # max_fitness = []
    # mean_fitness = []
    x = np.linspace(START, END, 200)
    plt.figure(figsize=(8, 6), dpi=80)
    plt.ion()
    plt.plot(x, F(x), 'b')
    plt.axis([-1.2, 2.2, -0.2, 4.2])

    for iter in range(MAX_ITER):
        pop_decoded = decoder(population, START, END)
        fitness = get_fitness(pop_decoded)

        if 'sca' in locals():
            sca.remove()
        sca = plt.scatter(pop_decoded, F(pop_decoded), color='r',
                          marker='o', alpha=0.8)
        plt.title("Genetic Algorithm: %d Iteration" % (iter+1))
        plt.pause(0.05)

        population = selector(population, pop_decoded, fitness)
        population_copy = population.copy()
        for parent in population:
            child = crossover(parent, population_copy, CRO_RATE)
            child = mutation(child, MUT_RATE)
            parent[:] = child

        # max_fitness.append(fitness.max())
        # mean_fitness.append(fitness.mean())
    plt.ioff()
    plt.show()

    # plt.plot(range(MAX_ITER), mean_fitness)
    # plt.show()


if __name__ == "__main__":
    main()
