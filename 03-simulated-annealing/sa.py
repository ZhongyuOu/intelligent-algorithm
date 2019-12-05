#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
np.random.seed(123)


def initialSolution(n):
    solution = np.arange(n)
    np.random.shuffle(solution)
    return solution


def findNeighbor(solution):
    neighbor = solution.copy()
    i, j = np.random.choice(len(solution), 2, replace=False)
    neighbor[i], neighbor[j] = solution[j], solution[i]
    return neighbor


def computeTotalDistance(solution, distMatrix):
    totalDistance = 0
    for i in range(len(solution) - 1):
        totalDistance += distMatrix[solution[i], solution[i + 1]]
    return totalDistance


def main():
    n = 31
    temperature = 100 * n
    r = 0.9
    innerIter = 120

    distMatrix = np.load('distMatrix.npy')
    solution = initialSolution(n)
    totalDistance = computeTotalDistance(solution, distMatrix)
    print('Initial solution:')
    for i in range(n - 1):
        print(solution[i], '->', end=' ')
    print(solution[-1])
    bestSolution = solution
    print('Total distance: %d' % totalDistance, end='\n\n')

    while temperature > 0.001:
        for i in range(innerIter):
            neighbor = findNeighbor(solution)
            newTotalDistance = computeTotalDistance(neighbor, distMatrix)
            delta = newTotalDistance - totalDistance
            if delta < 0:
                solution, totalDistance = neighbor, newTotalDistance
                bestSolution = neighbor
            elif np.exp(-delta / temperature) > np.random.rand():
                solution, totalDistance = neighbor, newTotalDistance
        temperature *= r

    print('Best solution:')
    for i in range(n - 1):
        print(bestSolution[i], '->', end=' ')
    print(bestSolution[-1])
    print('Total distance: %d' %
          computeTotalDistance(bestSolution, distMatrix))

    # Visualize
    data = np.load('ordinates.npz')
    x, y = data['x'], data['y']
    for i in range(n - 1):
        plt.plot([x[bestSolution[i]], x[bestSolution[i + 1]]],
                 [y[bestSolution[i]], y[bestSolution[i + 1]]],
                 '*-b',
                 mec='r')
    plt.plot([x[bestSolution[-1]], x[bestSolution[0]]],
             [y[bestSolution[-1]], y[bestSolution[0]]],
             '*-b',
             mec='r')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('Traveling Salesman Problem')
    plt.show()


if __name__ == '__main__':
    main()
