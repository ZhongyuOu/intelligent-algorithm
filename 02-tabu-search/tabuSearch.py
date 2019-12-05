#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from collections import deque


def initialSolution(n):
    return np.random.randint(2, size=n)


def createNeighborhood(solution):
    neighborhood = []
    for i in range(len(solution)):
        newSolution = solution.copy()
        newSolution[i] = 1 - newSolution[i]
        neighborhood.append(newSolution)
    return neighborhood


def createValueTable(neighborhood, values, weights, maxWeight):
    valueTable = []
    for neighbor in neighborhood:
        totalValue = computeTotalValue(neighbor, values, weights, maxWeight)
        valueTable.append(totalValue)
    return valueTable


def computeTotalValue(solution, values, weights, maxWeight):
    totalValue = np.dot(solution, values)
    totalWeight = np.dot(solution, weights)
    if totalWeight <= maxWeight:
        return totalValue
    else:
        return 0


def main():
    n = 7
    values = np.array([40, 50, 30, 10, 10, 40, 30])
    weights = np.array([40, 50, 10, 10, 30, 20, 60])
    maxWeight = 120
    maxGen = 10

    solution = initialSolution(n)
    bestSolution = solution
    aspirationLevel = computeTotalValue(bestSolution, values, weights,
                                        maxWeight)
    tabuList = deque([], maxlen=int(np.sqrt(n)))

    for gen in range(maxGen):
        neighborhood = createNeighborhood(solution)
        valueTable = createValueTable(neighborhood, values, weights, maxWeight)
        if max(valueTable) > aspirationLevel:
            solution = neighborhood[valueTable.index(max(valueTable))]
            bestSolution = solution
            aspirationLevel = max(valueTable)
        else:
            newNeighborhood = neighborhood.copy()
            for i in range(len(newNeighborhood)):
                for item in tabuList:
                    if all(newNeighborhood[i] == item):
                        neighborhood.pop(i)
            valueTable2 = createValueTable(neighborhood, values, weights,
                                           maxWeight)
            solution = neighborhood[valueTable2.index(max(valueTable2))]
        tabuList.append(solution)
    print("Best solution is ", bestSolution)
    print("Max value is ", aspirationLevel)


if __name__ == '__main__':
    main()
