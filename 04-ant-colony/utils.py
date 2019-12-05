import numpy as np


class Ant:
    def __init__(self, path):
        self.path = path
        self.length = self.calc_length(path)

    def calc_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            delta = (path[i].x - path[i + 1].x, path[i].y - path[i + 1].y)
            length += np.linalg.norm(delta)
        return length

    @staticmethod
    def calc_len(A, B):
        return np.linalg.norm((A.x - B.x, A.y - B.y))


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Path:
    def __init__(self, A):
        self.path = [A, A]

    def add_path(self, B):
        self.path.insert(-2, B)
