from utils import *
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()
np.random.RandomState(42)


class ACO(object):
    def __init__(self, ant_num=50, maxIter=200, alpha=1, beta=5, rho=0.1, Q=1):
        self.ants_num = ant_num
        self.maxIter = maxIter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

        self.deal_data("coordinates.npz")

        self.path_seed = np.zeros(self.ants_num).astype(int)
        self.ants_info = np.zeros((self.maxIter, self.ants_num))
        self.best_path = np.zeros(self.maxIter)

        self.solve()
        self.display()

    def deal_data(self, filename):
        data = np.load(filename)
        x, y = data["x"], data["y"]
        self.cities_num = len(x)
        self.cities = [City(x[i], y[i]) for i in range(self.cities_num)]
        self.city_dist_mat = np.zeros((self.cities_num, self.cities_num))
        for i in range(self.cities_num):
            A = self.cities[i]
            for j in range(i, self.cities_num):
                B = self.cities[j]
                self.city_dist_mat[i][j] = self.city_dist_mat[j][i] = Ant.calc_len(A, B)
        self.phero_mat = np.ones((self.cities_num, self.cities_num))
        self.eta_mat = 1 / (self.city_dist_mat + np.diag([np.inf] * self.cities_num))

    def solve(self):
        iterNum = 0
        while iterNum < self.maxIter:
            self.random_seed()
            delta_phero_mat = np.zeros((self.cities_num, self.cities_num))

            for i in range(self.ants_num):
                city_index1 = self.path_seed[i]
                ant_path = Path(self.cities[city_index1])
                tabu = [city_index1]
                non_tabu = list(set(range(self.cities_num)) - set(tabu))
                for j in range(self.cities_num - 1):
                    up_proba = np.zeros(self.cities_num - len(tabu))
                    for k in range(self.cities_num - len(tabu)):
                        up_proba[k] = np.power(
                            self.phero_mat[city_index1][non_tabu[k]], self.alpha
                        ) * np.power(self.eta_mat[city_index1][non_tabu[k]], self.beta)
                    proba = up_proba / sum(up_proba)
                    while True:
                        random_num = np.random.rand()
                        index_need = np.where(proba > random_num)[0]
                        if len(index_need) > 0:
                            city_index2 = non_tabu[index_need[0]]
                            break
                    ant_path.add_path(self.cities[city_index2])
                    tabu.append(city_index2)
                    non_tabu = list(set(range(self.cities_num)) - set(tabu))
                    city_index1 = city_index2
                self.ants_info[iterNum][i] = Ant(ant_path.path).length
                if iterNum == 0 and i == 0:
                    self.best_cities = ant_path.path
                else:
                    if self.ants_info[iterNum][i] < Ant(self.best_cities).length:
                        self.best_cities = ant_path.path
                tabu.append(tabu[0])
                for l in range(self.cities_num):
                    delta_phero_mat[tabu[l]][tabu[l + 1]] += (
                        self.Q / self.ants_info[iterNum][i]
                    )

            self.best_path[iterNum] = Ant(self.best_cities).length

            self.update_phero_mat(delta_phero_mat)
            iterNum += 1

    def update_phero_mat(self, delta):
        self.phero_mat = (1 - self.rho) * self.phero_mat + delta
        # self.phero_mat = np.where(self.phero_mat > self.phero_upper_bound, self.phero_upper_bound, self.phero_mat) # 判断是否超过浓度上限

    def random_seed(self):
        if self.ants_num <= self.cities_num:
            self.path_seed[:] = np.random.permutation(range(self.cities_num))[
                : self.ants_num
            ]
        else:
            self.path_seed[: self.cities_num] = np.random.permutation(
                range(self.cities_num)
            )
            temp_index = self.cities_num
            while temp_index + self.cities_num <= self.ants_num:
                self.path_seed[
                    temp_index : temp_index + self.cities_num
                ] = np.random.permutation(range(self.cities_num))
                temp_index += self.cities_num
            temp_left = self.ants_num % self.cities_num
            if temp_left != 0:
                self.path_seed[temp_index:] = np.random.permutation(
                    range(self.cities_num)
                )[:temp_left]

    def display(self):
        plt.figure(figsize=(8, 6))
        plt.plot(
            list(city.x for city in self.best_cities)+[self.best_cities[0].x],
            list(city.y for city in self.best_cities)+[self.best_cities[0].y],
            "b-",
        )
        plt.plot(
            list(city.x for city in self.best_cities),
            list(city.y for city in self.best_cities),
            "r.",
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig("ACO.png", dpi=500)
        plt.show()


if __name__ == '__main__':
    aco = ACO()
    print('Total distance: ', aco.best_path[-1])
