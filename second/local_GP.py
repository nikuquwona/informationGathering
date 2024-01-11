import numpy as np

class LocalGaussianDistribution:
    def __init__(self):
        self.data = {}  # 存储坐标点和信号值
        self.mu = np.array([0, 0])  # 当前均值
        self.sigma = np.array([0, 0])  # 当前标准差
        self.old_mu = np.array([0, 0])  # 上一次的均值
        self.old_sigma = np.array([0, 0])  # 上一次的标准差

    def update_distribution(self):
        # 如果有数据点，则计算新的均值和标准差
        if self.data:
            points = np.array(list(self.data.keys()))
            self.old_mu = self.mu
            self.old_sigma = self.sigma
            self.mu = np.mean(points, axis=0)
            self.sigma = np.std(points, axis=0)

    def add_point(self, x, y, signal):
        # 如果点不存在，则添加该点及其信号值
        if (x, y) not in self.data:
            self.data[(x, y)] = signal
            self.update_distribution()

    def get_distribution_parameters(self):
        return self.mu, self.sigma, self.old_mu, self.old_sigma

# 示例使用
local_gaussian = LocalGaussianDistribution()

local_gaussian.add_point(1, 2, 3)
local_gaussian.add_point(2, 3, 4)

mu, sigma, old_mu, old_sigma = local_gaussian.get_distribution_parameters()
print("Current mu:", mu)
print("Current sigma:", sigma)
print("Old mu:", old_mu)
print("Old sigma:", old_sigma)
