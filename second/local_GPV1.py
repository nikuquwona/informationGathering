import numpy as np

class LocalGaussianDistribution:
    def __init__(self):
        self.signals = []  # 存储信号值
        self.mu = None  # 当前信号值的均值向量
        self.cov = None  # 当前信号值的协方差矩阵

    def update_distribution(self):
        if self.signals:
            self.signals_array = np.array(self.signals)
            self.mu = np.mean(self.signals_array, axis=0)
            self.cov = np.cov(self.signals_array, rowvar=False)

    def add_point(self, x, y, z, signal):
        # 添加信号值及其对应的坐标
        self.signals.append([x, y, z, signal])
        self.update_distribution()

    def get_distribution_parameters(self):
        return self.mu, self.cov

# 示例使用
local_gaussian = LocalGaussianDistribution()

local_gaussian.add_point(1, 2, 3, 4)

mu, cov = local_gaussian.get_distribution_parameters()
print("Current mu:", mu)
print("Current covariance matrix:", cov)

local_gaussian.add_point(4, 5, 6, 7)

mu, cov = local_gaussian.get_distribution_parameters()
print("Current mu:", mu)
print("Current covariance matrix:", cov)


# import numpy as np

# class LocalGaussianDistribution:
#     def __init__(self):
#         self.data = {}  # 存储坐标点(x, y, z)和信号值
#         self.mu = np.array([0, 0, 0])  # 当前均值
#         self.sigma = np.array([0, 0, 0])  # 当前标准差
#         self.old_mu = np.array([0, 0, 0])  # 上一次的均值
#         self.old_sigma = np.array([0, 0, 0])  # 上一次的标准差

#     def update_distribution(self):
#         # 如果有数据点，则计算新的均值和标准差
#         if self.data:
#             points = np.array(list(self.data.keys()))
#             print("points",points)
#             self.old_mu = self.mu
#             self.old_sigma = self.sigma
#             self.mu = np.mean(points, axis=0)
#             self.sigma = np.std(points, axis=0)

#     def add_point(self, x, y, z, signal):
#         # 如果点不存在，则添加该点及其信号值
#         if (x, y, z) not in self.data:
#             self.data[(x, y, z)] = signal
#             self.update_distribution()

#     def get_distribution_parameters(self):
#         return self.mu, self.sigma, self.old_mu, self.old_sigma

# # 示例使用
# Local_gaussian = LocalGaussianDistribution()
# Local_gaussian.add_point(1, 2, 3, 4)
# Local_gaussian.add_point(4, 5, 6, 7)

# mu, sigma, old_mu, old_sigma = Local_gaussian.get_distribution_parameters()
# print("Current mu:", mu)
# print("Current sigma:", sigma)
# print("Old mu:", old_mu)
# print("Old sigma:", old_sigma)
