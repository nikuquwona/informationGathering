from matplotlib import pyplot as plt
from matplotlib.mlab import GaussianKDE
import numpy as np

class LocalGaussianDistribution:
    def __init__(self):
        self.points = []  # 存储坐标点 (x, y, z)
        self.weights = []  # 对应的信号值作为权重
        self.mu = np.array([0, 0, 0])  # 当前均值
        self.sigma = np.array([0, 0, 0])  # 当前标准差
        self.old_mu = np.array([0, 0, 0])  # 上一次的均值
        self.old_sigma = np.array([0, 0, 0])  # 上一次的标准差

    def update_distribution(self):
        if self.points:
            self.old_mu = self.mu
            self.old_sigma = self.sigma
            weighted_points = np.array(self.points) * np.array(self.weights)[:, None]
            total_weight = sum(self.weights)
            self.mu = np.sum(weighted_points, axis=0) / total_weight
            weighted_squares = (np.array(self.points) - self.mu) ** 2 * np.array(self.weights)[:, None]
            self.sigma = np.sqrt(np.sum(weighted_squares, axis=0) / total_weight)

    def add_point(self, x, y, z, signal):
        if (x, y, z) not in self.points:
            self.points.append((x, y, z))
            self.weights.append(signal)
            self.update_distribution()

    def get_distribution_parameters(self):
        return self.mu, self.sigma, self.old_mu, self.old_sigma
    def plot_3d_gaussian_distribution(self,num_samples=1000):
        mu=self.mu
        sigma=self.sigma,
        """
        绘制三维高斯分布图。

        :param mu: 均值向量，格式为 [mu_x, mu_y, mu_z]
        :param sigma: 方差向量，格式为 [sigma_x, sigma_y, sigma_z]
        :param num_samples: 生成的样本数量
        """
        # 生成高斯分布的样本
        samples = np.random.normal(mu, sigma, (num_samples, 3))

        # 创建三维散点图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2])

        # 设置图表标题和坐标轴标签
        ax.set_title('3D Gaussian Distribution')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        plt.show()

# 示例使用
local_gaussian = LocalGaussianDistribution()
local_gaussian.add_point(1, 2, 3, 4)
local_gaussian.add_point(4, 5, 6, 7)

mu, sigma, old_mu, old_sigma = local_gaussian.get_distribution_parameters()
print("Current mu:", mu)
print("Current sigma:", sigma)
print("Old mu:", old_mu)
print("Old sigma:", old_sigma)
local_gaussian.plot_3d_gaussian_distribution()
# local_gaussian.plot_distribution()
