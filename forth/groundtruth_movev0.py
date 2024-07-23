import sys
sys.path.append('.')
import math
from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter, convolve

# import numpy as np

class GroundTruth:
	""" This is a template for a ground truth. Every ground truth must inherit and implement all of its methods and variables. """

	
	def __init__(self, scenario_map: np.ndarray):
		self.vrms = 1.0
		self.signal_map = np.zeros_like(scenario_map)
		self.scenario_map = scenario_map
		self.users=[(77.57, 81.86), (20.20, 19.16), (26.85, 26.57), (17.72, 18.04), (56.67, 36.30), (20.69, 21.69), (23.87, 16.68), (56.15, 48.54), (84.07, 81.42), (18.61, 20.25), (63.45, 43.68), (88.75, 81.74), (79.39, 85.08), (18.27, 18.65), (16.99, 18.39), (59.69, 47.66), (55.53, 33.09), (69.67, 45.98), (86.68, 70.74), (54.22, 41.19), (19.87, 21.92), (62.34, 39.23), (23.40, 20.54), (28.35, 15.63), (74.16, 87.42), (61.68, 35.08), (81.78, 67.60), (16.11, 17.41), (70.92, 89.23), (76.76, 73.81), (80.63, 78.57), (77.33, 78.94), (59.82, 34.78), (80.44, 82.61), (15.31, 13.65), (55.06, 40.23), (62.58, 39.07), (90.30, 75.26), (17.72, 14.72), (15.21, 24.72), (61.33, 45.93), (77.93, 82.15), (20.05, 27.99), (54.87, 38.04), (81.28, 82.72), (65.31, 41.42), (79.92, 81.70), (21.40, 16.18), (63.01, 41.82), (27.89, 21.79), (74.68, 77.73)]
	
		
		# 假设 scenario_map 和 P 已经定义
		self.P = 1  # 信号强度系数，示例值

		# 遍历地图的每个点
		for x in range(scenario_map.shape[0]):
			for y in range(scenario_map.shape[1]):
				# 对于每个点，计算所有用户的信号强度并累加
				total_signal = 0
				for user_x, user_y in self.users:
					distance = np.sqrt((user_x - x) ** 2 + (user_y - y) ** 2)
					if distance != 0:  # 避免除以零
						signal_strength = self.P / (distance** 0.7)#1.5
						 # 添加高斯噪音
						signal_strength_with_noise = self.add_gaussian_noise(signal_strength)
                        
						total_signal += signal_strength_with_noise
				
				# 将累加的信号强度赋值给 signal_map
				self.signal_map[x, y] = total_signal
		# for x in range(self.scenario_map.shape[0]):
		# 	for y in range(self.scenario_map.shape[1]):
		# 		# self.signal_map[x, y] = total_signal
		# 		self.signal_map[x, y] = (self.signal_map[x, y] - self.signal_map.min()) / (self.signal_map.max() - self.signal_map.min())
		self.signal_map = gaussian_filter(self.signal_map, 0.8)
  		
	def add_gaussian_noise(self, signal_strength):
		mean = 0
		std_dev = 0.1  # 调整噪音的标准差以控制噪音的大小
		noise = np.random.normal(mean, std_dev)
		return signal_strength
		# return signal_strength + noise

	def step(self):
		updated_users = []
		for i in range(len(self.users)):
			user_x, user_y = self.users[i]
			# 生成速度数据
			Vx = np.random.normal(0, self.vrms)
			Vy = np.random.normal(0, self.vrms)
			# 根据速度和时间更新位置
			user_x += Vx * 0.5
			user_y += Vy * 0.5
			# user_x += Vx * 1
			# user_y += Vy * 1

			# 边界检测
			user_x = max(2, min(user_x, 97))
			user_y = max(2, min(user_y, 97))
			updated_users.append((user_x, user_y))
		self.users=updated_users
  
  
		# 遍历地图的每个点
		for x in range(self.scenario_map.shape[0]):
			for y in range(self.scenario_map.shape[1]):
				# 对于每个点，计算所有用户的信号强度并累加
				total_signal = 0
				for user_x, user_y in self.users:
					distance = np.sqrt((user_x - x) ** 2 + (user_y - y) ** 2)
					if distance != 0:  # 避免除以零
						signal_strength = self.P / (distance** 0.7)#1.5
						 # 添加高斯噪音
						signal_strength_with_noise = self.add_gaussian_noise(signal_strength)
                        
						total_signal += signal_strength_with_noise
				
				# 将累加的信号强度赋值给 signal_map
				self.signal_map[x, y] = total_signal
		return
	

	# 	raise NotImplementedError('This method has not being implemented yet.')
	def reset(self):
		# 51
		self.users=[(77.57, 81.86), (20.20, 19.16), (26.85, 26.57), (17.72, 18.04), (56.67, 36.30), (20.69, 21.69), (23.87, 16.68), (56.15, 48.54), (84.07, 81.42), (18.61, 20.25), (63.45, 43.68), (88.75, 81.74), (79.39, 85.08), (18.27, 18.65), (16.99, 18.39), (59.69, 47.66), (55.53, 33.09), (69.67, 45.98), (86.68, 70.74), (54.22, 41.19), (19.87, 21.92), (62.34, 39.23), (23.40, 20.54), (28.35, 15.63), (74.16, 87.42), (61.68, 35.08), (81.78, 67.60), (16.11, 17.41), (70.92, 89.23), (76.76, 73.81), (80.63, 78.57), (77.33, 78.94), (59.82, 34.78), (80.44, 82.61), (15.31, 13.65), (55.06, 40.23), (62.58, 39.07), (90.30, 75.26), (17.72, 14.72), (15.21, 24.72), (61.33, 45.93), (77.93, 82.15), (20.05, 27.99), (54.87, 38.04), (81.28, 82.72), (65.31, 41.42), (79.92, 81.70), (21.40, 16.18), (63.01, 41.82), (27.89, 21.79), (74.68, 77.73)]
		
  
		# # 遍历地图的每个点
		# for x in range(self.scenario_map.shape[0]):
		# 	for y in range(self.scenario_map.shape[1]):
		# 		# 对于每个点，计算所有用户的信号强度并累加
		# 		total_signal = 0
		# 		for user_x, user_y in self.users:
		# 			distance = np.sqrt((user_x - x) ** 2 + (user_y - y) ** 2)
		# 			if distance != 0:  # 避免除以零
		# 				signal_strength = self.P / (distance** 0.7)#1.5
		# 				 # 添加高斯噪音
		# 				signal_strength_with_noise = self.add_gaussian_noise(signal_strength)
                        
		# 				total_signal += signal_strength_with_noise
				
		# 		# 将累加的信号强度赋值给 signal_map
		# 		self.signal_map[x, y] = total_signal
	# def reset(self, random_gt: bool = False):
	# 	""" Reset ground Truth """
	# 	raise NotImplementedError('This method has not being implemented yet.')
	def read(self, position=None):
		if position is None:
			return self.signal_map
		else:
			position = np.asarray(position).astype(int)
			return self.signal_map[position[:,0], position[:,1]]
	# def read(self, position=None):
	# 	""" Read the complete ground truth or a certain position """
	# 	raise NotImplementedError('This method has not being implemented yet.')

	# def update_to_time(self, t):
	# 	""" Update the environment a number of steps """
	# 	raise NotImplementedError('This method has not being implemented yet.')



