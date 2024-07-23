import sys

import matplotlib
from matplotlib import animation
sys.path.append('.')
import math
from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter, convolve

# import numpy as np
# people_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dodgerblue","darkcyan", "forestgreen", "darkgreen"])
# background_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#69473e","dodgerblue"])


# 人群颜色映射（向人群聚集的颜色）
people_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkred", "orangered", "gold"])
# 背景颜色映射（白色系）
background_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "whitesmoke"])

fuelspill_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dodgerblue", "olive", "saddlebrown", "indigo"])

class GroundTruth:
	""" This is a template for a ground truth. Every ground truth must inherit and implement all of its methods and variables. """

	
	def __init__(self, scenario_map: np.ndarray):
		self.vrms = 0.8#1.0
		self.signal_map = np.zeros_like(scenario_map)
		self.scenario_map = scenario_map
		#30
		# self.users=[(20.30, 21.35), (61.59, 43.16), (12.37, 28.72), (26.85, 26.57), (75.92, 85.00), (77.96, 80.08), (8.58, 22.92), (81.77, 75.11), (19.87, 21.92), (68.42, 33.97), (85.04, 75.17), (52.71, 42.07), (65.10, 34.48), (56.36, 32.28), (18.27, 18.65), (24.25, 19.32), (17.16, 18.38), (85.31, 81.42), (58.80, 43.59), (60.79, 38.20), (30.15, 13.50), (61.80, 36.94), (79.56, 77.03), (76.56, 82.41), (54.79, 44.03), (83.83, 77.09), (12.78, 19.05), (83.01, 81.82), (28.35, 15.63), (75.37, 83.05)]
		# (86, 80),(25, 18),(50, 51), 687.2340425531913
		# 39 users
		# self.users=[(26.85, 26.57), (84.07, 81.42), (56.67, 36.30), (83.83, 77.09), (18.27, 18.65), (82.80, 72.84), (75.06, 80.23), (68.35, 44.05), (75.72, 78.45), (16.00, 21.73), (56.56, 42.41), (54.12, 37.94), (73.11, 80.28), (84.25, 80.39), (75.53, 73.09), (54.75, 48.69), (56.15, 48.54), (77.93, 82.15), (73.63, 77.79), (82.86, 72.77), (25.50, 25.38), (16.36, 12.28), (59.69, 47.66), (78.35, 78.93), (66.66, 48.48), (21.80, 16.94), (59.56, 37.03), (61.59, 43.16), (83.52, 77.91), (85.04, 75.17), (60.05, 47.99), (23.87, 16.68), (8.58, 22.92), (74.87, 78.04), (16.03, 11.14), (79.89, 76.70), (79.92, 81.70), (76.69, 86.90), (30.15, 13.50)]
		# (86, 78),(21, 18),(65, 82)   1178.0
		# 50 user
		self.users=[(77.57, 81.86), (20.20, 19.16), (26.85, 26.57), (17.72, 18.04), (56.67, 36.30), (20.69, 21.69), (23.87, 16.68), (56.15, 48.54), (84.07, 81.42), (18.61, 20.25), (63.45, 43.68), (88.75, 81.74), (79.39, 85.08), (18.27, 18.65), (16.99, 18.39), (59.69, 47.66), (55.53, 33.09), (69.67, 45.98), (86.68, 70.74), (54.22, 41.19), (19.87, 21.92), (62.34, 39.23), (23.40, 20.54), (28.35, 15.63), (74.16, 87.42), (61.68, 35.08), (81.78, 67.60), (16.11, 17.41), (70.92, 89.23), (76.76, 73.81), (80.63, 78.57), (77.33, 78.94), (59.82, 34.78), (80.44, 82.61), (15.31, 13.65), (55.06, 40.23), (62.58, 39.07), (90.30, 75.26), (17.72, 14.72), (15.21, 24.72), (61.33, 45.93), (77.93, 82.15), (20.05, 27.99), (54.87, 38.04), (81.28, 82.72), (65.31, 41.42), (79.92, 81.70), (21.40, 16.18), (63.01, 41.82), (27.89, 21.79), (74.68, 77.73)]		
		#  (18, 17),(80, 83),(65, 41)       1814.0
		# 60 user
		# self.users=[(20.93, 24.37), (83.46, 74.71), (16.00, 21.73), (70.92, 89.23), (28.35, 24.05), (63.83, 37.09), (12.78, 19.05), (83.09, 83.11), (57.33, 38.94), (24.24, 19.31), (26.68, 19.08), (61.77, 35.11), (71.76, 77.86), (59.56, 37.03), (78.03, 78.75), (78.28, 80.42), (57.12, 30.06), (83.08, 85.82), (16.36, 12.28), (63.45, 43.68), (62.80, 32.84), (78.21, 81.65), (15.94, 20.23), (72.15, 82.02), (59.82, 34.78), (23.87, 16.68), (79.81, 78.72), (82.21, 79.48), (77.08, 78.25), (23.26, 20.58), (79.51, 84.56), (22.08, 13.13), (54.52, 43.78), (53.33, 41.97), (82.33, 77.42), (19.87, 21.92), (18.80, 23.59), (79.72, 79.52), (57.16, 43.03), (61.33, 45.93), (19.54, 21.84), (84.78, 77.97), (26.85, 26.57), (54.22, 41.19), (16.15, 28.54), (15.21, 24.72), (63.52, 37.91), (18.44, 20.70), (65.04, 35.17), (74.68, 77.73), (64.92, 42.93), (83.34, 74.68), (61.68, 35.08), (88.75, 81.74), (82.59, 81.56), (78.37, 80.70), (57.77, 48.63), (77.15, 78.22), (81.78, 67.60), (77.08, 77.67)]
		#   (86, 80),(19, 22),(81, 59)   	2016.0.0
		#   70  
		# self.users=[(30.15, 13.50), (60.63, 38.57), (64.12, 41.43), (79.40, 84.82), (16.00, 21.73), (18.80, 23.59), (90.30, 75.26), (73.53, 83.58), (72.15, 82.02), (80.44, 82.61), (56.41, 36.92), (76.78, 76.36), (26.85, 26.57), (26.66, 28.48), (76.94, 87.42), (20.93, 24.37), (16.98, 20.14), (20.30, 21.35), (15.21, 24.72), (19.87, 21.92), (14.75, 28.69), (74.96, 76.73), (28.35, 24.05), (54.68, 37.73), (83.64, 80.26), (59.69, 47.66), (83.12, 80.02), (55.92, 45.00), (76.58, 73.57), (16.36, 12.28), (79.15, 78.23), (61.91, 43.03), (24.25, 19.32), (53.33, 41.97), (84.78, 77.97), (59.89, 36.70), (69.67, 45.98), (66.68, 30.74), (78.62, 78.96), (57.93, 42.15), (20.69, 21.69), (12.78, 19.05), (78.03, 78.75), (20.57, 21.80), (18.20, 25.47), (81.23, 76.02), (59.56, 37.03), (76.91, 86.87), (24.24, 19.31), (28.42, 13.97), (16.11, 17.41), (59.92, 41.70), (8.58, 22.92), (53.63, 37.79), (28.35, 15.63), (12.37, 28.72), (81.28, 82.72), (79.17, 76.39), (82.59, 81.56), (81.25, 75.56), (52.92, 42.73), (68.75, 41.74), (54.52, 43.78), (82.98, 77.94), (56.67, 36.30), (68.11, 82.71), (12.71, 22.07), (59.41, 38.67), (58.61, 40.44)]
		#   (24, 19),(87, 80),(59, 53)    1632.5581395348838
		self.grid = scenario_map
		self.fig = None
		self.visitable_positions = np.column_stack(np.where(scenario_map == 1))
  
		# 假设 scenario_map 和 P 已经定义
		self.P = 1  # 信号强度系数，示例值

		# 遍历地图的每个点
		# for x in range(scenario_map.shape[0]):
		# 	for y in range(scenario_map.shape[1]):
				
		# 		# block
		# 		# if x==0 or x==1 or x==98 or x==99 or y==0 or y==1 or y==98 or y==99:
		# 		# 	self.signal_map[x, y] = 0
		# 		# 	continue
					
		# 		# 对于每个点，计算所有用户的信号强度并累加
		# 		total_signal = 0
		# 		for user_x, user_y in self.users:
		# 			distance = np.sqrt((user_x - x) ** 2 + (user_y - y) ** 2)+0.001
		# 			if distance != 0 :#and distance<=10:#10  # 避免除以零
		# 				signal_strength = self.P / (distance** 0.5)#0.5
		# 				 # 添加高斯噪音
		# 				signal_strength_with_noise = self.add_gaussian_noise(signal_strength)
                        
		# 				total_signal += signal_strength_with_noise
		# 		# 将累加的信号强度赋值给 signal_map
		# 		self.signal_map[x, y] = total_signal
		def calculate_signal_map(self):
			X, Y = np.meshgrid(np.arange(self.scenario_map.shape[0]), np.arange(self.scenario_map.shape[1]), indexing='ij')
			for user_x, user_y in self.users:
				# 计算所有点到当前用户的距离
				distance = np.sqrt((X - user_x) ** 2 + (Y - user_y) ** 2) + 0.001
				# 创建一个布尔掩码，筛选距离小于等于10的点
				mask = distance <= 10
				# 计算满足条件的信号强度
				signal_strength = np.zeros_like(distance)
				signal_strength[mask] = self.P / (distance[mask] ** 0.5)
				# 添加高斯噪音
				signal_strength_with_noise = self.add_gaussian_noise(signal_strength)
				# 累加信号强度
				self.signal_map += signal_strength_with_noise
		self.signal_map = np.zeros(self.scenario_map.shape)
		calculate_signal_map(self)

		# for x in range(self.scenario_map.shape[0]):
		# 	for y in range(self.scenario_map.shape[1]):
		# 		# self.signal_map[x, y] = total_signal
		# 		self.signal_map[x, y] = (self.signal_map[x, y] - self.signal_map.min()) / (self.signal_map.max()*1.5 - self.signal_map.min())

		# 计算均值和标准差
		mean_value = np.mean(self.signal_map)
		std_dev = np.std(self.signal_map)

		# Z-score正则化
		self.signal_map = (self.signal_map - mean_value) / std_dev

		# 将值映射到0到1之间
		min_normalized_value = np.min(self.signal_map)
		max_normalized_value = np.max(self.signal_map)
		self.signal_map = (self.signal_map - min_normalized_value) / (max_normalized_value - min_normalized_value)

  
  
  
		
		# self.signal_map = gaussian_filter(self.signal_map, 0.8)
  		
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
			user_x += Vx #* 0.05
			user_y += Vy #* 0.05
			# user_x += Vx * 1
			# user_y += Vy * 1

			# 边界检测
			user_x = max(2, min(user_x, 97))
			user_y = max(2, min(user_y, 97))
			updated_users.append((user_x, user_y))
		self.users=updated_users
  
		# for x in range(self.scenario_map.shape[0]):
		# 	for y in range(self.scenario_map.shape[1]):
		# 		total_signal = 0
		# 		for user_x, user_y in self.users:
		# 			distance = np.sqrt((user_x - x) ** 2 + (user_y - y) ** 2)+0.001
		# 			if distance != 0 :#and distance<=10:#10  # 避免除以零
		# 				signal_strength = self.P / (distance** 0.5)#0.5
		# 				 # 添加高斯噪音
		# 				signal_strength_with_noise = self.add_gaussian_noise(signal_strength)
                        
		# 				total_signal += signal_strength_with_noise
		# 		# 将累加的信号强度赋值给 signal_map
		# 		self.signal_map[x, y] = total_signal
		def calculate_signal_map(self):
			X, Y = np.meshgrid(np.arange(self.scenario_map.shape[0]), np.arange(self.scenario_map.shape[1]), indexing='ij')
			for user_x, user_y in self.users:
				# 计算所有点到当前用户的距离
				distance = np.sqrt((X - user_x) ** 2 + (Y - user_y) ** 2) + 0.001
				# 创建一个布尔掩码，筛选距离小于等于10的点
				mask = distance <= 10
				# 计算满足条件的信号强度
				signal_strength = np.zeros_like(distance)
				signal_strength[mask] = self.P / (distance[mask] ** 0.5)
				# 添加高斯噪音
				signal_strength_with_noise = self.add_gaussian_noise(signal_strength)
				# 累加信号强度
				self.signal_map += signal_strength_with_noise
		self.signal_map = np.zeros(self.scenario_map.shape)
		calculate_signal_map(self)

		# 计算均值和标准差
		mean_value = np.mean(self.signal_map)
		std_dev = np.std(self.signal_map)

		# Z-score正则化
		self.signal_map = (self.signal_map - mean_value) / std_dev

		# 将值映射到0到1之间
		min_normalized_value = np.min(self.signal_map)
		max_normalized_value = np.max(self.signal_map)
		self.signal_map = (self.signal_map - min_normalized_value) / (max_normalized_value - min_normalized_value)
		return
	

	# 	raise NotImplementedError('This method has not being implemented yet.')
	def reset(self):
		self.users=[(77.57, 81.86), (20.20, 19.16), (26.85, 26.57), (17.72, 18.04), (56.67, 36.30), (20.69, 21.69), (23.87, 16.68), (56.15, 48.54), (84.07, 81.42), (18.61, 20.25), (63.45, 43.68), (88.75, 81.74), (79.39, 85.08), (18.27, 18.65), (16.99, 18.39), (59.69, 47.66), (55.53, 33.09), (69.67, 45.98), (86.68, 70.74), (54.22, 41.19), (19.87, 21.92), (62.34, 39.23), (23.40, 20.54), (28.35, 15.63), (74.16, 87.42), (61.68, 35.08), (81.78, 67.60), (16.11, 17.41), (70.92, 89.23), (76.76, 73.81), (80.63, 78.57), (77.33, 78.94), (59.82, 34.78), (80.44, 82.61), (15.31, 13.65), (55.06, 40.23), (62.58, 39.07), (90.30, 75.26), (17.72, 14.72), (15.21, 24.72), (61.33, 45.93), (77.93, 82.15), (20.05, 27.99), (54.87, 38.04), (81.28, 82.72), (65.31, 41.42), (79.92, 81.70), (21.40, 16.18), (63.01, 41.82), (27.89, 21.79), (74.68, 77.73)]		
		def calculate_signal_map(self):
			X, Y = np.meshgrid(np.arange(self.scenario_map.shape[0]), np.arange(self.scenario_map.shape[1]), indexing='ij')
			for user_x, user_y in self.users:
				# 计算所有点到当前用户的距离
				distance = np.sqrt((X - user_x) ** 2 + (Y - user_y) ** 2) + 0.001
				# 创建一个布尔掩码，筛选距离小于等于10的点
				mask = distance <= 10
				# 计算满足条件的信号强度
				signal_strength = np.zeros_like(distance)
				signal_strength[mask] = self.P / (distance[mask] ** 0.5)
				# 添加高斯噪音
				signal_strength_with_noise = self.add_gaussian_noise(signal_strength)
				# 累加信号强度
				self.signal_map += signal_strength_with_noise
		self.signal_map = np.zeros(self.scenario_map.shape)
		calculate_signal_map(self)
		# 计算均值和标准差
		mean_value = np.mean(self.signal_map)
		std_dev = np.std(self.signal_map)

		# Z-score正则化
		self.signal_map = (self.signal_map - mean_value) / std_dev

		# 将值映射到0到1之间
		min_normalized_value = np.min(self.signal_map)
		max_normalized_value = np.max(self.signal_map)
		self.signal_map = (self.signal_map - min_normalized_value) / (max_normalized_value - min_normalized_value)

		return
		# 51
		self.users=[(77.57, 81.86), (20.20, 19.16), (26.85, 26.57), (17.72, 18.04), (56.67, 36.30), (20.69, 21.69), (23.87, 16.68), (56.15, 48.54), (84.07, 81.42), (18.61, 20.25), (63.45, 43.68), (88.75, 81.74), (79.39, 85.08), (18.27, 18.65), (16.99, 18.39), (59.69, 47.66), (55.53, 33.09), (69.67, 45.98), (86.68, 70.74), (54.22, 41.19), (19.87, 21.92), (62.34, 39.23), (23.40, 20.54), (28.35, 15.63), (74.16, 87.42), (61.68, 35.08), (81.78, 67.60), (16.11, 17.41), (70.92, 89.23), (76.76, 73.81), (80.63, 78.57), (77.33, 78.94), (59.82, 34.78), (80.44, 82.61), (15.31, 13.65), (55.06, 40.23), (62.58, 39.07), (90.30, 75.26), (17.72, 14.72), (15.21, 24.72), (61.33, 45.93), (77.93, 82.15), (20.05, 27.99), (54.87, 38.04), (81.28, 82.72), (65.31, 41.42), (79.92, 81.70), (21.40, 16.18), (63.01, 41.82), (27.89, 21.79), (74.68, 77.73)]
		
  
		# 遍历地图的每个点
		for x in range(self.scenario_map.shape[0]):
			for y in range(self.scenario_map.shape[1]):
				# 对于每个点，计算所有用户的信号强度并累加
				total_signal = 0
				for user_x, user_y in self.users:
					distance = np.sqrt((user_x - x) ** 2 + (user_y - y) ** 2)
					if distance != 0 and distance<=10: # 避免除以零
						signal_strength = self.P / (distance** 0.5)#1.5
						 # 添加高斯噪音
						signal_strength_with_noise = self.add_gaussian_noise(signal_strength)
                        
						total_signal += signal_strength_with_noise
				
				# 将累加的信号强度赋值给 signal_map
				self.signal_map[x, y] = total_signal
		for x in range(self.scenario_map.shape[0]):
			for y in range(self.scenario_map.shape[1]):
				# self.signal_map[x, y] = total_signal
				self.signal_map[x, y] = (self.signal_map[x, y] - self.signal_map.min()) / (self.signal_map.max() - self.signal_map.min())
		
  		# self.signal_map = gaussian_filter(self.signal_map, 0.8)
	
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
	def render(self):

		f_map = self.signal_map
		f_map[self.grid == 0] = np.nan

		if self.fig is None:
			# current = self.current_field_fn(self.visitable_positions)
			self.fig, self.ax = plt.subplots(1,1)
			# self.ax.quiver(self.visitable_positions[::6,1], self.visitable_positions[::6,0], current[::6,1], -current[::6,0], color='black', alpha = 0.25)
			self.d = self.ax.imshow(f_map, cmap = people_colormap,)# vmin=0.0, vmax = 1.0

			background = self.grid.copy()
			background[background == 1] = np.nan
			self.ax.imshow(background, cmap=background_colormap)

		else:
			self.d.set_data(f_map)

		self.fig.canvas.draw()
		# self.fig.savefig('updated_plot.png')
		plt.pause(0.01)

def save_gif(self, filename, frames):
    ani = animation.ArtistAnimation(self.fig, frames, interval=100, blit=True)
    ani.save(filename, writer='imagemagick')
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    gt = GroundTruth(np.genfromtxt('Environment/Maps/example_map copy.csv'))
#, delimiter=','
    m = gt.reset()
    gt.render()
    frames = []
    for _ in range(50):#50
        gt.reset()
        for t in range(150):# 150
            m = gt.step()
            gt.render()
            
            
            
        