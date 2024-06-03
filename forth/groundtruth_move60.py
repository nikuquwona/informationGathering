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
		self.vrms = 1.0
		self.signal_map = np.zeros_like(scenario_map)
		self.scenario_map = scenario_map
		# self.users=[(77.57, 81.86), (20.20, 19.16), (26.85, 26.57), (17.72, 18.04), (56.67, 36.30), (20.69, 21.69), (23.87, 16.68), (56.15, 48.54), (84.07, 81.42), (18.61, 20.25), (63.45, 43.68), (88.75, 81.74), (79.39, 85.08), (18.27, 18.65), (16.99, 18.39), (59.69, 47.66), (55.53, 33.09), (69.67, 45.98), (86.68, 70.74), (54.22, 41.19), (19.87, 21.92), (62.34, 39.23), (23.40, 20.54), (28.35, 15.63), (74.16, 87.42), (61.68, 35.08), (81.78, 67.60), (16.11, 17.41), (70.92, 89.23), (76.76, 73.81), (80.63, 78.57), (77.33, 78.94), (59.82, 34.78), (80.44, 82.61), (15.31, 13.65), (55.06, 40.23), (62.58, 39.07), (90.30, 75.26), (17.72, 14.72), (15.21, 24.72), (61.33, 45.93), (77.93, 82.15), (20.05, 27.99), (54.87, 38.04), (81.28, 82.72), (65.31, 41.42), (79.92, 81.70), (21.40, 16.18), (63.01, 41.82), (27.89, 21.79), (74.68, 77.73)]
		self.users=[(20.93, 24.37), (83.46, 74.71), (16.00, 21.73), (70.92, 89.23), (28.35, 24.05), (63.83, 37.09), (12.78, 19.05), (83.09, 83.11), (57.33, 38.94), (24.24, 19.31), (26.68, 19.08), (61.77, 35.11), (71.76, 77.86), (59.56, 37.03), (78.03, 78.75), (78.28, 80.42), (57.12, 30.06), (83.08, 85.82), (16.36, 12.28), (63.45, 43.68), (62.80, 32.84), (78.21, 81.65), (15.94, 20.23), (72.15, 82.02), (59.82, 34.78), (23.87, 16.68), (79.81, 78.72), (82.21, 79.48), (77.08, 78.25), (23.26, 20.58), (79.51, 84.56), (22.08, 13.13), (54.52, 43.78), (53.33, 41.97), (82.33, 77.42), (19.87, 21.92), (18.80, 23.59), (79.72, 79.52), (57.16, 43.03), (61.33, 45.93), (19.54, 21.84), (84.78, 77.97), (26.85, 26.57), (54.22, 41.19), (16.15, 28.54), (15.21, 24.72), (63.52, 37.91), (18.44, 20.70), (65.04, 35.17), (74.68, 77.73), (64.92, 42.93), (83.34, 74.68), (61.68, 35.08), (88.75, 81.74), (82.59, 81.56), (78.37, 80.70), (57.77, 48.63), (77.15, 78.22), (81.78, 67.60), (77.08, 77.67)]

		self.grid = scenario_map
		self.fig = None
		self.visitable_positions = np.column_stack(np.where(scenario_map == 1))
  
		# 假设 scenario_map 和 P 已经定义
		self.P = 1  # 信号强度系数，示例值

		# 遍历地图的每个点
		for x in range(scenario_map.shape[0]):
			for y in range(scenario_map.shape[1]):
				# 对于每个点，计算所有用户的信号强度并累加
				total_signal = 0
				for user_x, user_y in self.users:
					distance = np.sqrt((user_x - x) ** 2 + (user_y - y) ** 2)
					if distance != 0 and distance<=10:  # 避免除以零
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
  		
	def add_gaussian_noise(self, signal_strength):
		mean = 0
		std_dev = 0.1  # 调整噪音的标准差以控制噪音的大小
		noise = np.random.normal(mean, std_dev)
		return signal_strength
		# return signal_strength + noise

	def step(self):
		# updated_users = []
		# for i in range(len(self.users)):
		# 	user_x, user_y = self.users[i]
		# 	# 生成速度数据
		# 	Vx = np.random.normal(0, self.vrms)
		# 	Vy = np.random.normal(0, self.vrms)
		# 	# 根据速度和时间更新位置
		# 	user_x += Vx * 0.5
		# 	user_y += Vy * 0.5
		# 	# user_x += Vx * 1
		# 	# user_y += Vy * 1

		# 	# 边界检测
		# 	user_x = max(2, min(user_x, 97))
		# 	user_y = max(2, min(user_y, 97))
		# 	updated_users.append((user_x, user_y))
		# self.users=updated_users
  
  
		# # 遍历地图的每个点
		# for x in range(self.scenario_map.shape[0]):
		# 	for y in range(self.scenario_map.shape[1]):
		# 		# 对于每个点，计算所有用户的信号强度并累加
		# 		total_signal = 0
		# 		for user_x, user_y in self.users:
		# 			distance = np.sqrt((user_x - x) ** 2 + (user_y - y) ** 2)
		# 			if distance != 0 and distance<=10:  # 避免除以零
		# 				signal_strength = self.P / (distance** 0.5)#1.5
		# 				 # 添加高斯噪音
		# 				signal_strength_with_noise = self.add_gaussian_noise(signal_strength)
                        
		# 				total_signal += signal_strength_with_noise
				
		# 		# 将累加的信号强度赋值给 signal_map
		# 		self.signal_map[x, y] = total_signal
		# for x in range(self.scenario_map.shape[0]):
		# 	for y in range(self.scenario_map.shape[1]):
		# 		# self.signal_map[x, y] = total_signal
		# 		self.signal_map[x, y] = (self.signal_map[x, y] - self.signal_map.min()) / (self.signal_map.max() - self.signal_map.min())
		
  		# # self.signal_map = gaussian_filter(self.signal_map, 0.8)
		return
	

	# 	raise NotImplementedError('This method has not being implemented yet.')
	def reset(self):
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
            
            
            
        