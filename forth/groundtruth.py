import sys
sys.path.append('.')
import math
from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt
import time
# import numpy as np

class GroundTruth:
	""" This is a template for a ground truth. Every ground truth must inherit and implement all of its methods and variables. """

	# sim_config_template = {
	# 	"seed": 0,
	# }

	def __init__(self, scenario_map: np.ndarray):

		# self.seed = sim_config['seed']
		# self.ground_truth_field = None
		self.signal_map = np.zeros_like(scenario_map)
		self.scenario_map = scenario_map
		# user 15
		# users=[(28.095737508379507, 34.26223818137524),
		# 		(31.534268858415665, 36.96599911650416),
		# 		(28.987847366016233, 39.938933089562966),
		# 		(32.11286120677997, 37.201298108043844),
		# 		(26.765425070521744, 34.1644206620949),
		# 		(45.749778214291226, 22.612149557436634),
		# 		(45.396274248005305, 20.614792298358264),
		# 		(50.642615825605006, 19.853023807082725),
		# 		(48.26626883173278, 15.509488390034488),
		# 		(54.425094652367605, 21.80724520028852),
		# 		(68.71555847946225, 41.95251215340648),
		# 		(69.30532824493405, 43.8309421065234),
		# 		(68.2989629979505, 42.05649377261501),
		# 		(73.96507288663165, 46.543965847603985),
		# 		(69.0837707884811, 48.828827829773275)]
		
  
  		# #user 50 v2·
		users=[
		(18.27, 18.65),
		(79.89, 76.70),
		(26.85, 26.57),
		(74.83, 83.49),
		(61.68, 35.08),
		(56.69, 46.90),
		(20.20, 19.16),
		(82.34, 80.40),
		(79.39, 85.08),
		(23.87, 16.68),
		(59.69, 47.66),
		(74.68, 77.73),
		(80.44, 82.61),
		(76.91, 86.87),
		(20.69, 21.69),
		(19.87, 21.92),
		(16.11, 17.41),
		(17.72, 18.04),
		(64.12, 41.43),
		(23.40, 20.54),
		(28.35, 15.63),
		(68.42, 33.97),
		(70.92, 89.23),
		(76.36, 73.44),
		(61.33, 45.93),
		(79.92, 81.70),
		(16.99, 18.39),
		(62.34, 39.23),
		(84.07, 81.42),
		(81.91, 83.03),
		(15.31, 13.65),
		(81.78, 67.60),
		(18.61, 20.25),
		(57.12, 30.06),
		(17.72, 14.72),
		(83.09, 83.11),
		(75.72, 78.45),
		(83.52, 77.91),
		(15.21, 24.72),
		(56.67, 36.30),
		(77.33, 79.76),
		(86.68, 70.74),
		(57.96, 40.08),
		(21.40, 16.18),
		(59.82, 34.78),
		(27.89, 21.79),
		(54.22, 41.19),
		(22.08, 13.13),
		(25.27, 19.20),
		(56.15, 48.54),
				]
		# users=[
		# (17.26, 17.86),
		# (79.83, 74.78),
		# (30.84, 30.39),
		# (71.82, 85.52),
		# (62.66, 32.23),
		# (54.77, 50.91),
		# (20.32, 18.68),
		# (83.70, 80.63),
		# (79.04, 88.04),
		# (26.11, 14.75),
		# (59.52, 52.12),
		# (71.59, 76.42),
		# (80.69, 84.12),
		# (75.11, 90.86),
		# (21.10, 22.67),
		# (19.80, 23.03),
		# (13.84, 15.91),
		# (16.40, 16.90),
		# (66.51, 42.25),
		# (25.38, 20.86),
		# (33.21, 13.09),
		# (73.32, 30.47),
		# (65.64, 94.60),
		# (74.25, 69.63),
		# (62.11, 49.38),
		# (79.88, 82.68),
		# (15.24, 17.46),
		# (63.70, 38.79),
		# (86.44, 82.24),
		# (83.01, 84.79),
		# (12.59, 9.96),
		# (82.82, 60.39),
		# (17.80, 20.40),
		# (55.45, 24.28),
		# (16.39, 11.65),
		# (84.89, 84.91),
		# (73.24, 77.55),
		# (85.56, 76.70),
		# (12.43, 27.46),
		# (54.73, 34.16),
		# (75.78, 79.63),
		# (90.57, 65.36),
		# (56.78, 40.12),
		# (22.21, 13.96),
		# (59.72, 31.74),
		# (32.47, 22.83),
		# (50.86, 41.89),
		# (23.29, 9.14),
		# (28.34, 18.73),
		# (53.91, 53.51)
		# ]
		# 假设 scenario_map 和 P 已经定义
		P = 1  # 信号强度系数，示例值

		# 遍历地图的每个点
		for x in range(scenario_map.shape[0]):
			for y in range(scenario_map.shape[1]):
				# 对于每个点，计算所有用户的信号强度并累加
				total_signal = 0
				for user_x, user_y in users:
					distance = np.sqrt((user_x - x) ** 2 + (user_y - y) ** 2)
					if distance != 0:  # 避免除以零
						signal_strength = P / (distance ** 1.5)#2.5
						total_signal += signal_strength
				
				# 将累加的信号强度赋值给 signal_map
				self.signal_map[x, y] = total_signal

		
		
	# def step(self):
	# 	raise NotImplementedError('This method has not being implemented yet.')

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



