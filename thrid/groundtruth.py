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
		users=[(28.095737508379507, 34.26223818137524),
				(31.534268858415665, 36.96599911650416),
				(28.987847366016233, 39.938933089562966),
				(32.11286120677997, 37.201298108043844),
				(26.765425070521744, 34.1644206620949),
				(45.749778214291226, 22.612149557436634),
				(45.396274248005305, 20.614792298358264),
				(50.642615825605006, 19.853023807082725),
				(48.26626883173278, 15.509488390034488),
				(54.425094652367605, 21.80724520028852),
				(68.71555847946225, 41.95251215340648),
				(69.30532824493405, 43.8309421065234),
				(68.2989629979505, 42.05649377261501),
				(73.96507288663165, 46.543965847603985),
				(69.0837707884811, 48.828827829773275)]
		

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
						signal_strength = P / (distance ** 2.5)
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



