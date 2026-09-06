import copy
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel
from scipy.spatial import distance_matrix
import multiprocessing as mp
import time


def _samples(x, y):
    """Normalize one scalar observation per 2D position, including empty batches."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    if x.size == 0:
        x = np.empty((0, 2), dtype=float)
    else:
        x = np.atleast_2d(x)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) != len(y):
        raise ValueError("Expected positions (N, 2) and one measurement per position")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Positions and measurements must be finite")
    return x, y


def _latest_samples(x, y):
    """Keep the most recent reading at each location in a changing signal field."""
    _, reverse_index = np.unique(x[::-1], axis=0, return_index=True)
    index = np.sort(len(x) - 1 - reverse_index)
    return x[index], y[index]


class BaseLocalGaussianProcess:
	
	""" Local Gaussian Process Regression Model """
	
	def __init__(self, position: np.ndarray, global_X: np.ndarray, local_index: np.ndarray, distance_threshold: float, kernel, alpha=1e-10, n_restarts_optimizer=0,):
		
		self.x = None
		self.y = None
		self.kernel = kernel
		self.alpha = alpha
		self.n_restarts_optimizer = n_restarts_optimizer
		self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha, n_restarts_optimizer=self.n_restarts_optimizer,normalize_y=False)
		self.position = position # The position of the local GP model

		# Every position of the scenario map is stored in the global_X #
		self.global_X = global_X
		# The local_X is the subset of the global_X that is close to the local GP model #
		self.local_X = global_X[local_index]
		# The local_index is the index of the local_X in the global_X #
		self.local_index = local_index

		self.local_mu = np.zeros(self.local_X.shape[0])
		self.local_sigma = np.sqrt(self.kernel.diag(self.local_X))

		self.distance_threshold = distance_threshold
		
		self.global_mu_map = np.zeros(self.global_X.shape[0])
		self.global_sigma_map = np.sqrt(self.kernel.diag(self.global_X))

		self.change_mu = 0
		self.change_sigma = 0
	
	def update(self, x, y):
		""" Concat the new sample to the old data """

		# Get the index of the new data that is far away from the local GP model #
		# The distance is computed using the L2 norm #
		x, y = _samples(x, y)
		self.change_mu = self.change_sigma = 0.0
		distance = np.linalg.norm(x - self.position, axis=1)
		index = np.where(distance <= self.distance_threshold)[0]
		#一堆位置
		#只更新局部的
		if index.size == 0:
			return self.global_mu_map, self.global_sigma_map
		else:
			x = x[index]
			y = y[index]
		
		if self.x is None:
			# If the local GP model is empty, then initialize the data #
			self.x = np.atleast_2d(x)#确保是二维的
			self.y = np.atleast_2d(y)
		else:
			
			self.x = np.vstack((self.x, x))
			self.y = np.vstack((self.y, y))

		# Erase the repeated data #
		self.x, self.y = _latest_samples(self.x, self.y)
		
		# Fit the new data #
		# If the model has been trained, then use the length scale of the previous model #
		self.gp.fit(self.x, self.y.flatten())

		# Copy the old mu and sigma #
		old_mu = self.local_mu.copy()
		old_sigma = self.local_sigma.copy()

		# Predict the new mu and sigma #
		self.local_mu, self.local_sigma = self.gp.predict(self.local_X, return_std=True)
		# 不是一个值，是一对点的预测值
  
		# Compute the change of mu and sigma #
		self.change_mu = np.abs(np.linalg.norm(self.local_mu - old_mu))
		self.change_sigma = np.abs(np.linalg.norm(self.local_sigma - old_sigma))

		# Update the global mu and sigma #
		self.global_mu_map[self.local_index] = self.local_mu
		self.global_sigma_map[self.local_index] = self.local_sigma
		
		return self.global_mu_map, self.global_sigma_map
	
	def reset(self):
		""" Reset the local GP model """
		
		self.x = None
		self.y = None
		self.local_mu = np.zeros_like(self.local_mu)
		self.local_sigma = np.sqrt(self.kernel.diag(self.local_X))
		self.global_mu_map = np.zeros_like(self.global_mu_map)
		self.global_sigma_map = np.sqrt(self.kernel.diag(self.global_X))
		self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha, n_restarts_optimizer=self.n_restarts_optimizer, normalize_y=False)
		self.change_mu = 0
		self.change_sigma = 0



class LocalGaussianProcessCoordinator:
	
	""" A coordinator for local Gaussian Process Regression Models.
	It receives the new data and give every local GP model the points depending on the distance between the new data and the local GP model.
	"""
	
	def __init__(self, gp_positions: np.ndarray, scenario_map:np.ndarray, kernel, alpha=1e-10, n_restarts_optimizer=0, distance_threshold=10):
		
		self.gp_positions = np.asarray(gp_positions, dtype=float)
		if self.gp_positions.ndim != 2 or self.gp_positions.shape[1] != 2 or len(self.gp_positions) == 0 or not np.isfinite(self.gp_positions).all():
			raise ValueError("At least one finite GP position (K, 2) is required")
		if not np.isfinite(distance_threshold) or distance_threshold <= 0:
			raise ValueError("distance_threshold must be positive and finite")

		# Select those positions of scenario map that are 1 using numpy #
		self.X = np.asarray(np.where(scenario_map == 1)).T
		
		""" Create a list to store the data """
		self.y = None
		self.x = None
		self.scenario_map = scenario_map
		self.distance_threshold  = distance_threshold # 半径
		self.mu_map = np.zeros_like(self.scenario_map, dtype=float)
		self.sigma_map = np.ones_like(self.scenario_map, dtype=float)

		""" Create N local GP models """
		self.gp_models = [BaseLocalGaussianProcess(position, self.X, self.compute_local_index(self.X, position), self.distance_threshold, kernel, alpha, n_restarts_optimizer) for position in gp_positions]
		self.gp_models_position_index = np.arange(len(self.gp_models))
		self.number_of_locals = len(self.gp_models)

		# Precompute  the distance matrix #
		self.distance_matrix_for_points = distance_matrix(self.X, self.gp_positions, p=2)

		# If a coeficient is larger than distance_threshold, then set it to inf #
		#self.distance_matrix_for_points[self.distance_matrix_for_points > self.distance_threshold] = np.inf

		# Compute the weight matrix - The size is (number_of_points, number_of_GP_models) #
		# Only blend experts that actually predict at a point. Subtract the
		# nearest distance before exponentiating to avoid underflow.
		support = np.asarray([gp.local_index for gp in self.gp_models]).T
		log_weight = np.where(support, -self.distance_matrix_for_points, -np.inf)
		self.covered = support.any(axis=1)
		nearest = np.max(log_weight, axis=1, keepdims=True)
		nearest[~self.covered] = 0.0
		weights = np.exp(log_weight - nearest)
		total = weights.sum(axis=1, keepdims=True)
		self.weight = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0).T
		self.prior_sigma = np.sqrt(kernel.diag(self.X))
		self.sigma_map[self.X[:, 0], self.X[:, 1]] = self.prior_sigma

		# MxN
  
		# To store the changes of the local GP models #
		self.changes = np.zeros(len(self.gp_models))
		self.changes_mu_map = np.zeros_like(self.scenario_map, dtype=float)
		self.changes_sigma_map = np.zeros_like(self.scenario_map, dtype=float)

	def compute_local_index(self, X: np.ndarray, position:np.ndarray):
		""" Compute the local positions X_local from the all positions X and the local GP model position """

		# X_local is the positions of the scenario that are closer than distance_threshold to the local GP model #
		indexes = np.linalg.norm(X - position, axis=1) <= self.distance_threshold*2

		return indexes

	@staticmethod
	def get_distance(position1, position2):
		return np.linalg.norm(position1 - position2, ord=2)
	
	def get_nearest_gp_index(self, position):# 得到最近的GP model
		""" Return the nearest GP model to the new data """
		
		distance = [self.get_distance(position, self.gp_models[i].position) for i in range(len(self.gp_models))]
		nearest_gp_index = np.argmin(distance)
		
		return nearest_gp_index
	
	def generate_mean_map(self):
		""" Generate a map to show the points that are nearest to the local GP models """
		# 生成一个展示最接近本地GP模型点的地图。这涉及到将不同模型的预测结果结合起来，以生成整体的预测。
		fussed_mu = np.zeros(self.X.shape[0])
		fussed_std = np.zeros(self.X.shape[0])

		for index, gp_model in enumerate(self.gp_models):
			fussed_mu += self.weight[index] * gp_model.global_mu_map # 附加一个权重，离得越近，权重越高
			fussed_std += self.weight[index] * gp_model.global_sigma_map
		#把所有高斯模型的mu std都加起来
		fussed_std[~self.covered] = self.prior_sigma[~self.covered]
		self.mu_map[self.X[:, 0], self.X[:, 1]] = fussed_mu

		self.sigma_map[self.X[:, 0], self.X[:, 1]] = fussed_std
	
		return self.mu_map, self.sigma_map
	
	def update(self, x, y):
		""" Update the local GP models with the new data that is nearest to the local GP model """
		
		x, y = _samples(x, y)
	
		# Update the local GP models in parallel using joblib #
		#Parallel(n_jobs=16, require='sharedmem')(delayed(gp_model.update)(x, y) for gp_model in self.gp_models)
		[gp_model.update(x, y) for gp_model in self.gp_models]

		# Concat the new data to the old data #
		if self.x is None:
			self.x = np.atleast_2d(x)
			self.y = np.atleast_2d(y)
		else:
			self.x = np.vstack((self.x, x))
			self.y = np.vstack((self.y, y))

		old_mu_map = self.mu_map.copy()
		old_sigma_map = self.sigma_map.copy()
		
		self.mu_map, self.sigma_map = self.generate_mean_map()#实际上对mu,sigma都进行了更新
		#多合一		

		# Store the changes of the local GP models #
		self.changes_mu_map = np.abs(self.mu_map - old_mu_map)
		self.changes_sigma_map = (self.sigma_map - old_sigma_map) / (1 + old_sigma_map)
										   
		return self.mu_map, self.sigma_map
	

	def reset(self):
		""" Reset the local GP models """
		
		self.x = None
		self.y = None
		self.mu_map = np.zeros_like(self.scenario_map, dtype=float)
		self.sigma_map = np.ones_like(self.scenario_map, dtype=float)
		
		self.sigma_map[self.X[:, 0], self.X[:, 1]] = self.prior_sigma
		self.changes_mu_map.fill(0)
		self.changes_sigma_map.fill(0)
		self.changes.fill(0)
		for gp_model in self.gp_models:
			gp_model.reset()

	def get_local_gp_indexes(self, position):
		""" Return the indexes of the local GP models that are closer than distance_threshold to the new data """

		# Compute the distance between the new data and the local GP models #
		nearest_gp_indexes = np.array([j for j in range(len(self.gp_models)) if self.get_distance(position, self.gp_models[j].position) <= self.distance_threshold])
		# 较近的一堆局部gp，小于半径，挺近的
		return nearest_gp_indexes
	
	def get_local_gp_changes(self, index = None):
		""" Return the changes in the local GP model """
		
		if index is None:
			np.array([gp_model.change_mu for gp_model in self.gp_models])
		else:
			return np.array([self.gp_models[i].change_mu for i in index])
		
	def get_changes_map(self):
		""" Return the mean changes in the surrounding of the position  """

		return self.changes_mu_map, self.changes_sigma_map
	
	def get_changes(self):
		""" Return the changes in the local GP models """
		
		return self.changes_mu_map[self.X[:, 0], self.X[:, 1]], self.changes_sigma_map[self.X[:, 0], self.X[:, 1]]
	
	def get_kernel_params(self):
		""" Return the kernel parameters """
		
		params = []
		for gp_model in self.gp_models:
			# Check if gpmodel has kernel or kernel_
			if hasattr(gp_model.gp, 'kernel_'):
				params.append(gp_model.gp.kernel_.get_params()['k1__k2__length_scale'])
			else:
				params.append(gp_model.gp.kernel.get_params()['k1__k2__length_scale'])
		
		return params
				
	
	
		
class GlobalGaussianProcessCoordinator:

		
	def __init__(self, scenario_map:np.ndarray, kernel, alpha=1e-10, n_restarts_optimizer=0, distance_threshold=10):

		# Select those positions of scenario map that are 1 using numpy #
		self.X = np.asarray(np.where(scenario_map == 1)).T
		
		""" Create N local GP models """
		self.gp_positions = np.array([[scenario_map.shape[0] // 2, scenario_map.shape[1] // 2]])
		self.gp_models = BaseLocalGaussianProcess(self.gp_positions[0], self.X, np.ones(self.X.shape[0]).astype(bool), 999999, kernel, alpha, n_restarts_optimizer)
		
		
		""" Create a list to store the data """
		self.y = None
		self.x = None
		self.scenario_map = scenario_map
		self.distance_threshold  = distance_threshold
		self.mu_map = np.zeros_like(self.scenario_map, dtype=float)
		self.sigma_map = np.ones_like(self.scenario_map, dtype=float)
  
		self.changes_mu_map = np.zeros_like(self.scenario_map, dtype=float)
		self.changes_sigma_map = np.zeros_like(self.scenario_map, dtype=float)
		# To store the changes of the local GP models #
		self.changes = 0
		self.mu, self.sigma = self.generate_nearest_map()
		self.sigma_map[self.X[:, 0], self.X[:, 1]] = self.sigma
			
	@staticmethod
	def get_distance(position1, position2):
		return np.linalg.norm(position1 - position2, ord=2)
	
	def get_nearest_gp_index(self, position):
		""" Return the nearest GP model to the new data """
		return 0
	
	def generate_nearest_map(self):
		"""Return the single global expert's mean and standard deviation."""
		return self.gp_models.global_mu_map.copy(), self.gp_models.global_sigma_map.copy()
	

	def update(self, x, y):
		""" Update the local GP models with the new data that is nearest to the local GP model """
		
		x, y = _samples(x, y)

		old_mu = self.gp_models.global_mu_map.copy()
		self.gp_models.update(x, y)
		self.changes = np.linalg.norm(self.gp_models.global_mu_map - old_mu, ord=2)
			
		# Concat the new data to the old data #
		if self.x is None:
			self.x = np.atleast_2d(x)
			self.y = np.atleast_2d(y)
		else:
			self.x = np.vstack((self.x, x))
			self.y = np.vstack((self.y, y))
			
		# Erase the repeated data #
		self.x, self.y = _latest_samples(self.x, self.y)
		
		# Compose the mu using the local GP models mu weighted by the uncertainty #
		self.mu = self.gp_models.global_mu_map
		self.sigma = self.gp_models.global_sigma_map
		
		
		
		# old_sigma_map=self.sigma_map
		old_mu_map = copy.deepcopy(self.mu_map)	
		old_sigma_map = copy.deepcopy(self.sigma_map)		
  
  		# Put every mu and sigma in a map #
		self.mu_map[self.X[:,0], self.X[:,1]] = self.mu
		self.sigma_map[self.X[:,0], self.X[:,1]] = self.sigma
				
		# Store the changes of the local GP models #
		self.changes_mu_map = np.abs(self.mu_map - old_mu_map)
		self.changes_sigma_map = (self.sigma_map - old_sigma_map) / (1 + old_sigma_map)
		
  
		return self.mu_map, self.sigma_map
	
	def reset(self):
		""" Reset the local GP models """
		
		self.x = None
		self.y = None
		self.mu_map = np.zeros_like(self.scenario_map, dtype=float)
		self.sigma_map = np.ones_like(self.scenario_map, dtype=float)
		
		self.gp_models.reset()
		self.mu, self.sigma = self.generate_nearest_map()
		self.sigma_map[self.X[:, 0], self.X[:, 1]] = self.sigma
		self.changes_mu_map.fill(0)
		self.changes_sigma_map.fill(0)
		self.changes = 0.0

	def get_local_gp_index(self, position):
		""" Return the index of the local GP model that is nearest to the position """
		
		return 0
	
	def get_local_gp_changes(self):
		""" Return the changes in the local GP model """
		
		return self.changes

	def get_kernel_params(self):
		
		params = []

		# Check if gpmodel has kernel or kernel_
		if hasattr(self.gp_models.gp, 'kernel_'):
			params.append(self.gp_models.gp.kernel_.get_params()['k1__k2__length_scale'])
		else:
			params.append(self.gp_models.gp.kernel.get_params()['k1__k2__length_scale'])
	
		return params

	def get_changes(self):
		""" Return the changes in the local GP models """
		
		return self.changes_mu_map[self.X[:, 0], self.X[:, 1]], self.changes_sigma_map[self.X[:, 0], self.X[:, 1]]
	
		
	