import sys
from thrid.GPmodel import GlobalGaussianProcessCoordinator, LocalGaussianProcessCoordinator

from thrid.groundtruth import GroundTruth
sys.path.append('.')
import numpy as np
# from Environment.GroundTruthsModels.AlgaeBloomGroundTruth import algae_bloom, algae_colormap
# from Environment.GroundTruthsModels.ShekelGroundTruth import shekel
# from Environment.GroundTruthsModels.NewFireFront import WildFiresSimulator
# from GPModel.GPmodel import LocalGaussianProcessCoordinator, GlobalGaussianProcessCoordinator
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
import gym
from scipy.spatial import distance_matrix
# from Environment.Wrappers.time_stacking_wrapper import MultiAgentTimeStackingMemory
import matplotlib.pyplot as plt


class DiscreteVehicle:

	def __init__(self, initial_position, n_actions, movement_length, navigation_map):
		
		""" Initial positions of the drones """
		np.random.seed(0)
		self.initial_position = initial_position
		self.position = np.copy(initial_position)

		""" Initialize the waypoints """
		self.waypoints = np.expand_dims(np.copy(initial_position), 0)

		""" Detection radius for the contmaination vision """
		self.navigation_map = navigation_map

		""" Reset other variables """
		self.distance = 0.0
		self.num_of_collisions = 0
		self.action_space = gym.spaces.Discrete(n_actions)
		self.angle_set = np.linspace(0, 2 * np.pi, n_actions, endpoint=False)
		self.movement_length = movement_length

		

	def move(self, action, valid=True):
		""" Move a vehicle in the direction of the action. If valid is False, the action is not performed. """

		angle = self.angle_set[action]
		movement = (np.round(np.array([np.cos(angle), np.sin(angle)])) * self.movement_length).astype(int)
		next_position = self.position + movement
		self.distance += np.linalg.norm(self.position - next_position)

		if self.check_collision(next_position) or not valid:
			collide = True
			self.num_of_collisions += 1
		else:
			collide = False
			self.position = next_position
			self.waypoints = np.vstack((self.waypoints, [self.position]))

		return collide

	def check_collision(self, next_position):

		if next_position[0] < 0 or next_position[0] >= self.navigation_map.shape[0] or next_position[1] < 0 or next_position[1] >= self.navigation_map.shape[1]:
			return True

		if self.navigation_map[int(next_position[0]), int(next_position[1])] == 0:
			return True  # There is a collision
		
		return False

	def reset(self, initial_position):
		""" Reset the agent - Position, detection mask, etc. """

		self.initial_position = initial_position
		self.position = np.copy(initial_position)
		self.waypoints = np.expand_dims(np.copy(initial_position), 0)
		self.distance = 0.0
		self.num_of_collisions = 0

	def check_action(self, action):
		""" Return True if the action leads to a collision """

		angle = self.angle_set[action]
		movement = (np.round(np.array([np.cos(angle), np.sin(angle)])) * self.movement_length).astype(int)
		next_position = self.position + movement

		return self.check_collision(next_position)

	def move_to_position(self, goal_position):
		""" Move to the given position """

		assert self.navigation_map[goal_position[0], goal_position[1]] == 1, "Invalid position to move"
		self.distance += np.linalg.norm(goal_position - self.position)
		""" Update the position """
		self.position = goal_position


class DiscreteFleet:

	def __init__(self,
				 number_of_vehicles,
				 n_actions,
				 fleet_initial_positions,
				 movement_length,
				 navigation_map):

		""" Coordinator of the movements of the fleet. Coordinates the common model, the distance between drones, etc. """
		np.random.seed(0)
		self.number_of_vehicles = number_of_vehicles
		self.initial_positions = fleet_initial_positions
		self.n_actions = n_actions
		self.movement_length = movement_length

		""" Create the vehicles object array """
		self.vehicles = [DiscreteVehicle(initial_position=fleet_initial_positions[k],
										 n_actions=n_actions,
										 movement_length=movement_length,
										 navigation_map=navigation_map) for k in range(self.number_of_vehicles)]

		self.agent_positions = np.asarray([veh.position for veh in self.vehicles])

		# Reset model variables 
		self.measured_values = None
		self.measured_locations = None

		self.fleet_collisions = 0
		self.danger_of_isolation = None


	@staticmethod
	def majority(arr: np.ndarray) -> bool:
		return arr.sum() >= len(arr) // 2

	def check_fleet_collision_within(self, veh_actions):
		""" Check if there is any collision between agents """
		
		new_positions = []

		for idx, veh_action in veh_actions.items():

			angle = self.vehicles[idx].angle_set[veh_action]
			movement = (np.round(np.array([np.cos(angle), np.sin(angle)])) * self.vehicles[idx].movement_length).astype(int)
			new_positions.append(list(self.vehicles[idx].position + movement))

		_, inverse_index, counts = np.unique(np.asarray(new_positions), return_inverse=True, return_counts=True, axis=0)

		# True if repeated #
		not_collision_within = counts[inverse_index] == 1

		return not_collision_within

	def move(self, fleet_actions):

		# Check if there are collisions between vehicles #
		# 计算新位置，如果同一位置出现多次就认为碰撞
  		# self_colliding_mask 是一个布尔数组
		self_colliding_mask = self.check_fleet_collision_within(fleet_actions)
  
		# Process the fleet actions and move the vehicles #
		# 进行移动，如果会碰撞就不移动，collision_array代表每个智能体是否碰撞了
		collision_array = {k: self.vehicles[k].move(fleet_actions[k], valid=valid) for k, valid in zip(list(fleet_actions.keys()), self_colliding_mask)}
		
  		# Update vector with agent positions #
    	# 更新新智能体的位置	
		self.agent_positions = np.asarray([veh.position for veh in self.vehicles])
		
    	# 统计总碰撞次数，计算done的时候要用
		self.fleet_collisions = np.sum([self.vehicles[k].num_of_collisions for k in range(self.number_of_vehicles)])

		return collision_array

	# def measure(self, gt_field):

	# 	"""
	# 	Take a measurement in the given N positions
	# 	:param gt_field:
	# 	:return: An numpy array with dims (N,2)
	# 	"""
	# 	positions = np.array([self.vehicles[k].position for k in range(self.number_of_vehicles)])

	# 	values = []
	# 	for pos in positions:
	# 		values.append([gt_field[int(pos[0]), int(pos[1])]])

	# 	if self.measured_locations is None:
	# 		self.measured_locations = positions
	# 		self.measured_values = values
	# 	else:
	# 		self.measured_locations = np.vstack((self.measured_locations, positions))
	# 		self.measured_values = np.vstack((self.measured_values, values))

	# 	return self.measured_values, self.measured_locations

	def reset(self, initial_positions=None):
		""" Reset the fleet """

		if initial_positions is None:
			initial_positions = self.initial_positions

		for k in range(self.number_of_vehicles):
			self.vehicles[k].reset(initial_position=initial_positions[k])

		self.agent_positions = np.asarray([veh.position for veh in self.vehicles])
		self.measured_values = None
		self.measured_locations = None
		self.fleet_collisions = 0
		
	def get_distances(self):
		return np.array([self.vehicles[k].distance for k in range(self.number_of_vehicles)])

	def check_collisions(self, test_actions):
		""" Array of bools (True if collision) """
		return [self.vehicles[k].check_action(test_actions[k]) for k in range(self.number_of_vehicles)]

	def move_fleet_to_positions(self, goal_list):
		""" Move the fleet to the given positions.
		 All goal positions must ve valid. """

		goal_list = np.atleast_2d(goal_list)

		for k in range(self.number_of_vehicles):
			self.vehicles[k].move_to_position(goal_position=goal_list[k])

	def get_distance_matrix(self):
		return distance_matrix(self.agent_positions, self.agent_positions)

	def get_positions(self):

		return np.asarray([veh.position for veh in self.vehicles])
	

class MultiagentInformationGathering:
	
	def __init__(self,
				 scenario_map: np.ndarray,
				 number_of_agents: int,
				 distance_between_locals: float,
				 radius_of_locals: float,
				 distance_budget: float,
				 distance_between_agents: float,
				 fleet_initial_positions = None,
				 fleet_initial_zones = None,
				 seed: int = 0,
				 movement_length: int = 1,
				 max_collisions: int = 1,
				#  ground_truth_type: str = 'algae_blooms',
				 frame_stacking = 0,
				 state_index_stacking = (0,1,2,3,4),
				 local = True,
				 reward_type = 'changes_mu',
				 ):
		
		""" 
		
		:param scenario_map: A numpy array with the scenario map. 1 is a valid position, 0 is an invalid position.
		:param number_of_agents: Number of agents in the fleet
		:param distance_between_locals: Distance between local GPs
		:param radius_of_locals: Radius of the local GPs
		:param distance_budget: Distance budget for the fleet
		:param distance_between_agents: Distance between agents
		:param fleet_initial_positions: Initial positions of the fleet. If None, random positions are chosen.
		:param seed: Seed for the random number generator
		:param movement_length: Length of every movement of the agents
		:param max_collisions: Maximum number of collisions allowed
		:param ground_truth_type: Type of ground truth. 'algae_blooms' or 'water_quality'
		:param frame_stacking: Number of frames to stack
		:param state_index_stacking: Indexes of the state to stack
		:param local: If True, the GP method is local
		
		"""
		
		# Set the seed
		self.seed = seed
		np.random.seed(seed)
		
		# Set the variables
		self.scenario_map = scenario_map
		self.visitable_locations = np.vstack(np.where(self.scenario_map != 0)).T
		self.number_of_agents = number_of_agents
		self.distance_budget = distance_budget # 100 最多能走多少步
		self.fleet_initial_positions = fleet_initial_positions
		self.movement_length = movement_length # 每次能移动的长度
		self.max_collisions = max_collisions
		# self.ground_truth_type = ground_truth_type
		self.fleet_initial_zones = fleet_initial_zones
		self.reward_type = reward_type

		self.max_steps = self.distance_budget // self.movement_length #50
		
		# Initial positions
		if fleet_initial_positions is None and fleet_initial_zones is None:
			random_positions_indx = np.random.choice(np.arange(0, len(self.visitable_locations)), number_of_agents, replace=False)
			self.initial_positions = self.visitable_locations[random_positions_indx]
		elif self.fleet_initial_zones is not None:
				self.random_inititial_positions = True
				# Obtain the initial positions as random valid positions inside the zones
				self.initial_positions = np.asarray([region[np.random.randint(0, len(region))] for region in self.fleet_initial_zones])
		else:
			self.initial_positions = fleet_initial_positions
   
		# Create the fleets 
		self.fleet = DiscreteFleet(number_of_vehicles=self.number_of_agents,
								   n_actions=8,
								   fleet_initial_positions=self.initial_positions,
								   movement_length=movement_length,
								   navigation_map=self.scenario_map)

		# # Ground truth selection
		# if ground_truth_type == 'shekel':
		# 	self.gt = shekel(self.scenario_map, max_number_of_peaks=4, is_bounded=True, seed=self.seed)
		# elif ground_truth_type == 'algae_bloom':
		# 	self.gt = algae_bloom(self.scenario_map, seed=self.seed)
		# elif ground_truth_type == 'wildfires':
		# 	self.gt = WildFiresSimulator(self.scenario_map, seed=self.seed)
		

		# else:
		# 	raise NotImplementedError("This Benchmark is not implemented. Choose one that is.")
		# print("self.gt")
		# print(self.gt)
		# # input()

		# Set the observation space
		# if frame_stacking != 0:
		# 	self.frame_stacking = MultiAgentTimeStackingMemory(n_agents = self.number_of_agents,
		# 	 													n_timesteps = frame_stacking - 1, 
		# 														state_indexes = state_index_stacking, 
		# 														n_channels = 5)
		# 	self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5 + len(state_index_stacking)*(frame_stacking - 1), *self.scenario_map.shape), dtype=np.float32)

		# else:
		self.frame_stacking = None
		self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5, *self.scenario_map.shape), dtype=np.float32)

		self.state_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5, *self.scenario_map.shape), dtype=np.float32)
		#这里为什么是5 也不清楚
		self.action_space = gym.spaces.Discrete(8)

		
		# Create a 2D grid of points with a distance of D between them
		self.distance_between_locals = distance_between_locals #一个固定距离
		x = np.arange(self.distance_between_locals//2, self.scenario_map.shape[0] - self.distance_between_locals//2, self.distance_between_locals)
		y = np.arange(self.distance_between_locals//2, self.scenario_map.shape[1] - self.distance_between_locals//2, self.distance_between_locals)
		X, Y = np.meshgrid(x, y)
		gp_positions = np.vstack((X.flatten(), Y.flatten())).T

		print("self.distance_between_locals//2",self.distance_between_locals//2)
		print("self.scenario_map.shape[0] - self.distance_between_locals//2",self.scenario_map.shape[0] - self.distance_between_locals//2)
		print("self.distance_between_locals",self.distance_between_locals)
		print("scenario_map",scenario_map)
		print("scenario_map[0]",scenario_map.shape)
		
		# input()
		# input() 
	# 	array([[ 3,  3],
    #    [10,  3],
    #    [17,  3],
    #    [24,  3],
    #    [31,  3],
    #    [ 3, 10],
    #    [10, 10],
    #    [17, 10],
    #    [24, 10],
    #    [31, 10],
    #    [ 3, 17],
    #    [10, 17],
    #    [17, 17],
    #    [24, 17],
    #    [31, 17],
    #    [ 3, 24],
    #    [10, 24],
    #    [17, 24],
    #    [24, 24],
    #    [31, 24],
    #    [ 3, 31],
    #    [10, 31],
    #    [17, 31],
    #    [24, 31],
    #    [31, 31]])
  		# 3 ,32 ,7
		# array([ 3, 10, 17, 24, 31])
  
		# Select the points that are are in 1 in the map
		gp_positions = gp_positions[self.scenario_map[gp_positions[:,0].astype(int), gp_positions[:,1].astype(int)] == 1]
		self.radius_of_locals = radius_of_locals
  		# Create the GP coordinator	
		self.local = local
		if self.local:
			self.gp_coordinator = LocalGaussianProcessCoordinator(gp_positions = gp_positions,
								kernel = C(1.0)*RBF(length_scale=5.0, length_scale_bounds=(0.5, 10.0)) + W(noise_level=1e-5, noise_level_bounds=(1e-5, 1e-5)),
								scenario_map = self.scenario_map,
								n_restarts_optimizer=0,
								alpha=1e-5,
								distance_threshold=radius_of_locals)
		else:
			self.gp_coordinator = GlobalGaussianProcessCoordinator(
								kernel = C(1.0)*RBF(length_scale=5.0, length_scale_bounds=(0.5, 10.0)) + W(noise_level=1e-5, noise_level_bounds=(1e-5, 1e-5)),
								scenario_map = self.scenario_map,
								n_restarts_optimizer=0,
								alpha=1e-5)
		

		self.fig = None
									

	def reset(self):
		""" Reset the variables of the environment """

		self.steps = 0

		if self.fig is not None:
			plt.close(self.fig)
			self.fig = None

		# Reset the ground truth #
		# self.gt.reset()
		# self.gt.step()

		# Initial positions
		if self.fleet_initial_positions is None and self.fleet_initial_zones is None:
			random_positions_indx = np.random.choice(np.arange(0, len(self.visitable_locations)), self.number_of_agents, replace=False)
			self.initial_positions = self.visitable_locations[random_positions_indx]
		elif self.fleet_initial_zones is not None:
				self.random_inititial_positions = True
				# Obtain the initial positions as random valid positions inside the zones
				self.initial_positions = np.asarray([region[np.random.randint(0, len(region))] for region in self.fleet_initial_zones])
		else:
			self.initial_positions = self.fleet_initial_positions

		self.fleet.reset(initial_positions=self.initial_positions)

		# Reset the GP coordinator #
		self.gp_coordinator.reset()

		# Take measurements #
		gt=GroundTruth(self.scenario_map)
		self.measurements = gt.read(self.fleet.agent_positions).reshape(-1,1)	

		# Update the GP coordinator #
		self.gp_coordinator.update(self.fleet.agent_positions, self.measurements)

		# Update the state of the agents #
		self.update_state()

		if self.fig is not None:
			plt.close(self.fig)
			self.fig = None

		# Return the state #
		return self.state if self.frame_stacking is None else self.frame_stacking.process(self.state)
	
	def step(self, actions):
		""" Take a step in the environment """

		self.steps += 1

		# Process action movement only for active agents #
		collision_mask = self.fleet.move(actions)
		
		# Collision mask to list 
		collision_mask = np.array(list(collision_mask.values()))

		# Take measurements #
		gt=GroundTruth(self.scenario_map)
		self.measurements = gt.read(self.fleet.agent_positions).reshape(-1,1)

		# Update the GP coordinator in those places without collision #
		self.gp_coordinator.update(self.fleet.agent_positions[np.logical_not(collision_mask)], self.measurements[np.logical_not(collision_mask)])

		# Compute the reward #
		reward = self.compute_reward(collisions=collision_mask)

		# Update the state of the agents #
		self.update_state()

		# Check if the episode is done #
		done = self.check_done()
		

		# Return the state #
		return self.state if self.frame_stacking is None else self.frame_stacking.process(self.state), reward, done, {}

	def update_state(self):
		""" Update the state of the environment """

		state = {}

		# State 0 -> Mu map
		mu_map = self.gp_coordinator.mu_map
		# State 1 -> Sigma map
		sigma_map = self.gp_coordinator.sigma_map
		# State 2 -> Agent 

		# Create fleet position #
		fleet_position_map = np.zeros_like(self.scenario_map)
		fleet_position_map[self.fleet.agent_positions[:,0], self.fleet.agent_positions[:,1]] = 1.0

		# State 3 and 4
		for i in range(self.number_of_agents):
			
			agent_observation_of_fleet = fleet_position_map.copy()
			agent_observation_of_fleet[self.fleet.agent_positions[i,0], self.fleet.agent_positions[i,1]] = 0.0

			agent_observation_of_position = np.zeros_like(self.scenario_map)
			agent_observation_of_position[self.fleet.agent_positions[i,0], self.fleet.agent_positions[i,1]] = 1.0
			
			'''
   			agent_observation_of_fleet：复制整个舰队位置图，但将当前代理的位置设置为 0.0。
			agent_observation_of_position：创建一个新图，仅在当前代理的位置处标记为 1.0。
      		'''
			state[i] = np.concatenate((
				np.clip(mu_map[np.newaxis], 0, 1),
				sigma_map[np.newaxis],
				agent_observation_of_fleet[np.newaxis],
				agent_observation_of_position[np.newaxis],
				self.scenario_map[np.newaxis].copy()
			))

		self.state = state

	def compute_reward(self, collisions):
		""" Compute the reward of the environment """


		# 1) obtain the changes in the surrogate model MU
		changes_mu_values, changes_sigma_values = self.gp_coordinator.get_changes()

		if self.reward_type == 'changes_mu':
			changes_values = changes_mu_values
		elif self.reward_type == 'changes_sigma':
			changes_values = changes_sigma_values
		else:
			raise ValueError('Invalid reward type')
		
		# 2) Compute the surroundings for every agent
		redundancy = np.zeros(self.gp_coordinator.X.shape[0]) #N
		agent_changes = np.zeros(self.number_of_agents)

		# Compute the redundancy and the changes for every agent
		for agent_id, position in enumerate(self.fleet.get_positions()):
			indexes = np.linalg.norm(self.gp_coordinator.X - position, axis=1) <= self.gp_coordinator.distance_threshold
			redundancy[indexes] += 1

		for agent_id, position in enumerate(self.fleet.get_positions()):	
			indexes = np.linalg.norm(self.gp_coordinator.X - position, axis=1) <= self.gp_coordinator.distance_threshold
			agent_changes[agent_id] = np.sum(np.abs(changes_values[indexes])/redundancy[indexes])

		# 3) Compute the distance between agents
		d_matrix = distance_matrix(self.fleet.agent_positions, self.fleet.agent_positions) # Compute the distance matrix
		distance_between_agents = d_matrix.copy() # Copy the distance matrix
		distance_between_agents[distance_between_agents <= 1] = 1.0 # Clip the min to 1.0 
		distance_between_agents[distance_between_agents > self.radius_of_locals] = np.inf # If the distance is greater than the radius of the local gp, set it to infinity
		np.fill_diagonal(distance_between_agents, 1.0) # Set the diagonal to 1.0 to simplify the computation of the redundancy
		distance_between_agents = 1.0 / distance_between_agents # Compute the inverse of the distance
		redundancy = np.sum(distance_between_agents, axis=1) # Compute the redundancy of each agent

		# Compute penalizations for being too close to other agents
		np.fill_diagonal(d_matrix, np.inf)
		penalization = np.sum(d_matrix <= self.movement_length, axis=1).astype(int)
		
		# 4) Compute the reward
		if self.local:
			reward = {agent_id: agent_changes[agent_id] - penalization[agent_id] for agent_id in range(self.number_of_agents)}
		else:
			reward = {agent_id: agent_changes for agent_id in range(self.number_of_agents)}

		# 5) Add a penalty for collisions
		for agent_id in range(self.number_of_agents):
			if collisions[agent_id] == 1:
				reward[agent_id] = -1

		return reward

	def check_done(self):

		# Check if the episode is done #
		done = {agent_id: False for agent_id in range(self.number_of_agents)}

		# Check if the episode is done #
		"""
		if self.fleet.fleet_collisions > self.max_collisions or any(self.fleet.get_distances() >= self.distance_budget):
			done = {agent_id: True for agent_id in range(self.number_of_agents)}
		"""

		if self.fleet.fleet_collisions > self.max_collisions or self.steps >= self.max_steps:

			done = {agent_id: True for agent_id in range(self.number_of_agents)}


		return done
		
  
	def get_error(self):
		""" Compute the MSE error """
		
		# Compute the error #
		gt=GroundTruth(self.scenario_map)
		error = np.sum(np.abs((gt.read() - self.gp_coordinator.mu_map)))
		return error
		#render he  get_error delete
		
		
