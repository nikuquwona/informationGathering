import sys
sys.path.append('.')
from InformationGatheringEnvironment import MultiagentInformationGathering
# from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np
import argparse





scenario_map = np.genfromtxt('Maps/example_map.csv') 
# 一堆0和1





# Generate initial positions with squares of size 3 x 3 around positions
center_initial_zones = np.array([[30,15], [50,0], [70,15]])   ## 30，15   50，0    70，15
# 9 positions in the sorrounding of the center
area_initial_zones = np.array([[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]])
# Generate the initial positions with the sum of the center and the area
fleet_initial_zones = np.array([area_initial_zones + center_initial_zones[i] for i in range(len(center_initial_zones))])
#创建一个以每个中心点为中心的3x3方阵的所有坐标。


env = MultiagentInformationGathering(
			# scenario_map = scenario_map, # 合法的位置
# 0，75 
# -25 ，25
			number_of_agents = 3,
			distance_between_locals = 10, #7
			radius_of_locals = np.sqrt(2) * 10 / 2, # 半径
			distance_budget = 100,
			distance_between_agents = 1,
			fleet_initial_zones=fleet_initial_zones,
			fleet_initial_positions=None,
			seed = 0,
			movement_length = 2,
   #Length of every movement of the agents
			max_collisions = 5,
   #Maximum number of collisions allowed
			ground_truth_type = 'algae_bloom',
			local = True,
			reward_type='changes_mu'#
)

agent = MultiAgentDuelingDQNAgent(env = env,
			memory_size = 500_000,
			batch_size = 64,
			target_update = 1000,
			soft_update = True,
			tau = 0.001,
			epsilon_values = [1.0, 0.05],
			epsilon_interval = [0.0, 0.5],
			learning_starts = 100,
			gamma = 0.99,
			lr = 1e-4,
			# NN parameters
			number_of_features = 512,
			logdir=f'runs/DuelingDQN_{GT}_{reward}_{N}_vehicles',
			log_name="DQL",
			save_every=1000,
			train_every=10,
			masked_actions= True,
			# device='cuda:1',
			device='cpu',
			seed = 0,
			eval_every = 200,
			eval_episodes = 50,)

agent.train(10000)
