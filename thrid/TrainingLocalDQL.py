import sys
sys.path.append('.')
from thrid.DuelingDQNAgent import MultiAgentDuelingDQNAgent

from thrid.InformationGatheringEnvironment import MultiagentInformationGathering
# from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, required=False,default=3)
# agent的数量
parser.add_argument('--R', type=str, required=False,default='changes_mu')
# 奖励的类型

# 环境的类型，是ALGAE还是WQP GT 彻底删掉
args = parser.parse_args()

# 设置为75，50吧，现在不是，大概是76-4，76-4
scenario_map = np.genfromtxt('Maps/example_map copy.csv') 
# 一堆0和1

N = args.N
reward = args.R

# Distance between local GPs
D = 7# 考虑设为10

# Generate initial positions with squares of size 3 x 3 around positions
center_initial_zones = np.array([[30,15], [50,0], [70,15]])   ## 30，15   50，0    70，15
# center_initial_zones = np.array([[17,9], [22,8], [28,9]])   ## 30，15   50，0    70，15

# 9 positions in the sorrounding of the center
area_initial_zones = np.array([[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]])
# Generate the initial positions with the sum of the center and the area
fleet_initial_zones = np.array([area_initial_zones + center_initial_zones[i] for i in range(len(center_initial_zones))])
#创建一个以每个中心点为中心的3x3方阵的所有坐标。


env = MultiagentInformationGathering(
			scenario_map = scenario_map,
			number_of_agents = N,
			distance_between_locals = D,
			radius_of_locals = np.sqrt(2) * D / 2,
			distance_budget = 100,
			distance_between_agents = 1,
			fleet_initial_zones=fleet_initial_zones,
			fleet_initial_positions=None,
			seed = 0,
			movement_length = 1,#2
			max_collisions = 5,
			# ground_truth_type = GT,
			local = True,
			reward_type=reward
)
# print("test")
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
			logdir=f'./thrid/runs/DuelingDQN_{reward}_{N}_vehicles_test0010_1211_test2',
			log_name="DQL",
			save_every=1000,
			train_every=10,
			masked_actions= True,
			# device='cuda:1',
			device='cpu',
			seed = 0,
			eval_every = 200,
			eval_episodes = 50,)
# print("test2")
agent.train(10000)
