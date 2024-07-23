import sys
sys.path.append('.')
from forth.DuelingDQNAgent import MultiAgentDuelingDQNAgent
from torch.utils.tensorboard import SummaryWriter
from forth.InformationGatheringEnvironment import MultiagentInformationGathering
import numpy as np
import argparse
from forth.PPO_dis_main import main 

'''
90 91 92
现在把UAV位置都放右上角，增加步数100（50到100），D change ，gamma change （0.95-0.983333333333333）

把步数和位置改回来，第二个的位置改好一些 gamma改回来
接着D change
95 96 97-98 不错，多搞几个




re：做一个大胆试验，都放在右上角，增加步数100、，每次走2步 gamma不变，然后删去地图限制#and distance<=10:
	1:bad 效果   
	re： 112 local  113 global
			感觉上是去掉限制后要加大gamma才行，不然会找小的奖励 
			或者加上限制也可以 : 加上限制，缩小步数70 gamma=0.966666666666667
		仿照111  114local 去掉限制  保持100/2步数，gamma还是0.95 ;但111感觉收敛性一般，尝试降低econf0.03-》0.01



尝试把第一个agent离得远一些，结果居然很差（可能是gauss filter的问题）
尝试把门限10，放大一些


local\global\ddpg   \random

纯random的做一下，不要learn，加快速度

'''
agent_num=3

'''
init
'''

parser = argparse.ArgumentParser()

parser.add_argument('--N', type=int, required=False,default=agent_num)
# agent的数量
parser.add_argument('--R', type=str, required=False,default='changes_mu')

parser.add_argument('--number', type=int, required=True,default=2024060)
parser.add_argument('--users_count', type=str, required=True,default='50')
parser.add_argument('--local', type=bool, required=True,default=True)
parser.add_argument('--D', type=int, required=False,default=7)
parser.add_argument('--lr', type=float, required=False,default=3e-4)
parser.add_argument('--re', type=str, required=False,default='IG')#SG


# 5 10 15
# 奖励的类型

# 环境的类型，是ALGAE还是WQP GT 彻底删掉
args = parser.parse_args()

# 设置为75，50吧，现在不是，大概是76-4，76-4
scenario_map = np.genfromtxt('Maps/example_map copy.csv') 
# 76，76:D为7  100个
# 76，76:D为10 49个
# 一堆0和1
lr=args.lr
N = args.N
reward = args.R

# Distance between local GPs
#D = 7# 考虑设为10
# D = 10# 考虑设为10

# distance_between_locals
D = args.D

# Generate initial positions with squares of size 3 x 3 around positions
# center_initial_zones = np.array([[30,15], [50,50], [70,75]])   ## 30，15   50，0    70，15
# center_initial_zones = np.array([[30,15], [50,15], [70,15]])   ## 30，15   50，0    70，15
# center_initial_zones = np.array([[30,30], [50,50], [75,90]]) 
# center_initial_zones = np.array([[17,9], [22,8], [28,9]]) 
# center_initial_zones = np.array([[29,65], [28,70], [29,75]])

# center_initial_zones = np.array([[20,40], [40,60], [80,60]])
center_initial_zones = np.array([[20,40], [40,40], [80,60]])
# center_initial_zones = np.array([[20,50], [40,40], [80,60]])
# center_initial_zones = np.array([[20,80], [18,82], [22,78]])  #not work


# change 突出冗余
center_initial_zones = np.array([[40,40], [40,40], [40,40]])


fleet_initial_pos=center_initial_zones

env = MultiagentInformationGathering(
			scenario_map = scenario_map,
			number_of_agents = N,
			distance_between_locals = D,
			radius_of_locals = np.sqrt(2) * D / 2,
			distance_budget = 100,#100-2#70,#50      #300,#100
			distance_between_agents = 1,
			fleet_initial_zones=None,
			fleet_initial_positions=fleet_initial_pos,
			seed = 0,
			movement_length = 2,#2 #1
			max_collisions = 5,#5,
			ground_truth_type = args.users_count,
			local = args.local,
			reward_type=reward,
			re=args.re,
)

'''
change
'''
# parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
parser.add_argument("--max_train_steps", type=int, default=int(10e4), help=" Maximum number of training steps")
# 1e4 10k
# 经过测试 700k 就差不多  将 3e6改为7e5 3e4
parser.add_argument("--evaluate_freq", type=float, default=1000, help="Evaluate the policy every 'evaluate_freq' steps")
# 100
#5e3
#1e3
#500
#1500
parser.add_argument("--save_freq", type=int, default=5000, help="Save frequency")
#20
# parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
#2048 1024 128
# 1000 /2
# 64
parser.add_argument("--mini_batch_size", type=int, default=256, help="Minibatch size")
# 64
# 64
parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
# 20 -64
    #   2048
# 256
# 64
parser.add_argument("--lr_a", type=float, default=lr, help="Learning rate of actor")
parser.add_argument("--lr_c", type=float, default=lr, help="Learning rate of critic")
#3e-3
#3e-2
#3e-4
parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")#0.98333
# 0.96
# 0.995
# 0.99 #0.995
parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")#0.03
parser.add_argument("--use_lr_decay", type=bool, default=False, help="Trick 6:learning rate Decay")
parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
parser.add_argument("--use_tanh", type=float, default=False, help="Trick 10: tanh activation function")
parser.add_argument("--PriorityReplayBuffer", type=bool, default=False, help="")
parser.add_argument("--learningStart", type=int, default=3000, help="3000")
parser.add_argument("--trainFreq", type=int, default=100, help="")
parser.add_argument("--random", type=bool, default=False, help="")

# args.random
#trainFreq

args = parser.parse_args()
print("lr",args.lr_a)
env_name = ['test','LocalGP', 'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']
env_index = 1
#002 目的是重新修正了MAP的bug
#		增加evalue freq

#003 初始位置改变

# 考虑budget
#2024053
main(args, env_name=env_name[env_index], number=args.number, seed=0,env=env)#number
'''
 40 正常on_policy 51users 修改了env 
python3 -W ignore forth/TrainingLocalDQLv4.py
 41 256/64/4      |  up learing rate  | dont state norm
 42 增大步数 100-500 
 43 collision =0 差
 44  考虑优先级buffer加回来 
	 再次增大步数he碰撞次数
	 
  45	增大了memory，应该减少步数
  
  46    先增大随机性，可能有一个epo就全随机	    
  		#//#增大初始学习步数
  47     删除随机性
		调大train freq置 （200-》300）
  48    gamma0.99 to 0.9833
		提高 entropy_coef 0.01->0.1 增强探索
		改为on policy
        
  49      提高了mini64-》256 提高学习率
		  提高 entropy_coef 0.01->0.5
    
 50  gamma 0.96  25? 
 51  gamma 0.9   10?
 #model/actor_LocalGP_2024051_2_model_best.pth
 #model/critic_LocalGP_2024051_2_model_best.pth
 
 52 gamma 0.9 ec 0.01 + use_lr_decay + 3e-3>3e-4?感觉不追求收敛好像用处一般？
 
 
 
 
 53 
	gamma 0.96改0.9改0.95
	entropy_coef 0.01改0.03
 
 
 
 应该是一下 优先级buffer，会不会好使？
	
 
 “39个用户”
[(26.85, 26.57), (84.07, 81.42), (56.67, 36.30), (83.83, 77.09), (18.27, 18.65), (82.80, 72.84), (75.06, 80.23), (68.35, 44.05), (75.72, 78.45), (16.00, 21.73), (56.56, 42.41), (54.12, 37.94), (73.11, 80.28), (84.25, 80.39), (75.53, 73.09), (54.75, 48.69), (56.15, 48.54), (77.93, 82.15), (73.63, 77.79), (82.86, 72.77), (25.50, 25.38), (16.36, 12.28), (59.69, 47.66), (78.35, 78.93), (66.66, 48.48), (21.80, 16.94), (59.56, 37.03), (61.59, 43.16), (83.52, 77.91), (85.04, 75.17), (60.05, 47.99), (23.87, 16.68), (8.58, 22.92), (74.87, 78.04), (16.03, 11.14), (79.89, 76.70), (79.92, 81.70), (76.69, 86.90), (30.15, 13.50)]
 “60个用户”
[(20.93, 24.37), (83.46, 74.71), (16.00, 21.73), (70.92, 89.23), (28.35, 24.05), (63.83, 37.09), (12.78, 19.05), (83.09, 83.11), (57.33, 38.94), (24.24, 19.31), (26.68, 19.08), (61.77, 35.11), (71.76, 77.86), (59.56, 37.03), (78.03, 78.75), (78.28, 80.42), (57.12, 30.06), (83.08, 85.82), (16.36, 12.28), (63.45, 43.68), (62.80, 32.84), (78.21, 81.65), (15.94, 20.23), (72.15, 82.02), (59.82, 34.78), (23.87, 16.68), (79.81, 78.72), (82.21, 79.48), (77.08, 78.25), (23.26, 20.58), (79.51, 84.56), (22.08, 13.13), (54.52, 43.78), (53.33, 41.97), (82.33, 77.42), (19.87, 21.92), (18.80, 23.59), (79.72, 79.52), (57.16, 43.03), (61.33, 45.93), (19.54, 21.84), (84.78, 77.97), (26.85, 26.57), (54.22, 41.19), (16.15, 28.54), (15.21, 24.72), (63.52, 37.91), (18.44, 20.70), (65.04, 35.17), (74.68, 77.73), (64.92, 42.93), (83.34, 74.68), (61.68, 35.08), (88.75, 81.74), (82.59, 81.56), (78.37, 80.70), (57.77, 48.63), (77.15, 78.22), (81.78, 67.60), (77.08, 77.67)]
 
 
 ？ 学习率
  1_0000 /3000（600steps） +600（200steps）
	1024







 32 10_000? 随机抽样 多个正则化 batchsize 2048
 31 5_0000	尝试把碰撞次数去掉，一次碰撞就撞车
 30 
		people  0.7衰减  并进行正则化
  		agent   多层网络cuda:0  加入随机探索
		加入优先级replay buffer，增大buffer数量50_0000
  		100/2 共50步  
		修改出生点

	5000 *50=250 000
	center_initial_zones = np.array([[17,9], [22,8], [28,9]]) 


	如果实在没有进展，要考虑就用DDPG做global的对比
	
	是不是还是得2.5

'''



#001 51 users 重点在于步数增加到3000
#002    重点修改了网络，包括relu、featurenum、hiddensize等
#003    改回去
# 004  jiezhe 003
# 004  不接着 把碰撞次数应该加回来，步数300
	#    1.5改2.5  3e-3变3e-4
	#    改变一下初始位置
 # 005 测试一下512 
 # 006 测试一下512 2.5 -》 2
 # 007  还是2.5，加上噪音,不好
 
 # 39 个用户
 #[(36.85, 56.57), (74.07, 31.42), (46.67, 56.30), (73.83, 27.09), (28.27, 48.65), (72.80, 22.84), (65.06, 30.23), (58.35, 64.05), (65.72, 28.45), (26.00, 51.73), (46.56, 62.41), (44.12, 57.94), (63.11, 30.28), (74.25, 30.39), (65.53, 23.09), (44.75, 68.69), (46.15, 68.54), (67.93, 32.15), (63.63, 27.79), (72.86, 22.77), (35.50, 55.38), (26.36, 42.28), (49.69, 67.66), (68.35, 28.93), (56.66, 68.48), (31.80, 46.94), (49.56, 57.03), (51.59, 63.16), (73.52, 27.91), (75.04, 25.17), (50.05, 67.99), (33.87, 46.68), (18.58, 52.92), (64.87, 28.04), (26.03, 41.14), (69.89, 26.70), (69.92, 31.70), (66.69, 36.90), (40.15, 43.50)]

 # 009  不加噪音 修改user 60 
 # 		修改初始位置第一个 512-》256
 #      一次性走两步

#010   70个用户   300步和2 
#011   30个用户    修改gamma 修改步数
#      学习率3e-4 ->1e-4
# 
# 012  test  将batchsize扩大
			#  将featur降低256-》64
			#  减少一层
   
   
   # 013  tanh 降低学习率
		#   51个用户
		# self.users=[(77.57, 81.86), (20.20, 19.16), (26.85, 26.57), (17.72, 18.04), (56.67, 36.30), (20.69, 21.69), (23.87, 16.68), (56.15, 48.54), (84.07, 81.42), (18.61, 20.25), (63.45, 43.68), (88.75, 81.74), (79.39, 85.08), (18.27, 18.65), (16.99, 18.39), (59.69, 47.66), (55.53, 33.09), (69.67, 45.98), (86.68, 70.74), (54.22, 41.19), (19.87, 21.92), (62.34, 39.23), (23.40, 20.54), (28.35, 15.63), (74.16, 87.42), (61.68, 35.08), (81.78, 67.60), (16.11, 17.41), (70.92, 89.23), (76.76, 73.81), (80.63, 78.57), (77.33, 78.94), (59.82, 34.78), (80.44, 82.61), (15.31, 13.65), (55.06, 40.23), (62.58, 39.07), (90.30, 75.26), (17.72, 14.72), (15.21, 24.72), (61.33, 45.93), (77.93, 82.15), (20.05, 27.99), (54.87, 38.04), (81.28, 82.72), (65.31, 41.42), (79.92, 81.70), (21.40, 16.18), (63.01, 41.82), (27.89, 21.79), (74.68, 77.73)]
		# 做参照组，与global对比
  
	# 014 51user local 加入探索机制	降低batchsize
	# 集群 1 的中心坐标为：(15, 14)
	# 集群 2 的中心坐标为：(62, 38)
	# 集群 3 的中心坐标为：(79, 84)
 
	#015 global 一块时修改global，一块是修改了replay buffer1
			# 还有 writer scaller
   
   
   # 016 global 对global进行修改，看能否正常运行
				#可以要记录每个sum_of_reward,碰撞时总的看的
    
    # 017 local=True   ,duibi1  1024
	# 018  感觉要调一下第一个agent的位置，(15,15)-->(30.30)
	#      调整位置还是不好，什么原因？
	
	# 019  local=true
 	#      去掉了探索机制，似乎导致曲线不稳定
	#      去掉了use_reward_scaling，以及state norm，现在没有norm和scale

	# 21 local 
	# 22 global

 
	








'''
修改了随机噪音、然后用户数9 ，做一个实验，吧feature 256降低64，加快
			减小步数到100，然后D 20缩小到10
			
			batchsize1024->512
   
   
71:			学习率3e-3 变 0.01
			折扣系数 0.995-->0.96--》0.99
			mini_batch_size 256-64
			eva 1000-》batchsize 512  改 500-》batchsize 250
			radius_of_locals 增大 20 算了，修改回来
			衰减系数 1.5->0.5->0.005->1.5
   
			增大用户数 到51个

'''





# env.number_of_agents
print("env.number_of_agents",env.number_of_agents)
print("env.initial_positions",env.initial_positions)
print("as",env.action_space)
print("ob",env.observation_space)

obs_dim = env.observation_space.shape
action_dim = env.action_space.n
print("obs_dim",obs_dim)#(5, 76, 38)
print("action_dim",action_dim)
print("env.reset()",env.reset()[0].shape) # {0: 1: 2:} (5,76,38)
print("env.reset()",env.reset()[0][3])
indices = np.where(env.reset()[0][3] == 1)
print("indice",list(zip(indices[0], indices[1])))
# input()

print("ending")
'''
change
'''

# # print("test")
# agent = MultiAgentDuelingDQNAgent(env = env,
# 			memory_size = 500_000,
# 			batch_size = 128,#64,
# 			target_update = 1000,
# 			soft_update = True,
# 			tau = 0.001,
# 			epsilon_values = [1.0, 0.05],
# 			epsilon_interval = [0.0, 0.5],
# 			learning_starts = 100,
# 			gamma = 0.99,
# 			lr = 1e-4,
# 			# NN parameters
# 			number_of_features = 128,#512,
# 			#logdir=f'./forth/runs/DuelingDQN_{reward}_{N}_vehicles_test0010_1211_test2',
# 			logdir=f'./forth/runs/DuelingDQN_{reward}_{N}_vehicles_test_1_12_00:07',
#    			log_name="DQL",
# 			save_every=1000,
# 			train_every=1,#10,
# 			masked_actions= True,
# 			# device='cuda:1',
# 			device='cpu',
# 			seed = 0,
# 			eval_every = 200,
# 			eval_episodes = 50,)
# # print("test2")
# agent.train(10000)
'''
vehicles_test_1_11_20:10        
vehicles_test_1_11_23:47     hidden_size = 128
vehicles_test_1_12_00:07	 batchsize=128

修改为PPO吧 分布式的MAPPO
	主要用到了env.step 以及一些环境信息
	现在是离散动作,forth的修改先以离散为主还是连续为主呢
 	
  	先尝试修改连续和三维，同时保留离散吧
	|但是env现在都是离散的，不好改，讲道理离散的更好收敛，先写个离散的版本吧
  
  
	原来的env是分离的，每个都单独对应一个agent
	现在是对应整体agent的env



	这个(5, 76, 38)到底是什么意思
		答：state[i] = np.concatenate((
				np.clip(mu_map[np.newaxis], 0, 1),
				sigma_map[np.newaxis],
				agent_observation_of_fleet[np.newaxis],
				agent_observation_of_position[np.newaxis],
				self.scenario_map[np.newaxis].copy()
			))
   
		5知道了 76也知道了
		38呢？ 答：y为38 

		5个地图 均值图、方差图、当前agents位置、当前agent位置、
  
  
		select_action
			对每个agent的这五维都进行DRL，选择出一个离散动作

		知道为什么GPU效率不高了，因为内部有选择cpu
		回来把Autodl上那个再运行一下
			把cpu改了，然后跑12个小时
			最好加回来网络层数吗？
   
   
		replaybuffer是个问题
		s,s_,reward,done的处理也是个问题
			之前是 state【13】 action【13】 reward 1 done false
			现在是 state {N:(5,76,38)} action {N:1} reward {N:-0.2} done {N:false}
			确定一下它网络的 (5,76,38) 1 -0.2 false
			solution：data_list = list(data_dict.values())
   
   
		env中done的设置问题，牵扯dw
  
		每一次最大max step的问题 现在是在网络侧已经做了
			删去环境侧的最大步数？
			删去网络侧的最大步数？ 
   				先删网络侧试一下
				目前使网络侧步数和环境侧对齐为100

		没有GPU跑不动 2048 update
			改成64
   
		路径还是点
			原文是什么 再研究一下
  
  
		又发现一个问题：	
  			环境编写有问题，
     			user的坐标 , map的坐标 ,对应关系是什么
				现在应该把y从38换成75,现在到不了那里，反而会导致碰撞！！！！
					最好都改成(100,100)
     
		ppo版本要把评估时间缩短
			# batch_size 应该缩小3倍
			｜evaluate 大概除以5倍
		
		找个GPU泡一下 PPO的代码
  
  
		答案：起点：[30,15], [50,0], [70,15]
# 49，20
# 70，45
# 30，36
		起点初始位置有点问题

			[30,15], [50,15], [70,15]
			互相间判断惩罚应该大一些 5把
   
		记得吧autodl上的代码保存住

   /usr/local/bin/python3 -W ignore forth/TrainingLocalDQL.py
   
   
   
		todo：
			把最大步数加大
   
		1.迁移到GPU
   
		2.发现最优位置不太对，化热图等发现确实有问题，进行调整
			不对，是对的，肯定和起始位置不同
				Cluster 1: (19.950819672131175, 49.0655737704918)
				Cluster 2: (45.006172839506206, 69.93209876543207)
				Cluster 3: (36.51612903225806, 29.39999999999998)
		
				Cluster 1: (69.93209876543207, 45.006172839506206)
				Cluster 2: (29.39999999999998, 36.51612903225806)
				Cluster 3: (49.0655737704918, 19.950819672131175)
		
		
				这是我目前的最优位置 ans_max 43696.43203788462 ans_state [[(28, 40)], [(50, 34)], [(72, 41)]]
  
			但还是有一个点没到聚集点那目前两个办法
				1.调整agent初始位置
				2.把users位置改了
				
    		不对 我应该最后把mu_map 和 sigma_map 打印出来 因为最后step多了 它可能也会变


		把mu_map 和 sigma_map进行了输出
		突然想到每次eva的map都重置了，看一下原来的DDQN的评估思路

		接下来得把map如何表示搞一下
  
		007
		#把cnn缩小了
		减少hidden width
  
		008
		缩小batch size
		每次done结束时输出mu——map和sigma--map
  
  
		009
		变大batch size 和步数100-1000
		变大eva——freq和max_train_steps
  
		011
		变回batch size 和步数
		变回eva——freq和max train steps
		把feature_变回去
  
  
		#011 ans_max 339.8792868117923 ans_state [[(36, 18)], [(50, 20)], [(63, 21)]]
		#011出现一直38的问题

		#012 缩小为50步 步伐增大为2
		# batchsize -128 为了让他多更新   还是增大
		# 缩小学习旅
		# gamma
'''
		#013 将network hidden 128 改为 256
'''
.3.15

跑一些对比方案
	正常的
 	已知用户分布的部署策略，累计奖励和最终结果
		想一下这个具体怎么做，env和agent两块，agent好说，几乎不用变，只是env返回我一个reward改为功率的最大
		当然这个是不是可以没有过程，单独就是计算一个值，用户集中点的功率 和 估算出的集中点的功率 
			三个集群的中心坐标：
# 			集群 1 的中心坐标为：(69, 43)
# 			集群 2 的中心坐标为：(70, 49)
# 			集群 3 的中心坐标为：(50, 19)
#    or
			三个集群的中心坐标：
			集群 1 的中心坐标为：(69, 43)
			集群 2 的中心坐标为：(50, 19)
			集群 3 的中心坐标为：(30, 38)
   (30, 38)(50, 19)(69, 43)
   (29, 40)(51, 20)(69, 49)
		我也许可以直接遍历表格输出最大的三个点
			三个集群的中心坐标：
			集群 1 的中心坐标为：(29, 40)
			集群 2 的中心坐标为：(69, 49)
			集群 3 的中心坐标为：(51, 20)
		测试的时候发现衰减系数太大了,改为0.4,因为我地图小。。
        
        
        
  问题：还做local的对比吗，对比global、local数量
  
  第一次run 13 记得改final_mu_map_
  13 修改了 hidden为256
  14	 先增大minibatchsize 64-》256
  15 还和013一样，不过输出部署位置
  
  16 修改了衰减系数 重新测试 ，third全改为forth 15user
  
  17 修改回来2.5吧，重新测试
  
  020050 增加50user ，但 学习率降低3e-4
  021050  学习率3e-2
  
  
  041050  增大batchsize
  
  感觉还得增大batchsize，因为会陷入局部解集
  
  042050  #增加移动性
  
  042051  # 一个epo动一次，碰撞旅这么高，是不是用户太多在边界外了
  
  042052  # 碰撞旅高，那我不移动呢，效果还行，只是变化不明显
  ans_max 274.1968881409159 ans_state [[(22, 13)], [(66, 3)], [(56, 31)]]
  
  042053  # gt.step的时候加入边界检测，降低移动大小
  			这次看碰撞率还高吗，为什么会碰撞
  
  042054  # 碰撞机率降下来，但还不收敛，扩大轮次
		  # 如果还不行，要考虑调整衰减系数了
		  # 或者靠谱一点，还是间隙中移动？
  
  042055	#衰减系数  2.5-》1.5
  
  042056   # env.gt.step 只有训练结束后做一次step
		   # 同时扩大batch->扩大失败
		   # 衰减还是2.5 ,把衰减降低 1
	       # 降低学习率 3e-2 -> 3e-3
           # 记录碰撞机率 训练时的
  
  042057   # 衰减降低为 0.5
  
  042058   # 不移动 
  
  042059   # 移动 ,发现1不行，改回1.5，但增大步数
  
  042060   # 改为0.7 ,0.1; ，扩大local gp 数量  D=3，
#   		 难怪之前改了衰减有问题，没改完全
		   # 先不移动	
		   # 扩大步数 100-200
		   # 修改起始位置	
     实在不行就把衰减改回来
		到目前为止改过的内容：衰减系数、学习率、GP、步数、起始位置
   
   042061 
		衰减2.5
		步数50，2  -》100，1
  		然后D=10
    	起始位置复原
		学习率复原 3e-4
     
     distance_budget 步数搞错了 ，可能是原来100/2 改为50/2或100/1 的问题吗？
     改成100/2  50
	 2048 ，64    特征树256- hidden 64
	 换回之前的gt
  ``	0.9995

  移动效果对比，如果大范围移动呢
  
  之后还可以统计碰撞率
  
  之后想办法在夸大batchsize的前提下保证不超内存
  KMAPPO建模；然后想办法与KMAPPO对比
  
'''