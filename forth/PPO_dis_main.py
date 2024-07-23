import logging
import random
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import torch
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer,PriorityReplayBuffer
from ppo_discrete import PPO_discrete,Actor,Critic
def save_map(args,final_map):
    final_mu_map=final_map[0]
    final_sigma_map=final_map[1]
    
    # 指定保存文件的路径和文件名
    file_path = "final_mu_map_"+str(args.number)+".txt"
    # 使用NumPy的savetxt函数保存数组到文本文件
    np.savetxt(file_path, final_mu_map)
    # print(f"已保存文件: {file_path}")
    
    
    # 指定保存文件的路径和文件名
    file_path = "final_sigma_map_"+str(args.number)+".txt"
    # 使用NumPy的savetxt函数保存数组到文本文件
    np.savetxt(file_path, final_sigma_map)
    # print(f"已保存文件: {file_path}")
def save_eva_map(args,final_map):
    final_mu_map=final_map[0]
    final_sigma_map=final_map[1]
    
    # 指定保存文件的路径和文件名
    file_path = "final_mu_map_"+str(args.number)+"_eva.txt"
    # 使用NumPy的savetxt函数保存数组到文本文件
    np.savetxt(file_path, final_mu_map)
    # print(f"已保存文件: {file_path}")
    
    
    # 指定保存文件的路径和文件名
    file_path = "final_sigma_map_"+str(args.number)+"_eva.txt"
    # 使用NumPy的savetxt函数保存数组到文本文件
    np.savetxt(file_path, final_sigma_map)
    # print(f"已保存文件: {file_path}")

def save_eva_pos(args,path):
    # final_mu_map=final_map[0]
    # final_sigma_map=final_map[1]
    from datetime import datetime
    current_time = datetime.now()
    timestamp_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # # 指定保存文件的路径和文件名
    
    file_path1 = "forth/path/final_eva_path1_"+str(args.number)+"_eva_"+'.txt'
    file_path2 = "forth/path/final_eva_path2_"+str(args.number)+"_eva_"+'.txt'
    file_path3 = "forth/path/final_eva_path3_"+str(args.number)+"_eva_"+'.txt'
    # file_path = "/Users/lhd/Desktop/algov3/forth/path/final_eva_path_2024005_eva_"+timestamp_string+'.txt'
    path1=[]
    path2=[]
    path3=[]
    for p in path:
        path1.append(p[0])
        path2.append(p[1])
        path3.append(p[2])
    
    np.savetxt(file_path1, path1)
    np.savetxt(file_path2, path2)
    np.savetxt(file_path3, path3)
    # 使用NumPy的savetxt函数保存数组到文本文件
    # import pickle
    # with open(file_path, 'wb') as f:
    #     pickle.dump(path, f)
    
def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    evaluate_collision=0
    eva_path=[]
    for time in range(times):
        s = env.reset()
        if args.use_state_norm:
            s_norm = {key: state_norm(value, update=False) for key, value in s.items()}
            s=s_norm
        # if args.use_state_norm:
        #     s = state_norm(s, update=False)  # During the evaluating,update=False
        
        
        # done = False
        done = {i:False for i in range(env.number_of_agents)}
			
        episode_reward = 0
        step=0
        eposide_collision=0
        # recordForS_=s
        # while not done:
        # run 一个 epo
        while not all(done.values()):# 只要有一个done为true就会结束
            # a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
                    # if args.policy_dist == "Beta":
                    #     action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
                    # else:
            # action = a
            actions = {agent_id: agent.evaluate(state) for agent_id, state in s.items()}
            # actions = {agent_id: self.predict_action(state, deterministic) for agent_id, state in states.items()}
            # 简单说一下， 网络时针对每个agent的，得出的是每个agent的策略
            # 同一step用每个agent的状态去跑网络，顺序决策，得到集合actions，然后运行env
            s_, r, done, _ = env.step(actions)
            if time==times-1:
                eva_path.append(env.fleet.agent_positions)
            eposide_collision+=env.fleet.fleet_collisions
            # recordForS_=s_[]
            map=s_[0]
            step+=1
            if step==args.max_episode_steps:
                # done=True
                done ={agent_id: True for agent_id, d in done.items()}
            # if args.use_state_norm:
            #     s_ = state_norm(s_, update=False)
            if args.use_state_norm:
                s_norm = {key: state_norm(value, update=False) for key, value in s_.items()}
                s_=s_norm
                
            # episode_reward += r
            for _, value in r.items():
                episode_reward+=value
            s = s_
            if time==times-1 and (not all(done.values())==False):
                save_eva_map(args,map)
                save_eva_pos(args,eva_path)
        evaluate_reward += episode_reward
        evaluate_collision+=eposide_collision

    return evaluate_reward / times, evaluate_collision / times ,map[0],map[1],eva_path
# def evaluate_policy_eval(args, env, agent, state_norm):
#     path=[]
    
    
#     times = 3
#     evaluate_reward = 0
#     for _ in range(times):
#         s = env.reset()
#         if args.use_state_norm:
#             s = state_norm(s, update=False)  # During the evaluating,update=False
#         done = False
#         episode_reward = 0
#         step=0
#         while not done:
#             a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
#             # if args.policy_dist == "Beta":
#             #     action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
#             # else:
#             action = a
#             s_, r, done, _ = env.step(action)
#             ###记录path
#             path.append(s_.tolist())
#             ###
#             step+=1
#             if step==args.max_episode_steps:
#                 done=True
#             if args.use_state_norm:
#                 s_ = state_norm(s_, update=False)
#             episode_reward += r
#             s = s_
#         evaluate_reward += episode_reward

#     return evaluate_reward / times,path


def main(args, env_name, number, seed
        #  ,agent_num=3
        #  ,users_pos=[(48,-3),(48,3)]
         ,env
         ,eval=False
         ):
    
    # Set random seed
    # env.seed(seed)
    # env.action_space.seed(seed)
    # env_evaluate.seed(seed)
    # env_evaluate.action_space.seed(seed)
    args.number=number
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    args.agent_num=env.number_of_agents#agent_num
    
    # users=[(40,3),(44,0),(42,1),(50,5)]
    # 不知道user pos
    # users=[(28.095737508379507, 9.262238181375238), (31.534268858415665, 11.965999116504157), (28.987847366016233, 14.938933089562964), (32.11286120677997, 12.201298108043847), (26.765425070521744, 9.164420662094894), (45.749778214291226, -2.3878504425633658), (45.396274248005305, -4.385207701641737), (50.642615825605006, -5.146976192917276), (48.26626883173278, -9.490511609965512), (54.425094652367605, -3.1927547997114782), (68.71555847946225, 16.95251215340648), (69.30532824493405, 18.830942106523402), (68.2989629979505, 17.05649377261501), (73.96507288663165, 21.54396584760398), (69.0837707884811, 23.828827829773275)]
    # users=users_pos
    
    # groups =group_points(users, args.agent_num) 
    # print("groups",groups)
    
    # input()
    #[[(50, 5)], [(44, 0), (42, 1)], [(40, 3)]]
    # [
    #  [(68.71555847946225, 16.95251215340648), (69.30532824493405, 18.830942106523402), (68.2989629979505, 17.05649377261501), (73.96507288663165, 21.54396584760398), (69.0837707884811, 23.828827829773275)], 
    #  [(45.749778214291226, -2.3878504425633658), (45.396274248005305, -4.385207701641737), (50.642615825605006, -5.146976192917276), (48.26626883173278, -9.490511609965512), (54.425094652367605, -3.1927547997114782)], 
    #  [(28.095737508379507, 9.262238181375238), (31.534268858415665, 11.965999116504157), (28.987847366016233, 14.938933089562964), (32.11286120677997, 12.201298108043847), (26.765425070521744, 9.164420662094894)]
    # ]
    # input()
    start_pos=[]
    for pos in env.initial_positions:
        x=pos[0]
        y=pos[1]
        z=5 # 初始高度默认为5
        start_pos.append((x,y,z))
    # start_pos=[(30,15,5),(50,0,5),(70,15,5)]
    
    env=env
    
    import copy     
    env_evaluate=copy.deepcopy(env)
    # envs=[]
    # env_evaluates=[]
    # for i in range(args.agent_num):
    #     envs.append(ENV(groups[i]),start_pos[i])
    #     env_evaluates.append(ENV(groups[i]),start_pos[i])
    # env = ENV([(40,3),(44,0),(42,1)])
    # env_evaluate = ENV([(40,3),(44,0),(42,1)])
    args.PriorityReplayBuffer=args.PriorityReplayBuffer
    obs_dim = env.observation_space.shape
    action_dim = env.action_space.n
    args.state_dim = obs_dim#13#8#env.observation_space.shape[0]  
    # obs_dim (5, 76, 38)
    args.action_dim = action_dim#13#8#env.action_space.shape[0]  13 
    # 8
    args.max_action = 1#float(env.action_space.high[0]) # -1,1 
    # 0，1
    args.max_episode_steps = 50#35#50#100  
    #500#500#50#50#100#10000#env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    evaluate_collisions=[] 
    total_steps = 0  # Record the total steps during the training

    # replay_buffers=[]
    # agents=[]
    # for _ in range(args.agent_num):
    #     replay_buffers.append(ReplayBuffer(args))
    #     agents.append(PPO_discrete(args))
    if args.PriorityReplayBuffer==True:
        replay_buffer = PriorityReplayBuffer(args)
    else:
        replay_buffer = ReplayBuffer(args)
    # replay_buffer = PriorityReplayBuffer(args)
    agent = PPO_discrete(args)

    bestReward=0
    ## load
    # print("加载65模型")
    # agent.actor.load_state_dict(torch.load(r"model/actor_LocalGP_2024065_2_model_best.pth"))
    # agent.critic.load_state_dict(torch.load(r"model/critic_LocalGP_2024065_2_model_best.pth"))
    # print("加载成功🏅")
    ##
    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/PPO_discrete/env_{}_number_{}_seed_{}'.format(env_name, number, seed))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    ans_max=-999
    ans_state=[]
    ans_gp_map=None
    ###
    reward_record=[]
    eposide_collisions_training=[]
    
    paths=[]
    for _ in range(args.agent_num):
        paths.append([])
        # path_1=[]
        # path_2=[]
        # path_3=[]
    ###
    # randomWander=0
    while total_steps < args.max_train_steps:
        # s=[]
        # for env in envs:
        #     s.append(env.reset())
        # env.gt.step() #v1
        s=env.reset() #{N:(5，76，38）}
        eposide_collision_training=0
        # print(s)
        # input()
        # s = env.reset() # 返回的是什么
        # path=[]

        for i in range(args.agent_num):
                # env.reset()[0]
            indices = np.where(s[i][3] == 1)
            indice=list(zip(indices[0], indices[1]))
            # paths[i]=[]
            # paths[i].append(s[i].tolist())
            paths[i].append(indice)
        # path_1=[]
        # path_2=[]
        # path_3=[]
        
        # path_1.append(s[0].tolist())
        # path_2.append(s[1].tolist())
        # path_3.append(s[2].tolist())
        
        # if args.use_state_norm:
        #     # for i in range(args.agent_num):
        #     #     s[i]=state_norm(s[i])
        #     s = state_norm(s)
            
        if args.use_state_norm:
            s_norm = {key: state_norm(value, update=False) for key, value in s.items()}
            s=s_norm    
            
            
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        # done = False
        done = {i:False for i in range(env.number_of_agents)}
        
        # if eval==True:
        #     # load model
        #     # for i in range(args.agent_num):
        #     #     agents[i].actor.load_state_dict(torch.load('./model/actor_'+str(env_name)+'_'+str(number)+'_'+str(i)+'_model.pth'))
        #     #     print("********加载成功***********")
        #     #     print()
        #     #     # './model/actor_'+str(env_name)+'_'+str(number)+'_'+str(i)+'_model.pth'
        #     #     # critic.load_state_dict(torch.load('critic_model.pth'))
        #     #     _,res_path=evaluate_policy_eval(args, env_evaluates[i], agents[i], state_norm)
        #     #     paths[i].extend(res_path)
        #     agent.actor.load_state_dict(torch.load('./model/actor_'+str(env_name)+'_'+str(number)+'_'+str(i)+'_model.pth'))
        #     print("********加载成功***********")
        #     print()
        #     # './model/actor_'+str(env_name)+'_'+str(number)+'_'+str(i)+'_model.pth'
        #     # critic.load_state_dict(torch.load('critic_model.pth'))
        #     _,res_path=evaluate_policy_eval(args, env_evaluate, agent, state_norm)
        #     paths.extend(res_path)
        #     break
        #     pass
        
        # while not done:
        # 有一个为true，他也还会执行，全为true时会推出
        sum_strength=0
        
        avg_sum_reward0=0
        avg_sum_reward1=0
        avg_sum_reward2=0
        # randomWander=0
        epsilon=0.1
        # random_wander=False
        # if random.random() < epsilon :
        #     random_wander=True
        while not all(done.values()):
            

            episode_steps += 1
            # a=[]
            # a_logprob=[]
            # for i in range(args.agent_num):
            #     t1,t2=agents[i].choose_action(s[i])
            #     a.append(t1)
            #     a_logprob.append(t2)
            
            actions = {i:0 for i in range(env.number_of_agents)}
            actions_logprobs = {i:0 for i in range(env.number_of_agents)}
            for agent_id, state in s.items():
                if random.random() < epsilon or args.random==True:
                    actions[agent_id]= env.action_space.sample()#agent.choose_action(state)
                    actions_logprobs[agent_id]=0.0
                else:
                    actions[agent_id], actions_logprobs[agent_id] = agent.choose_action(state)
                # actions[agent_id], actions_logprobs[agent_id] = agent.choose_action(state)                
                # print("a",actions[agent_id],"a_logprob",actions_logprobs[agent_id],end=' ')
            
            # if random.random() < epsilon:
            #     # Explore: Randomly select an action
            #     action = env.action_space.sample()  # Assuming discrete action space
            #     action_logprob = 0.0  # Log prob is not relevant for random actions
            # else:
            #     # Exploit: Choose action based on agent's policy
            #     action, action_logprob = agent.choose_action(state)
            
            # print()
            # print()

            # a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            # print("a",a,"a_logprob",a_logprob)
            
            
            # if args.policy_dist == "Beta":
            #     action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            # else:
            action = actions #（0，1）
                
            # s_=[]
            # r=[]
            # done=[]
            
            # for i in range(args.agent_num):
            #     t1, t2, t3, _ =envs[i].step(action[i])
            #     s_.append(t1)
            #     r.append(t2)
            #     done.append(t3)
                
            s_, r, done, _ = env.step(action)
            
            avg_sum_reward0+=r[0]
            avg_sum_reward1+=r[1]
            avg_sum_reward2+=r[2]
            
            sum_strength+=env.gt.read(env.fleet.agent_positions).reshape(-1,1).sum()	
            
            eposide_collision_training+=env.fleet.fleet_collisions
            r_any=r
            # print("r",r)
            # r {0: 0.23410831412663755, 1: -1, 2: 0.1041172912146944}
            # input()
            ###
            # if episode_steps==1:
            # logging.basicConfig(filename='./forth/log/runlogV2'+'.log', level=logging.INFO, 
            #                     format='%(asctime)s - %(levelname)s - %(message)s')
            # if episode_steps==1:
            #     logging.info("")
            #     logging.info("")
            #     logging.info(f"total: {total_steps}")
                
            # logging.info(f"steps: {episode_steps}")
            # logging.info(f"state: {env.fleet.get_positions()}")#env.fleet.get_positions()
            # logging.info(f"action: {actions}")
            # logging.info(f"reward: {r}")
            
            if total_steps>=args.max_train_steps-args.max_episode_steps:
                r_tmp=0
                for _, value in r.items():
                    r_tmp+=value
                # reward_record.append(r[0])
                reward_record.append(r_tmp)
            
            for i in range(args.agent_num):
                # paths[i]=[]
                indices = np.where(s_[i][3] == 1)
                indice=list(zip(indices[0], indices[1]))
                # paths[i].append(s_[i].tolist())
                paths[i].append(indice)
            # path_1.append(s_[0].tolist())
            # path_2.append(s_[1].tolist())
            # path_3.append(s_[2].tolist())
            
            # print('path',path)
            
            # state_to_print=s_[0]#s_[0]
            state_to_print=[]
            for i in range(args.agent_num):
                indices = np.where(s_[i][3] == 1)
                indice=list(zip(indices[0], indices[1]))
                state_to_print.append(indice)
            gp_map=s_[0]
            
            if episode_steps==args.max_episode_steps:
                # for i in range(args.agent_num):
                #     done[i] =True
                
                done ={agent_id: True for agent_id, d in done.items()}
                # done=True
                # ddone=True
                
                
                r_tmp=0
                for _, value in r.items():
                    r_tmp+=value
                
                # if r_tmp>ans_max:
                    # ans_max=r_tmp
                    # ans_state=state_to_print
                    # ans_gp_map=gp_map
                    
                    # save_map(ans_gp_map)
            ###
            # for i in range(args.agent_num):
            if args.use_state_norm:
                s_norm = {key: state_norm(value, update=False) for key, value in s_.items()}
                s_ =s_norm
                # s_ = state_norm(s_)
            if args.use_reward_norm:
                r_norm = {key: reward_norm(value, update=False) for key, value in r.items()}
                r= r_norm
                # r = reward_norm(r)
            elif args.use_reward_scaling:
                r_scale = {key: reward_scaling(value) for key, value in r.items()}
                r= r_scale
                
                # r = reward_scaling(r)


            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            # all(done.values())
            
            
            dw = {i:False for i in range(env.number_of_agents)}
            for i in range(env.number_of_agents):
                if done[i] and episode_steps != args.max_episode_steps:
                    dw[i]=True
                else: 
                    dw[i]=False
            # if all(done.values()) and episode_steps != args.max_episode_steps:
            # # if done and episode_steps != args.max_episode_steps:
            #     dw = True
            # else:
            #     dw = False

            # Take the 'action'，but store the original 'a'（especially for Beta）
            # replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            # for i in range(args.agent_num):
            #     replay_buffers[i].store(s[i], a[i], a_logprob[i], r[i], s_[i], dw, done[i])
            #     s[i]=s_[i]
            # s = s_
            
            # 全得转化格式
            s_format = list(s.values()) # [(5,76,38),(5,76,38),(5,76,38)]
            actions_format=list(actions.values())
            actions_logprobs_format=list(actions_logprobs.values())
            r_format = list(r.values())
            s_next_format = list(s_.values())
            dw_format=list(dw.values())
            done_format=list(done.values())
            if args.random==False:
                if args.PriorityReplayBuffer==True:
                    for i in range(env.number_of_agents):
                        replay_buffer.store(s_format[i], actions_format[i], actions_logprobs_format[i], r_format[i], s_next_format[i], dw_format[i], done_format[i])
                        if replay_buffer.len > args.learningStart: # learning start
                                if total_steps %args.trainFreq==0:
                                    a,c=agent.update(replay_buffer, total_steps)
                                    writer.add_scalar('actor_loss_{}'.format(env_name),  a, global_step=total_steps)
                                    writer.add_scalar('critic_loss_{}'.format(env_name),  c, global_step=total_steps)
                    
                            
                else:
                    for i in range(env.number_of_agents):
                        replay_buffer.store(s_format[i], actions_format[i], actions_logprobs_format[i], r_format[i], s_next_format[i], dw_format[i], done_format[i])
                        if replay_buffer.count == args.batch_size: 
                            a,c=agent.update(replay_buffer, total_steps)
                            writer.add_scalar('actor_loss_{}'.format(env_name),  a, global_step=total_steps)
                            writer.add_scalar('critic_loss_{}'.format(env_name),  c, global_step=total_steps)
                            replay_buffer.count = 0
            
            
            
            
            s = s_
            

            
            
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            # for i in range(args.agent_num):
            #     if replay_buffers[i].count == args.batch_size:
            #         agents[i].update(replay_buffers[i], total_steps)
                    # replay_buffers[i].count = 0

            # if replay_buffer.count == args.batch_size:
            #     agent.update(replay_buffer, total_steps)
            #     replay_buffer.count = 0
            
            
            # Evaluate the policy every 'evaluate_freq' steps
            # print("total_steps",total_steps)
            
            # if total_steps%100==0:
            #     # now = datetime.now()
            #     now = datetime.now().strftime("%Y-%m-%d %H:%M")
            #     # print("total_steps",total_steps,"reward",r[0],r[1],r[2],"s",state_to_print[:3])
            #     print("time",now,"total_steps",total_steps,"reward",r,"s",state_to_print)#state_to_print
            if (not all(done.values()))==False:
                eposide_collisions_training.append(eposide_collision_training)
                 # 记录碰撞率 collision_rate_training
                writer.add_scalar('step_collisions_training_{}'.format(env_name),  eposide_collisions_training[-1], global_step=total_steps)
                
                # 3 fenbiejilu
                writer.add_scalar('avg_sum_reward_0_{}'.format(env_name), avg_sum_reward0, global_step=total_steps)
                writer.add_scalar('avg_sum_reward_1_{}'.format(env_name), avg_sum_reward1, global_step=total_steps)
                writer.add_scalar('avg_sum_reward_2_{}'.format(env_name), avg_sum_reward2, global_step=total_steps)
                writer.add_scalar('train_rewards_{}'.format(env_name), avg_sum_reward0+avg_sum_reward1+avg_sum_reward2, global_step=total_steps)
                
                # 记录strength sum_strength
                writer.add_scalar('strength_sum{}'.format(env_name), sum_strength, global_step=total_steps)
                
                now = datetime.now().strftime("%H:%M")
                print(now,str(total_steps)+"/"+str(args.max_train_steps)+":"+str(episode_steps),state_to_print,"mean_reward",(avg_sum_reward0+avg_sum_reward1+avg_sum_reward2)/3)#state_to_print
            
            if total_steps==1 or total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward=0
                evaluate_collision=0
                # for i in range(args.agent_num):
                #     evaluate_reward+=evaluate_policy(args, env_evaluates[i], agents[i], state_norm)
                evaluate_reward,evaluate_collision,mu_map,sig_map,path=evaluate_policy(args, env_evaluate, agent, state_norm)
                
                
                evaluate_reward=evaluate_reward/args.agent_num
                evaluate_collision=evaluate_collision/args.agent_num
                
                # evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                evaluate_collisions.append(evaluate_collision)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))

                # now = datetime.now()
                # print("time",now,"total_steps",total_steps,"reward",r,"s",state_to_print)#state_to_print
                
                writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
                # average_reward = np.mean(evaluate_rewards[-1])  # Calculate the average
                # writer.add_scalar('step_rewards_{}'.format(env_name), average_reward, global_step=total_steps)
                
                # 记录碰撞率 collision_rate
                writer.add_scalar('step_collisions_{}'.format(env_name), evaluate_collisions[-1], global_step=total_steps)
                  
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    np.save('./data_train/PPO_discrete_env_{}_number_{}_seed_{}.npy'.format(env_name, number, seed), np.array(evaluate_rewards))
                
                # Save the agent network
                # 保存模型
                # for i in range(args.agent_num):
                #     torch.save(agents[i].actor.state_dict(),'./model/actor_'+str(env_name)+'_'+str(number)+'_'+str(i)+'_model.pth')
                #     torch.save(agents[i].critic.state_dict(),'./model/critic_'+str(env_name)+'_'+str(number)+'_'+str(i)+'_model.pth')
                if evaluate_rewards[-1]>bestReward:
                    bestReward=evaluate_rewards[-1]
                    torch.save(agent.actor.state_dict(),'./model/actor_'+str(env_name)+'_'+str(number)+'_'+str(i)+'_model_best.pth')
                    torch.save(agent.critic.state_dict(),'./model/critic_'+str(env_name)+'_'+str(number)+'_'+str(i)+'_model_best.pth')
                    # 同时记录一下路径和map
                    file_path = "best_mu_map_"+str(args.number)+"_eva.txt"
                    np.savetxt(file_path, mu_map)
                    file_path = "best_sigma_map_"+str(args.number)+"_eva.txt"
                    np.savetxt(file_path, sig_map)
                    file_path1 = "best_path_1_"+str(args.number)+"_eva.txt"
                    file_path2 = "best_path_2_"+str(args.number)+"_eva.txt"
                    file_path3 = "best_path_3_"+str(args.number)+"_eva.txt"
                    path1=[]
                    path2=[]
                    path3=[]
                    for p in path:
                        path1.append(p[0])
                        path2.append(p[1])
                        path3.append(p[2])
                    
                    np.savetxt(file_path1, path1)
                    np.savetxt(file_path2, path2)
                    np.savetxt(file_path3, path3)
                # env.gt.step() #v1  4.1
                # torch.save(agent.state_dict(), 'your_model_path.pth')

            if (not all(done.values())==False):
                save_map(args,gp_map)   
    #输出最终结果
    # print(groups)
    # print('ans_max',ans_max,'ans_state',ans_state)
    # save_map(gp_map)
    #输出最后一回合它的路径  reward_record
    
    # # agent.actor_loss
    # print('len',len(agents[0].actor_loss))
    # # input()
    # plt.plot(np.arange(len(agents[0].actor_loss)), agent.actor_loss)
    # plt.ylabel('actor loss')
    # plt.xlabel('steps')
    # # plt.savefig('figs/actor_loss.png',dpi='1000')
    # plt.show()

    # plt.plot(np.arange(len(agent.critic_loss)), agent.critic_loss)
    # plt.ylabel('critic loss')
    # plt.xlabel('steps')
    # # plt.savefig('figs/critic_loss.png',dpi='1000')
    # plt.show()
    
    # print()
    # plt.plot(np.arange(len(reward_record)), reward_record)
    # plt.ylabel('Reward')
    # plt.xlabel('steps')
    # # plt.savefig('figs/reward_record40.png')
    # plt.show()
    
    # 41 单用户(48,3)
    # 42 单用户(40,3)
    # 43 双用户(40,3),(44,0)
    
    # 45 10 users
    for i in range(args.agent_num):
        filepath="./log/filename_"+str(number)+"_"+str(i)+".txt"
        with open(filepath, "a") as file:
            print('len(path_)'+str(i),len(paths[i]))
            file.write(str(paths[i]))
            file.close()    
    # with open("./log/filename_101_1.txt", "a") as file:
    #     print('len(path_1)',len(path_1))
    #     file.write(str(path_1))
    #     file.close()
    # with open("./log/filename_101_2.txt", "a") as file:
    #     print('len(path_2)',len(path_2))
    #     file.write(str(path_2))
    #     file.close()
    # with open("./log/filename_101_3.txt", "a") as file:
    #     print('len(path_3)',len(path_3))
    #     file.write(str(path_3))
    #     file.close()
    
    ans_path=[]
    for i in range(args.agent_num):
        ans_path.append(paths[i][-1])
    return ans_path









