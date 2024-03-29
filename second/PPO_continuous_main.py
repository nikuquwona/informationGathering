from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous ,Actor_Gaussian,Critic
from env import env as ENV

def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        step=0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ = env.step(action)
            step+=1
            if step==args.max_episode_steps:
                done=True
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times
def evaluate_policy_eval(args, env, agent, state_norm):
    path=[]
    
    
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        step=0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ = env.step(action)
            ###记录path
            path.append(s_.tolist())
            ###
            step+=1
            if step==args.max_episode_steps:
                done=True
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times,path

def group_points(coordinates, M, random_seed=0):
        # 创建K-means模型   
        kmeans = KMeans(n_clusters=M,random_state=random_seed)

        # 拟合模型
        kmeans.fit(coordinates)

        # 获取每个点的标签
        labels = kmeans.labels_

        # 根据标签分组
        groups = [[] for _ in range(M)]
        for i, label in enumerate(labels):
            groups[label].append(coordinates[i])

        return groups

def main(args, env_name, number, seed
         ,agent_num=3
         ,user_num=2
         ,users_pos=[(48,-3),(48,3)]
         ,eval=False
         ):
    
    # Set random seed
    # env.seed(seed)
    # env.action_space.seed(seed)
    # env_evaluate.seed(seed)
    # env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    args.agent_num=agent_num
    
    # users=[(40,3),(44,0),(42,1),(50,5)]
    users=[(28.095737508379507, 9.262238181375238), (31.534268858415665, 11.965999116504157), (28.987847366016233, 14.938933089562964), (32.11286120677997, 12.201298108043847), (26.765425070521744, 9.164420662094894), (45.749778214291226, -2.3878504425633658), (45.396274248005305, -4.385207701641737), (50.642615825605006, -5.146976192917276), (48.26626883173278, -9.490511609965512), (54.425094652367605, -3.1927547997114782), (68.71555847946225, 16.95251215340648), (69.30532824493405, 18.830942106523402), (68.2989629979505, 17.05649377261501), (73.96507288663165, 21.54396584760398), (69.0837707884811, 23.828827829773275)]
    users=users_pos
    groups =group_points(users, args.agent_num)
    
    print("groups",groups)
    
    # input()
    #[[(50, 5)], [(44, 0), (42, 1)], [(40, 3)]]
    # [
    #  [(68.71555847946225, 16.95251215340648), (69.30532824493405, 18.830942106523402), (68.2989629979505, 17.05649377261501), (73.96507288663165, 21.54396584760398), (69.0837707884811, 23.828827829773275)], 
    #  [(45.749778214291226, -2.3878504425633658), (45.396274248005305, -4.385207701641737), (50.642615825605006, -5.146976192917276), (48.26626883173278, -9.490511609965512), (54.425094652367605, -3.1927547997114782)], 
    #  [(28.095737508379507, 9.262238181375238), (31.534268858415665, 11.965999116504157), (28.987847366016233, 14.938933089562964), (32.11286120677997, 12.201298108043847), (26.765425070521744, 9.164420662094894)]
    # ]
    # input()
    start_pos=[(30,15,5),(50,0,5),(70,15,5)]
    envs=[]
    env_evaluates=[]
    for i in range(args.agent_num):
        envs.append(ENV(groups[i]),start_pos[i])
        env_evaluates.append(ENV(groups[i]),start_pos[i])
    # env = ENV([(40,3),(44,0),(42,1)])
    # env_evaluate = ENV([(40,3),(44,0),(42,1)])
    
    
    args.state_dim = 13#8#env.observation_space.shape[0]
    args.action_dim = 13#8#env.action_space.shape[0]
    args.max_action = 1#float(env.action_space.high[0]) # -1,1
    args.max_episode_steps = 10000#env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffers=[]
    agents=[]
    for _ in range(args.agent_num):
        replay_buffers.append(ReplayBuffer(args))
        agents.append(PPO_continuous(args))
    # replay_buffer = ReplayBuffer(args)
    # agent = PPO_continuous(args)

    ## load
        # 保存和加载模型还没有

    ##
    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}'.format(env_name, args.policy_dist, number, seed))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    ans_max=-999
    ans_state=[]
    ###
    reward_record=[]
    
    paths=[]
    for _ in range(args.agent_num):
        paths.append([])
        # path_1=[]
        # path_2=[]
        # path_3=[]
    ###
    while total_steps < args.max_train_steps:
        s=[]
        for env in envs:
            s.append(env.reset())
        # print(s)
        # input()
        # s = env.reset() # 返回的是什么
        # path=[]
        for i in range(args.agent_num):
            paths[i]=[]
            paths[i].append(s[i].tolist())
        # path_1=[]
        # path_2=[]
        # path_3=[]
        
        # path_1.append(s[0].tolist())
        # path_2.append(s[1].tolist())
        # path_3.append(s[2].tolist())
        
        if args.use_state_norm:
            for i in range(args.agent_num):
                s[i]=state_norm(s[i])
            # s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        ddone = False
        
        if eval==True:
            # load model
            for i in range(args.agent_num):
                agents[i].actor.load_state_dict(torch.load('./model/actor_'+str(env_name)+'_'+str(number)+'_'+str(i)+'_model.pth'))
                print("********加载成功***********")
                print()
                # './model/actor_'+str(env_name)+'_'+str(number)+'_'+str(i)+'_model.pth'
                # critic.load_state_dict(torch.load('critic_model.pth'))
                _,res_path=evaluate_policy_eval(args, env_evaluates[i], agents[i], state_norm)
                paths[i].extend(res_path)
            break
            pass
        
        while not ddone:
            episode_steps += 1
            a=[]
            a_logprob=[]
            for i in range(args.agent_num):
                t1,t2=agents[i].choose_action(s[i])
                a.append(t1)
                a_logprob.append(t2)
                
            # a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            # print("a",a,"a_logprob",a_logprob)
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
                
            s_=[]
            r=[]
            done=[]
            
            for i in range(args.agent_num):
                t1, t2, t3, _ =envs[i].step(action[i])
                s_.append(t1)
                r.append(t2)
                done.append(t3)
                
            # s_, r, done, _ = env.step(action)

            ###
            if total_steps>=args.max_train_steps-args.max_episode_steps:
                reward_record.append(r[0])
            
            for i in range(args.agent_num):
                # paths[i]=[]
                paths[i].append(s_[i].tolist())
            # path_1.append(s_[0].tolist())
            # path_2.append(s_[1].tolist())
            # path_3.append(s_[2].tolist())
            
            # print('path',path)
            state_to_print=s_[0]#s_[0]
            if episode_steps==args.max_episode_steps:
                for i in range(args.agent_num):
                    done[i] =True
                ddone=True
                r_sum=0
                for i in range(args.agent_num):
                    r_sum+=r[i]
                if r[0]>ans_max:
                    ans_max=r[0]
                    ans_state=state_to_print
            ###
            for i in range(args.agent_num):
                if args.use_state_norm:
                    s_[i] = state_norm(s_[i])
                if args.use_reward_norm:
                    r[i] = reward_norm(r[i])
                elif args.use_reward_scaling:
                    r[i] = reward_scaling(r[i])

            
            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done[0] and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            # Take the 'action'，but store the original 'a'（especially for Beta）
            # replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            for i in range(args.agent_num):
                replay_buffers[i].store(s[i], a[i], a_logprob[i], r[i], s_[i], dw, done[i])
                s[i]=s_[i]
            # s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            for i in range(args.agent_num):
                if replay_buffers[i].count == args.batch_size:
                    agents[i].update(replay_buffers[i], total_steps)
                    replay_buffers[i].count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            # if total_steps%1000==0:
            #     print("total_steps",total_steps,"reward",r,"s",state_to_print[:3])
                
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward=0
                for i in range(args.agent_num):
                    evaluate_reward+=evaluate_policy(args, env_evaluates[i], agents[i], state_norm)
                evaluate_reward=evaluate_reward/args.agent_num
                
                # evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                
                print("total_steps",total_steps,"reward",r,"s",state_to_print)#state_to_print
                
                writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
                  
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    np.save('./data_train/PPO_continuous_{}_env_{}_number_{}_seed_{}.npy'.format(args.policy_dist, env_name, number, seed), np.array(evaluate_rewards))
                
                # Save the agent network
                # 保存模型
                for i in range(args.agent_num):
                    torch.save(agents[i].actor.state_dict(),'./model/actor_'+str(env_name)+'_'+str(number)+'_'+str(i)+'_model.pth')
                    torch.save(agents[i].critic.state_dict(),'./model/critic_'+str(env_name)+'_'+str(number)+'_'+str(i)+'_model.pth')

                
                # torch.save(agent.state_dict(), 'your_model_path.pth')

    #输出最终结果
    print(groups)
    print('ans_max',ans_max,'ans_state',ans_state)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(300), help=" Maximum number of training steps")
    # 经过测试 700k 就差不多  将 3e6改为7e5
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    env_name = ['test','BipedalWalker-v3', 'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']
    env_index = 0
    main(args, env_name=env_name[env_index], number=101, seed=10)
    
    # 26 yidong
    # 27 buyi dong

    # 28 -1 -1 -1 (48,20)
    # 29 -1 -1 -1 (48,20) (48,-20)
    
    
    
    
    
    # ***************
    # 画出轨迹图🗺️，
    # 先考虑单用户情况，看一下它的最终优化轨迹如何
    # 然后考虑集中的多用户情况，画一个圈，把最终优化轨迹展现
    
    