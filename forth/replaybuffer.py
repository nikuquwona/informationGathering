import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, args):
        self.s = np.zeros((args.batch_size,*args.state_dim))
        self.a = np.zeros((args.batch_size, 1))
        self.a_logprob = np.zeros((args.batch_size, 1))
        self.r = np.zeros((args.batch_size, 1))
        
        self.s_ = np.zeros((args.batch_size, *args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0
        

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        # self.r[self.count] = r.flatten()[0]  # Flatten and get the first element

        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1
        # self.len+=1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.long)  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done


class PriorityReplayBuffer:
    def __init__(self, args):
        self.s = np.zeros((500_000,*args.state_dim))
        self.a = np.zeros((500_000, 1))
        self.a_logprob = np.zeros((500_000, 1))
        self.r = np.zeros((500_000, 1))
        self.s_ = np.zeros((500_000, *args.state_dim))
        self.dw = np.zeros((500_000, 1))
        self.done = np.zeros((500_000, 1))
        
        # 添加优先级数组
        self.priorities = np.zeros(500_000)
        
        self.count = 0
        self.len=0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        
        # 存储优先级
        # new is always important 
        self.priorities[self.count] = 10#0.1 # default
        
        self.count += 1
        self.len   += 1
        if self.count==500_000:
            self.count = 0          

    def sample(self, batch_size):
        # 根据优先级进行采样
        probabilities = self.priorities / np.sum(self.priorities)
        indices = np.random.choice(np.arange(min(self.len,500_000)), size=batch_size, replace=False, p=probabilities[:min(self.len,500_000)])
        # indices = np.random.choice(np.arange(min(self.len,500_000)), size=batch_size, replace=False)
        
        s = torch.tensor(self.s[indices], dtype=torch.float)
        a = torch.tensor(self.a[indices], dtype=torch.long)
        a_logprob = torch.tensor(self.a_logprob[indices], dtype=torch.float)
        r = torch.tensor(self.r[indices], dtype=torch.float)
        s_ = torch.tensor(self.s_[indices], dtype=torch.float)
        dw = torch.tensor(self.dw[indices], dtype=torch.float)
        done = torch.tensor(self.done[indices], dtype=torch.float)
        
        return s, a, a_logprob, r, s_, dw, done,indices
    def update_priorities_batch(self,indices ,adv):
        '''
        adv tensor([[ 4.9820],
                    [ 3.7248],
                    [ 0.6749],
                    ...,
                    [-0.7104],
                    [-0.6963],
                    [-0.9336]])
        adv torch.Size([1024, 1])
        '''
         # 将adv转换为numpy数组
        adv = adv.cpu().detach()#.numpy()
        
        adv_mean = torch.mean(adv)
        adv_std = torch.std(adv)
        normalized_adv = (adv - adv_mean) / (adv_std + 1e-8)
        # print("normalized_adv.",normalized_adv.min())
        # 加上常数以确保优先级为非负值
        priorities = normalized_adv + 5
        
        # 更新经验池中对应索引的优先级
        for idx, priority in zip(indices, priorities):
            if priority<0:
                priority=0
            self.priorities[idx] = priority
        # print("test")
        # print("self.priorities",self.priorities[indices])
        # input()
        # 补充 
    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.long)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done