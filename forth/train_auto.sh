# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024095 --users_count 50 --local True --D 5
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024096 --users_count 50 --local True --D 10
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024097 --users_count 50 --local True --D 15
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024098 --users_count 50 --local True --D 15

# local
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024103 --users_count 50 --local True --D 5
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024103 --users_count 50 --local False --D 5

# random
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024110 --users_count 50 --local True --D 5 #--random True
# 
# 111 not work
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024112 --users_count 50 --local True --D 5 #--random True
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024113 --users_count 50 --local False --D 5 #--random True

# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024114 --users_count 50 --local True --D 5 
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024120 --users_count 50 --local True --D 5 
# 要不要在y坐标轴加个5  100/2-》50/1  去掉随机性  增大eva_freq

# 改部署、去随机：
# lr    3e-4 
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024121 --users_count 50 --local True --D 5 --lr 0.0003
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024122 --users_count 50 --local True --D 5 --lr 0.003
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024123 --users_count 50 --local True --D 5 --lr 0.03
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024124 --users_count 50 --local True --D 5 --lr 0.3

# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024125 --users_count 50 --local True --D 5 --lr 0.0001
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024126 --users_count 50 --local True --D 5 --lr 0.00003

# 改D 
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024130 --users_count 50 --local True --D 7 --lr 0.0003   # 
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024131 --users_count 50 --local True --D 12 --lr 0.0003  # not good

# 5 7 10 12 15
#(400, 2) (196, 2) #(81, 2) # 64 #36
#400 196 81 64 36
# 0.4 0.196 0.081 0.064 0.036

# 这里修一个bug ，env_envlate 没隔离，没吃reset会出错
# 还有一个bug ，距离为0 ，加0.001

#   这个实验不好做，
#   做两个实验，就是略微动态环境下，记录每一时刻的信号强度和覆盖率变化，localGP在使用信息增益情况下，和不使用信息增益使用信号强度，的区别。
# 每轮之间reset的时候 位置会发生改变
# none dynamic  反正都是每轮间隙移动，我直接让用户不移动
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024132 --users_count 50 --local True --D 5 --lr 0.0003 --re IG
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024133 --users_count 50 --local True --D 5 --lr 0.0003 --re SG
# dynamic 但是过程中动，无人机走一步，用户走一步，每轮用户回到原点
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024134 --users_count 50 --local True --D 5 --lr 0.0003 --re IG
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024135 --users_count 50 --local True --D 5 --lr 0.0003 --re SG
# very good 就是在移动过程中用户移动，但是凸显不了IG的优势

# 修改起始点，突出触发冗余，  只注重最终结果，还要考虑是不是加大冗余因子
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024136 --users_count 50 --local True --D 5 --lr 0.0003 --re IG 
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024137 --users_count 50 --local True --D 5 --lr 0.0003 --re SG 
# run two at once
# 132 和 137 做对比

# 是不是做一下平均 ，我现在一半奖励是探索 ，一半是现有信息增益（0.1）
python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024139 --users_count 50 --local True --D 5 --lr 0.0003 --re IS 
# 我在思考是不是应该均衡奖励
# 然后 最后评估的事模型估计和信号强度，二者都达到较有状态

# 不行我就找两个132 137得了


# 我又改回来了

#120  lr 3e-4

#121  lr 3e-3

#122  lr 3e-2

#122  lr 3e-1



# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024090 --users_count 50 --local True --D 5
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024091 --users_count 50 --local True --D 10
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024092 --users_count 50 --local True --D 15





# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024060 --users_count 39 --local True
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024061 --users_count 60 --local True
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024062 --users_count 50 --local True

# # global

# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024063 --users_count 39 --local False
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024064 --users_count 60 --local False
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024065 --users_count 50 --local False



# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024072 --users_count 60 --local True
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024073 --users_count 60 --local False

# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024074 --users_count 50 --local True
# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024075 --users_count 50 --local False



# python3 -W ignore forth/TrainingLocalDQLv4.py --number 2024080 --users_count 50 --local True
