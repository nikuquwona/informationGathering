
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import copy
import time
import  matplotlib.pyplot as plt
import gym
import  random
import math
import numpy as np


def gatherSignalInformation(state):
    # 用户真实的分布
    users=[(28.095737508379507, 9.262238181375238), (31.534268858415665, 11.965999116504157), (28.987847366016233, 14.938933089562964), (32.11286120677997, 12.201298108043847), (26.765425070521744, 9.164420662094894), (45.749778214291226, -2.3878504425633658), (45.396274248005305, -4.385207701641737), (50.642615825605006, -5.146976192917276), (48.26626883173278, -9.490511609965512), (54.425094652367605, -3.1927547997114782), (68.71555847946225, 16.95251215340648), (69.30532824493405, 18.830942106523402), (68.2989629979505, 17.05649377261501), (73.96507288663165, 21.54396584760398), (69.0837707884811, 23.828827829773275)]

    # 当前agent的位置
    agent_pos=state
    agent_pos_x=state[0]
    agent_pos_y=state[1]
    agent_pos_z=state[2]
    
    distance=[]
    # 绝对距离
    for user in users:
        point1=np.array([user[0],user[1],0])
        point2=np.array([agent_pos_x,agent_pos_y,agent_pos_z])
        distance_temp=np.linalg.norm(point1 - point2)
        distance.append(distance_temp)
    
    # y=Pt −FSPL(dB)
    def calculate_received_signal_strength(Pt, d, f, c=3e8):
        """
        Calculate the received signal strength based on the Free Space Path Loss model.

        :param Pt: Transmit power in dBm
        :param d: Distance between transmitter and receiver in meters
        :param f: Frequency of the signal in Hertz
        :param c: Speed of light in vacuum (default is 3e8 m/s)
        :return: Received signal strength in dBm
        """
        # Convert transmit power from dBm to linear scale
        Pt_linear = 10 ** (Pt / 10)

        # Calculate Free Space Path Loss (FSPL)
        FSPL = 20 * np.log10(d) + 20 * np.log10(f) + 20 * np.log10(4 * np.pi / c)

        # Calculate received power in linear scale and convert back to dBm
        Pr_linear = Pt_linear / (10 ** (FSPL / 10))
        Pr = 10 * np.log10(Pr_linear)

        return Pr

    Pt = 1  # Transmit power in dBm
    f =  1e9 # Frequency in Hz (e.g., 2.4 GHz for WiFi)
    received_signal_strength_sum=0
    
    for d in distance:
        
        # d = 100  # Distance in meters
    
        received_signal_strength = calculate_received_signal_strength(Pt, d, f)
        received_signal_strength_sum+=received_signal_strength

    return  received_signal_strength_sum
    
    pass


'''
    Pt = 30  # Transmit power in dBm
    d = 100  # Distance in meters
    f = 2.4e9  # Frequency in Hz (e.g., 2.4 GHz for WiFi)
'''