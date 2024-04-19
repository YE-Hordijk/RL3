# Johan
'''
def Reinforce_Learn(pi, theta, eta):
    initialize theta
    while not converged:
        grad = 0
        for m in range(M)
            sample trace h_0{s_0,a_0,r_0,s_1,...,s_n+1} according to policy pi(a|s)
            R = 0
            for t in reversed(range(n))
                R = r_t + self.gamma * R
                grad += R* rho log pi(a_t|s_t)
        theta <- theta + eta * grad
    return pi
'''
import gymnasium as gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import argparse
from collections import namedtuple, deque
from itertools import count, product
from scipy.signal import savgol_filter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init

class Policy_Net(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(3, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, 8)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
        
class REINFORCE():
    def __init__(self):
        self.env = gym.make("LunarLander-v2", render_mode="human")
        self.max_episodes = 300
        
    def update_policy():
        
    def Reinforce_Learn(pi, theta, eta):
    #initialize theta
    for episode in range(self.max_episodes):
        state = env.reset()
        grad = 0
        for m in range(M)
            sample trace h_0{s_0,a_0,r_0,s_1,...,s_n+1} according to policy pi(a|s)
            R = 0
            for t in reversed(range(n))
                R = r_t + self.gamma * R
                #rho = entropy ding
                grad += R * rho * log * pi(a_t|s_t)
        theta = theta + eta * grad
    return pi
    
