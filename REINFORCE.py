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

from torch.autograd import Variable
from torch.distributions import Categorical

class Policy_Net(nn.Module):
    def __init__(self):
        super(Policy_Net, self).__init__()
        self.layer1 = nn.Linear(8, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        x = nn.functional.softmax(x, dim=1)
        return x
        
class REINFORCE():
    def __init__(self):
        self.env = gym.make("LunarLander-v2", render_mode="human")
        self.max_episodes = 10
        self.gamma = 0.99
        self.LearningRate = 0.001
        self.PolicyNet = Policy_Net()
        self.optimizer = optim.AdamW(self.PolicyNet.parameters(), lr=self.LearningRate, amsgrad=True)
        
    def select_action(self, state):
        #dit misschien nog anders doen?
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.PolicyNet(state).squeeze()
        log_probs = torch.log(action_probs)
        cpu_action_probs = action_probs.detach().cpu().numpy()
        action = np.random.choice(np.arange(4), p=cpu_action_probs)
        return action
        
    #def update_parameters():
        
    def Reinforce_Learn(self):#(pi, theta, eta):
    #initialize theta
        done = False
        count = 0
        while not done:
            state, info = self.env.reset()
            grad = 0
            totalreward = 0
            state_t = []
            reward_t = []
            for m in range(self.max_episodes):
                count += 1
                action = self.select_action(state)
                #sample trace h_0{s_0,a_0,r_0,s_1,...,s_n+1} according to policy pi(a|s)
                state_t.append(torch.from_numpy(state).float().unsqueeze(0))
                next_state, reward, terminated, truncated, info = self.env.step(action)
                totalreward += reward
                reward_t.append(reward)
                R = 0
                for t in reversed(range(count)):
                    R = reward_t[t] + self.gamma * R
                    print("R", R)
                    #rho = entropy ding
                    action_probs = self.PolicyNet(state_t[t]).squeeze()
                    entropy = ()
                    grad += R * torch.log(self.PolicyNet(state_t[t]).squeeze()) #*rho
                    print("grad", grad)
            #theta = theta + eta * grad
                state = next_state
            done = True
        return pi


    
