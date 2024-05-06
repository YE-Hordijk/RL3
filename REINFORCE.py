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
        self.layer1 = nn.Linear(8, 128)
        self.dropout = nn.Dropout(p=0.4)
        self.layer2 = nn.Linear(128, 4)
        #self.layer1 = nn.Linear(8, 128)
        #self.layer2 = nn.Linear(128, 128)
        #self.layer3 = nn.Linear(128, 4)

        #self.saved_log_probs = []
        #self.rewards = []

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.layer2(x)
        return F.softmax(action_scores, dim=1)
        #x = F.relu(self.layer1(x))
        #x = F.relu(self.layer2(x))
        #x = self.layer3(x)
        #x = F.softmax(x, dim=1)#.to(torch.float64)
        #return x
        
class REINFORCE():
    def __init__(self, LearningRate=0.01, epsilon = 0.1):
        self.env = gym.make("LunarLander-v2")#, render_mode="human")
        self.max_episodes = 1000
        self.gamma = 0.99
        self.epsilon = 0.1
        self.LearningRate = 0.01
        self.PolicyNet = Policy_Net()
        self.optimizer = optim.Adam(self.PolicyNet.parameters(), lr=self.LearningRate)
        self.eps = np.finfo(np.float32).eps.item()
        self.totalrewards = []
        self.losses = []
        
        
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.PolicyNet(state)
        #print(probs)
        if torch.isnan(probs).any(): # Check for NaN values in the probabilities
            print(f"\033[41X\033[0m", end="")
            probs = torch.tensor([0.25, 0.25, 0.25, 0.25], requires_grad=True)
            softmax_tensor = F.softmax(probs, dim=0)
            probs = softmax_tensor.view(1, -1)
            print(probs)
        entropy = (- probs * torch.log(probs)).sum()
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob, entropy
        
    def select_action_tomke(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        action_probs = self.PolicyNet(state).squeeze()
        print(action_probs)
        log_prob = torch.log(action_probs)
        entropy = (- action_probs * torch.log(action_probs)).sum()
        cpu_action_probs = action_probs.detach().cpu().numpy()
        action = np.random.choice(np.arange(4), p=cpu_action_probs)
        log_prob = log_prob[action]
        return action, log_prob, entropy
        
        
    def Reinforce_Learn(self, episodes=None):
        if episodes:
            max_episodes = episodes
        else:
            max_episodes = self.max_episodes
        for m in range(max_episodes):
            #state, info = self.env.reset()
            #totalreward = 0
            reward_t = []
            log_probs = []
            entropies = []
            #terminated, truncated = [False, False]
            print("ep", m)
            totalreward = self.episode(reward_t, log_probs, entropies)
            self.totalrewards.append(totalreward)
            '''
            while not (terminated or truncated):#for t in range(1, 1000):#
                action, log_prob, entropy = self.select_action(state)
                state, reward, terminated, truncated, info = self.env.step(action)
                reward_t.append(reward)
                totalreward += reward
                log_probs.append(log_prob)
                entropies.append(entropy)
                if (terminated or truncated):
                    self.totalrewards.append(totalreward)
                    break
            '''
            R = 0
            grad = []
            returns = deque()
            for t in reversed(range(len(reward_t))):
                R = reward_t[t] + self.gamma * R
                returns.appendleft(R)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std())# + self.eps)
            for log_prob, R, entropy in zip(log_probs, returns, entropies):
                grad.append((-log_prob * R)-(self.epsilon*entropy))
            self.optimizer.zero_grad()
            grad = torch.cat(grad).sum()
            grad.backward()
            self.optimizer.step()
        return self.totalrewards

    def episode(self, reward_t, log_probs, entropies, deterministic=False):
        state, info = self.env.reset()
        totalreward = 0
        terminated, truncated = [False, False]
        while not (terminated or truncated):#for t in range(1, 1000):#
            action, log_prob, entropy = self.select_action(state)
            state, reward, terminated, truncated, info = self.env.step(action)
            reward_t.append(reward)
            totalreward += reward
            log_probs.append(log_prob)
            entropies.append(entropy)
        return totalreward

def experiment(nrEpisodes, interval, nrTestEpisodes):
    r = REINFORCE()
    rewards = []
    # per iteration, we perform `interval` episodes, so divide nrEpisodes by interval
    for i in range(nrEpisodes // interval):
        # first perform `interval` episodes of learning
        r.Reinforce_Learn(episodes=interval)
        # then perform `nrTestEpisodes` test episodes deterministically to get results for this nr of episodes
        avg = 0
        for _ in range(nrTestEpisodes):
            avg += r.episode([], [], [], deterministic=True)
        rewards.append(avg / nrTestEpisodes)
        print((i+1)*interval, '=', avg/nrTestEpisodes)
        # and then repeat
    return rewards
    
def parameters(nrEpisodes, interval, nrTestEpisodes, LearningRate, epsilon):
    r = REINFORCE(LearningRate, epsilon)
    rewards = []
    # per iteration, we perform `interval` episodes, so divide nrEpisodes by interval
    for i in range(nrEpisodes // interval):
        # first perform `interval` episodes of learning
        r.Reinforce_Learn(episodes=interval)
        # then perform `nrTestEpisodes` test episodes deterministically to get results for this nr of episodes
        avg = 0
        for _ in range(nrTestEpisodes):
            avg += r.episode([], [], [], deterministic=True)
        rewards.append(avg / nrTestEpisodes)
        print((i+1)*interval, '=', avg/nrTestEpisodes)
        # and then repeat
    return rewards

if __name__ == "__main__":
    if 0:
        r = REINFORCE("human")
    else:
        r = REINFORCE()
    rewards = r.Reinforce_Learn()
    np.save("reinforce.npy", rewards)
