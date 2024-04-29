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
    def __init__(self, state_dim):
        super(Policy_Net, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, 4)
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        x = self.common(state)
        policy = nn.functional.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return policy, value

class ActorCritic():
    def __init__(self):
        self.env = gym.make("LunarLander-v2")#, render_mode="human")
        self.state_dim = self.env.observation_space.shape[0]
        self.max_episodes = 1000
        self.discount = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        self.model = Policy_Net(self.state_dim)
        self.actor_optimizer = optim.Adam(self.model.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.model.parameters(), lr=self.critic_lr)

        self.rewards = []
    
    def learn(self):
        for episode in range(self.max_episodes):
            state, info = self.env.reset()
            ep_reward = 0
            log_probs = []
            values = []
            rewards = []
            terminated = truncated = false

            while not (terminated or truncated):
                state_tensor = torch.from_numpy(state).unsqueeze(0)
                action_probs, value = self.model(state_tensor).squeeze()
                m = Categorical(action_probs) # TODO: nieuw
                action = m.sample() # TODO: nieuw
                log_prob = m.log_prob(action)
                next_state, reward, truncated, info = self.env.step(action.item())
                ep_reward += reward

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                state = next_state

            self.rewards.push(ep_reward)

            # TODO: dit begrijpen
            returns = []
            advantage = 0
            for r in rewards[::-1]:
                advantage = r + discount_factor * advantage
                returns.insert(0, advantage)
            returns = torch.tensor(returns)
            log_probs = torch.cat(log_probs)
            values = torch.cat(values)

            # compute actor and critic losses
            actor_loss = -(log_probs * (returns - values.detach())).mean()
            critic_loss = nn.functional.mse_loss(values, returns)
            total_loss = actor_loss + critic_loss

            # Optimize
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            total_loss.backward()
            actor_optimizer.step()
            critic_optimizer.step()

            if episode % 100 == 0:
                print(f"Episode: {episode}, Total Reward: {episode_reward}")

        return self.rewards

ac = ActorCritic()
policy = ac.learn()
print("end", policy)
