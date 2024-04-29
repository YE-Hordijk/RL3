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
            nn.Linear(state_dim, 128),
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
    def __init__(self, render_mode=None):
        self.env = gym.make("LunarLander-v2", render_mode=render_mode)
        self.render_mode = render_mode
        self.state_dim = self.env.observation_space.shape[0]
        self.max_episodes = 1000
        self.discount = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.0005

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
            terminated = truncated = False

            while not (terminated or truncated):
                state_tensor = torch.from_numpy(state).unsqueeze(0)
                action_probs, value = self.model(state_tensor)
                m = Categorical(action_probs) # TODO: nieuw
                action = m.sample() # TODO: nieuw
                log_prob = m.log_prob(action)
                next_state, reward, terminated, truncated, info = self.env.step(action.item())
                ep_reward += reward

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                state = next_state

            self.rewards.append(ep_reward)

            returns = []
            advantage = 0
            for r in rewards[::-1]:
                # go through rewards in reverse direction, to use the correct value for advantage
                advantage = r + self.discount * advantage
                returns.insert(0, advantage)
            # TODO: dit begrijpen
            returns = torch.tensor(returns, dtype=torch.float32)
            log_probs = torch.cat(log_probs)
            values = torch.cat(values).squeeze()

            # compute actor and critic losses
            actor_loss = -(log_probs * (returns - values.detach())).mean()
            critic_loss = nn.functional.mse_loss(values, returns)
            total_loss = actor_loss + critic_loss

            # Optimize
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            if episode % 100 == 0 or self.render_mode == 'human':
                print(f"Episode: {episode}, Total Reward: {ep_reward}")

        return self.rewards

if __name__ == "__main__":
    if 0:
        ac = ActorCritic("human")
    else:
        ac = ActorCritic()
    policy = ac.learn()
    print("end", policy)
