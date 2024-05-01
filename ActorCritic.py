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

class ActorNet(nn.Module):
    def __init__(self, state_dim):
        super(ActorNet, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Dropout(0.10),
            nn.PReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, state):
        return nn.functional.softmax(self.actor(state), dim=-1)

class CriticNet(nn.Module):
    def __init__(self, state_dim):
        super(CriticNet, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Dropout(0.10),
            nn.PReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.critic(state)

class ActorCritic():
    def __init__(self, render_mode=None, bootstrapping=True, baseline_subtraction=True, steps=5):
        self.env = gym.make("LunarLander-v2", render_mode=render_mode)
        self.render_mode = render_mode
        self.state_dim = self.env.observation_space.shape[0]
        self.max_episodes = 3001
        self.gamma = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.0005
        self.n_step = steps

        self.bootstrapping = bootstrapping
        self.baseline_subtraction = baseline_subtraction

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_model = ActorNet(self.state_dim).to(self.device)
        self.critic_model = CriticNet(self.state_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.critic_lr)

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
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
                action_probs = self.actor_model(state_tensor)
                value = self.critic_model(state_tensor)
                m = Categorical(action_probs)
                action = m.sample()
                log_prob = m.log_prob(action)
                next_state, reward, terminated, truncated, info = self.env.step(action.item())
                ep_reward += reward

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                state = next_state

            self.rewards.append(ep_reward)

            returns = []
            if self.bootstrapping:
                n_return = 0
                for i in range(len(rewards)-1, -1, -1):
                    r = rewards[i]
                    for j in range(self.n_step):
                        if i + j < len(rewards):
                            n_return = r + self.gamma * n_return
                            r *= self.gamma
                    returns.insert(0, n_return)
            else:
                advantage = 0
                for r in rewards[::-1]:
                    # go through rewards in reverse direction, to use the correct value for advantage
                    advantage = r + self.gamma * advantage
                    returns.insert(0, advantage)

            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
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

            if episode % 1000 == 0 or self.render_mode == 'human':
                print(f"Episode: {episode}, Total Reward: {ep_reward}, Average: {np.mean(self.rewards[-100:])}")

        return self.rewards

if __name__ == "__main__":
    if 0:
        ac = ActorCritic("human")
    else:
        for i in range(1, 10):
            print("steps:", i)
            ac = ActorCritic(steps=i)
            ac.learn()
    rewards = ac.learn()
    np.save("ac.npy", rewards)
