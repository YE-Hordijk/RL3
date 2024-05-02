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
    def __init__(self, episodes=3001, bootstrapping=True, baseline_subtraction=True, render_mode=None):
        self.env = gym.make("LunarLander-v2", render_mode=render_mode)
        self.render_mode = render_mode
        self.state_dim = self.env.observation_space.shape[0]
        self.max_episodes = episodes
        self.gamma = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.0005
        self.n_step = 5

        self.bootstrapping = bootstrapping
        self.baseline_subtraction = baseline_subtraction

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_model = ActorNet(self.state_dim).to(self.device)
        self.critic_model = CriticNet(self.state_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.critic_lr)

        self.rewards = []
    
    def learn(self, episodes):
        if episodes:
            max_episodes = episodes
        else:
            max_episodes = self.max_episodes
        for episode in range(max_episodes):
            log_probs = []
            values = []
            rewards = []

            ep_reward = self.episode(log_probs, values, rewards)

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

            if episode % 100 == 0 or self.render_mode == 'human':
                print(f"Episode: {episode}, Total Reward: {ep_reward}, Average: {np.mean(self.rewards[-100:])}")

        return self.rewards

    def episode(self, log_probs, values, rewards, deterministic=False):
        state, info = self.env.reset()
        ep_reward = 0
        terminated = truncated = False

        while not (terminated or truncated):
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            if deterministic:
                # TODO: maken
                action = self.actor_model.forward(state_tensor) # TODO: causes error because this gives a tensor
                value = self.critic_model.forward(state_tensor)
                log_prob = 0
            else:
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
        return ep_reward

def experiment(nrEpisodes, interval, nrTestEpisodes):
    ac = ActorCritic()
    rewards = []
    # per iteration, we perform `interval` episodes, so divide nrEpisodes by interval
    for i in range(nrEpisodes // interval):
        # first perform `interval` episodes of learning
        ac.learn(episodes=interval)
        # then perform `nrTestEpisodes` test episodes deterministically to get results for this nr of episodes
        avg = 0
        for _ in range(nrTestEpisodes):
            avg += ac.episode([], [], [], deterministic=True)
        rewards.append(avg / nrTestEpisodes)
        # and then repeat, 
    return rewards

if __name__ == "__main__":
    if 0:
        ac = ActorCritic("human")
    else:
        ac = ActorCritic()
    rewards = ac.learn()
    np.save("ac.npy", rewards)
