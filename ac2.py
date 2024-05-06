# New version of Actor Critic based on the 4-5-2024 version of REINFORCE

# Jace
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

#*******************************************************************************
class ActorNet(nn.Module): # Policy network
    def __init__(self):
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(8, 128), 
            nn.Dropout(p=0.2), 
            nn.PReLU(), 
            nn.Linear(128, 4),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.actor(x)

#*******************************************************************************

class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(8, 128),
            nn.Dropout(0.15),
            nn.PReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.critic(state)

#*******************************************************************************

class ActorCritic():
    def __init__(self, bootstrapping=False, baseline_subtraction=True, render_mode=None, epsilon = 0.01):
        self.env = gym.make("LunarLander-v2")#, render_mode="human")
        #self.env = gym.make("CartPole-v1")
        self.max_episodes = 800
        self.gamma = 0.99
        self.actor_lr = 0.005
        self.critic_lr = 0.05

        self.bootstrapping = bootstrapping
        self.baseline_subtraction = baseline_subtraction
        
        self.render_mode = render_mode
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_model = ActorNet()#.to(self.device)
        self.critic_model = CriticNet()#.to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.critic_lr)
        self.n_step = 5


    #===========================================================================

    def select_action(self, state_tensor):
        probs = self.actor_model(state_tensor) # Put state through network and get action probabilities
        if torch.isnan(probs).any(): # Check for NaN values in the probabilities
            print(f"\033[41X\033[0m", end="")
            probs = torch.tensor([0.25, 0.25, 0.25, 0.25], requires_grad=True)
            softmax_tensor = F.softmax(probs, dim=0)
            probs = softmax_tensor.view(1, -1)

        entropy = (- probs * torch.log(probs)).sum() # Calculate the entropy of the action distribution
        m = Categorical(probs) # Create a Categorical distribution from the action probabilities
        action = m.sample() # Sample an action from the categorical distribution
        log_prob = m.log_prob(action) # Calculate the log probability of the sampled action
        return action.item(), log_prob, entropy 

    #===========================================================================

    def learn(self, episodes=None):
        max_episodes = episodes if episodes else self.max_episodes
        total_rewards = [] # Keep track of total rewards per episode for evaluation # XXX DIT WEGHALEN TOCH VOOR UITEINDELIJKE EVAUATIE?

        for episode in range(max_episodes+1):
            rewards = [] # Returns from the environment
            values = [] # Returns from the critic network
            log_probs = []
            entropies = []

            ep_total_reward = self.episode(rewards, log_probs, entropies, values)
            total_rewards.append(ep_total_reward)

            #=================== Calculating the Returns =======================

            returns = []
            n_return = 0

            if self.bootstrapping: # With bootstrapping
                for t in range(0, len(rewards)):
                    end_index = min(t+self.n_step, len(rewards)-1)
                    discounted_r = [r * self.gamma ** index for index, r in enumerate(rewards[t: end_index])]
                    returns.append( sum( discounted_r )  + ((self.gamma ** self.n_step) * values[end_index-1]) )

            else: # No bootstrapping
                for r in rewards[::-1]:
                    n_return = r + self.gamma * n_return # go through rewards in reverse direction, to use the correct value for advantage
                    returns.insert(0, n_return)

            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std()) # Normalize

            #====================== Computing Losses ===========================

            # Converting to torch tensors
            returns = returns.clone().detach() # XXX IPV returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            log_probs = torch.cat(log_probs)
            values = torch.cat(values).squeeze(-1) # XXX Maybe remove the -1

            # Determine advantages
            if self.baseline_subtraction: advantages = returns - values.detach()
            else: advantages = returns
            advantages = (advantages - advantages.mean()) / advantages.std() # Normalize


            # Compute actor and critic losses
            actor_loss = -(log_probs * (advantages)).sum()
            critic_loss = nn.functional.mse_loss(values, returns) # F.smooth_l1_loss(returns, values).sum()
            #critic_loss = F.smooth_l1_loss(returns, values).sum()


            # Cast critic_loss to right datatype https://discuss.pytorch.org/t/backward-does-not-work/168732
            critic_loss = torch.tensor([critic_loss.item()]).to(dtype = torch.float32).requires_grad_(True)




            # Set the gradients to zero
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            # Backpropagation
            actor_loss.backward()
            critic_loss.backward()

            # Gadient clipping
            clipping_value = 1 # chosen arbitrary value
            torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), clipping_value)
            torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), clipping_value)

            # Optimization (Update) the network
            self.actor_optimizer.step()
            self.critic_optimizer.step()



            if episode > 100 and (episode % 10 == 0 or self.render_mode == 'human'):
                print(f"Episode: {episode}, Total Reward: {ep_total_reward}, Average: {np.mean(total_rewards[-100:])}")

        return total_rewards

    #===========================================================================

    def episode(self, rewards, log_probs, entropies, values, deterministic=False):
        state, info = self.env.reset() # Reset the environment
        ep_total_reward = 0 # The cumulative reward within this episode
        terminated = truncated = False # Set termination criteria

        while not (terminated or truncated): # Run episode until the lander crashes or max timestaps is reached
            #state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device) # Convert state to float type PyTorch tensor & add batch dimension
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            # Get critic (value) prediction form the network
            value = self.critic_model(state_tensor) # TODO 
            
            action, log_prob, entropy = self.select_action(state_tensor)
            next_state, reward, terminated, truncated, info = self.env.step(action) # Take a step in the environment


            ep_total_reward += reward # Add the reward to the reward for this episode

            log_probs.append(log_prob)
            values.append(value) # TODO 
            rewards.append(reward)
            entropies.append(entropy)
            state = next_state
        return ep_total_reward

#*******************************************************************************

def experiment(nrEpisodes, interval, nrTestEpisodes):
    r = ActorCritic()
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

#*******************************************************************************

def parameters(nrEpisodes, interval, nrTestEpisodes, epsilon):
    r = ActorCritic(epsilon)
    rewards = []
    # per iteration, we perform `interval` episodes, so divide nrEpisodes by interval
    for i in range(nrEpisodes // interval):
        # first perform `interval` episodes of learning
        r.learn(episodes=interval)
        # then perform `nrTestEpisodes` test episodes deterministically to get results for this nr of episodes
        avg = 0
        for _ in range(nrTestEpisodes):
            avg += r.episode([], [], [], deterministic=True)
        rewards.append(avg / nrTestEpisodes)
        print((i+1)*interval, '=', avg/nrTestEpisodes)
        # and then repeat
    return rewards

#*******************************************************************************

def playout(algo):
    env = gym.make("LunarLander-v2", render_mode="human")
    state, info = env.reset(seed=42)
    model = algo.actor_model
    for _ in range(20): # 20 episodes
        state, info = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            #action = env.action_space.sample()  # this is where you would insert your policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, _, _ = algo.select_action(state_tensor)
            state, reward, terminated, truncated, info = env.step(action)


    env.close()

#*******************************************************************************

if __name__ == "__main__":
    r = ActorCritic() # "human"
    rewards = r.learn()
    np.save("ac2.npy", rewards)
    print("\a")
    playout(r)





