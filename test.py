import argparse
import gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()



class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(8, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 4)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

class REINFORCE():
    def __init__(self):
        self.env = gym.make('LunarLander-v2')
        self.max_episodes = 500
        self.gamma = 0.99
        self.LearningRate = 0.01
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.LearningRate)
        self.eps = np.finfo(np.float32).eps.item()


    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        #print(probs)
        return action.item()


    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = deque()
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            returns.appendleft(R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.policy.saved_log_probs, returns):
            #print(log_prob)
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]


    def main(self):
        running_reward = 10
        totalrewards = []
        for i_episode in range(self.max_episodes):#count(1):
            state, _ = self.env.reset()
            ep_reward = 0
            for t in range(1, 1000):  # Don't infinite loop while learning
                action = self.select_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                self.policy.rewards.append(reward)
                ep_reward += reward
                if (terminated or truncated):
                    totalrewards.append(ep_reward)
                    break

            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            self.finish_episode()
            if i_episode % args.log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                      i_episode, ep_reward, running_reward))
            if running_reward > self.env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(running_reward, t))
                break
        
            
        plt.plot(range(len(totalrewards)), totalrewards)
        plt.xlabel("Episode")
        plt.ylabel("Timestep")
        plt.show()
        plt.savefig("test.png")
                
                
                
                
                
                
                
                
                
                
                
        

r = REINFORCE() # initialize the models
policy = r.main()
print("end", policy)
#if __name__ == '__main__':
#    main()
