#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:34:23 2023

@author: john
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque
import numpy as np
import random
import math
import matplotlib
import matplotlib.pyplot as plt


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class DQN(nn.Module):
    
    def __init__(self, n_observation, n_actions, hidden_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observation, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)
    
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

      
class Agent:
    def __init__(self, n_observations, n_actions, hidden_size, 
                 learning_rate, gamma, epsilon_start, epsilon_end, 
                 epsilon_decay, tau, device):
    
        self.n_observations = n_observations
        self.n_actions      = n_actions
        self.hidden_size    = hidden_size
        self.learning_rate  = learning_rate
        self.gamma          = gamma
        self.epsilon_start  = epsilon_start
        self.epsilon_end    = epsilon_end
        self.epsilon_decay  = epsilon_decay
        self.epsilon        = epsilon_start
        self.memory_size    = 10000
        self.memory         = ReplayMemory(self.memory_size)
        self.batch_size     = 32
        self.update_freq    = 1000
        self.steps          = 0
        self.anneal_steps   = 1e6
        self.device         = device
        self.tau            = tau # soft update value
        
        self.q_network      = DQN(n_observations, n_actions, 
                                  hidden_size).to(device)
        self.target_network = DQN(n_observations, n_actions, 
                                  hidden_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        
        self.loss_fn        = nn.MSELoss()
        self.optimizer      = optim.AdamW(self.q_network.parameters(),
                                          lr = learning_rate, amsgrad=True)
        # self.optimizer      = optim.RMSprop(self.q_network.parameters(),
        #                                     lr = learning_rate, alpha = 0.95,
        #                                     eps = 0.01, momentum = 0.95)
       
        
    def choose_action(self, state):
        sample       = np.random.rand()
        eps_threshold= self.epsilon_end + \
                        (self.epsilon_start - self.epsilon_end) * \
                            math.exp(-1 * self.steps / self.epsilon_decay)
        self.steps  += 1
        self.epsilon = eps_threshold
        self.epsilon = max(self.epsilon_end, self.epsilon_start - 
                           (self.steps / self.anneal_steps) 
                           * (self.epsilon_start - self.epsilon_end))
        if sample < self.epsilon:
            return torch.tensor([[np.random.choice(self.n_actions)]], 
                                device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.q_network(state).max(1)[1].view(1, 1)
                
    


    def plot_rewards(self, episode_rewards, show_result=False):
        
        rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Mean Episodic Returns')
        
        # Take 100 episode averages and plot them too
        if len(rewards_t) >= 100:
            plt.figure(1)
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
    
        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf()) 
        
   
            
    def replay_and_learn(self):
        if len(self.memory) < self.batch_size:
            return 
        transitions    = self.memory.sample(self.batch_size)
        
        batch          = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                    batch.next_state)), 
                                    device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        
        state_batch    = torch.cat(batch.state)
        action_batch   = torch.cat(batch.action)
        reward_batch   = torch.cat(batch.reward)
    
       
        state_action_values = self.q_network(state_batch).gather(1, action_batch)
    
       
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        expected_state_action_values =  expected_state_action_values.float()
        # Compute Huber loss
        
        # criterion = nn.SmoothL1Loss()
        # loss      = criterion(state_action_values, 
        #             expected_state_action_values.unsqueeze(1))
        
        # Computer MSE loss
        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()
        
    def update_target_network(self):
        target_net_state_dict = self.target_network.state_dict()
        policy_net_state_dict = self.q_network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau \
                                    + target_net_state_dict[key]*(1-self.tau)
        self.target_network.load_state_dict(target_net_state_dict)

        
class policy_net:
    def __init__(self, env, hidden_size, device, weight_file):
        self.NN  = DQN(env.num_obs, env.num_actions, hidden_size).to(device)
        self.device = device
        self.NN.load_state_dict(torch.load(weight_file)) 
        
    def query(self, observation):
        observation  = torch.tensor(observation, dtype=torch.float32, 
                                    device=self.device).unsqueeze(0)
        with torch.no_grad():
            action =  self.NN(observation).max(1)[1].view(1, 1)
        return action.item()       
           
      
        
    