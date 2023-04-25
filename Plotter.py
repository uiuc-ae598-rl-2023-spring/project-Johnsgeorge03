#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:35:09 2023

@author: john
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from scipy.special import legendre
from agent_network import DQN

class Plotter:
    def __init__(self, env, weight_file, hidden_size, device):
        self.env = env
        self.NN  = DQN(env.num_obs, env.num_actions, hidden_size).to(device)
        self.NN.load_state_dict(torch.load(weight_file))
        self.device = device
        
    def policy(self, state):
        state  = torch.tensor(state, dtype=torch.float32, device= self.device)\
                 .unsqueeze(0)
        with torch.no_grad():
            action = self.NN(state).max(1)[1].view(1, 1).item()
        return action
    
    
    def plot_learning_curve(self, gamma, source_file, dest_file_ud, dest_file_d):
        with open(source_file) as f:
            lines = f.readlines()

        # Remove any leading or trailing whitespace characters from each line
        lines = [line.strip() for line in lines]

        # Split each line into a list of integers
        episode_list = [[float(num) for num in line[1:-1].split(',')] 
                        for line in lines]
        num_eps              = len(episode_list)
        discounted_returns   = np.zeros(num_eps)
        undiscounted_returns = np.zeros(num_eps)
        for e in range(num_eps):
            returns   = 0
            u_returns = 0
            for i in range(len(episode_list[e])):
                returns   += (gamma**i) * episode_list[e][i]
                u_returns += episode_list[e][i]
            discounted_returns[e]   = returns
            undiscounted_returns[e] = u_returns
        
        mean_ud_returns = []
        sd_ud_returns   = []
        
        mean_d_returns  = []
        sd_d_returns    = []
        for i in range(num_eps):
            if i < 100:
                mean_ud_returns.append(np.mean(undiscounted_returns[0:i+1]))
                sd_ud_returns.append(np.std(undiscounted_returns[0:i+1]))
                
                mean_d_returns.append(np.mean(discounted_returns[0:i+1]))
                sd_d_returns.append(np.std(discounted_returns[0:i+1]))
                
            else:
                mean_ud_returns.append(np.mean(undiscounted_returns[i-100:i]))
                sd_ud_returns.append(np.std(undiscounted_returns[i-100:i]))
                
                mean_d_returns.append(np.mean(discounted_returns[i-100:i]))
                sd_d_returns.append(np.std(discounted_returns[i-100:i]))
                
        
        sns.set(rc={'figure.figsize':(10,6)})
     
        row_means = np.array(mean_d_returns)
        row_stds  = np.array(sd_d_returns)

        # create a pandas DataFrame with columns for x-axis, y-axis, lower bound, and upper bound
        df = pd.DataFrame({'x': range(num_eps),
                            'y': row_means,
                            'lower': row_means - row_stds,
                            'upper': row_means + row_stds})

        # plot the mean values with a variance band using Seaborn's lineplot
        sns.lineplot(data=df, x='x', y='y', ci='sd', lw = 4, color = 'r')

        # plot the variance band as a shaded area
        plt.fill_between(df['x'], df['lower'], df['upper'], alpha=0.2)

        # set the plot labels and legend
        plt.ylim(bottom=-250, top=200)
        plt.xlabel('Episodes')
        plt.ylabel('Mean Discounted Returns')
        plt.title("DQN learning curve")
        plt.tight_layout()
        plt.savefig(dest_file_d)
        # show the plot
        plt.show()

        
        row_means = np.array(mean_ud_returns)
        row_stds  = np.array(sd_ud_returns)

        # create a pandas DataFrame with columns for x-axis, y-axis, lower bound, and upper bound
        df = pd.DataFrame({'x': range(num_eps),
                            'y': row_means,
                            'lower': row_means - row_stds,
                            'upper': row_means + row_stds})

        # plot the mean values with a variance band using Seaborn's lineplot
        sns.lineplot(data=df, x='x', y='y', ci='sd', lw = 4)

        # plot the variance band as a shaded area
        plt.fill_between(df['x'], df['lower'], df['upper'], alpha=0.2)

        # set the plot labels and legend
        plt.ylim(bottom=-250, top=200)
        plt.xlabel('Episodes')
        plt.ylabel('Mean Undiscounted Returns')
        plt.title("DQN learning curve")
        plt.tight_layout()
        plt.savefig(dest_file_ud)
        # show the plot
        plt.show()
                
            
    def plot_model_deployment(self, mesh_x, mesh_xx, mesh_u, n_elems, max_elem,
                              true_x, true_u, filename):
        r_cycle = len(mesh_x)
        fig, axes = plt.subplots(int(r_cycle/2), 2, 
                                 figsize=(20, 4 * r_cycle/2))
        for i in range(r_cycle):
            row = int(i / 2)
            col = int(i % 2)
            axes[row, col].plot(mesh_x[i], -0.1 + 0*mesh_x[i], marker ='.',
                                markersize = 20, label = 'mesh', color ='k')
            u =  mesh_u[i]
            xx = mesh_xx[i]
            c = self.env.solver.Vinv @ u
            m = 40
            v = np.zeros((m, xx.shape[1]))
            zz = np.zeros((m, xx.shape[1]))
            z = np.linspace(-1, 1, m)
            for k in range(xx.shape[1]):
                zz[:, k] = xx[0, k] + (xx[self.env.solver.p, k] - xx[0, k])*(z+1)/2
                for q in range(self.env.solver.p+1):
                    v[:, k] += c[q, k] * (legendre(q)/np.sqrt(2/(2*q + 1)))(z)

            zz = zz.ravel(order='F')
            axes[row, col].plot(zz, v.ravel(order='F'), lw = 3, color='r', 
                                label = 'numerical solution')
            # mask = np.arange(len(true_x)) % 4 == 0
            axes[row, col].plot(true_x, true_u, linestyle='--', dashes=(1, 3), 
                                lw = 5, color='b', label = 'true solution')
            axes[row, col].grid(True)
            axes[row, col].set_title('Cycle {}: {}/{} cells used'.format(i+1, 
                                       n_elems[i], max_elem))
            axes[row, col].set_ylim(bottom=-0.2, top=1.1)
        # handles, labels = axes.get_legend_handles_labels()
        labels= ['mesh', 'numerical solution', 'true solution']
        fig.legend(labels, loc='upper center', ncol=3)
        fig.tight_layout()
        fig.savefig(filename)
    
    
    
    
        
    
    


