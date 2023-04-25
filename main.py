# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mesh import *
from PDEworld import *
from solver import *
from PDE import *
from agent_network import *
import seaborn as sns
import pandas as pd
import os

############################
#       PARAMETERS         #
############################
plt.ion()
### COMPUTATIONAL DOMAIN ###                  
a            = 0.0
b            = 8.0
dt           = 1e-5
T            = 1e-5



########## PDE #############
pde_type     = "advection"
pde          = PDE(T, dt, pde_type, a, b)

class sub_pde(PDE):
    def initial_condition(self, z):
        uu   = np.zeros_like(z)
        mu   = self.xa*0.5 + self.xb*0.5
        sd   = 0.25
        for i in range(uu.shape[1]):
            uu[:, i] = np.exp(-((z[:, i] - mu)**2)/(2 * (sd)**2)) \
                # /(sd * np.sqrt(2*np.pi))
        return uu
pde          = sub_pde(T, dt, pde_type, a, b)    

########## SOLVER ##########
# slope_lim must be true if burgers is used
order        = 3
slope_lim    = False
DGFEM        = Solver(order, pde, slope_lim)


########### GRID ###########
root         = Node(a, b)
root.is_root = True
random       = False
refine_cycle = 2
mesh         = MeshTree(root, refine_cycle)



###### ENVIRONMENT #########
step_solve   = True
env          = PDEworld(mesh, DGFEM, step_solve)
env.scaling_f= 100
env.max_elements  = 25

######### AGENT ############
hidden_size         = 64
gamma               = 0.99
learning_rate       = 0.0001
epsilon_start       = 1.0
epsilon_end         = 0.01
epsilon_decay       = 1000 # not used
tau                 = 1.0

device              = ""
if torch.backends.mps.is_available():
            device  = "mps"
elif torch.cuda.is_available():
            device  = "cuda"
else:
            device  = "cpu"
            
device              = "cpu" # turns out mps is so slow 
agent               = Agent(env.num_obs, env.num_actions, hidden_size, 
                            learning_rate, gamma, epsilon_start, epsilon_end, 
                            epsilon_decay, tau, device)

agent.batch_size    = 64
agent.memory_size   = 10000
agent.update_freq   = 1
agent.anneal_steps  = 10000
# update freq = 500
# no replay batch size = 128


########### DIRECTORIES #######
test_dir            = 'test_new/'
weight_dir          = test_dir + 'NN_weights/no_target/'
fig_dir             = test_dir + 'figures/no_target/'
reward_dir          = test_dir + 'rewards/'
reward_file         = 'rewards_no_tar.txt'
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(weight_dir, exist_ok=True)
os.makedirs(reward_dir, exist_ok=True)


########## TRAIN ##############
num_episodes        = 2000
scores              = []
mean_scores         = []
max_mean            = 0
all_udissc_r        = []

for i in range(num_episodes):
    total_r = 0
    returns = 0
    
    observation, done = env.reset(random)
    observation  = torch.tensor(observation, dtype=torch.float32, 
                                device=agent.device).unsqueeze(0)
    done            = False
    total_reward    = 0
    rewards_list    = []
    while not done:
        action      = agent.choose_action(observation) 
        assert(action.item() in np.arange(env.num_actions))
        obs, done, r = env.step(action.item())
        rewards_list.append(r)
        total_reward += r
        if done:
            s1  = None
        else:
            s1  = torch.tensor(obs, dtype = torch.float32, device = device) \
                 .unsqueeze(0)
        r        = torch.tensor([r], device = agent.device) 
        agent.memory.push(observation, action, s1 , r)
        observation   = s1
        agent.replay_and_learn()
        if agent.steps % agent.update_freq == 0:
            agent.update_target_network()
        

    all_udissc_r.append(rewards_list)
    scores.append(total_reward)
    agent.plot_rewards(scores)
    mean = np.mean(scores[-100:])
    mean_scores.append(mean)
    if mean > max_mean:
        torch.save(agent.q_network.state_dict(), 
                    weight_dir + 'qnet_model_weights_max_mean.pth')
        torch.save(agent.target_network.state_dict(), 
                    weight_dir + 'tnet_model_weights_max_mean.pth')
        max_mean = mean
        
    print('Episode {}, Total iterations {}, Total Reward {:.2f}, Mean Reward {:.2f}, Epsilon: {:.2f}'
          .format(i + 1, agent.steps,  
          total_reward, np.mean(scores[-100:]), agent.epsilon))
    
    plt.figure()
    plt.plot(env.s, -0.1 + 0*env.s,'-o', label = "new_grid")
    plt.plot(env.prev_s, 0*env.prev_s,'-o', label = "old_grid")
    env.plotmodal(env.xx, env.solver.PDE.initial_condition(env.xx),
              color='c', label = "initial")
    env.plotmodal(env.xx, env.sol, color ='b', label = "new_soln")
    env.plotmodal(env.prev_xx, env.prev_sol, color ='k', label = "old_soln")
    plt.legend(loc = 'best')
    plt.title("Modal, episode = {}".format(i + 1))
    plt.grid()
    plt.show()


print("COMPLETE")
agent.plot_rewards(scores, show_result = True)
plt.ioff
plt.show()
    
    


## WRITE REWARDS AND MODEL PARAMS TO FILES
torch.save(agent.q_network.state_dict(), weight_dir + 'qnet_model_weights_end.pth')
torch.save(agent.target_network.state_dict(), weight_dir + 'tnet_model_weights_end.pth')
with open(reward_dir + reward_file, 'w+') as f:
    for items in all_udissc_r:
        f.write('%s\n' %items)
    print("File written successfully")
f.close()

## PLOTS
import Plotter
weight_file   = weight_dir + 'qnet_model_weights_end.pth'
plotter       = Plotter.Plotter(env, weight_file, hidden_size, device)
plotter.plot_learning_curve(gamma, reward_dir + reward_file, 
                            fig_dir + 'undiscounted_learning_curve.png', 
                            fig_dir + 'discounted_learning_curve.png')








