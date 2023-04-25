#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 23:20:53 2023

@author: john
"""


import numpy as np
import matplotlib.pyplot as plt
from mesh import *
from PDEworld import *
from solver import *
from PDE import *
from agent_network import *
from deployment_model import Tester
import seaborn as sns
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
        uu    = np.zeros_like(z)
        mu    = self.xa*0.5 + self.xb*0.5
        for i in range(uu.shape[1]):
            uu[:, i] = np.exp(-((z[:, i] - mu)**2)/(2 * (0.05)))
        for i in range(uu.shape[1]):
            if z[0, i ] >= (mu - np.pi/2) and \
                z[-1, i] <= (mu + np.pi/2):
                uu[:, i] = np.sin((z[:, i] - mu + np.pi/2))
            else:
                uu[:, i] = 0.0
        return uu
    def true_solution(self, T, n_pts):
        x     = np.linspace(self.xa, self.xb, n_pts)
        u     = np.zeros_like(x)
        mu    = self.xa*0.5 + self.xb*0.5
        for i in range(n_pts):
            if x[i] >= (mu - np.pi/2 + T) and \
                x[i] <= (mu + np.pi/2 + T):
                u[i] = np.sin((x[i] - mu + np.pi/2))
            else:
                u[i] = 0.0
        return x, u
    
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


######### AGENT ############

hidden_size  = 64
device       = "cpu"
test_dir     = 'test4_adamw_batch64_50dn_2keps_1/'
weight_dir   = test_dir + 'NN_weights/no_target/'
fig_dir      = test_dir + 'figures/no_target/'
weight_file  = weight_dir + 'qnet_model_weights_end.pth'
figfilename  = 'deploy40_end.png'
p_net        = policy_net(env, hidden_size, device, weight_file)

####### MODEL ##############
mod          = Tester(pde, DGFEM, mesh, p_net)
mod.max_elements = 40
r_cycle = 8

mesh_x  = []
mesh_xx = []
mesh_u  = []
n_elems = []
true_x, true_u = pde.true_solution(T, 75)
sns.set()
plt.figure(figsize=(8, 4))
plt.plot(mod.x, -0.1 + 0*mod.x, marker ='.',
                    markersize = 20, label = 'mesh', color ='k')
plt.plot(mod.xx.ravel(order='F'), mod.sol.ravel(order='F'), lw = 3, color='r', 
                    label = 'numerical solution')
plt.plot(true_x, true_u, linestyle='--', dashes=(1, 3), 
                    lw = 5, color='b', label = 'true solution')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(fig_dir + 'initialgrid.png')
plt.show()
for i in range(r_cycle):
    mod.deploy()
    print("num_elems: ", mod.n_elems)
    
    mesh_x.append(mod.new_x)
    mesh_xx.append(mod.new_xx)
    mesh_u.append(mod.new_sol)
    if i != r_cycle - 1:
        mod.reset()
    print(" ----- ")
    n_elems.append(mod.n_elems)

## PLOTS
import Plotter
sns.set()
true_x, true_u = pde.true_solution(T, 75)
plotter        = Plotter.Plotter(env, weight_file, hidden_size, device)
plotter.plot_model_deployment(mesh_x, mesh_xx, mesh_u, n_elems, 
                              mod.max_elements, true_x, true_u, 
                              fig_dir + figfilename)

# GOOD ones 
# test2_relu_rms_mse-end, 
# test3_relu_rms_mse_batch64-end/max
# test3_relu_rms_mse_batch128-end, 

# test4_relu_adamw_mse_batch64_25dn - end
# test4_relu_adamw_mse_batch64_35dn - end/max, 
# test4_relu_adamw_mse_batch64_40dn - max
# test4_relu_adamw_mse_batch64_50dn - max

# test4_relu_adamw_mse_batch128_35dn - end
# test4_relu_adamw_mse_batch128_40dn - end


# OK
# test3_relu_adamw_mse_batch64-end


# Really bad - test4_relu_rms_mse_batch64_40dn



