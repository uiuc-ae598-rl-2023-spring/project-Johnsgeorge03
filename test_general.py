#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 19:33:57 2023

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
### COMPUTATIONAL DOMAIN ###                  
a            = 0.0
b            = 8.0
dt           = 1e-10
T            = 1e-10

########## PDE #############
pde_type     = "advection"
pde          = PDE(T, dt, pde_type, a, b)
class sub_pde(PDE):
    def initial_condition(self, z):
        uu = np.zeros_like(z)
        mu = self.xa*0.5 + self.xb*0.5
        # for i in range(uu.shape[1]):
        #     uu[:, i] = np.exp(-((z[:, i] - mu)**2)/(2 * (0.1)))
        for i in range(uu.shape[1]):
            if z[0, i ] >= (mu - np.pi/2) and \
                z[-1, i] <= (mu + np.pi/2):
                uu[:, i] = -np.sin((z[:, i] - mu - np.pi/2))
            else:
                uu[:, i] = 0.0
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
refine_cycle = 3
mesh         = MeshTree(root, refine_cycle)


###### ENVIRONMENT #########
step_solve   = True
env          = PDEworld(mesh, DGFEM, step_solve)
env.scaling_f= 100
env.max_elements = 25

######### AGENT ############
hidden_size  = 64
device       = "cpu"
weight_file  = 'NN_weights/no_target/qnet_model_weights_end.pth'
p_net        = policy_net(env, hidden_size, device, weight_file)

mod          = Tester(pde, DGFEM, mesh, p_net)
mod.max_elements = 25
elem = 0
obs  = mod.generate_observation(elem)
print("observation for :", elem, obs)
plt.figure()
plt.plot(mod.x, 0*mod.x,'-o', label = "old_grid")
# mod.deploy()
# mod.reset()
jump  = mod.calculate_jump()
elems = (mod.x[0:-1] + mod.x[1:]) * 0.5
plt.plot(mod.x, -1e-7 + 0*mod.x,'-o', label = "new_grid")
plt.plot(elems, jump, '-o',  label = "abs jump")
plt.plot(elems, np.mean(jump) + 0*elems, label="mean jump")
plt.legend()
plt.grid()

c_reward = []
r_reward = []
x = env.grid.active_nodes.copy()
elems = np.arange(env.num_elems)
for i in range(2**refine_cycle):  
    env.observation['element'] = [i, x[i], x[i+1]]
    obs, done, r = env.step(2)
    r_reward.append(r)
    env.reset(random)
    env.observation['element'] = [i, x[i], x[i+1]]
    obs, done , r = env.step(1)
    c_reward.append(r)
    env.reset(random)
    
plt.figure()
plt.plot(elems, r_reward, label = "refine_reward")
plt.plot(elems, c_reward, label = "coarse_reward")
plt.legend()
plt.grid()  


plt.figure()
plt.plot(env.xx.ravel(order='F'), env.sol.ravel(order='F'), label = "solution")
plt.legend()
plt.grid() 
# e_id  = 2
# mesh.coarsen(x[e_id], x[e_id + 1])
# mod          = Tester(pde, DGFEM, mesh, p_net)
# jump  = mod.calculate_jump()
# elems = np.arange(jump.shape[0]) + 1
# plt.figure()
# plt.plot(elems, jump, label = "abs jump")
# plt.plot(elems, np.mean(jump) + 0*elems)
# plt.legend()
# plt.grid()
# mesh.reset()


# env.step(1)

# idx       = env.grid.del_idx[0]
# u_coarse  = env.sol
# u_refine  = env.prev_sol

# x_coarse  = env.xx
# x_refine  = env.prev_xx

# xx        = np.linspace(env.s[idx - 1], env.s[idx], 100)

# u_r_elem  = env.interpolate(u_refine[:, idx - 1], x_refine[:, idx - 1], xx)
# u_r_elem2 = env.interpolate(u_refine[:, idx], x_refine[:, idx], xx)
# u_c_elem  = env.interpolate(u_coarse[:, idx - 1], x_coarse[:, idx - 1], xx)
# f         = np.abs(u_r_elem + u_r_elem2 - u_c_elem)
# plt.figure()
# plt.plot(xx, u_r_elem, label = 'refine1')
# plt.plot(xx, u_r_elem2, label = 'refine 2')
# plt.plot(xx, u_c_elem, label = 'coarse')
# plt.plot(xx, f, label = 'f')
# plt.legend(loc = 'best')
# plt.grid()

# for i in range(1):
#     mesh.coarsen(mesh.active_nodes[0], mesh.active_nodes[1])
# mesh.coarsen(mesh.active_nodes[1], mesh.active_nodes[2])
# mesh.coarsen(mesh.active_nodes[1], mesh.active_nodes[2])
# u            = DGFEM.solve(env.s, env.xx)
# uinit        = pde.initial_condition(env.xx)
# xx           = env.xx.ravel(order='F') 
# plt.figure()
# plt.plot(env.s, -0.1 + 0*env.s,'-o', label = "new_grid")
# plt.plot(xx, u.ravel(order = 'F'), label = "sol")
# plt.plot(xx, uinit.ravel(order = 'F'), label = "init" )
# plt.legend(loc = 'best')
# plt.grid()

# plt.figure()
# plt.plot(env.s, -0.1 + 0*env.s,'-o', label = "new_grid")
# env.plotmodal(env.xx, u, 'r', label = "sol")
# env.plotmodal(env.xx, uinit, 'k', label = "init" )
# plt.legend(loc = 'best')
# plt.grid()
