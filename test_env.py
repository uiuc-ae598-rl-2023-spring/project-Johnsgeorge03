# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mesh import *
from PDEworld import *
from solver import *
from PDE import *
from agent_network import *
############################
#       PARAMETERS         #
############################

### COMPUTATIONAL DOMAIN ###                  
a            = 0.0
b            = 8.0
dt           = 0.001
T            = 2


########## PDE #############
pde_type     = "advection"
pde          = PDE(T, dt, pde_type, a, b)


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
env.scaling_f= 200
env.max_elements  = 25


count      = 0
cum_reward = 0
num_episodes = 5
c_r_list   =[]
color      = ['b', 'r', 'c', 'k', 'g', 'm', 'y']
for eps in range(num_episodes):
    score      = 0
    done       = False
    count      = 0
    cum_reward = 0
    #while not done:
    obs, done  = env.reset(random)
    plt.figure(figsize = (10, 8))
    ttle = 'base'
    for i in range(7):
        rand = np.random.randint(1, 10)
        plt.plot(env.s, 0*env.s + i/5, '-o', color = color[i], label = ttle)
        env.plotmodal(env.xx, env.sol, color = color[i], label = "sol, {}".format(i))
        if rand > 5 :
            obs, done, r = env.step(1)
            print("coarsen , success --> ", env.grid.success)
            ttle = "coarsen, {}, {}".format(env.prev_obs['element'][0], env.grid.success)
            
        elif rand <= 5:
            obs, done, r = env.step(2)
            print("refine, success --> ", env.grid.success)
            ttle = "refine, {}, {}".format(env.prev_obs['element'][0], env.grid.success)
            
        # else:
        #     print("do nothing: ")
        #     ttle = "do_nothing, {}, {}".format(env.observation['element'][0], env.grid.success)
        #     obs, done, r = env.step(0)
            
   
        cum_reward += r
        print("u_error:  ", env.u_error)
        print("reward: ", r)
        print("\n")
        # plt.plot(env.s, -0.1 + 0*env.s,'-o', label = "new_grid")
        # plt.plot(env.prev_s, 0*env.prev_s,'-o', label = "old_grid")
        
        # env.plotmodal(env.xx, env.solver.PDE.initial_condition(env.xx),
        #           color='c', label = "initial")
        
        # env.plotmodal(env.prev_xx, env.prev_sol, color ='k', label = "old_soln")
    print("---end episode----")
    plt.legend(loc = 'best')
    plt.title("Modal, episode = {}, iteration = {}".format(eps, i))
    plt.grid()
    plt.show()
       
    c_r_list.append(cum_reward)
    count+=1   
    obs, done = env.reset(random)
    # print("No. of iterations: ", count, "Episode No, ", i)
    
    
# plt.grid()
# plt.legend()  
    # plt.figure()
    # plt.plot(env.s, -0.1 + 0*env.s,'-o', label = "new_grid")
    # plt.plot(env.prev_s, 0*env.prev_s,'-o', label = "old_grid")
    # env.plotnodal(env.xx, env.solver.PDE.initial_condition(env.xx),
    #           color='c', label = "initial")
    # env.plotnodal(env.xx, env.sol, color ='b', label = "new_soln")
    # env.plotnodal(env.prev_xx, env.prev_sol, color ='k', label = "old_soln")
    # plt.legend(loc = 'best')
    # plt.title("Nodal, episode = {}, iteration = {}".format(i, count))
    # plt.grid()
    # plt.show()
