# -*- coding: utf-8 -*-

import numpy as np
import copy
class Tester():
    def __init__(self, pde, solver, grid, p_net):
        self.pde     = pde
        self.solver  = solver
        self.grid    = grid
        self.cp_grid = copy.deepcopy(grid)
        self.p_net   = p_net
        self.max_elements = 50
        self.x       = self.grid.active_nodes
        self.xx      = self.generate_full_mesh(self.x)
        self.new_x   = None
        self.new_xx  = None
        self.sol     = self.solver.solve(self.x, self.xx)
        self.new_sol = None
        self.n_elems = self.x.shape[0] - 1
        self.rp      = self.n_elems/self.max_elements
        
    def calculate_jump(self):
        k_jump     = np.zeros(self.sol.shape[1])
        k_jump[0]  = np.abs(self.sol[-1,0] - self.sol[0,1]) + \
                     np.abs(self.sol[-1,-1] - self.sol[0,0]) 
        k_jump[-1] = np.abs(-self.sol[-1, 0] + self.sol[-2, -1]) + \
                     np.abs(self.sol[-1,-1] - self.sol[0,0])
        for i in range(1, self.n_elems - 1):
            jump = np.abs(-self.sol[0,i] + self.sol[-1, i-1]) \
                + np.abs(self.sol[-1, i] - self.sol[0, i+1])
            k_jump[i] = jump
        return k_jump
    
    def sort_elements(self, k_jump):
        sort_idx = np.flip(np.argsort(k_jump))
        return sort_idx
    
    def generate_observation(self, idx):
        jump     = []
        
        I        = np.arange(self.n_elems)
        Ip1      = np.roll(I, -1)
        i        = I[idx]
        ip1      = np.roll(I, -1)[idx]
        ip2      = np.roll(I, -2)[idx]
        im1      = np.roll(I, 1)[idx]
        im2      = np.roll(I, 2)[idx]
        
        j_km2    = np.abs(self.sol[-1, im2] - self.sol[0, im1])
        j_km1    = np.abs(self.sol[-1, im1] - self.sol[0, i])
        j_kp1    = np.abs(self.sol[-1, i]   - self.sol[0, ip1])
        j_kp2    = np.abs(self.sol[-1, ip1] - self.sol[0, ip2])
        
        
        avg_jump = np.sum(np.abs(self.sol[-1, I] - self.sol[0, Ip1]))
        avg_jump = 2 * avg_jump/(self.n_elems)
        jump.extend([j_km2 + j_km1, j_km1 + j_kp1, j_kp1 + j_kp2])
        obs      = np.zeros(5)
        obs[0:3] = jump
        obs[3]   = avg_jump
        obs[4]   = self.rp
        return obs
    
    def generate_full_mesh(self, x):
        xx        = np.zeros((self.solver.p + 1, x.shape[0] - 1))
        for e in range(x.shape[0] - 1):
            xx[:, e] = x[e] \
            + (x[e+1] - x[e]) * 0.5 * (self.solver.qx + 1) 
        return xx
    
    def execute_action(self, action, idx):
        x1, x2   = self.x[idx], self.x[idx + 1]
        node     = self.cp_grid.search(x1, x2)
        if node != None:
            if action == 2:
                self.cp_grid.refine(x1, x2)
                self.new_x = self.cp_grid.active_nodes
            elif action == 1:
                self.cp_grid.coarsen(x1, x2)
                self.new_x = self.cp_grid.active_nodes
            else:
                self.new_x = self.cp_grid.active_nodes
        self.rp  = (len(self.new_x) - 1)/self.max_elements
        
    
    def deploy(self):
        k_jump      = self.calculate_jump()
        sorted_idx  = self.sort_elements(k_jump)
        
        for i in sorted_idx:
            observation = self.generate_observation(i)
            action      = int(self.p_net.query(observation))
            print("element: ", i, "action; ", action)
            self.execute_action(action, i)
            
        self.new_xx  = self.generate_full_mesh(self.new_x)
        self.new_sol = self.solver.solve(self.new_x, self.new_xx)
         
    def reset(self):
        self.x       = self.new_x
        self.xx      = self.new_xx
        self.new_x   = None
        self.new_xx  = None
        self.sol     = self.new_sol
        self.new_sol = None
        self.n_elems = self.x.shape[0] - 1
        self.rp      = self.n_elems/self.max_elements
        
    
    