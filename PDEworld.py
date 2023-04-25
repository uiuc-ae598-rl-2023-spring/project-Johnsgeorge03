# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import lagrange
from scipy.special import legendre
import math
import copy 
import matplotlib.pyplot as plt
class PDEworld:
    def __init__(self, grid, solver, step_solve):
        
        # 0 - do-nothing
        # 1 - coarsen
        # 2 - refine
        self.action_space  = np.array([0, 1, 2], dtype=np.int16)
        self.T             = 0.5
        self.dt            = 1e-3
        self.grid          = grid
        self.solver        = solver
        self.num_actions   = 3
        self.max_num_steps = 200
        self.max_elements  = 25
        self.scaling_f     = 10 # scaling factor
        self.num_obs       = 5
        self.step_solve    = step_solve
        
        # Below variables are modified when env is reset
        self.iteration     = 0
        self.s             = self.grid.active_nodes.copy()
        self.prev_s        = None
        
        self.num_elems     = len(self.s) - 1
        self.xx            = np.zeros((self.solver.p + 1, self.num_elems))
        self.generate_full_mesh()  
        self.prev_xx       = None
        self.sol           = None
        if self.step_solve:
            u_init         = self.solver.PDE.initial_condition(self.xx)
            self.solver.prevt_sol = u_init.copy()
            self.sol       = self.solver.step_solve(self.s, self.xx, u_init)
        else:
            self.sol       = self.solver.solve(self.s, self.xx)
        self.prev_sol      = None
        
        self.rp            = self.num_elems/self.max_elements
        self.prev_rp       = None
        
        self.observation   = {
                'element': self.get_random_element(),
                'jump': [],
                'avg_jump': 0,
                'rp':self.rp
                }
        
        self.prev_obs      = {}
        
        self.prev_action   = None
        self.u_error       = 0
        self.obs           = None
        self.dn_cntr       = 0
        
    def reset(self, random):
        self.iteration   = 0
        self.dn_cntr     = 0
        self.grid.reset()
        if random:
            self.grid.random_refine(4)
        self.prev_s      = None
        self.s           = self.grid.active_nodes.copy()
        self.prev_s      = None
        
        self.num_elems   = len(self.s) - 1
        
        self.xx          = np.zeros((self.solver.p + 1, self.num_elems))
        self.generate_full_mesh()  
        self.prev_xx     = None
        
        self.solver.t    = 0
        self.sol         = None
        if self.step_solve:
            u_init       = self.solver.PDE.initial_condition(self.xx)
            self.solver.prevt_sol = u_init
            self.sol     = self.solver.step_solve(self.s, self.xx, u_init)
        else:
            self.sol     = self.solver.solve(self.s, self.xx)
        self.prev_sol    = None
        
        self.rp          = self.num_elems/self.max_elements
        self.prev_rp     = None
        
        self.observation = {
                'element': self.get_random_element(),
                'jump': [],
                'avg_jump': 0,
                'rp':self.rp
                }
        self.generate_observation(self.observation['element'])
        self.unpack_observation()
        self.prev_action = None
        self.prev_obs    = {}
        self.u_error     = 0
        done             = False
        return self.obs, done
    
    def step(self, action):
        self.iteration   += 1
        [idx, x1, x2]    = self.observation['element']
        self.prev_rp     = self.rp
        self.prev_s      = self.s.copy()
        self.prev_sol    = self.sol.copy()
        self.prev_xx     = self.xx.copy()
        self.prev_obs    = copy.deepcopy(self.observation)
        self.u_error     = 0
        r                = 0
        done             = False
        u_init           = None
        if action == 1 and self.num_elems > 1:
            # coarsen
            
            self.grid.coarsen(x1, x2)
            if self.grid.success:
                self.s         = self.grid.active_nodes.copy()
                self.num_elems = len(self.s) - 1
                self.rp        = self.num_elems / self.max_elements
                
                self.generate_full_mesh()
                if self.step_solve:
                    del_elems  = self.grid.del_idx
                    a          = self.solver.prevt_sol.copy()
                    cidx       = del_elems[0] - 1
                    xr         = self.prev_xx.copy()
                    xc1        = self.s[cidx]
                    xc2        = self.s[cidx + 1]
                    e_xc       = xc1 + (xc2 - xc1) * 0.5 * (self.solver.qx + 1)
                    urall      = np.zeros_like(e_xc)
                    for i in del_elems:
                        urall += self.interpolate(a[:, i], xr[:, i], e_xc)
                        
                    urall += self.interpolate(a[:, cidx], xr[:, cidx], e_xc)
                    mask       = np.full_like(a[0, :], True, dtype = bool)
                    mask[del_elems] = False
                    
                    u_init     = a[:, mask]
                    u_init[:, cidx] = urall
                    self.sol   = self.solver.step_solve(self.s, self.xx, u_init)
                else:
                    self.sol   = self.solver.solve(self.s, self.xx)
                
                self.calculate_u_error(action)
                r              = self.reward(action)
            
        elif action == 2:
            # refine
            self.grid.refine(x1, x2)
            self.s         = self.grid.active_nodes.copy()
            self.num_elems = len(self.s) - 1
            self.rp        = self.num_elems / self.max_elements
            self.generate_full_mesh()
            if self.step_solve:
                a         = self.solver.prevt_sol.copy()
                mid       = (x1 + x2) / 2
                e_uc      = a[:, idx]
                e_xc      = self.prev_xx[:, idx]
                u_init    = np.zeros_like(self.xx)
                e_xr1     = x1 + (mid - x1) * 0.5 * (self.solver.qx + 1)
                e_xr2     = mid + (x2 - mid) * 0.5 * (self.solver.qx + 1)
                e_ur1     = self.interpolate(e_uc, e_xc, e_xr1)
                e_ur2     = self.interpolate(e_uc, e_xc, e_xr2)
                u_init[:, 0:idx] = a[:, 0:idx]
                u_init[:, idx] = e_ur1
                u_init[:, idx + 1] = e_ur2
                u_init[:, idx + 2:] = a[:, idx + 1:]
                self.sol   = self.solver.step_solve(self.s, self.xx, u_init)
            else:
                self.sol   = self.solver.solve(self.s, self.xx)
                
            self.calculate_u_error(action)
            r              = self.reward(action)
        else:
            # do nothing
            pass
        
        
        
        K                 = self.get_random_element()
        self.generate_observation(K)
        self.unpack_observation()
        if self.rp >= 1:
            print("Max comp. resource")
            r      = -1e3
            done   = True
            
        if self.iteration >= self.max_num_steps:
            print("max iteration reached")
            done   = True
        if self.prev_action == 0 and action == 0:
            self.dn_cntr += 1
        else:
            self.dn_cntr  = 0
        
        if self.dn_cntr > 50:
            print("too many do_nothing")
            done = True
            
        self.prev_action = action
        return self.obs, done, r
     
    def advance_one_time_step(self):
        
        pass
    def reward(self, action):
        '''
        Generates the reward for the action

        Parameters
        ----------
        action : belongs to [0, 1, 2]

        Returns
        -------
        r : reward value

        '''
        rac     = self.barrier_func(self.rp) - self.barrier_func(self.prev_rp)
        rsa     = np.log(self.u_error + self.grid.mach_eps) \
                    - np.log(self.grid.mach_eps)
        r       = 0
        if action == 2: # refine
            r = rsa - self.scaling_f * rac
        elif action == 1: # coarsen
            r = -rsa - self.scaling_f * rac
        return r
        
    
    def unpack_observation(self):
        '''
        Unpacks the observation dictionary and generates in the input vector
        for the neural network. Input vector is stored in self.obs
        
        Returns
        -------
        None.

        '''
        self.obs        = np.zeros(self.num_obs)
        self.obs[0:3]   = self.observation['jump']
        self.obs[3]     = self.observation['avg_jump']
        self.obs[4]     = self.observation['rp']
        
    def barrier_func(self, p):
        '''
        The barrier function penalizes the increasing resouce usage. 
        Used to compute reward.

        Parameters
        ----------
        p : fraction of resource used

        Returns
        -------
        the barrier function

        '''
        ans = 0
        if p >= 1:
            ans = 1e17
        else:
            ans = np.sqrt(p)/(1 - p)
        return ans
        
        
    def get_random_element(self):
        '''
        Picks an element using a uniform distribution

        Returns
        -------
        list 
              idx : index of element
              x1, x2: coordinates of the element

        '''
        idx    = np.random.randint(self.num_elems)
        x1, x2 = self.s[idx], self.s[idx + 1]
        return [idx, x1, x2]
    
    
    def g(self, r, a, b):  
        alpha = (a + b)/2
        beta  = b - alpha
        return alpha + beta * r
    
    def gxtor(self, x, a, b):
        alpha = (a + b)/2
        beta  = b - alpha
        return (x - alpha)/beta
    
    def calculate_u_error(self, action):
        '''
        Calculate the error between previous numerical solution and 
        current one using interpolation.
            
        Parameters
        ----------
        action : action type 

        Returns
        -------
        '''
    
        
        if action == 1: # coarsen
            del_elems = self.grid.del_idx
            ur    = self.prev_sol.copy()
            uc    = self.sol.copy()
            xc    = self.xx.copy()
            xr    = self.prev_xx.copy()
            idx   = del_elems[0] - 1
            Ng    = int(math.ceil((self.solver.p + 1)/2)) * 10
            r, w  = self.GaussQuad(Ng*30)
            a     = self.s[idx]
            b     = self.s[idx + 1]
            xx    = self.g(r, a, b)
            uce   = self.interpolate(uc[:, idx], xc[:, idx], xx)
            urall = np.zeros_like(uce)
            for i in del_elems:
                urall += self.interpolate(ur[:, i], xr[:, i], xx)
            urall += self.interpolate(ur[:, idx], xr[:, idx], xx)
            self.u_error = np.sum(w*np.abs(urall - uce)) * (b - (a + b)/2)
            
            for i in range(idx):
                a     = self.prev_s[i]
                b     = self.prev_s[i + 1]
                xx    = self.g(r, a, b)
                ure   = self.interpolate(ur[:, i], xr[:, i], xx)
                uce   = self.interpolate(uc[:, i], xc[:, i], xx) 
                self.u_error += np.sum(w*np.abs(ure - uce)) \
                            * (b - (a + b)/2)
            
            for i in range(idx + 1, self.num_elems):
                a     = self.prev_s[i]
                b     = self.prev_s[i + 1]
                xx    = self.g(r, a, b)
                ure   = self.interpolate(ur[:, i + len(del_elems)], 
                                         xr[:, i + len(del_elems)], xx)
                uce   = self.interpolate(uc[:, i], xc[:, i], xx) 
                self.u_error += np.sum(w*np.abs(ure - uce)) \
                            * (b - (a + b)/2)
            
        elif action == 2: # refine
            ur    = self.sol.copy()
            uc    = self.prev_sol.copy()
            xr    = self.xx.copy()
            xc    = self.prev_xx.copy()
            idx   = self.observation['element'][0]
            Ng    = int(math.ceil((self.solver.p + 1)/2)) * 10
            r, w  = self.GaussQuad(Ng)
            a     = self.prev_s[idx]
            b     = self.prev_s[idx + 1]
            xx    = self.g(r, a, b)
            ure1  = self.interpolate(ur[:, idx], xr[:, idx], xx)
            ure2  = self.interpolate(ur[:, idx + 1], xr[:, idx + 1], xx)
            uce   = self.interpolate(uc[:, idx], xc[:, idx], xx)
            self.u_error = np.sum(w*np.abs(ure1 + ure2 - uce)) \
                            * (b - (a + b)/2)
            
            for i in range(idx):
                a     = self.prev_s[i]
                b     = self.prev_s[i + 1]
                xx    = self.g(r, a, b)
                ure   = self.interpolate(ur[:, i], xr[:, i], xx)
                uce   = self.interpolate(uc[:, i], xc[:, i], xx) 
                self.u_error += np.sum(w*np.abs(ure - uce)) \
                            * (b - (a + b)/2)
            
            for i in range(idx + 1, self.num_elems - 1):
                a     = self.prev_s[i]
                b     = self.prev_s[i + 1]
                xx    = self.g(r, a, b)
                ure   = self.interpolate(ur[:, i+1], xr[:, i+1], xx)
                uce   = self.interpolate(uc[:, i], xc[:, i], xx) 
                self.u_error += np.sum(w * np.abs(ure - uce)) \
                            * (b - (a + b)/2)
                
    def interpolate(self, ue, xe, xeval):
        """
        Parameters
        ----------
        ue : coefficient of the elements
        xe : points in the element
        xeval : nodes at which f is evaluated (-1, 1)

        Returns
        -------
        v : evaluated value

        """
        a    = xe[0]
        b    = xe[-1]
        mask = np.where((xeval >= a) & (xeval <= b), True, False)
        
        z  = np.where(mask == True, self.gxtor(xeval, a, b), 0)
        c  = self.solver.Vinv @ ue
        v  = np.zeros_like(z)
    
        for q in range(self.solver.p+1):
            v[:] += c[q] * (legendre(q)/np.sqrt(2/(2*q + 1)))(z)
        v  = np.where(mask == True, v, 0.0)
        return v 
   
                
    def generate_full_mesh(self):
        self.xx        = np.zeros((self.solver.p + 1, self.num_elems))
        for e in range(self.num_elems):
            self.xx[:, e] = self.s[e] \
            + (self.s[e+1] - self.s[e]) * 0.5 * (self.solver.qx + 1) 
    
    
    
    def generate_observation(self, K):
        
        self.observation['element'] = K
        self.observation['rp']      = self.rp
        idx = K[0]
        self.observation['jump']    = []
        # if idx - 2 >= 0:
        #     j_km2  = np.abs(self.sol[-1, idx - 2] - self.sol[0, idx - 1])
        # else:
        #     j_km2  = 0
        # if idx - 1 >=0:
        #     j_km1  = np.abs(self.sol[-1, idx - 1] - self.sol[0, idx])
        # else:
        #     j_km1  = 0
        # if idx + 1 < self.num_elems:
        #     j_kp1  = np.abs(self.sol[-1, idx]     - self.sol[0, idx + 1])
        # else:
        #     j_kp1  = 0
        # if idx + 2 < self.num_elems:    
        #     j_kp2  = np.abs(self.sol[-1, idx + 1] - self.sol[0, idx + 2])
        # else:
        #     j_kp2  = 0
        
        # jump = 0
        # for i in range(self.num_elems - 1):
        #     jump += np.abs(self.sol[-1, i] - self.sol[0, i + 1])
        # jump *= 2
        
        I      = np.arange(self.num_elems)
        Ip1    = np.roll(I, -1)
        i      = I[idx]
        ip1    = np.roll(I, -1)[idx]
        ip2    = np.roll(I, -2)[idx]
        im1    = np.roll(I, 1)[idx]
        im2    = np.roll(I, 2)[idx]
        
        j_km2  = np.abs(self.sol[-1, im2] - self.sol[0, im1])
        j_km1  = np.abs(self.sol[-1, im1] - self.sol[0, i])
        j_kp1  = np.abs(self.sol[-1, i]   - self.sol[0, ip1])
        j_kp2  = np.abs(self.sol[-1, ip1] - self.sol[0, ip2])
        
        
        jump   = 2 * np.sum(np.abs(self.sol[-1, I] - self.sol[0, Ip1]))
        
        
        self.observation['jump'].extend([j_km2 + j_km1, j_km1 + j_kp1, 
                                         j_kp1 + j_kp2])
        self.observation['avg_jump'] = jump/(self.num_elems)
    
        
    def GaussQuad(self, Ng):
        r_alpha, w_alpha = np.polynomial.legendre.leggauss(Ng)

        return r_alpha, w_alpha
    
    def plotnodal(self, xx, u, color, label):
        """Plot all of the nodal values."""
        
        plt.plot(self.xx.ravel(order='F'), self.sol.ravel(order='F'), 
                 color, label = label, lw = 1)
        

    def plotmodal(self, xx, u, color, label):
        """High fidelity modal."""
        c = self.solver.Vinv @ u
        m = 40
        v = np.zeros((m, xx.shape[1]))
        zz = np.zeros((m, xx.shape[1]))
        z = np.linspace(-1, 1, m)
        for k in range(xx.shape[1]):
            zz[:, k] = xx[0, k] + (xx[self.solver.p, k] - xx[0, k])*(z+1)/2
            for q in range(self.solver.p+1):
                v[:, k] += c[q, k] * (legendre(q)/np.sqrt(2/(2*q + 1)))(z)

        zz = zz.ravel(order='F')
        plt.plot(zz, v.ravel(order='F'), color, label = label, lw = 1)
        
        