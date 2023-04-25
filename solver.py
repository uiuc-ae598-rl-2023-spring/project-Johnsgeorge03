# -*- coding: utf-8 -*-
import numpy as np
import modepy as mp
from scipy.special import legendre
import matplotlib.pyplot as plt

class Solver:
    def __init__(self, order, PDE, slope_lim):
       
        # order of polynomial for integration
        self.p    = order   
        # Local quadrature points
        self.qx   = mp.quadrature.jacobi_gauss.legendre_gauss_lobatto_nodes(self.p) 
        # Details about the PDE , f, df, initial conditions, T, dt etc.
        self.PDE  = PDE
        
        
        
        # Vandermonde matrix and its derivative
        self.V  = np.zeros((self.p+1, self.p+1))
        self.dV = np.zeros((self.p+1, self.p+1))
        
        for q in range(self.p + 1):
            poly    = self.legendre_poly(q)
            dpoly   = poly.deriv()
            self.V[:, q] = poly(self.qx)
            self.dV[:, q] = dpoly(self.qx)
        self.Vinv   = np.linalg.inv(self.V)
        self.M      = (self.Vinv.T) @ self.Vinv
        self.Minv   = self.V@(self.V.T)
        self.S      = self.M @ self.dV @ self.Vinv
        self.D      = self.dV @ self.Vinv
        self.t      = 0
        self.slope_limit = slope_lim
        
        self.prevt_sol = None
        
    def legendre_poly(self, degree):
        return legendre(degree)/np.sqrt(2/(2*degree + 1)) 
        
    def solve(self, x, xx):
        self.n_elems = len(x) - 1
        self.xx      = xx
        self.x       = x
        f_name       = "Lax-Fredrichs"
        r_name       = "minmod"
        u            = self.PDE.initial_condition(self.xx)
        t            = 0
        #dx     = np.max(x[1:] - x[0:-1])
        while t <= self.PDE.T:
            dt      = self.PDE.dt#1e-3#0.5*dx/np.max(np.abs(self.PDE.df(u)))
            u1      = u + dt * self.L(u, f_name)
            if self.slope_limit:
                u1      = self.slopelimiter(u1, r_name)
            u2      = (1/4) * (3 * u + u1 + dt * self.L(u1, f_name))
            if self.slope_limit:
                u2      = self.slopelimiter(u2, r_name)
            u[:]    = (1/3) * (u + 2 * u2 + 2 * dt * self.L(u2, f_name))
            if self.slope_limit:
                u       = self.slopelimiter(u, r_name)
            t      += dt
        return u
    
    def step_solve(self, x, xx, u_init):
        self.prevt_sol = u_init
        self.n_elems   = len(x) - 1
        self.xx        = xx
        self.x         = x
        f_name         = "Lax-Fredrichs"
        r_name         = "minmod"
        # u              = u_init.copy()
        u              = self.PDE.initial_condition(self.xx)
        
        dt             = self.PDE.dt#0.5*dx/np.max(np.abs(self.PDE.df(u)))
        u1             = u + dt * self.L(u, f_name)
        if self.slope_limit:
            u1      = self.slopelimiter(u1, r_name)
        u2             = (1/4) * (3 * u + u1 + dt * self.L(u1, f_name))
        if self.slope_limit:
            u2      = self.slopelimiter(u2, r_name)
        u[:]           = (1/3) * (u + 2 * u2 + 2 * dt * self.L(u2, f_name))
        if self.slope_limit:
            u       = self.slopelimiter(u, r_name)
        #print(np.linalg.norm(u - self.prevt_sol), "norm val")
        # uinit_temp  = self.PDE.initial_condition(self.xx)
        # print(np.linalg.norm(u_init - uinit_temp), "norm b/w initial conditions")
        # plt.figure()
        # plt.plot(xx.ravel(order = 'F'), u_init.ravel(order ='F'), label='interpolated initial sol')
        # plt.plot(xx.ravel(order = 'F'), u.ravel(order = 'F'), label='advected sol from inter initial sol')
        # plt.plot(xx.ravel(order = 'F'), uinit_temp.ravel(order ='F'), label='original initial sol')
        # plt.legend()
        # plt.grid()
        return u
    
    def advance_one_dt(self, x, u_init):
        self.prevt_sol = u_init
        self.n_elems   = len(x) - 1
        self.xx        = np.zeros((self.p + 1, self.n_elems))
        self.x         = x
        for e in range(self.n_elems):
            self.xx[:, e] = x[e] + (x[e+1] - x[e]) * 0.5 * (self.qx + 1)
            
        
        f_name  = "Lax-Fredrichs"
        r_name  = "minmod"
        u       = u_init
        
        dt      = self.PDE.dt#0.5*dx/np.max(np.abs(self.PDE.df(u)))
        u1      = u + dt * self.L(u, f_name)
        if self.slope_limit:
            u1      = self.slopelimiter(u1, r_name)
        u2      = (1/4) * (3 * u + u1 + dt * self.L(u1, f_name))
        if self.slope_limit:
            u2      = self.slopelimiter(u2, r_name)
        u[:]    = (1/3) * (u + 2 * u2 + 2 * dt * self.L(u2, f_name))
        if self.slope_limit:
            u       = self.slopelimiter(u, r_name)
        self.t  += dt
        
        
    def L(self, u, f_name):
        I         = np.arange(self.n_elems)
        Im1       = np.roll(I, 1)
        Ip1       = np.roll(I, -1)
        flux      = np.zeros_like(u)
        u_l_i     = u[0, I]
        u_l_e     = u[self.p, Im1]
        u_r_i     = u[self.p, I]
        u_r_e     = u[0, Ip1]
        
        flux[0, :]= self.flux(f_name, u_l_i, u_l_e, -1, 1)
        flux[self.p, :]= -self.flux(f_name, u_r_i, u_r_e, 1, -1)
        
        h         = self.x[1:] - self.x[0:-1]
        Lu        = self.Minv @ self.S.T @ self.PDE.f(u) * 2 / h \
                    + self.Minv @ flux * 2 / h
        
        return Lu
    
    
    def flux(self, name, u_i, u_e, n_i, n_e):
        '''

        Parameters
        ----------
        name : The type of flux reconstruction.
        u_i  : u in same element near boudnary
        u_e  : u in the neighbouring element near boundary
        n_i  : normal to the internal element at the boundary
        n_e  : normal to the external element at the boundary

        Returns
        -------
        The flux

        '''
        if name == "Lax-Fredrichs":
            avg_flx = (self.PDE.f(u_i) + self.PDE.f(u_e)) / 2
            jumpu   = n_i * u_i + n_e * u_e
            alpha   = np.max([np.abs(self.PDE.df(u_i)), np.abs(self.PDE.df(u_e))])
            return avg_flx + (alpha / 2) * jumpu
        
    
    def reconstruction(self, a, b, c, name):
        
        if name == "minmod":
            a = a.ravel()
            b = b.ravel()
            c = c.ravel()
            
            I = np.where(np.abs(np.sign(a) + np.sign(b) + np.sign(c)) == 3)[0]
            
            mm = np.zeros_like(a)
            
            mm[I] = np.sign(a[I]) * np.vstack((np.abs(a[I]), 
                                               np.abs(b[I]),
                                               np.abs(c[I]))).min(axis=0)
            
            return mm
            
    def slopelimiter(self, u, r_name):
        
        I         = np.arange(self.n_elems)
        Im1       = np.roll(I, 1)
        Ip1       = np.roll(I, -1)
        
        uhat      = self.Vinv @ u
        u1        = uhat.copy()
        u1[2:, :] = 0
        u1        = self.V @ u1
        ux        = (u1[-1, :] - u1[0, :])/(self.x[1:] - self.x[0:-1])
        ux        = ux.ravel()
        
        uhat[1:, :] = 0
        ubar      = (self.V @ uhat)[0, :].ravel()
        ubarm1    = ubar[Im1]
        ubarp1    = ubar[Ip1]
        
        b         = (ubarp1 - ubar)/(self.x[1:] - self.x[0:-1])
        c         = (ubar - ubarm1)/(self.x[1:] - self.x[0:-1])
        n_slope   = self.reconstruction(ux, b, c, r_name)
        
        xbar      = (self.xx[-1, :] + self.xx[0, :]) / 2
        
        bshape    = (self.p + 1, self.n_elems)
        xcenter   = self.xx - xbar
        ubar      = np.broadcast_to(ubar, bshape)
        n_slope   = np.broadcast_to(n_slope, bshape)
        newu      = ubar + xcenter * n_slope
        
        return newu
        
    
            
            
            