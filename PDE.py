# -*- coding: utf-8 -*-
import numpy as np

class PDE:
    
    def __init__(self, T, dt, name, xa, xb):
       self.T    = T
       self.dt   = dt
       self.name = name
       self.xa   = xa
       self.xb   = xb
    def f(self, u):
        if self.name == "burgers":
            return u**2/2
        elif self.name == "advection":
            return u
        
    def df(self, u):
        if self.name == "burgers":
            return u
        elif self.name == "advection":
            return 1.0
    
    def initial_condition(self, z):
        uu = np.zeros_like(z)
        mu    = self.xa*0.5 + self.xb*0.5
        for i in range(uu.shape[1]):
            uu[:, i] = np.exp(-((z[:, i] - mu)**2)/(2 * (0.2)))
        return uu
        