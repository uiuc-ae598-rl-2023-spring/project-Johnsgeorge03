# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mesh import *

### COMPUTATIONAL DOMAIN ###                  
a            = 0.0
b            = 8.0

########### GRID ###########
root         = Node(a, b)
root.is_root = True
random       = False
refine_cycle = 2
mesh         = MeshTree(root, refine_cycle)

x    = mesh.active_nodes
plt.figure(figsize = (10, 8))
ttle = 'base'
for i in range(10):
    rand = np.random.randint(1, 11)
    x    = mesh.active_nodes
    plt.plot(x, 0*x + i, '-o', label = ttle)
    if rand > 5:
        idx = np.random.randint(len(mesh.active_nodes) - 1)
        mesh.coarsen(mesh.active_nodes[idx], mesh.active_nodes[idx + 1])
        ttle = "coarsen, {}, {}".format(idx, mesh.success)
    else:
        idx = np.random.randint(len(mesh.active_nodes) - 1)
        mesh.refine(mesh.active_nodes[idx], mesh.active_nodes[idx + 1])
        ttle = "refine, {}, {}".format(idx, mesh.success)
    
   

plt.grid()
plt.legend()