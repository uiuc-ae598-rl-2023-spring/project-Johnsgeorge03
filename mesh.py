#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:42:47 2023

@author: john
"""

import numpy as np

class Node:
    def __init__(self, x1, x2):
        #self.ID = None
        self.x1      = x1
        self.x2      = x2
        self.left    = None
        self.right   = None
        self.parent  = None
        self.active  = True
        self.is_root = False



class MeshTree:
    def __init__(self, root, rf_cycle):
        self.root         = root
        self.mach_eps     = 1e-16
        self.refine_cycle = rf_cycle
        
        self.active_nodes = np.array([root.x1, root.x2])
        self.leaf_nodes   = [root]
        self.del_idx      = np.array([])
        self.generate_grid()
        self.success      = True
        
    def generate_grid(self):
        for i in range(self.refine_cycle):
            x = np.copy(self.active_nodes)
            for i in range(len(x) - 1):
                self.refine(x[i], x[i+1])
    
    def random_refine(self, n_refine):
        for i in range(n_refine):
            idx    = np.random.randint(len(self.active_nodes) - 1)
            x1, x2 = self.active_nodes[idx], self.active_nodes[idx + 1]
            self.refine(x1, x2)
     
    def random_coarsen(self, n_coarsen):
        for i in range(n_coarsen):
            idx    = np.random.randint(len(self.active_nodes) - 1)
            x1, x2 = self.active_nodes[idx], self.active_nodes[idx + 1]
            self.coarsen(x1, x2)
            
    def refine(self, x1, x2):
        # assumes the given node is active
        node = self.search(x1, x2)
        assert(node != None)
        if node != None: 
            self.success = True
        
        mid        = (node.x1 + node.x2)/2
        left_node  = Node(node.x1, mid)
        left_node.parent = node
    
        
        right_node = Node(mid, node.x2)
        right_node.parent = node
        
        node.left   = left_node
        node.right  = right_node
        node.active = False
        idx = np.where(abs(self.active_nodes - x2) < self.mach_eps)
        self.active_nodes = np.insert(self.active_nodes, idx[0], mid)
        
        self.leaf_nodes.extend([left_node, right_node])
        
    def coarsen(self, x1, x2):
        # assumes given node is active
        node   = self.search(x1, x2)
        assert(node != None)
        if node != None:
            self.success = True
        parent = node.parent
        if node.is_root or not parent.left.active or not parent.right.active:
            self.success = False
            return
        
        parent.left   = None
        parent.right  = None
        parent.active = True
        self.del_idx  = np.where((self.active_nodes > parent.x1)
                                              &(self.active_nodes < parent.x2))[0]
        self.active_nodes = np.delete(self.active_nodes, self.del_idx)
        del_nodes = []
        for i in range(len(self.leaf_nodes)):
            nodes = self.leaf_nodes[i]
            mid   = (nodes.x1 + nodes.x2)/2
            if mid < parent.x2 and mid > parent.x1:
                del_nodes.append(i)
                
        for index in sorted(del_nodes, reverse=True):
            del self.leaf_nodes[index]
        
        self.leaf_nodes.append(parent)
        
    def search(self, x1, x2):
        for nodes in self.leaf_nodes:
            if abs(nodes.x2 - x2) < self.mach_eps \
                and abs(nodes.x1 - x1) < self.mach_eps:
                    return nodes
                    
    def reset(self):
        self.root.left    = None
        self.root.right   = None
        self.root.active  = True
        self.active_nodes = np.array([self.root.x1, self.root.x2])
        self.leaf_nodes   = [self.root]
        self.success      = True
        self.del_idx      = np.array([])
        self.generate_grid()
                