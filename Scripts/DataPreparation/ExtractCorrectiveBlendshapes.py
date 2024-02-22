# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:16:54 2023

@author: Stevo Rackovic

This script is to be executed in Autodesk Maya console, with a metahuman (or some other) avatar.
"""

data_dir = r'..\FacialAnimation\Data' # Put a path to your data directory
m = 130  # Put the number of your character blendhsapes
m2 = 642 # Put a number of all the controllers under the blendshape node
names = [] # copy the names from a text file here 

# -----------------------------------------------------------------------------

import numpy as np
import pymel.core as pycore
import os

bs_node = pycore.PyNode("head_lod0_mesh_blendShapes") 
neutral_mesh = pycore.PyNode("head_lod0_mesh")
# Before runing the next line, I need to select controllers node:
bs_node2 = pycore.PyNode("CTRL_expressions") 

deltas = np.load(os.path.join(data_dir,'deltas.npy')).T
neutral = np.load(os.path.join(data_dir,'neutral.npy'))

for nm in names:
    bs_node2.setAttr(nm,0)
    
### First level corrections:

first_level = {}
for i in range(m-1):
    bs_node2.setAttr(names[i],.5)
    for j in range(i+1,m):
        bs_node2.setAttr(names[j],.5)
        candidates = []
        c_values = []
        for c_shape in range(m2):
            if bs_node.w[c_shape].get() == .25:
                candidates.append(c_shape)
                c_values.append(bs_node.w[c_shape].get())
        if len(candidates) > 0:
            first_level[(i,j)] = [candidates, c_values]
        bs_node2.setAttr(names[j],0)
    bs_node2.setAttr(names[i],0)
    
fli = list(first_level.items())
keys = np.array([fli[i][0] for i in range(len(fli))])
corr_keys = np.array([fli[i][1][0] for i in range(len(fli))])

corr_shapes = []
for i in range(len(fli)):
    k1,k2 = keys[i][0],keys[i][1]
    bs_node2.setAttr(names[k1],1)    
    bs_node2.setAttr(names[k2],1)
    pred = neutral + deltas[k1] + deltas[k2]
    true = np.array([[x, y, z] for x, y, z in neutral_mesh.getPoints()], dtype=np.float64).flatten()
    corr_shapes.append(true-pred)
    bs_node2.setAttr(names[k1],0)    
    bs_node2.setAttr(names[k2],0)
    
### Second level corrections:

second_level = {}
for i in range(len(fli)):
    k1 = fli[i][0][0]
    k2 = fli[i][0][1]
    bs_node2.setAttr(names[k1],.5)
    bs_node2.setAttr(names[k2],.5)
    for j in range(m):
        if j != k1 and j!= k2:
            bs_node2.setAttr(names[j],.5)
            candidates = []
            c_values = []
            for c_shape in range(m2):
                if bs_node.w[c_shape].get() == .125:
                    candidates.append(c_shape)
                    c_values.append(bs_node.w[c_shape].get())
            if len(candidates) > 0:
                second_level[tuple(sorted((k1,k2,j)))] = [candidates, c_values]
            bs_node2.setAttr(names[j],0)
    bs_node2.setAttr(names[k1],0)    
    bs_node2.setAttr(names[k2],0)
    
sli = list(second_level.items())
keys2 = np.array([sli[i][0] for i in range(len(sli))])
corr_keys2 = np.array([sli[i][1][0] for i in range(len(sli))])
 
corr_shapes2 = []
for i in range(len(sli)):
    k1,k2,k3 = keys2[i][0],keys2[i][1],keys2[i][2]
    bs_node2.setAttr(names[k1],1)    
    bs_node2.setAttr(names[k2],1)
    bs_node2.setAttr(names[k3],1)
    pred = neutral + deltas[k1] + deltas[k2] + deltas[k3]
    if (k1,k2) in first_level:
        idx = first_level.keys().index((k1,k2))
        pred += corr_shapes[idx]
    if (k1,k3) in first_level:
        idx = first_level.keys().index((k1,k3))
        pred += corr_shapes[idx]
    if (k2,k3) in first_level:
        idx = first_level.keys().index((k2,k3))
        pred += corr_shapes[idx]
    true = np.array([[x, y, z] for x, y, z in neutral_mesh.getPoints()], dtype=np.float64).flatten()
    corr_shapes2.append(true-pred)
    bs_node2.setAttr(names[k1],0)    
    bs_node2.setAttr(names[k2],0)
    bs_node2.setAttr(names[k3],0)
    
### Third level corrections:

# Likewise, I asume that only that the only candidates are those triplets that already appeared in the second level corrections.
third_level = {}
for i in range(len(sli)):
    k1 = sli[i][0][0]
    k2 = sli[i][0][1]
    k3 = sli[i][0][2]
    bs_node2.setAttr(names[k1],.5)
    bs_node2.setAttr(names[k2],.5)
    bs_node2.setAttr(names[k3],.5)
    for j in range(m):
        if j != k1 and j!= k2 and j!=k3:
            bs_node2.setAttr(names[j],.5)
            candidates = []
            c_values = []
            for c_shape in range(m2):
                if bs_node.w[c_shape].get() == .0625:
                    candidates.append(c_shape)
                    c_values.append(bs_node.w[c_shape].get())
            if len(candidates) > 0:
                third_level[tuple(sorted((k1,k2,k3,j)))] = [candidates, c_values]
            bs_node2.setAttr(names[j],0)
    bs_node2.setAttr(names[k1],0)    
    bs_node2.setAttr(names[k2],0)
    bs_node2.setAttr(names[k3],0)
    
tli = list(third_level.items())
keys3 = np.array([tli[i][0] for i in range(len(tli))])
corr_keys3 = np.array([tli[i][1][0] for i in range(len(tli))])

corr_shapes3 = []
for i in range(len(tli)):
    k1,k2,k3,k4 = keys3[i][0],keys3[i][1],keys3[i][2],keys3[i][3]
    bs_node2.setAttr(names[k1],1)    
    bs_node2.setAttr(names[k2],1)
    bs_node2.setAttr(names[k3],1)
    bs_node2.setAttr(names[k4],1)
    pred = neutral + deltas[k1] + deltas[k2] + deltas[k3] + deltas[k4]
    if (k1,k2) in first_level:
        idx = first_level.keys().index((k1,k2))
        pred += corr_shapes[idx]
    if (k1,k3) in first_level:
        idx = first_level.keys().index((k1,k3))
        pred += corr_shapes[idx]
    if (k1,k4) in first_level:
        idx = first_level.keys().index((k1,k4))
        pred += corr_shapes[idx]
    if (k2,k3) in first_level:
        idx = first_level.keys().index((k2,k3))
        pred += corr_shapes[idx]
    if (k2,k4) in first_level:
        idx = first_level.keys().index((k2,k4))
        pred += corr_shapes[idx]
    if (k3,k4) in first_level:
        idx = first_level.keys().index((k3,k4))
        pred += corr_shapes[idx]
    if (k1,k2,k3) in second_level:
        idx = second_level.keys().index((k1,k2,k3))
        pred += corr_shapes2[idx]
    if (k1,k2,k4) in second_level:
        idx = second_level.keys().index((k1,k2,k4))
        pred += corr_shapes2[idx]
    if (k1,k3,k4) in second_level:
        idx = second_level.keys().index((k1,k3,k4))
        pred += corr_shapes2[idx]
    if (k2,k3,k4) in second_level:
        idx = second_level.keys().index((k2,k3,k4))
        pred += corr_shapes2[idx]
    true = np.array([[x, y, z] for x, y, z in neutral_mesh.getPoints()], dtype=np.float64).flatten()
    corr_shapes3.append(true-pred)
    bs_node2.setAttr(names[k1],0)    
    bs_node2.setAttr(names[k2],0)
    bs_node2.setAttr(names[k3],0)
    bs_node2.setAttr(names[k4],0)
    
### We assume there are no higher levels of corrections

### Save the extracted corrective terms and their indices
corr_shapes = np.array(corr_shapes)
corr_shapes2 = np.array(corr_shapes2)
corr_shapes3 = np.array(corr_shapes3)
np.save(os.path.join(data_dir,'bs1.npy'),corr_shapes)
np.save(os.path.join(data_dir,'bs2.npy'),corr_shapes2)
np.save(os.path.join(data_dir,'bs3.npy'),corr_shapes3)
np.save(os.path.join(data_dir,'keys1.npy'),keys)
np.save(os.path.join(data_dir,'keys2.npy'),keys2)
np.save(os.path.join(data_dir,'keys3.npy'),keys3)

print('Number of corrective terms of the first level: ', len(corr_shapes))
print('Number of corrective terms of the second level: ', len(corr_shapes2))
print('Number of corrective terms of the third level: ', len(corr_shapes3))
print('Data extracted successfully, stored at ', data_dir)

