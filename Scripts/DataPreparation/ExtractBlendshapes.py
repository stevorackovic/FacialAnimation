# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:59:06 2023

@author: Stevo Rackovic

This script is to be executed in Autodesk Maya console, with a metahuman (or some other) avatar.
"""

N = 100   # Set here the number of frames of your animaiton
n = 72147 # Set here the number of vertices (times 3) of your avatar. The default value for metahumans is given here
data_dir = r'..\FacialAnimation\Data' # Put a path to your data directory

# -----------------------------------------------------------------------------
print('Extracting data with the following properties:')
print('Number of vertices in the mesh: ', int(n/3))
print('Number of the frames in the animation: ', N)

import os
import numpy as np
import pymel.core as pycore

bs_node = pycore.PyNode("head_lod0_mesh_blendShapes") 
neutral_mesh = pycore.PyNode("head_lod0_mesh") # mesh of the face - this might not be in a neutral position at the momment, so later we will have to set all the weights to 0 and obtain a real neutral face.
# Before running the next line, we need to select this node:
bs_node2 = pycore.PyNode("CTRL_expressions") 
names0 = [str(atr) for atr in pycore.listAttr(keyable=True)] # This should give over 200 names, so we need to check which ones produce zero offset (puppils) or are connected to the shoulder area.
m = len(names0)

## First we extract weights for each frame:
W = np.zeros((N,m)) # number of frames we have in this model
for i in range(N):
    pycore.currentTime(i)
    W[i]=[bs_node2.getAttr(nm) for nm in names0]
    
# Now we can also remove the controllers with zero offset:
W_offset = np.mean(W,axis=0)
W = W[:,W_offset>0]
names = []
for i in range(len(W_offset)):
    if W_offset[i] > 0:
        names.append(names0[i]) # this yields somewhere under 150 controllers for metahumans
m = len(names)
print('Number of blendshapes: ', m)
        
# Now go back to the first frame and set all weights to zero, to extract the neutral face
pycore.currentTime(0)
for nm in names0:
    bs_node2.setAttr(nm,0)
neutral_verts = np.array([[x, y, z] for x, y, z in neutral_mesh.getPoints()], dtype=np.float64).flatten()

# Extract each (delta) blendshape:
# The idea is to activate a single controller to 1 (or whatever is the max value) and all the rest to zero, and just take vertices
deltas = []
for nm in names:
    bs_node2.setAttr(nm,1)
    shape_points = np.array([[x, y, z] for x, y, z in neutral_mesh.getPoints()], dtype=np.float64).flatten()
    shape_delta = shape_points - neutral_verts
    deltas.append(shape_delta)
    bs_node2.setAttr(nm,0)
deltas = np.array(deltas)

# Now remove all those that do not affest the mesh (they are ususally connected only to pupils or teeth)
D_offset = np.sum(np.abs(deltas),axis=1)
deltas = deltas[D_offset>0]
names1 = []
for i in range(len(D_offset)):
    if D_offset[i] > 0:
        names1.append(names[i]) # this yields 130 controllers
W = W[:,D_offset>0]

np.save(os.path.join(data_dir,'weights.npy'),W)
np.save(os.path.join(data_dir,'deltas.npy'),deltas.T)
np.save(os.path.join(data_dir,'neutral.npy'),neutral_verts)

print('Data extracted successfully, stored at ', data_dir)

