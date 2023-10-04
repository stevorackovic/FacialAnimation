# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 17:47:58 2023

@author: Stevo Rackovic

"""

train_frames = 10           # this will take the first 'train_frames' from 'weights.npy' matrix as a training set
num_iter = 7                # the number of iterations of the CD solver
lmbd =  5                   # the regularization parameter of the objective funciton

# -----------------------------------------------------------------------------
import numpy as np
import os
from CDFunctions import minimization
work_dir = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = os.path.join(work_dir,'Data')
predictions_dir = os.path.join(work_dir,'Predictions\CoordinateDescent')
from HelperFunctions import quartic_rig, ctr_order

neutral = np.load(os.path.join(data_dir,'neutral.npy'))
deltas = np.load(os.path.join(data_dir,'deltas.npy'))
n,m = deltas.shape
bs1  = np.load(os.path.join(data_dir,'bs1.npy'))
bs2  = np.load(os.path.join(data_dir,'bs2.npy'))
bs3  = np.load(os.path.join(data_dir,'bs3.npy'))
keys1  = np.load(os.path.join(data_dir,'keys1.npy'))
keys2  = np.load(os.path.join(data_dir,'keys2.npy'))
keys3  = np.load(os.path.join(data_dir,'keys3.npy'))
order = ctr_order(deltas)

print('Dimensions of the data:')
print('Number of vertices in the mesh: ', int(n/3))
print('Number of blendshapes: ', m)
print('Number of corrective terms of the first level: ', len(keys1))
print('Number of corrective terms of the second level: ', len(keys2))
print('Number of corrective terms of the third level: ', len(keys3))

weights = np.load(os.path.join(data_dir,'weights.npy'))
weights_train = weights[:train_frames]
N = len(weights_train)
target_meshes = np.array([quartic_rig(C, deltas, bs1, bs2, bs3, keys1, keys2, keys3) for C in weights_train])

print('Total animation frames: ', len(weights))
print('Training animation frames: ', len(weights_train))

Predictions = []
for frame in range(N):
    target_mesh = target_meshes[frame]
    pred = minimization(num_iter,np.zeros(m),deltas,target_mesh,bs1,bs2,bs3,keys1,keys2,keys3,order,lmbd)
    Predictions.append(np.copy(pred))
filename = f'Pred_num_iter_{num_iter}_lmbd_{lmbd}.npy'
np.save(os.path.join(predictions_dir, filename), np.array(Predictions))
print('')
print('Predictions made and stored at ', predictions_dir)
