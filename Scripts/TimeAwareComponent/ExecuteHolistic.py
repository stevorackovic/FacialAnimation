# -*- coding: utf-8 -*-
"""
Created on Wed Oct 4 16:54:17 2023

@author: Stevo Rackovic

"""

train_frames = 10           # this will take the first 'train_frames' from 'weights.npy' matrix as a training set
num_iter_max = 10           # the maximum number of iterations of the CD solver
num_iter_min = 5            # the minimum number of iterations of the CD solver
lmbd1 =  1                  # the sparsity regularization parameter of the objective funciton
lmbd2 =  1                  # the temporal smoothness regularization parameter of the objective funciton
T = 10                      # Interval batch size

# -----------------------------------------------------------------------------
import numpy as np
import os
from TACFunctions import solver_holistic, banded_matrix, banded_matrix_add
work_dir = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = os.path.join(work_dir,'Data')
predictions_dir = os.path.join(work_dir,'Predictions\TimeAwareComponent')
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
target_meshes = np.array([quartic_rig(C, deltas, bs1, bs2, bs3, keys1, keys2, keys3) for C in weights_train]).T
F = banded_matrix(T)
F_tilde, vector_add_e, vector_add_g = banded_matrix_add(F)

print('Total animation frames: ', len(weights))
print('Training animation frames: ', len(weights_train))

Predictions = []
for frame in range(N):
    target_mesh = target_meshes[frame]
    pred = solver_holistic(N,T,target_meshes,m,deltas,bs1, bs2, bs3, keys1, keys2, keys3,order,F,F_tilde,vector_add_e,vector_add_g,lmbd1,lmbd2,num_iter_max,num_iter_min)    
    Predictions.append(np.copy(pred))
filename = f'Pred_num_iter_max_{num_iter_max}_num_iter_min_{num_iter_min}_lmbd1_{lmbd1}_lmbd2_{lmbd2}_T_{T}.npy'
np.save(os.path.join(predictions_dir, filename), np.array(Predictions))
print('')
print('Predictions made and stored at ', predictions_dir)
