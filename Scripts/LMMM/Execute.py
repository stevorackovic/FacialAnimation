# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 17:15:32 2023

@author: Stevo Rackovic

"""

train_frames = 10           # this will take the first 'train_frames' from 'weights.npy' matrix as a training set
num_iter = 200              # the number of iterations of the CD solver
lmbd =  5                   # the regularization parameter of the objective funciton
tolerance = 0.0005          # stopping criteria (if the norm of the difference of the two consecutive vectors i s less than tolerance, break)
n_batches = 40              # for parallel computing within the function for computing the terms and coefficients of the surogate
initialization = 'CO'       # Initialization method. If 'CO', use the speudoinverse warm start of Cetinaslan aand Orvalho. If '0', use zero-initialization

# -----------------------------------------------------------------------------
import numpy as np
import os
from LMMMFunctions import minimization
work_dir = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = os.path.join(work_dir,'Data')
predictions_dir = os.path.join(work_dir,'Predictions\LMMM')
from HelperFunctions import quartic_rig

neutral = np.load(os.path.join(data_dir,'neutral.npy'))
deltas = np.load(os.path.join(data_dir,'deltas.npy'))
n,m = deltas.shape
bs1  = np.load(os.path.join(data_dir,'bs1.npy'))
bs2  = np.load(os.path.join(data_dir,'bs2.npy'))
bs3  = np.load(os.path.join(data_dir,'bs3.npy'))
keys1  = np.load(os.path.join(data_dir,'keys1.npy'))
keys2  = np.load(os.path.join(data_dir,'keys2.npy'))
keys3  = np.load(os.path.join(data_dir,'keys3.npy'))
eig_max_D = np.load(os.path.join(data_dir,'eigen_max.npy'))   # the largest eigenvalues
eig_min_D = np.load(os.path.join(data_dir,'eigen_min.npy'))   # the smallest eigenvalues
sigma_D   = np.load(os.path.join(data_dir,'singular.npy'))    # the largest singular values

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
print('')

if initialization == 'CO':
    # Solution with warm-start (initializaed at the pseudoinverse solution of Cetinaslan and Orvalho)
    print('Solving for CO initialization')
    Bpsd = (np.linalg.inv(deltas.T.dot(deltas) + lmbd*np.eye(m))).dot(deltas.T)
    Predictions = []
    for frame in range(N):
        target_mesh = target_meshes[frame]
        x0 = Bpsd.dot(target_mesh)
        x0[x0<0] = 0
        x0[x0>1] = 1
        res = minimization(num_iter,x0,deltas,target_mesh,eig_max_D,eig_min_D,sigma_D,lmbd,bs1,keys1,n_batches)
        Predictions.append(res)
    filename = f'Pred_initialization_{initialization}_lmbd_{lmbd}.npy'
    np.save(os.path.join(predictions_dir, filename), np.array(Predictions)) 
    print('')
    print('Predictions made and stored at ', predictions_dir)
    
if initialization == '0':
    # Solution with warm-start (initializaed at the pseudoinverse solution of Cetinaslan and Orvalho)
    print('Solving for 0-initialization')
    Bpsd = (np.linalg.inv(deltas.T.dot(deltas) + lmbd*np.eye(m))).dot(deltas.T)
    Predictions = []
    for frame in range(N):
        target_mesh = target_meshes[frame]
        x0 = np.zeros(m)
        res = minimization(num_iter,x0,deltas,target_mesh,eig_max_D,eig_min_D,sigma_D,lmbd,bs1,keys1,n_batches)
        Predictions.append(res)
    filename = f'Pred_initialization_{initialization}_lmbd_{lmbd}.npy'
    np.save(os.path.join(predictions_dir, filename), np.array(Predictions)) 
    print('')
    print('Predictions made and stored at ', predictions_dir)