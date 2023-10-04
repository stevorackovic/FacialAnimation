# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:59:53 2023

@author: Stevo Rackovic

"""

clustering_method = 'RSJDA' # the method used for clustering, either RSJD or RSJDA
number_of_clusters = 24     # the number of clusters produced by the specified method. This should be available in hte file names, in the folder ..\Data\Clusters
train_frames = 10           # this will take the first 'train_frames' from 'weights.npy' matrix as a training set
num_iter = 7                # the number of iterations of the CD solver
lmbd =  5                   # the regularization parameter of the objective funciton

# -----------------------------------------------------------------------------
import numpy as np
import os
from BlockCoordinateDescent import minimization
from ClusteredSetting import split_data
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
data_dir = os.path.join(work_dir,'Data')
clusters_dir = os.path.join(work_dir,'Data\Clusters')
predictions_dir = os.path.join(work_dir,'Predictions\DistributedSolution\Clustering')
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

print('Dimensions of the data:')
print('Number of vertices in the mesh: ', int(n/3))
print('Number of blendshapes: ', m)
print('Number of corrective terms of the first level: ', len(keys1))
print('Number of corrective terms of the second level: ', len(keys2))
print('Number of corrective terms of the third level: ', len(keys3))
print('Selected number of clusters: ', number_of_clusters)

weights = np.load(os.path.join(data_dir,'weights.npy'))
weights_train = weights[:train_frames]
N = len(weights_train)
target_meshes = np.array([quartic_rig(C, deltas, bs1, bs2, bs3, keys1, keys2, keys3) for C in weights_train])

print('Total animation frames: ', len(weights))
print('Training animation frames: ', len(weights_train))

# Load Clusters: 
    
vtx_list, ctr_list,coord_list = [],[],[]
for clstr in range(number_of_clusters):
    coord_clstr = []
    vtx_clstr = np.load(os.path.join(data_dir,'Clusters/'+clustering_method+'_'+str(number_of_clusters)+'_vtx_cluster_'+str(clstr)+'.npy'))
    ctr_clstr = np.load(os.path.join(data_dir,'Clusters/'+clustering_method+'_'+str(number_of_clusters)+'_ctr_cluster_'+str(clstr)+'.npy'))
    for v in vtx_clstr:
        for i in range(3):
            coord_clstr.append(3*v+i)
    vtx_list.append(vtx_clstr)
    ctr_list.append(ctr_clstr)
    coord_list.append(coord_clstr)
deltas_list, n_list, m_list, order_list, denominator, bs1_list, bs2_list, bs3_list = split_data(ctr_list, coord_list, deltas, bs1, bs2, bs3)
lmbd_list = [lmbd/number_of_clusters for _ in range(number_of_clusters)]
print('')
print('Clusters loaded, and data split accordingly.')

# Solve the rig inversion problem:
    
Predictions = []
for frame in range(N):
    target_mesh = target_meshes[frame]
    C = np.zeros(m)
    pred = minimization(C, deltas_list, target_mesh, order_list, lmbd_list, num_iter, ctr_list, coord_list, bs1_list, bs2_list, bs3_list, keys1, keys2, keys3)
    Predictions.append(pred)
filename = f'Pred_{clustering_method}_num_iter_{num_iter}_lmbd_{lmbd}.npy'
np.save(os.path.join(predictions_dir, filename), np.array(Predictions))

print('')
print('Predictions made and stored at ', predictions_dir)
