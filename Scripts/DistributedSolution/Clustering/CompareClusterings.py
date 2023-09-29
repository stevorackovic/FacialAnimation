# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:45:01 2023

@author: Stevo Rackovic

"""

cluster_number_choice = [4,10,20,50,102]
number_of_repetitions = 1000
method1='RSJD'
method2='RSJDA'

import numpy as np
import matplotlib.pyplot as plt
import os
from ClusteringRSJDA import complete_clustering as clusteringRSJDA
from ClusteringRSJD import complete_clustering as clusteringRSJD
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
data_dir = os.path.join(work_dir,'Data')
os.chdir(os.path.join(work_dir,'Scripts'))
from HelperFunctions import compute_error_density, compute_density

neutral = np.load(os.path.join(data_dir,'neutral.npy'))
deltas = np.load(os.path.join(data_dir,'deltas.npy'))
n,m = deltas.shape

def EvaluatingClusterings(number_of_repetitions,cluster_number_choice,deltas,neutral,method='RSJD'):
    n,m = deltas.shape
    Res_tuple = [] # A matrix that will store five values for each clustering:
        # 0. - number of non-empty clusters, 1. - number of clusters, 2. - ReconstructionError, 3. - Density, 4. Inter-Density 
    for rep in range(number_of_repetitions):
        num_clstr = np.random.choice(cluster_number_choice)
        if method=='RSJD':
            clust_dict = clusteringRSJD(deltas,num_clstr,neutral,m,factor=0.9)
        else:
            clust_dict = clusteringRSJDA(deltas,num_clstr,neutral,m,factor=0.9)
        vrtcs_list,ctr_list,num_clusters, num_clusters_nonempty = [],[],0,0
        for i in range(len(clust_dict.keys())):
            num_clusters += 1
            vrtcs_list.append(clust_dict[i][1])
            ctr_list.append(clust_dict[i][0])
            if len(clust_dict[i][0]):
                num_clusters_nonempty += 1
        coord_list = []
        for clstr in range(num_clusters):
            cl = []
            for i in vrtcs_list[clstr]:
                for j in range(3):
                    cl.append(3*i+j)
            coord_list.append(cl)
        ReconstructionError, Density, AssignmentMatrix, _ = compute_error_density(vrtcs_list,ctr_list,coord_list,deltas)
        InterDensity = compute_density(vrtcs_list,AssignmentMatrix)
        Res_tuple.append([num_clusters_nonempty, num_clusters, ReconstructionError, Density, InterDensity])
    Res_tuple = np.array(Res_tuple)
    return Res_tuple

Res_tuple = EvaluatingClusterings(number_of_repetitions,cluster_number_choice,deltas,neutral,method='RSJD')
num_clusters_nonempty, errors, density, interdensity = Res_tuple[:,0], Res_tuple[:,2], Res_tuple[:,3]/(n*m), Res_tuple[:,4]/(n*m)
Res_tuple_adjusted = EvaluatingClusterings(number_of_repetitions,cluster_number_choice,deltas,neutral,method='RSJDA')
num_clusters_nonempty_adjusted, errors_adjusted, density_adjusted, interdensity_adjusted = Res_tuple_adjusted[:,0], Res_tuple_adjusted[:,2], Res_tuple_adjusted[:,3]/(n*m), Res_tuple_adjusted[:,4]/(n*m)

fig, axes = plt.subplots(1,2,figsize=(8,4))
axes[0].scatter(density, errors, color='g', label='$RSJD$')
axes[0].scatter(density_adjusted, errors_adjusted, color='r', label='$RSJD_A$')
axes[0].legend()
axes[0].set_xlabel('Density ($E_D$)')
axes[0].set_ylabel('Reconstruction Error ($E_R$)')

axes[1].scatter(interdensity, errors, color='g', label='$RSJD$')
axes[1].scatter(interdensity_adjusted, errors_adjusted, color='r', label='$RSJD_A$')
axes[1].legend()
axes[1].set_xlabel('Inter-Density ($E_{ID}$)')
plt.show()