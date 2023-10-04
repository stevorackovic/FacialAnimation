# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:45:01 2023

@author: Stevo Rackovic

"""

number_of_clusters = 25

# -----------------------------------------------------------------------------
import numpy as np
import os
from ClusteringRSJD import complete_clustering as clusteringRSJD
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
data_dir = os.path.join(work_dir,'Data')
clusters_dir = os.path.join(work_dir,'Data\Clusters')
os.chdir(os.path.join(work_dir,'Scripts'))
from HelperFunctions import compute_error_density, compute_density

neutral = np.load(os.path.join(data_dir,'neutral.npy'))
deltas = np.load(os.path.join(data_dir,'deltas.npy'))
n,m = deltas.shape

print('Dimensions of the data:')
print('Number of vertices in the mesh: ', int(n/3))
print('Number of blendshapes: ', m)
print('Selected number of clusters: ', number_of_clusters)

clust_dict = clusteringRSJD(deltas,number_of_clusters,neutral,m,factor=0.8)
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
print('')
print('Number of non-empty clusters: ', num_clusters_nonempty)
print('Reconstruction Error ($E_R$): ', ReconstructionError)
print('Density ($E_D$): ', Density/(n*m))
print('Inter-Density ($E_{ID}$): ', InterDensity/(n*m))

os.chdir(clusters_dir)
counter = 0
for i in range(num_clusters):
    if len(ctr_list[i])>0:
        np.save('RSJD_'+str(num_clusters_nonempty)+'_vtx_cluster_'+str(counter)+'.npy',vrtcs_list[i])
        np.save('RSJD_'+str(num_clusters_nonempty)+'_ctr_cluster_'+str(counter)+'.npy',ctr_list[i])
        counter += 1
print('Clusters created successfully, stored at ', clusters_dir)
