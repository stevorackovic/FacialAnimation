# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:49:38 2023

@author: Stevo Rackovic

"""

import numpy as np
from sklearn.cluster import KMeans
    
################################################################# Mesh Clusters

def mesh_clustering(deltas, num_clusters, neutral):
    '''
	The function preprocesses the original blendshape matrix to give a matrix 
    of offsets, and performs K-means over it, to produce mex clusters.

    Parameters
    ----------
    deltas : np.array(n,m)
        Deltas blendshape matrix.
    num_clusters : int
        Desired number of mesh clusters.
    neutral : np.array(n)
        Vectorized neutral face (resting pose).

    Returns
    -------
    vtx_labels : np.array(int(n/3))
        Vector of labels corresponginf to the mesh clusters of each mesh vertex.
    offset : np.array(int(n/3),m)
        Matrix of offsets, where an element in row i and column j represents the 
        Eucledian distance of the i-th vertex of the j-th blendshape from its 
        corresponding position in the neutral face.

    '''
    offset = np.sqrt(deltas[::3]**2)+(deltas[1::3]**2)+(deltas[2::3]**2)
    offset_scaled = offset/offset.max(0)
    kmeans_labels = KMeans(n_clusters=num_clusters).fit(offset_scaled).labels_
    vtx_clusters = [np.where(kmeans_labels==clstr)[0] for clstr in np.unique(kmeans_labels)]
    num_clusters = len(vtx_clusters)
    return vtx_clusters, num_clusters, offset

########################################################### Controller Clusters
    
def column_assignment(num_clusters, m, offset, vtx_clusters):
    '''
    Takes labels from vertex clustering, and assigns columns to corresponding 
	clusters. The idea is to create a new matrix, that has single row for each 
    mesh cluster, and entries are average values for controllers over that 
    cluster. Then, it perform k-means with k=2 over each column, to split 
    clusters into those with significant effect and the others.

    Parameters
    ----------
    num_clusters : int
        Desired number of mesh clusters.
    m : int
        Number of blendshapes.
    offset : np.array(int(n/3),m)
        Matrix of offsets, where an element in row i and column j represents the 
        Eucledian distance of the i-th vertex of the j-th blendshape from its 
        corresponding position in the neutral face.
    vtx_labels : np.array(int(n/3))
        Vector of labels corresponginf to the mesh clusters of each mesh vertex.

    Returns
    -------
    column_clusters : dict
        A dictionary with whose entries are the indices of the mesh clusters 
        where a given blendshape is assigned.

    '''
    Compress_mtx = np.zeros((num_clusters, m)) # here I store the estimated effects
    Index_mtx = np.zeros((m,num_clusters)) # here just an indicator if the controller is assigned or not
    for clstr in range(num_clusters):
        compress_row = np.mean(offset[vtx_clusters[clstr]], axis=0) # 
        Compress_mtx[clstr] += compress_row
        
    for ctr in range(m):
        compress_column = Compress_mtx[:,ctr]
        mean_cc = np.mean(compress_column)
        index_ctr = np.where(compress_column>mean_cc)[0]
        Index_mtx[ctr][index_ctr] += 1
        
    ctr_clusters = []
    Index_mtx2 = np.zeros((num_clusters,m))
    for clstr in range(num_clusters):
        initial_ctr = np.where(Index_mtx[:,clstr]>0)[0]
        if len(initial_ctr)>0:
            compress_row = Compress_mtx[clstr]
            threshold = np.min(compress_row[initial_ctr])
            adjusted_ctr = np.where(compress_row>=threshold)[0]
            Index_mtx2[clstr][adjusted_ctr] += 1
        ctr_clusters.append(np.where(Index_mtx2[clstr]==1)[0])
    
    return ctr_clusters

################################################## Overlapping Clusters Merging

def overlapping_factor(cl1,cl2,clust_dict):
    ''' 
	Observes clusters cl1 and cl2 and returns the percentage of the overlap (in 
	terms of the controllers)
    '''
    lst1,lst2 = clust_dict[cl1][0],clust_dict[cl2][0]
    if len(lst1)==0 or len(lst2)==0:
        return 0
    lst_overlap = list(set(lst1) & set(lst2))
    factor = max(len(lst_overlap)/len(lst1),len(lst_overlap)/len(lst2))
    return factor

def max_overlapping_factor(clust_dict):
    '''
	Traverse all the cluster pairs and find the one with the highest overalapping factor.
    '''
    num_clusters = len(clust_dict.keys())
    max_factor = 0
    pair = (0,0)
    for cl1 in range(num_clusters-1):
        for cl2 in range(cl1+1,num_clusters):
            factor = overlapping_factor(cl1,cl2,clust_dict)
            if factor > max_factor:
                max_factor = factor
                pair = (cl1,cl2)
    return max_factor, pair

def merge_overlapping(clust_dict,tol_factor,offset):
    ''' 
	Check if overlapping factor is above a given tolerance:
	-If yes - return a new dict with the two clusters merged and a flag=True.
	-If no  - retunr the same dict and a flag=False.
	'''
    max_factor, (cl1,cl2) = max_overlapping_factor(clust_dict)
    if max_factor <= tol_factor:
        return clust_dict, False
    new_dict = {}
    num_clusters = len(clust_dict)
    cl = 0
    for i in range(num_clusters):
        if i not in (cl1,cl2):
            new_dict[cl] = clust_dict[i]
            cl += 1
    ctr_union = list(set(clust_dict[cl1][0]) | set(clust_dict[cl2][0]))
    mesh_union = np.array(list(set(clust_dict[cl1][1]) | set(clust_dict[cl2][1])))
    new_dict[cl] = [ctr_union, mesh_union]
    return new_dict, True

def merging(clust_dict,tol_factor,offset):
    ''' 
	Clusters are merged until we get flag==False.
    '''
    flag = True 
    while flag:
        clust_dict,flag = merge_overlapping(clust_dict,tol_factor,offset)
    return clust_dict

############################################# Combined Mesh-Controller Clusters

def complete_clustering(deltas,num_clusters,neutral,m,factor=1,merge=True):
    '''
    Function for clustering the blendshape face model in both mesh space (rows) 
    and controller space (columns).

    Parameters
    ----------
    deltas : np.array(n,m)
        Deltas blendshape matrix.
    num_clusters : int
        Desired number of mesh clusters.
    neutral : np.array(n)
        Vectorized neutral face (resting pose).
    m : int
        Number of blendshapes.
    factor : float, optional
        If 'merge' is set to 'True', this will define the threshold for the overlapping 
        between the two clusters before they get merged. 0 <= factor <= 1. The default is 1.
    merge : bool, optional
        Specifies wether to merge the clusters with large number of overlapping 
        components or not. The default is True.

    Returns
    -------
    clust_dict : TYPE
        DESCRIPTION.

    '''
    vtx_clusters, num_clusters, offset = mesh_clustering(deltas,num_clusters,neutral)
    column_clusters = column_assignment(num_clusters, m, offset, vtx_clusters)
    clust_dict = {cl:[column_clusters[cl],vtx_clusters[cl]] for cl in range(num_clusters)}
    if merge == False:
        return clust_dict
    else:
        clust_dict = merging(clust_dict,factor,offset)
        return clust_dict