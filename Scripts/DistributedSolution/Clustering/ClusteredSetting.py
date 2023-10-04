# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:37:09 2023

@author: Stevo Rackovic

"""

import os
import sys
import numpy as np
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
script_dir = os.path.join(work_dir,'Scripts')
sys.path.append(script_dir)
from HelperFunctions import ctr_order

def load_clusters(num_clusters, method='RSJD'):
    '''
    Function for loading the clusters.

    Parameters
    ----------
    num_clusters : int
        Number of clusters. It will correspond to the first number in the file 
        name of the stored clusters.
    method : str
        Method whose clusters should be loaded. Either 'RSJD' or 'RSJDA'.

    Returns
    -------
    vtx_list : list
        Each entry of the list will be a vector contatining the indices of the 
        vertices that correpsond to the considered cluster.
    ctr_list : list
        Each entry of the list will be a vector contatining the indices of the 
        blendshapes that correpsond to the considered cluster.
    coord_list : list
        Each entry of the list will be a vector contatining the indices of the 
        vertex-coordinates that correpsond to the considered cluster.
    num_clusters : int
        Number of non-empty clusters.

    '''
    vtx_list   = []
    ctr_list   = []
    coord_list = []
    for i in range(num_clusters):
        ctr_cluster = np.load(os.path.join(work_dir,'Data/Clusters/'+method+'_'+str(num_clusters)+'_ctr_cluster_'+str(i)+'.npy'))
        if len(ctr_cluster)>0:    
            vtx_cluster = np.load(os.path.join(work_dir,'Data/Clusters/'+method+'_'+str(num_clusters)+'_vtx_cluster_'+str(i)+'.npy'))
            vtx_list.append(vtx_cluster)
            ctr_list.append(ctr_cluster)
            coords = []
            for vtx in vtx_cluster:
                for j in range(3):
                    coords.append(vtx*3+j)
            coord_list.append(coords)
    num_clusters = len(ctr_list) # The number of clusters after the empty ones are excluded
    return vtx_list, ctr_list, coord_list, num_clusters
    
def split_data(ctr_list, coord_list, deltas, bs1, bs2, bs3): 
    '''
    Function that takes the blendshape model and elements, and split them 
    accoridng to the clustering of the blendshpae mode, to obtain set of 
    smaller submodels.

    Parameters
    ----------
    ctr_list : list
        Each entry of the list will be a vector contatining the indices of the 
        blendshapes that correpsond to the considered cluster.
    coord_list : list
        Each entry of the list will be a vector contatining the indices of the 
        vertex-coordinates that correpsond to the considered cluster.
    deltas : np.array(n,m)
        Deltas blendshape matrix.
    bs1 : np.array(m1,n)
        Matrix containing (m1) corrective terms of the first level.
    bs2 : np.array(m2,n)
        Matrix containing (m2) corrective terms of the second level.
    bs3 : np.array(m3,n)
        Matrix containing (m3) corrective terms of the third level.

    Returns
    -------
    deltas_list : list
        Each element of the list is a submatrix of deltas, that correspond to 
        only the rows assigned to that cluster.
    n_list : list
        Number of vertices in each cluster.
    m_list : list
        Number of blendshapes in each cluster.
    order_list : list
        Ordering of the blendshapes in each cluster.
    denominator : np.array(m)
        A vector contating teh multiplicity of blendshapes over each cluster.
    bs1_list : list
        Each element of the list is a submatrix of bs1, that correspond to 
        only the rows assigned to that cluster.
    bs2_list : list
        Each element of the list is a submatrix of bs2, that correspond to 
        only the rows assigned to that cluster.
    bs3_list : list
        Each element of the list is a submatrix of bs3, that correspond to 
        only the rows assigned to that cluster.
        
    '''
    num_clusters = len(ctr_list)
    m = deltas.shape[1]
    deltas_list  = []
    n_list       = []
    m_list       = []
    bs1_list     = []
    bs2_list     = []
    bs3_list     = []
    order_list   = []
    denominator  = np.zeros(m)
    for i in range(num_clusters):
        denominator[ctr_list[i]] += 1
        deltas_list.append(deltas[coord_list[i]])
        n_list.append(len(coord_list[i]))
        m_list.append(len(ctr_list[i]))
        bs1_list.append(bs1[:,coord_list[i]])
        bs2_list.append(bs2[:,coord_list[i]])
        bs3_list.append(bs3[:,coord_list[i]])

        order = ctr_order(deltas_list[i][:,ctr_list[i]]) # to order them, we only compare the columns that are to be optimized
        order_list.append(order)
    denominator[denominator==0] += 1 # This never happens with our clustering, but with that one of [Romeo 2020] or [Seol 2011] it might happen
    return deltas_list, n_list, m_list, order_list, denominator, bs1_list, bs2_list, bs3_list

def lambda_ratios1(deltas_list, ctr_list, lmbd=1):
    '''
    Computse the suggested rations between the regularization parameters of 
    different clusters, based on the cluster size. (Non-ADMM case)

    Parameters
    ----------
    deltas_list : list
        Each element of the list is a submatrix of deltas, that correspond to 
        only the rows assigned to that cluster.
    ctr_list : list
        Each entry of the list will be a vector contatining the indices of the 
        blendshapes that correpsond to the considered cluster.
    lmbd : float, optional
        Regularization parameter. lmbd > 0. The default is 1.

    Returns
    -------
    lmbd_list : np.array(num_clusters)
        Vector of suggested regularizer for each cluster.

    '''
    ratios = []
    for clstr in range(len(deltas_list)):
        dlts_c = deltas_list[clstr][:,ctr_list[clstr]]
        ctr_c = ctr_list[clstr]
        ratios.append(0.5*(np.linalg.norm(dlts_c.mean(1))**2)/len(ctr_c))
    lmbd_list =  lmbd * np.array(ratios)
    return lmbd_list

def lambda_ratios2(deltas_list, ctr_list, lmbd=1):
    '''
    Computse the suggested rations between the regularization parameters of 
    different clusters, based on the cluster size. (ADMM case)

    Parameters
    ----------
    deltas_list : list
        Each element of the list is a submatrix of deltas, that correspond to 
        only the rows assigned to that cluster.
    ctr_list : list
        Each entry of the list will be a vector contatining the indices of the 
        blendshapes that correpsond to the considered cluster.
    lmbd : float, optional
        Regularization parameter. lmbd > 0. The default is 1.

    Returns
    -------
    lmbd_list : np.array(num_clusters)
        Vector of suggested regularizer for each cluster.

    '''
    ratios = []
    for clstr in range(len(deltas_list)):
        dlts_c = deltas_list[clstr]
        ctr_c = ctr_list[clstr]
        ratios.append(0.5*(np.linalg.norm(dlts_c.mean(1))**2)/len(ctr_c))
    lmbd_list =  lmbd * np.array(ratios)
    return lmbd_list

    