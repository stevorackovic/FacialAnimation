# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:28:00 2023

@author: Stevo Rackovic

"""

import numpy as np
import os
import sys
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
script_dir = os.path.join(work_dir,'Scripts')
sys.path.append(script_dir)
from HelperFunctions import quartic_rig

def compute_increment(C, ctr_cluster, deltas, bs1, bs2, bs3, keys1, keys2, keys3, target_mesh, order, lmbd):
    '''
    This function computes the increments for blendshape weights belonging to the 
    specified cluster ctr_cluster, using coordinate descent. All the weights are 
    sent to the cluster, but only those of the ctr_cluster are optimized.


    Parameters
    ----------
    C : np.array(m)
        Vector of the controller activation weights, where m is the number of 
        blendshapes.
    ctr_cluster : np.array(m_k) 
        Elements of the array are indices of the blenshapes that are assigned to 
        the considered cluster; where m_k is the number of blendshapes in the 
        selected cluster.
    deltas : np.array(n,m)
        Deltas blendshape matrix.
    bs1 : np.array(m1,n)
        Matrix containing (m1) corrective terms of the first level.
    bs2 : np.array(m2,n)
        Matrix containing (m2) corrective terms of the second level.
    bs3 : np.array(m3,n)
        Matrix containing (m3) corrective terms of the third level.
    keys1 : np.array(m1,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs1.
    keys2 : np.array(m2,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs2.
    keys3 : np.array(m3,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs3.
    target_mesh : np.array(n)
        Ground-truth mesh vector.
    order : np.array(m)
        Vector of indices of the belndshapes, ordered by their respective magnitude 
        of deformation.
    lmbd : float
        Regularization aprameter. lambda>=0.

    Returns
    -------
    x : np.array(m)
        Vector of the controller activation weights with added increments.

    '''
    x = 0. + C
    for ctr in ctr_cluster[order]:
        # first the corrective terms where ctr is included
        C_ctr_1 = 0. + x
        C_ctr_1[ctr] = 1.
        indices1 = np.where(keys1==ctr)
        coef1 = np.dot(x[keys1[indices1[0],1-indices1[1]]], bs1[indices1[0]]) + deltas[:,ctr]
        indices2 = np.where(keys2==ctr)[0] # second level
        coef1 += np.dot(C_ctr_1[keys2[indices2]].prod(1), bs2[indices2])
        indices3 = np.where(keys3==ctr)[0] # third level
        coef1 += np.dot(C_ctr_1[keys3[indices3]].prod(1), bs3[indices3])
        # then the corrective cases where ctr is not included
        C_no_ctr = 0. + x
        C_no_ctr[ctr] = 0.
        coef2 = quartic_rig(C_no_ctr,deltas,bs1,bs2,bs3,keys1,keys2,keys3)
        coef2 -= target_mesh
        x_ctr = -(lmbd + np.dot(coef1,coef2))/(np.linalg.norm(coef1)**2)
        x_ctr = min(max(x_ctr,0),1) # 
        x[ctr] = 0. + x_ctr
    return x

def minimization_single(C, ctr_cluster, deltas, target_mesh, order, bs1, bs2, bs3, keys1, keys2, keys3, lmbd=0., num_iter=1):
    '''
    This function performs multiple CD steps (defined in the function compute_increment)
    over the specified face cluster.

    Parameters
    ----------
    C : np.array(m)
        Vector of the controller activation weights, where m is the number of 
        blendshapes.
    ctr_cluster : np.array(m_k) 
        Elements of the array are indices of the blenshapes that are assigned to 
        the considered cluster; where m_k is the number of blendshapes in the 
        selected cluster.
    deltas : np.array(n,m)
        Deltas blendshape matrix.
    target_mesh : np.array(n)
        Ground-truth mesh vector.
    order : np.array(m)
        Vector of indices of the belndshapes, ordered by their respective magnitude 
        of deformation.
    bs1 : np.array(m1,n)
        Matrix containing (m1) corrective terms of the first level.
    bs2 : np.array(m2,n)
        Matrix containing (m2) corrective terms of the second level.
    bs3 : np.array(m3,n)
        Matrix containing (m3) corrective terms of the third level.
    keys1 : np.array(m1,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs1.
    keys2 : np.array(m2,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs2.
    keys3 : np.array(m3,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs3.
    lmbd : float, optional
        Regularization aprameter. lambda>=0. The default is 0..
    num_iter : int, optional
        max number of iterations before termination. The default is 1.

    Returns
    -------
    C_subvector : np.array(m_k)
        The updated vector of weights, containing only the controllers belonging
        to the specified cluster; where m_k is the number of blendshapes in the 
        selected cluster..

    '''
    Cnew = 0. + C
    for i in range(num_iter):
        x = compute_increment(Cnew,ctr_cluster,deltas,bs1,bs2,bs3,keys1,keys2,keys3,target_mesh,order,lmbd)
        Cnew = 0. + x 
    C_subvector = x[ctr_cluster]
    return C_subvector

def vec_merge(x_list, ctr_list, denominator):
    '''
    Merges the results (weigth subvectors) of all the clusters.

    Parameters
    ----------
    x_list : list
        Each entry of the list will be a vector contatining the weights of the 
        blendshapes that correpsond to the considered cluster.
    ctr_list : list
        Each entry of the list will be a vector contatining the indices of the 
        blendshapes that correpsond to the considered cluster.
    denominator : np.array(m)
        A vector contating teh multiplicity of blendshapes over each cluster.

    Returns
    -------
    w : TYPE
        DESCRIPTION.

    '''
    w = np.zeros(len(denominator))
    for i in range(len(x_list)):
        w[ctr_list[i]] += x_list[i]
    w /= denominator
    return w

def minimization(C, deltas_list, target_mesh, order_list, lmbd_list, num_iter, ctr_list, coord_list, bs1_list, bs2_list, bs3_list, keys1, keys2, keys3):
    '''
    Performs minimization by applying the function minimization_single over each 
    face cluster, and merging the results using the function vec_merge.

    Parameters
    ----------
    C : np.array(m)
        Vector of the initial controller activation weights, where m is the number of 
        blendshapes.
    deltas : np.array(n,m)
        Deltas blendshape matrix.
    target_mesh : np.array(n)
        Ground-truth mesh vector.
    order_list : list
        Each element of the list is a vector of indices of the belndshapes, 
        assigned to the corresponding face cluster, ordered by their respective 
        magnitude of deformation.
    lmbd_list : list
        List of regularization aprameters corresponding to each face cluster.
    num_iter : int
        max number of iterations before termination.
    ctr_list : list
        Each entry of the list will be a vector contatining the indices of the 
        blendshapes that correpsond to the considered cluster.
    coord_list : list
        Each entry of the list will be a vector contatining the indices of the 
        vertex-coordinates that correpsond to the considered cluster.
    bs1 : np.array(m1,n)
        Matrix containing (m1) corrective terms of the first level.
    bs2 : np.array(m2,n)
        Matrix containing (m2) corrective terms of the second level.
    bs3 : np.array(m3,n)
        Matrix containing (m3) corrective terms of the third level.
    keys1 : np.array(m1,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs1.
    keys2 : np.array(m2,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs2.
    keys3 : np.array(m3,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs3.

    Returns
    -------
    C_new :  np.array(m)
        Vector of the estimated controller activation weights.

    '''
    num_clusters = len(deltas_list)
    x_list = []
    denominator = 0. * C
    for clstr in range(num_clusters):
        C_clstr = 0.*C
        C_clstr[ctr_list[clstr]] += C[ctr_list[clstr]]
        x = minimization_single(C_clstr, ctr_list[clstr], deltas_list[clstr], target_mesh[coord_list[clstr]], order_list[clstr], bs1_list[clstr], bs2_list[clstr], bs3_list[clstr], keys1, keys2, keys3, lmbd_list[clstr], num_iter)
        x_list.append(x)
        denominator[ctr_list[clstr]] += 1 
    denominator[denominator==0] += 1 # This should never happen with our clustering, but with that one of [Romeo 2020] or [Seol 2011] it might happen
    C_new = vec_merge(x_list,ctr_list,denominator)
    return C_new
        
    
        
    
    
    
