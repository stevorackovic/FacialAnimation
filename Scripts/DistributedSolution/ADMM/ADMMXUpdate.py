# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:52:57 2023

@author: Stevo Rackovic

"""

import os
import sys
import numpy as np
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
script_dir = os.path.join(work_dir,'Scripts')
sys.path.append(script_dir)
from HelperFunctions import quartic_rig

def compute_increment(C, Z, U, deltas, bs1, bs2, bs3, keys1, keys2, keys3, target_mesh, ctr_list, order, rho):
    '''
    X-update of the ADMM formulation - single iteration

    Parameters
    ----------
    C : np.array(m)
        Vector of the initial controller activation weights, where m is the number of 
        blendshapes.
    Z : np.array(m)
        Z-vector of the ADMM forumlation.
    U : np.array(m)
        U-vector of the ADMM forumlation.
    deltas : np.array(n_k,m_k)
        Deltas blendshape sub-matrix, corresponding to the specified cluster, where
        n_k and m_k are the number of vertices and controllers in that cluster.
    bs1 : np.array(m1,n_k)
        Matrix containing (m1) corrective terms of the first level, with only the
        vertices assigned to the corresponding face cluster.
    bs2 : np.array(m2,n_k)
        Matrix containing (m2) corrective terms of the second level,  with only the
        vertices assigned to the corresponding face cluster.
    bs3 : np.array(m3,n_k)
        Matrix containing (m3) corrective terms of the third level, with only the
        vertices assigned to the corresponding face cluster.
    keys1 : np.array(m1,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs1.
    keys2 : np.array(m2,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs2.
    keys3 : np.array(m3,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs3.
    target_mesh : np.array(n_k)
        Ground-truth mesh sub-vector.
    ctr_list : list
        Each entry of the list will be a vector contatining the indices of the 
        blendshapes that correpsond to the considered cluster.
    order : np.array(m_k)
        Vector of indices of the belndshapes, assigned to the corresponding face 
        cluster, ordered by their respective magnitude of deformation.
    rho : float
        ADMM regularization parameter.

    Returns
    -------
    x : np.array(m_k)
        Subvector of controller weights, corresponing to the blendshapes assigned 
        to the face cluster.

    '''
    x = 0. + C
    for i in order:
        ctr = ctr_list[i]
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
        x_ctr = -(np.dot(coef1,coef2) + rho*(U[i]-Z[ctr]))/(np.linalg.norm(coef1)**2 + rho)
        x_ctr = max(min(x_ctr,1),0)
        x[ctr] = 0. + x_ctr
    return x

def minimization(Z, U, ctr_list, deltas, target_mesh, order, rho, bs1, bs2, bs3, keys1, keys2, keys3, num_iter=1):
    '''
    X-update of the ADMM formulation - multiple iteraions

    Parameters
    ----------
    Z : np.array(m)
        Z-vector of the ADMM forumlation.
    U : np.array(m)
        U-vector of the ADMM forumlation.
    ctr_list : list
        Each entry of the list will be a vector contatining the indices of the 
        blendshapes that correpsond to the considered cluster.
    deltas : np.array(n_k,m_k)
        Deltas blendshape sub-matrix, corresponding to the specified cluster, where
        n_k and m_k are the number of vertices and controllers in that cluster.
    target_mesh : np.array(n_k)
        Ground-truth mesh sub-vector.
    order : np.array(m_k)
        Vector of indices of the belndshapes, assigned to the corresponding face 
        cluster, ordered by their respective magnitude of deformation.
    rho : float
        ADMM regularization parameter.
    bs1 : np.array(m1,n_k)
        Matrix containing (m1) corrective terms of the first level, with only the
        vertices assigned to the corresponding face cluster.
    bs2 : np.array(m2,n_k)
        Matrix containing (m2) corrective terms of the second level,  with only the
        vertices assigned to the corresponding face cluster.
    bs3 : np.array(m3,n_k)
        Matrix containing (m3) corrective terms of the third level, with only the
        vertices assigned to the corresponding face cluster.
    keys1 : np.array(m1,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs1.
    keys2 : np.array(m2,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs2.
    keys3 : np.array(m3,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs3.
    num_iter : int, optional
        Number of iterations of the X-update. The default is 1.

    Returns
    -------
    Output : np.array(m_k)
        Subvector of controller weights, corresponing to the blendshapes assigned 
        to the face cluster

    '''
    C_new = 0. + Z
    for i in range(num_iter):
        x = compute_increment(C_new, Z, U, deltas, bs1, bs2, bs3, keys1, keys2, keys3, target_mesh, ctr_list, order, rho)
        Cnew = 0. + x
    Output = Cnew[ctr_list]
    return Output