# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 17:40:26 2023

@author: Stevo Rackovic

"""

import numpy as np
from HelperFunctions import quartic_rig

def compute_increment(C,deltas,bs1,bs2,bs3,keys1,keys2,keys3,target_mesh,order,lmbd):
    '''
    A single CD step of the proposed inverse rig solver

    Parameters
    ----------
    C : np.array(m)
        Initialization vector of the controller activation weights, where m is 
        the number of blendshapes.
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
        Estimated vector of the controller activation weights.

    '''
    x = 0. + C
    for ctr in order:
        # Coeficients.
        C_ctr_1 = 0. + x
        C_ctr_1[ctr] = 1.
        # First level corrective terms where 'ctr' is taking part:
        indices1 = np.where(keys1==ctr)
        coef1_1 = np.dot(x[keys1[indices1[0],1-indices1[1]]], bs1[indices1[0]])
        # Second level corrections:
        indices2 = np.where(keys2==ctr)[0]        
        coef1_2 = np.dot(C_ctr_1[keys2[indices2]].prod(1), bs2[indices2])
        # Third level corrections
        indices3 = np.where(keys3==ctr)[0]        
        coef1_3 = np.dot(C_ctr_1[keys3[indices3]].prod(1), bs3[indices3])        
        # there we also add a corresponding blendshape vector
        coef1 = coef1_1+coef1_2+coef1_3+deltas[:,ctr]
        # 2. Part of the rig without 'ctr' is obtained by setting controller ctr to zero:
        C_no_ctr = 0. + x
        C_no_ctr[ctr] = 0.
        coef2 = quartic_rig(C_no_ctr,deltas,bs1,bs2,bs3,keys1,keys2,keys3)
        # there we also add a target mesh
        coef2 -= target_mesh
        x_ctr = -(lmbd + np.dot(coef1,coef2))/(np.linalg.norm(coef1)**2)
        x_ctr = min(max(x_ctr,0),1)
        x[ctr] = 0. + x_ctr
    return x

def minimization(num_iter,C,deltas,target_mesh,bs1,bs2,bs3,keys1,keys2,keys3,order,lmbd):
    '''
    CD-based solver for the inverse rig problem

    Parameters
    ----------
    num_iter : int
        Number of algorithm iterations.
    C : np.array(m)
        Initialization vector of the controller activation weights, where m is 
        the number of blendshapes.
    deltas : np.array(n,m)
        Deltas blendshape matrix.
    target_mesh : np.array(n)
        Ground-truth mesh vector.
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
    order : np.array(m)
        Vector of indices of the belndshapes, ordered by their respective magnitude 
        of deformation.
    lmbd : float
        Regularization aprameter. lambda>=0.
        
    Returns
    -------
    x : np.array(m)
        Estimated vector of the controller activation weights.

    '''
    n,m = deltas.shape
    x = 0. + C
    for i in range(num_iter):
        Cnew = 0. + x
        x = compute_increment(Cnew,deltas,bs1,bs2,bs3,keys1,keys2,keys3,target_mesh,order,lmbd)
    return x


