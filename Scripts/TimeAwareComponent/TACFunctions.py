# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:26:48 2024

@author: Stevo Rackovic

"""

import numpy as np
from Holistic_CD_Interval import minimization as minimzation_holistic, banded_matrix, banded_matrix_add


def solver_holistic(N,T,target_meshes,m,deltas,bs1,bs2,bs3,keys1,keys2,keys3,order,F,F_tilde,vector_add_e,vector_add_g,lmbd1,lmbd2,num_iter_max=10,num_iter_min=5):
    '''
    

    Parameters
    ----------
    N : int
        The total number of frames in the animation.
    T : int
        Granularity - If you want to split the total animation sequence into 
        subintervals, this decides the size. It must be T<=N.
    target_mesh : np.array(n)
        Ground-truth mesh vector.
    m : int
        Numebr of blendshapes.
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
    order : TYPE
        DESCRIPTION.
    F : np.array(T,T)
        Banded matrix.
    F_tilde : np.array(T,T)
        Edge-case banded matrix.
    vector_add_e : TYPE
        DESCRIPTION.
    vector_add_g : TYPE
        DESCRIPTION.
    lmbd1 : float
        Sparsity regularizer.
    lmbd2 : float
        Temporal smoothness regularizer.
    num_iter_max : int, optional
        The max number of iterations befor teh algorithm terminates. The 
        default is 10.
    num_iter_min : int, optional
        The min number of iterations befor teh algorithm terminates. The 
        default is 5.

    Returns
    -------
    X : np.array(m)
        Estimated vector of the controller activation weights.

    '''
    X = []
    gran = int(N/T)
    for t in range(gran):
        target_t = target_meshes[:,t*T:(t+1)*T]
        if t==0:
            X_t, _ = minimzation_holistic(np.zeros((m,T)), deltas, bs1, bs2, bs3, keys1, keys2, keys3, target_t, order, F,F_tilde,vector_add_e,vector_add_g,lmbd1, lmbd2, num_iter_max,num_iter_min)
        else:
            X_t, _ = minimzation_holistic(np.zeros((m,T)), deltas, bs1, bs2, bs3, keys1, keys2, keys3, target_t, order, F,F_tilde,vector_add_e,vector_add_g, lmbd1, lmbd2, num_iter_max,num_iter_min, past_known=True,v_ultimo=X_t[:,-1],v_penultimo=X_t[:,-2])            
        X.append(X_t)
    X = np.concatenate(X,1)
    return X