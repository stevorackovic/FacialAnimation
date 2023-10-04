# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:42:11 2023

@author: Stevo Rackovic

"""

import numpy as np
from ADMMXUpdate import minimization

def z_update(x_list, u_list, idx_list, denominator, lmbd_list, rho_list):
    '''
    Z-update of the ADMM formulation
    '''
    w = np.zeros(len(denominator))
    for i in range(len(x_list)):
        w[idx_list[i]] += x_list[i] + u_list[i] - (lmbd_list[i]/rho_list[i])
    w /= denominator
    return w

def u_update(U, X, Z):
    '''
    U-update of the ADMM formulation
    '''
    return X + U - Z

def ADMM(C, deltas_list, ctr_list, coord_list, target_mesh, order_list, lmbd_list, m_list, bs1_list, bs2_list, bs3_list, keys1, keys2, keys3, rho_list, denominator, outer_iter=10, inner_iter=1):
    '''
    Performs minimization by applying the ADMM formulation over face clusters. 

    Parameters
    ----------
    C : np.array(m)
        Vector of the initial controller activation weights, where m is the number of 
        blendshapes.
    deltas_list : list
        Each element of the list is a submatrix of deltas, that correspond to 
        only the rows assigned to that cluster.
    ctr_list : list
        Each entry of the list will be a vector contatining the indices of the 
        blendshapes that correpsond to the considered cluster.
    coord_list : list
        Each entry of the list will be a vector contatining the indices of the 
        vertex-coordinates that correpsond to the considered cluster.
    target_mesh : np.array(n)
        Ground-truth mesh vector.
    order_list : list
        Each element of the list is a vector of indices of the belndshapes, 
        assigned to the corresponding face cluster, ordered by their respective 
        magnitude of deformation.
    lmbd_list : list
        List of regularization aprameters corresponding to each face cluster.
    m_list : list
        Number of blendshapes in each cluster.
    bs1_list : list
        Each element of the list is a submatrix of bs1, that correspond to 
        only the rows assigned to that cluster.
    bs2_list : list
        Each element of the list is a submatrix of bs2, that correspond to 
        only the rows assigned to that cluster.
    bs3_list : list
        Each element of the list is a submatrix of bs3, that correspond to 
        only the rows assigned to that cluster.
    keys1 : np.array(m1,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs1.
    keys2 : np.array(m2,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs2.
    keys3 : np.array(m3,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs3.
    rho_list : list
        List of ADMM regularization aprameters.
    denominator : np.array(m)
        A vector contating teh multiplicity of blendshapes over each cluster.
    outer_iter : int, optional
        Number of iterations of the outter loop. The default is 10.
    inner_iter : int, optional
        Number of iterations of the inner loop. The default is 1.

    Returns
    -------
    Z : np.array(m)
        Vector of the estimated controller activation weights.

    '''
    num_clusters = len(ctr_list)
    Z = 0. + C
    u_list = [np.zeros(m_list[clstr]) for clstr in range(num_clusters)]
    for _ in range(outer_iter):
        x_list = []
        for clstr in range(num_clusters): 
            # This part can be implemented in parallel, where each machine would then communicate the values x_c,u_c i z_c
            X_c = 0. + Z[ctr_list[clstr]]
            Z_c = 0. + Z[ctr_list[clstr]] 
            U_c = 0. + u_list[clstr]
            
            target_mesh_c = target_mesh[coord_list[clstr]]
            deltas_c = deltas_list[clstr]
            
            bs1_c   = bs1_list[clstr]
            bs2_c   = bs2_list[clstr]
            bs3_c   = bs3_list[clstr]
            rho_c   = rho_list[clstr]
            order_c = order_list[clstr]
            
            U_c = u_update(U_c,X_c,Z_c)
            X_c = minimization(Z, U_c, ctr_list[clstr], deltas_c, target_mesh_c, order_c, rho_c, bs1_c, bs2_c, bs3_c, keys1, keys2, keys3, inner_iter)
            u_list[clstr]= 0. + U_c
            x_list.append(0. + X_c)
        Z = z_update(x_list,u_list,ctr_list,denominator,lmbd_list,rho_list)
    Z[Z<0] = 0
    Z[Z>1] = 1 
    return Z
