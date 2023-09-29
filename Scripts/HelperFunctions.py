# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:07:51 2023

@author: Stevo Rackovic
"""

import os
import numpy as np

def quartic_rig(C, deltas, bs1, bs2, bs3, keys1, keys2, keys3):
    '''
    Computes a quartic rig given a weight vector C.

    Parameters
    ----------
    C : np.array(m)
        Vector of the controller activation weights, where m is the number of 
        blendshapes.
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

    '''
    return deltas.dot(C) + bs1.T.dot(C[keys1].prod(1)) + bs2.T.dot(C[keys2].prod(1)) + bs3.T.dot(C[keys3].prod(1))

def quadratic_rig(C,deltas,bs1,keys1):
    '''
    Computes a quadratic rig (approximation) given a weight vector C.

    Parameters
    ----------
    C : np.array(m)
        Vector of the controller activation weights, where m is the number of 
        blendshapes.
    deltas : np.array(n,m)
        Deltas blendshape matrix.
    bs1 : np.array(m1,n)
        Matrix containing (m1) corrective terms of the first level.
    keys1 : np.array(m1,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs1.

    '''
    return deltas.dot(C) + bs1.T.dot(C[keys1[:,0]]*C[keys1[:,1]])

def objective_function(C,deltas,bs1,keys1,target_mesh,lmbd):
    '''
    Gives a value of the objective cost. 

    Parameters
    ----------
    C : np.array(m)
        Vector of the controller activation weights, where m is the number of 
        blendshapes.
    deltas : np.array(n,m)
        Deltas blendshape matrix.
    bs1 : np.array(m1,n)
        Matrix containing (m1) corrective terms of the first level.
    keys1 : np.array(m1,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs1.
    target_mesh : np.array(n)
        Ground-truth mesh vector.
    lmbd : float
        Regularization aprameter. lambda>=0.

    '''
    pred_mesh = quadratic_rig(C,deltas,bs1,keys1)
    return np.linalg.norm(pred_mesh-target_mesh)**2 + lmbd*np.sum(C)

# -----------------------------------------------------------------------------
# Functions for evaluating the results

def rmse(mesh1,mesh2,percentile=95):
    '''
    Computes root squared error between predicted and ground-truth meshes, and 
    returns mean, max, and percentile value of the obtained vector.

    Parameters
    ----------
    mesh1 : np.array(n)
        Predicted mesh vector, with n elements.
    mesh2 : np.array(n)
        Ground-truth mesh vector.
    percentile : int, optional
        Percentile value of the error vector to return. 0<=percentile<=100. 
        The default is 95.

    '''
    error = (mesh1[::3]-mesh2[::3])**2 + (mesh1[1::3]-mesh2[1::3])**2 + (mesh1[2::3]-mesh2[2::3])**2
    error = np.sqrt(error)
    return np.mean(error), np.max(error), np.percentile(error,percentile)

def error_cardinality(path, file_name, target_meshes, deltas, bs1, bs2, bs3, keys1, keys2, keys3):
    '''
    Computes the evaluation metrics; inn specific mesh error (mean and max), 
    cardinality (i.e., L0 norm) of the weight vector, L1 norm of the weight 
    vector as well as roughness penalty per blendhspape (inversely proportional 
    to temporal smoothness).

    Parameters
    ----------
    path : str
        Path to the predictions diectory.
    file_name : str
        Name of the predictions file.
    target_meshes : np.array(N,m)
        matrix contatining predicted weights, where N is the numebr of frames 
        and m is the nuber of blendshapes.
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

    Returns
    -------
    Error_mean : float
        Average mesh error.
    Error_max : float
        Maximal mesh error.
    Cardinality : float
        Cardinality of the predicted weight vector (L0 norm).
    L1norm : float
        L1 norm of the predicted weigth vector.

    '''
    Ypred = np.load(os.path.join(path, file_name))
    Ypred[Ypred<0]=0
    Ypred[Ypred>1]=1
    Xpred = np.array([quartic_rig(Ypred[frame], deltas, bs1, bs2, bs3, keys1, keys2, keys3) for frame in range(len(Ypred))])
    err0 = [rmse(Xpred[frame],target_meshes[frame])[0] for frame in range(len(target_meshes))]
    err1 = [rmse(Xpred[frame],target_meshes[frame])[1] for frame in range(len(target_meshes))]
    Error_mean = np.mean(err0)
    Error_max = np.mean(err1)
    crd = [np.sum(Ypred[frame]>0) for frame in range(len(Ypred))]
    Cardinality = np.mean(crd)
    nrm = [np.linalg.norm(Ypred[frame],1) for frame in range(len(Ypred))]
    L1norm = np.mean(nrm)
    return Error_mean, Error_max, Cardinality, L1norm

# -----------------------------------------------------------------------------
# Functions for SSKLN:

def optimal_w(s,b):
    w = s.dot(b)/(b.dot(b))
    if w<=1 and w>=0:
        return w
    else:
        obj_0, obj_1 = np.linalg.norm(s), np.linalg.norm(s-b)
        if obj_1 < obj_0:
            return 1
        else:
            return 0
        
def optimal_C(deltas,msh,order,m):
    C = np.zeros(m)
    for i in order:
        b = deltas[:,i]
        C[i] += optimal_w(msh,b)
        msh = msh - C[i]*b
    return C

def ctr_order(deltas):
    offset = deltas[::3]**2 + deltas[1::3]**2 + deltas[2::3]**2
    offset = np.sum(offset,0)
    order  = np.argsort(-offset)
    return order

# -----------------------------------------------------------------------------
# Functions for clustering

def compute_error_density(vrtcs_list,ctr_list,coord_list,deltas):
    num_clusters = len(vrtcs_list)
    n,m = deltas.shape
    Error        = 0.
    Information  = 0.
    Density      = 0.
    AssignmentMatrix = np.zeros((num_clusters,m))
    for clstr in range(num_clusters):
        ctr_clst = ctr_list[clstr]
        ctr_clst_cmp = [i for i in range(m) if i not in ctr_clst]
        vtx_clstr = vrtcs_list[clstr]
        coord_clstr = coord_list[clstr]
        dlt_clstr = deltas[coord_clstr][:,ctr_clst] # - Submatrix correspodning to the cluster 'clstr'
        dlt_clstr_cmp = deltas[coord_clstr][:,ctr_clst_cmp] # - A discarded submatrix of the cluster 'clstr'
        info = np.sum(dlt_clstr**2)
        err  = np.sum(dlt_clstr_cmp**2)
        AssignmentMatrix[clstr][ctr_clst] += 1
        Error += err
        Information += info
        dns = len(vtx_clstr) * len(ctr_clst)
        Density += dns
    ReconstructionError = Error/Information
    return ReconstructionError, Density, AssignmentMatrix.T, Error
    
def compute_density(vrtcs_list,AssignmentMatrix):
    m,num_clusters = AssignmentMatrix.shape
    InterDensity = 0
    for ctr in range(m):
        Mrow = AssignmentMatrix[ctr]
        if np.sum(Mrow) > 1:
            for clstr in range(num_clusters):
                InterDensity += len(vrtcs_list[clstr])*Mrow[clstr]
    return InterDensity
        
