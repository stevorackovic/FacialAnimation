# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:33:03 2023

@author: Stevo Rackovic

Holistic coordinate descent. This is based on the version of CD we used before, 
but adjusted so that it takes temporal smothness into consideration as well. 
With this in mind, instead of solving for a single weight vector, funcitons 
take as input a matrix of weights, covering an interval of T time frames.

This script uses scipy optimizer to solve teh constrained quadratic problem

"""

import numpy as np
from scipy.optimize import minimize

def quartic_rig(C, deltas, bs1, bs2, bs3, keys1, keys2, keys3):
    '''
    Computes a quartic rig for multiple time frames, given a weight matrix C.

    Parameters
    ----------
    C : np.array(m,T)
        Matrix of activation weigths, where m is the number of blendshapes, and 
        T is the number of time frames.
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
    np.array(n), resulting face mesh
    '''
    return deltas.dot(C) + bs1.T.dot(C[keys1].prod(1)) + bs2.T.dot(C[keys2].prod(1)) + bs3.T.dot(C[keys3].prod(1))

def root_squared_error(mesh1, mesh2, perc=100):
    '''
    Computes RSE between two face meshes. One can choose to see max error or 
    any percentile.

    Parameters
    ----------
    mesh1 : np.array(n)
        Predicted mesh vector, with n elements.
    mesh2 : np.array(n)
        Ground-truth mesh vector.
    perc : int, optional
        Percentile value of the error vector to return. 0<=percentile<=100. 
        The default is 100, i.e., gives only max error.

    Returns
    -------
    error : float
        Error.
    '''
    error = np.sqrt((mesh1[::3]-mesh2[::3])**2 + (mesh1[1::3]-mesh2[1::3])**2 + (mesh1[2::3]-mesh2[2::3])**2)
    error = np.percentile(error, perc)
    return error

def add_noise(n,sigma,p):
    '''
    Adds random noise to the mesh. It will visit each vertex, and with prob p 
    add random noise to each of its three coordiantes.

    Parameters
    ----------
    n : int
        Number of vertices.
    sigma : float
        Standard deviation for added noise.
    p : float, 0<=p<=1
        Percentage of corrupted vertices.

    Returns
    -------
    noise : np.array(3*n)
        Corrupted face mesh.
    '''
    n3 = int(n/3)
    noise = np.zeros(n)
    for i in range(n3):
        if np.random.rand()<=p:
            for j in range(3):
                noise[3*i+j] += (sigma**2)*np.random.rand()
    return noise

def coordinate_order(deltas):
    '''
    Gives order of the blendshapes based on their magnitude of deformation.

    Parameters
    ----------
    deltas : np.array(n,m)
        Deltas blendshape matrix.

    Returns
    -------
    order : np.array(m)
        Ordered blendshape indices.
    '''
    D = deltas[::3]**2 + deltas[1::3]**2 + deltas[2::3]**2 
    effect = D.mean(0)
    order = np.argsort(-effect)
    return order

def banded_matrix(T):
    F = np.eye(T)*6 + np.diag(np.full(T - 1, -4), k=1) + np.diag(np.full(T - 1, -4), k=-1) + np.diag(np.full(T - 2, 1), k=2) + np.diag(np.full(T - 2, 1), k=-2)
    F[0,0]=1
    F[-1,-1]=1
    F[1,1]=5
    F[-2,-2]=5
    F[0,1]=-2
    F[1,0]=-2
    F[-1,-2]=-2
    F[-2,-1]=-2
    F = F.astype(int)
    return F

def banded_matrix_add(F):
    F_tilde = np.copy(F)
    F_tilde[0,0]+=5
    F_tilde[1,1]+=1
    F_tilde[0,1]-=2
    F_tilde[1,0]-=2
    vector_add_e = np.zeros(len(F))
    vector_add_e[0] += 1
    vector_add_g = np.zeros(len(F))
    vector_add_g[0]-=4
    vector_add_g[1]+=1
    return F_tilde, vector_add_e, vector_add_g

def spearate_coefficients(C, ctr, deltas, bs1, bs2, bs3, keys1, keys2, keys3, target_mesh):
    '''
    Computes coefficients for the objective for a signle controller ctr. These 
    coefficients are vectors, as they correpsond to full time sequence.

    Parameters
    ----------
    C : np.array(m,T)
        Matrix of activation weigths, where m is the number of blendshapes, and 
        T is the number of time frames.
    ctr : int
        Index of the observed blendshape.
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
    target_mesh : np.array(n,T)
        Ground-truth mesh matrix.

    Returns
    -------
    Phi : np.array(T)
        Quadratic coefficient of the data fidelity term (this is a vector, but 
        the actual term will be diagonal matrix with Phi on its diagonal).
    Theta : np.array(T)
        Linear coefficient of the data fidelity term.
    '''

    # first the corrective terms where ctr is included
    C_ctr_1 = 0. + C
    C_ctr_1[ctr] = 1.
    indices1 = np.where(keys1==ctr)
    coef1 = np.dot(C[keys1[indices1[0],1-indices1[1]]].T, bs1[indices1[0]]) + deltas[:,ctr]
    indices2 = np.where(keys2==ctr)[0] # second level
    coef1 += np.dot(C_ctr_1[keys2[indices2]].prod(1).T, bs2[indices2])
    indices3 = np.where(keys3==ctr)[0] # third level
    coef1 += np.dot(C_ctr_1[keys3[indices3]].prod(1).T, bs3[indices3])
    # then the corrective cases where ctr is not included
    C_no_ctr = 0. + C
    C_no_ctr[ctr] = 0.
    coef2 = quartic_rig(C_no_ctr,deltas,bs1,bs2,bs3,keys1,keys2,keys3)
    coef2 -= target_mesh
    
    Phi = np.linalg.norm(coef1,axis=1)**2/len(deltas)                          # i just added /len(deltas) here and in teh next line, to scale the fidelity term proportional to the number of vertices
    Theta = np.sum(coef1*coef2.T,1)/len(deltas)
    del coef1
    del coef2
    return Phi, Theta

def objective_function(x,Phi,F,Theta,lmbd1,lmbd2):
    '''
    Minimiztion objective for a single controller.

    Parameters
    ----------
    x : variable vector
    Phi : np.array(T)
        Quadratic coefficient of the data fidelity term (this is a vector, but 
        the actual term will be diagonal matrix with Phi on its diagonal).
    F : np.array(T,T)
        Banded matrix.
    Theta : np.array(T)
        Linear coefficient of the data fidelity term.
    lmbd1 : float, lmbd1>=0
        Coefficient for regularizing sparsity.
    lmbd2 : float, lmbd2>=0
        Coefficient for regularizing temporal smoothness.

    Returns
    -------
    Objetive function
    '''
    return x.T @ (np.diag(Phi) + lmbd2 * F) @ x + x@(2*Theta+lmbd1)

def objective_function_add(x,Phi,F_tilde,Theta,lmbd1,lmbd2,v_ultimo_ctr,v_penultimo_ctr,vector_add_e,vector_add_g):
    ''' Similar to objective_function, but assuming we know those two last timeframes '''
    return x.T @ (np.diag(Phi) + lmbd2 * F_tilde) @ x + x@(2*Theta+lmbd1 + 2*lmbd2*v_penultimo_ctr*vector_add_e+2*lmbd2*v_ultimo_ctr*vector_add_g)


def minimize_i(C, ctr, deltas, bs1, bs2, bs3, keys1, keys2, keys3, target_mesh, lmbd1, lmbd2, F, x0):
    '''
    Minimizes the objective for a single controller (over T time frames).

    Parameters
    ----------
    C : np.array(m,T)
        Matrix of activation weigths, where m is the number of blendshapes, and 
        T is the number of time frames.
    ctr : int
        Index of the observed blendshape.
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
    target_mesh : np.array(n,T)
        Ground-truth mesh matrix.
    lmbd1 : float, lmbd1>=0
        Coefficient for regularizing sparsity.
    lmbd2 : float, lmbd2>=0
        Coefficient for regularizing temporal smoothness.
    F : np.array(T,T)
        Banded matrix.
    x0 : np.array(T)
        Initial weights vector (of a controller ctr over T time frames).

    Returns
    -------
    result : np.array(T)
        Estimated weights vector (of a controller ctr over T time frames).
    '''
    Phi, Theta = spearate_coefficients(C, ctr, deltas, bs1, bs2, bs3, keys1, keys2, keys3, target_mesh)
    bounds = [(0, 1)] * len(F) 
    result = minimize(lambda x:objective_function(x,Phi,F,Theta,lmbd1,lmbd2), x0, bounds=bounds).x
    #result[result<0]=0
    #result[result>1]=1
    return result

def minimize_i_add(C, ctr, deltas, bs1, bs2, bs3, keys1, keys2, keys3, target_mesh, lmbd1, lmbd2, F_tilde, x0, v_ultimo_ctr, v_penultimo_ctr, vector_add_e, vector_add_g):
    '''
    Similar to minimize_i, but assumes we already know values for the previous time interval
    
    v_ultimo_ctr : float
        The weigth of the controller ctr, estimated for the last frame of teh previous interval
    v_penultimo_ctr : float
        The weigth of the controller ctr, estimated for the second to last frame of the previous interval
    '''
    Phi, Theta = spearate_coefficients(C, ctr, deltas, bs1, bs2, bs3, keys1, keys2, keys3, target_mesh)
    bounds = [(0, 1)] * len(F_tilde) 
    result = minimize(lambda x:objective_function_add(x,Phi,F_tilde,Theta,lmbd1,lmbd2,v_ultimo_ctr,v_penultimo_ctr,vector_add_e,vector_add_g), x0, bounds=bounds).x
    #result[result<0]=0
    #result[result>1]=1
    return result

def minimization(C, deltas, bs1, bs2, bs3, keys1, keys2, keys3, target_mesh, order, F, F_tilde, vector_add_e, vector_add_g, lmbd1=0., lmbd2=0., num_iter_max=5, num_iter_min=1, past_known=False, v_ultimo=np.empty(0),v_penultimo=np.empty(0), return_cost=False):
    '''
    Complete minimization funciton. Does num_iter iterations, and in each 
    iteration visits all the controllers in a specified oreder.

    Parameters
    ----------
    C : np.array(m,T)
        Initial matrix of activation weigths, where m is the number of 
        blendshapes, and T is the number of time frames.
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
    target_mesh : np.array(n,T)
        Ground-truth mesh matrix.
    order : np.array(m)
        Ordered blendshape indices.
    F : np.array(T,T)
        Banded matrix.
    lmbd1 : float, lmbd1>=0
        Coefficient for regularizing sparsity.The default is 0..
    lmbd2 : float, lmbd2>=0
        Coefficient for regularizing temporal smoothness. The default is 0..
    num_iter : int, optional
        Number of iterations. The default is 1.

    Returns
    -------
    C : np.array(m,T)
        Estimated matrix of activation weigths.
    '''
    Cost_list = []
    lmbd1 /= len(C)
    cost_old = np.inf
    for i in range(num_iter_max):
        C_old = np.copy(C)
        for ctr in order:
            x0 = np.copy(C[ctr])
            if past_known:
                x = minimize_i_add(C, ctr, deltas, bs1, bs2, bs3, keys1, keys2, keys3, target_mesh, lmbd1, lmbd2, F_tilde, x0, v_ultimo[ctr], v_penultimo[ctr], vector_add_e, vector_add_g)
            else:
                x = minimize_i(C, ctr, deltas, bs1, bs2, bs3, keys1, keys2, keys3, target_mesh, lmbd1, lmbd2, F, x0)
            C[ctr] = np.copy(x)
        
        cost = np.sum((quartic_rig(C, deltas, bs1, bs2, bs3, keys1, keys2, keys3)-target_mesh)**2) + lmbd1*np.sum(C) + lmbd2*np.sum((C[:,:-2]+C[:,2:]-2*C[:,1:-1])**2)
        if i >= num_iter_min:
            if cost >= cost_old:
                C = np.copy(C_old)
                break
        if return_cost:
            Cost_list.append(cost)
        cost_old = 0.+cost
    return C, Cost_list