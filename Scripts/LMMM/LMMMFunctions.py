# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 17:11:32 2023

@author: Stevo Rackovic

"""

import numpy as np
from HelperFunctions import quadratic_rig, objective_function
import numba
from numba import prange

@numba.jit(nopython=True, parallel=True)
def compute_dot_product(mx1, mx2, result, deltas, batch_size, n_batches):
    for i in prange(n_batches):
        start = i * batch_size
        end = start + batch_size
        result[start:end] = (mx2[start:end].dot(mx1) / 2) + deltas[start:end]
        
def precompute_one_hot_keys1(keys1, m):
    rows = np.arange(keys1.shape[0])
    one_hot_keys1 = np.zeros((m, keys1.shape[0]))
    one_hot_keys1[keys1[:, 0], rows] = 1
    return one_hot_keys1

def terms_and_coefficients(target_mesh,C,n,m,eig_max_D,eig_min_D,sigma_D,deltas,lmbd,bs1,keys1,one_hot_keys1,n_batches,batch_size):
    '''
    Computes coefficients for the upper bound polinomial to be minimized.

    Parameters
    ----------
    target_mesh : np.array(n)
        Ground-truth mesh vector.
    C : np.array(m)
        Vector of the controller activation weights.
    n : int
        Length of the mesh vector.
    m : int
        Number of blendshapes.
    eig_max_D : np.array(n)
        Max eigen values for correspodning rows of the blendshape matrix.
    eig_min_D : np.array(n)
        Min eigen values for correspodning rows of the blendshape matrix.
    sigma_D : np.array(n)
        Max singular values for correspodning rows of the blendshape matrix.
    deltas : np.array(n,m)
        Deltas blendshape matrix.
    lmbd : float
        Regularization aprameter. lambda>=0.
    bs1 : np.array(m1,n)
        Matrix containing (m1) corrective terms of the first level.
    keys1 : np.array(m1,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs1.
    one_hot_keys1 : TYPE
        DESCRIPTION.
    n_batches : int
        number of batches for the parallel computation of the dotproduct in 
        terms_and_coefficients.
    batch_size : int
        batch_size = n // n_batches.

    Returns
    -------
    coef1 : np.array(m)
        Vector of linear coefficients for the upper bound polinomial.
    coef2 : float
        Quadratic coefficient for the upper bound polinomial.
    coef4 : float
        Quartic coefficient for the upper bound polinomial.

    '''
    term_p = quadratic_rig(C,deltas,bs1,keys1) - target_mesh
    
    C_mapped = C[keys1[:, 1]]
    expanded_C = one_hot_keys1 * C_mapped
    term_q = np.zeros((n, m))
    compute_dot_product(expanded_C.T, bs1.T, term_q, deltas, batch_size, n_batches)
        
    coef1 = 2*term_p.dot(term_q) + lmbd
    term_r = np.zeros(n)
    term_r[term_p>0] += eig_max_D[term_p>0]
    term_r[term_p<0] += eig_min_D[term_p<0]
    coef2 = 2*(np.sum(term_p*term_r) + np.sum(term_q**2))
    coef4 = 2*m*np.sum(sigma_D**2)
    return coef1,coef2,coef4

def compute_increment(C,m,coef1,coef2,coef4):
    '''
    Takes coefficients for the polinomial, and then visit one controller 
    at a time, to find an increment that minimizes the upper bound.

    Parameters
    ----------
    C : np.array(m)
        Vector of the controller activation weights.
    m : int
        Number of blendshapes.
    coef1 : np.array(m)
        Vector of linear coefficients for the upper bound polinomial.
    coef2 : float
        Quadratic coefficient for the upper bound polinomial.
    coef4 : float
        Quartic coefficient for the upper bound polinomial.

    Returns
    -------
    increment : np.array(m)
        Optimal increment vector for the weigths vctor C.

    '''
    increment = np.zeros(m)
    for ctr in range(m):
        min_x = -C[ctr] # check the borders (of the feasible set) first
        min_y = coef4*(min_x**4) + coef2*(min_x**2) + coef1[ctr]*min_x
        candidate_x = 1-C[ctr]
        candidate_y = coef4*(candidate_x**4) + coef2*(candidate_x**2) + coef1[ctr]*candidate_x
        if candidate_y < min_y:
            min_y = candidate_y
            min_x = candidate_x
        # then we check potential extreme values.
        # If it is within the feasible set (-C, 1-C), I check if the value is lower than at the border.
        # Use a closed form solution
        term0 = 27*((4*coef4)**2)*coef1[ctr]
        term1 = (term0)**2 - 4*((-3*(4*coef4)*(2*coef2))**3)
        if term1 >= 0: # if this is negative, a root is complex, so we dismiss it 
            candidate_x = -1/(3*(4*coef4))*np.cbrt(.5*(term0+np.sqrt(term1))) - 1/(3*(4*coef4))*np.cbrt(.5*(term0-np.sqrt(term1)))
            if candidate_x > -C[ctr] and candidate_x < 1-C[ctr]:
                candidate_y = coef4*(candidate_x**4) + coef2*(candidate_x**2) + coef1[ctr]*candidate_x
                if candidate_y < min_y:
                    min_y = candidate_y
                    min_x = candidate_x               
        increment[ctr] += min_x
    return increment
    
def minimization(num_iter,C,deltas,target_mesh,eig_max_D,eig_min_D,sigma_D,lmbd,bs1,keys1,n_batches,tolerance=0.0005,accelerated=True):
    '''
    We use previously define functions to minimize the upper bound function.

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
    eig_max_D : np.array(n)
        Max eigen values for correspodning rows of the blendshape matrix.
    eig_min_D : np.array(n)
        Min eigen values for correspodning rows of the blendshape matrix.
    sigma_D : np.array(n)
        Max singular values for correspodning rows of the blendshape matrix.
    lmbd : float
        Regularization aprameter. lambda>=0.
    bs1 : np.array(m1,n)
        Matrix containing (m1) corrective terms of the first level.
    keys1 : np.array(m1,2) with int entries
        Each row corresponds to the indices of the blendshapes that invoke the 
        correspodning corrective term from bs1.
    n_batches : int
        number of batches for the parallel computation of the dotproduct in 
        terms_and_coefficients.

    Returns
    -------
    C : np.array(m)
        Vector of the controller activation weights.

    '''
    acceleration_factors = [50,20,10,5]
    cost_old = objective_function(C,deltas,bs1,keys1,target_mesh,lmbd)
    n,m = deltas.shape
    batch_size = n // n_batches
    one_hot_keys1 = precompute_one_hot_keys1(keys1, m)
    for i in range(num_iter):
        coef1,coef2,coef4 = terms_and_coefficients(target_mesh,C,n,m,eig_max_D,eig_min_D,sigma_D,deltas,lmbd,bs1,keys1,one_hot_keys1,n_batches,batch_size)
        increment = compute_increment(C,m,coef1,coef2,coef4)
        if accelerated:
            C0 = C + increment
            cost = objective_function(C0,deltas,bs1,keys1,target_mesh,lmbd)
            for factor in acceleration_factors:
                C1 = C + factor*increment
                C1[C1<0] = 0
                C1[C1>1] = 1
                cost1 = objective_function(C1,deltas,bs1,keys1,target_mesh,lmbd)
                if cost1 < cost:
                    increment = factor*increment
                    cost = cost1
                    break
            
        C += increment
        cost_new = objective_function(C,deltas,bs1,keys1,target_mesh,lmbd)
        if np.abs(cost_new-cost_old) <= tolerance:
            break
        cost_old = np.copy(cost_new)

    return C

