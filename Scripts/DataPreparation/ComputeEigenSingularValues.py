# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:52:23 2023

@author: Stevo Rackovic

"""

import os
import numpy as np
from scipy.linalg import eigh
work_dir = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = os.path.join(work_dir,'Data')

def H_initialization(vtx,bs1,keys1):
    H = np.zeros((m,m))
    for i in range(len(keys1)):
        tpl = keys1[i]
        H[tpl[0],tpl[1]] = bs1[i][vtx]/2
        H[tpl[1],tpl[0]] = bs1[i][vtx]/2
    return H

neutral = np.load(os.path.join(data_dir,'neutral.npy'))
deltas = np.load(os.path.join(data_dir,'deltas.npy'))
weights = np.load(os.path.join(data_dir,'weights.npy'))
bs1 = np.load(os.path.join(data_dir,'bs1.npy'))
keys1 = np.load(os.path.join(data_dir,'keys1.npy'))
n,m = deltas.shape

D_chs = np.zeros(((n,m,m)))
for vtx in range(n):
    D = H_initialization(vtx,bs1,keys1)
    D_chs[vtx] += D
    
lambdas_high = np.zeros(n)
lambdas_low = np.zeros(n)
for i in range(n):
    egvs = eigh(D_chs[i],eigvals=(0,m-1),eigvals_only =True)
    lambdas_high[i] += egvs[-1]
    lambdas_low[i] += egvs[0]
np.save(os.path.join(data_dir,'eigen_max.npy'),lambdas_high)
np.save(os.path.join(data_dir,'eigen_min.npy'),lambdas_low)

sigmas_D = np.zeros(n)
for i in range(n):
    _, s, _ = np.linalg.svd(D_chs[i])
    sigmas_D[i] += s[0]
np.save(os.path.join(data_dir,'singular.npy'),sigmas_D)

