# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:52:23 2023

@author: Stevo Rackovic

"""

N = 100   # Set here the number of frames of your animaiton
n = 9000  # Set here the number of vertices (times 3) of your avatar. 
m = 60    # Put the number of your character blendhsapes
m1, m2, m3 = 50, 25, 10 # Set the number of corrective terms of first, second and third level, respectively

# -----------------------------------------------------------------------------

n = n//3*3 # To make sure it is divisible by 3
import os
import numpy as np
import random
work_dir = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = os.path.join(work_dir,'Data')

deltas = np.random.randn(n,m)
neutral = np.random.randn(n)
W = np.random.randn(N,m)
W[W<0]=0
W[W>1]=1

np.save(os.path.join(data_dir,'weights.npy'),W)
np.save(os.path.join(data_dir,'deltas.npy'),deltas)
np.save(os.path.join(data_dir,'neutral.npy'),neutral)

bs1, bs2, bs3 = np.random.randn(m1,n), np.random.randn(m2,n), np.random.randn(m3,n)
keys1, keys2, keys3 = np.array([sorted(random.sample(range(m), 2)) for _ in range(m1)]), np.array([sorted(random.sample(range(m), 3)) for _ in range(m2)]), np.array([sorted(random.sample(range(m), 4)) for _ in range(m3)])

np.save(os.path.join(data_dir,'bs1.npy'),bs1)
np.save(os.path.join(data_dir,'bs2.npy'),bs2)
np.save(os.path.join(data_dir,'bs3.npy'),bs3)
np.save(os.path.join(data_dir,'keys1.npy'),keys1)
np.save(os.path.join(data_dir,'keys2.npy'),keys2)
np.save(os.path.join(data_dir,'keys3.npy'),keys3)


