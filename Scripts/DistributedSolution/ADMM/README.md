Required modules:

```python
numpy
```

```ADMMXUpdate.py``` - Contains X-update function for the ADMM formulation

```ADMMFunctions.py``` - Contains ADMM functions for solving the inverse rig problem

```Execute.py``` - Executes the solver using the available clusters

Set the desired hyperparameter values in the script ```Execute.py```:

```python
clustering_method = 'RSJDA' # the method used for clustering, either RSJD or RSJDA
number_of_clusters = 24     # the number of clusters produced by the specified method. This should be available in hte file names, in the folder ..\Data\Clusters
train_frames = 10           # this will take the first 'train_frames' from 'weights.npy' matrix as a training set
outer_iter = 10             # the number of ADMM iterations
inner_iter = 1              # the number of iterations of the CD solver within one ADMM iteration
lmbd =  5                   # the regularization parameter of the objective funciton
rho =  50                   # the ADMM regularization parameter
```

Run the script 

```bash
python Execute.py
```



      
