This folder contains the results of the ADMM-based inverse rig solution, from the script ../DistributedSolution/ADMM/Execute.py
The results are stored as a numpy array, where each row represents a predicted weight vector for a given frame of animaiton. The file name is structured as follows:
Pred_{clustering_method}_outer_iter_{outer_iter}_inner_iter_{inner_iter}_lmbd_{lmbd}_rho_{rho}.npy
where:
clustering_method is the method used for clustering, either RSJD or RSJDA
outer_iter is the number of iterations of the ADMM solver
inner_iter is the number of iterations of the CD solver within each ADMM iteration
lmbd is the regularization parameter of the objective funciton
rho is the ADMM regularization parameter