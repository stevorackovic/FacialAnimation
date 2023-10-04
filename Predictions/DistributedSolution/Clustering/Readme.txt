This folder contains the results of the clustering-based inverse rig solution, from the script ../DistributedSolution/Clustering/Execute.py
The results are stored as a numpy array, where each row represents a predicted weight vector for a given frame of animaiton. The file name is structured as follows:
Pred_{clustering_method}_num_iter_{num_iter}_lmbd_{lmbd}.npy
where:
clustering_method is the method used for clustering, either RSJD or RSJDA
num_iter is the number of iterations of the CD solver
lmbd is the regularization parameter of the objective funciton