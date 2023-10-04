This folder contains the results of the CD-based inverse rig solution, from the script ../CoordinateDescent/Execute.py
The results are stored as a numpy array, where each row represents a predicted weight vector for a given frame of animaiton. The file name is structured as follows:
Pred_num_iter_{num_iter}_lmbd_{lmbd}.npy
where:
num_iter is the number of iterations of the algorithm
lmbd is the regularization parameter of the objective funciton