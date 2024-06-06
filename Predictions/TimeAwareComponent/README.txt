This folder contains the results of the TimeAwareComponent inverse rig solution, from the script ../TimeAwareComponent/ExecuteHolistic.py
The results are stored as a numpy array, where each row represents a predicted weight vector for a given frame of animaiton. The file name is structured as follows:
Pred_num_iter_max_{num_iter_max}_num_iter_min_{num_iter_min}_lmbd1_{lmbd1}_lmbd2_{lmbd2}_T_{T}.npy
where:
num_iter_max is the max number of iterations
num_iter_min is the min number of iterations
lmbd1 is the sparsity regularization parameter of the objective funciton
lmbd2 is the temporal smoothness regularization parameter of the objective funciton
T is the granularity of the animation interval split