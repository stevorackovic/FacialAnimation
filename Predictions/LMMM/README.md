This folder contains the results of the LMMM inverse rig solution, from the script ../LMMM/Execute.py

The results are stored as a numpy array, where each row represents a predicted weight vector for a given frame of animaiton. The file name is structured as follows:

Pred_initialization_{initialization}_lmbd_{lmbd}.npy

where:

initialization is the initialization method ('CO' or '0')

lmbd is the regularization parameter of the objective funciton
