Required modules:

```python
numpy
numba
```

Set the desired hyperparameter values in the script ```Execute.py```:

```python
train_frames = 10           # this will take the first 'train_frames' from 'weights.npy' matrix as a training set
num_iter = 200              # the number of iterations of the CD solver
lmbd =  5                   # the regularization parameter of the objective funciton
tolerance = 0.0005          # stopping criteria (if the norm of the difference of the two consecutive vectors i s less than tolerance, break)
n_batches = 40              # for parallel computing within the function for computing the terms and coefficients of the surogate
initialization = 'CO'       # Initialization method. If 'CO', use the speudoinverse warm start of Cetinaslan aand Orvalho. If '0', use zero-initialization
```

Run the script 

```bash
python Execute.py
```
