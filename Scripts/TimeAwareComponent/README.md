Required modules:

```python
numpy
scipy
```

Set the desired hyperparameter values in the script ```Execute.py```:

```python
train_frames = 10           # this will take the first 'train_frames' from 'weights.npy' matrix as a training set
num_iter_max = 10           # the maximum number of iterations of the CD solver
num_iter_min = 5            # the minimum number of iterations of the CD solver
lmbd1 =  1                  # the sparsity regularization parameter of the objective funciton
lmbd2 =  1                  # the temporal smoothness regularization parameter of the objective funciton
T = 10                      # Interval batch size
```

Run the script 

```bash
python Execute.py
```
