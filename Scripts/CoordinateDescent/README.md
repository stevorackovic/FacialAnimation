Required modules:

```python
numpy
```

Set the desired hyperparameter values in the script ```Execute.py```:

```python
train_frames = 10           # this will take the first 'train_frames' from 'weights.npy' matrix as a training set
num_iter = 7                # the number of iterations of the CD solver
lmbd =  5                   # the regularization parameter of the objective funciton
```

Run the script 

```bash
python Execute.py
```
