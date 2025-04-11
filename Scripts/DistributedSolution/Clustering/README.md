Here we consider two approaches:
1.) RSJD is the method proposed in S. Racković, C. Soares, D. Jakovetić, Z. Desnica and R. Ljubobratović, "Clustering of the Blendshape Facial Model," 2021 29th European Signal Processing Conference (EUSIPCO), Dublin, Ireland, 2021, pp. 1556-1560, doi: 10.23919/EUSIPCO54536.2021.9616061. 
2.) RSJDA is the adjustement of the above method, proposed in S. Racković, C. Soares, D. Jakovetić, "Distributed Solution of the Blendshape Rig Inversion Problem," SIGGRAPH Asia 2023 Technical Communications, Sydney, NSW, Australia, 2023, doi: 10.1145/3610543.3626166. 

Required modules:

```python
numpy
sklearn
```

```ClusteringRSJD.py``` - Contains the functions for clustering the blendshape face model based on RSJD

```ClusteringRSJDA.py``` - Contains the functions for clustering the blendshape face model based on RSJDA

```CompareClusterings.py``` - Performs the clustering for each of the two methods multiple times and for varying number of clusters, and plots the tradeoffs between the reconstruction error and density of the clustered blendshape matrix.The user should specify the number of repetitions and a list with desired numbers of clusters.

Set the desired hyperparameter values in the script ```CompareClusterings.py```:

```python
cluster_number_choice = [4,10,20,50,102]
number_of_repetitions = 1000
method1='RSJD'
method2='RSJDA'
```

Run the script 

```bash
python CompareClusterings.py
```

Based on the results, decide on the optimal number of clusters for each approach, that is to be used in the following scripts.

```CreateClustersRSJD.py``` - Creates and stores the set of face clusters based on RSJD. The user should specify a desired number of clusters.

Set the estimated optimal number of clusters in the script ```CreateClustersRSJD.py```

```python
number_of_clusters = 25
```

Run the script 

```bash
python CreateClustersRSJD.py
```

```CreateClustersRSJDA.py``` - Creates and stores the set of face clusters based on RSJDA. The user should specify a desired number of clusters.

Set the estimated optimal number of clusters in the script ```CreateClustersRSJDA.py```

```python
number_of_clusters = 25
```

Run the script 

```bash
python CreateClustersRSJDA.py
```

```ClusteredSetting.py``` - Contains funcitons for formating the data according to the available clusters.

```BlockCoordinateDescent.py``` - Contains functions for minimizing the wights usign the coordinate descent with respect to the available clusters.

```Execute.py``` - Executes the solver using the available clusters. You can skip the prior steps, and only execute this script, if you already know your optimal number of clusters.

Set the desired hyperparameter values:

```python
clustering_method = 'RSJDA' # the method used for clustering, either RSJD or RSJDA
number_of_clusters = 24     # the number of clusters produced by the specified method. This should be available in hte file names, in the folder ..\Data\Clusters
train_frames = 10           # this will take the first 'train_frames' from 'weights.npy' matrix as a training set
num_iter = 7                # the number of iterations of the CD solver
lmbd =  5                   # the regularization parameter of the objective funciton
```

Run the script 

```bash
python Execute.py
```


