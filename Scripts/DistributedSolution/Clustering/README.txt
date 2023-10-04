Here we consider two approaches:
1.) RSJD is the method proposed in S. Racković, C. Soares, D. Jakovetić, Z. Desnica and R. Ljubobratović, "Clustering of the Blendshape Facial Model," 2021 29th European Signal Processing Conference (EUSIPCO), Dublin, Ireland, 2021, pp. 1556-1560, doi: 10.23919/EUSIPCO54536.2021.9616061. 
2.) RSJDA is the adjustement of the above method, proposed in S. Racković, C. Soares, D. Jakovetić, "Distributed Solution of the Blendshape Rig Inversion Problem," SIGGRAPH Asia 2023 Technical Communications, Sydney, NSW, Australia, 2023, doi: 10.1145/3610543.3626166. 

        ClusteringRSJD.py - Contains the functions for clustering the blendshape face model based on RSJD
       ClusteringRSJDA.py - Contains the functions for clustering the blendshape face model based on RSJDA
    CompareClusterings.py - Performs the clustering for each of the two methods multiple times and for varying number of clusters, and plots the tradeoffs between the reconstruction error and density of the clustered blendshape matrix.The user should specify the number of repetitions and a list with desired numbers of clusters.
    CreateClustersRSJD.py - Creates and stores the set of face clusters based on RSJD. The user should specify a desired number of clusters.
   CreateClustersRSJDA.py - Creates and stores the set of face clusters based on RSJDA. The user should specify a desired number of clusters.
      ClusteredSetting.py - Contains funcitons for formating the data according to the available clusters.
BlockCoordinateDescent.py - Contains functions for minimizing the wights usign the coordinate descent with respect to the available clusters.
               Execute.py - Executes the solver using the available clusters