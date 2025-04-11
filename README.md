This repository covers the work of several scientific papers in the field of blendshape animation of the human face, that together build my PhD thesis, titled "...".

## LMMM - Levenberg–Marquardt Majorization-Minimization-based Solution

THis chapter consists of two papers, solving the inverse rig problem on a base of Majorization-Minimization, in order to simplyfy originally complex, nonconvex objective. the corresponing scripts can be found in ../Scripts/LMMM repo.

#### A Majorization--Minimization-Based Method for Nonconvex Inverse Rig Problems in Facial Animation: Algorithm Derivation

https://link.springer.com/article/10.1007/s11590-023-02012-w

This paper gives a mathematical derivation of the algorithm, with convergence guarantees. 

```bibtex
@article{rackovic2024majorization,
  title={A majorization--minimization-based method for nonconvex inverse rig problems in facial animation: algorithm derivation},
  author={Rackovi{\'c}, Stevo and Soares, Cl{\'a}udia and Jakoveti{\'c}, Du{\v{s}}an and Desnica, Zoranka},
  journal={Optimization Letters},
  volume={18},
  number={2},
  pages={545--559},
  year={2024},
  publisher={Springer}
}
```

#### Accurate and Interpretable Solution of the Inverse Rig for Realistic Blendshape Models with Quadratic Corrective Terms

https://arxiv.org/abs/2302.04843

Complementing the previous paper, this one quantitativelly explores the results and benchmarks the algoritm with SOTA approaches over animated sequences. 

```bibtex
@article{rackovic2023accurate,
  title={Accurate and Interpretable Solution of the Inverse Rig for Realistic Blendshape Models with Quadratic Corrective Terms},
  author={Rackovi{\'c}, Stevo and Soares, Cl{\'a}udia and Jakoveti{\'c}, Du{\v{s}}an and Desnica, Zoranka},
  journal={arXiv preprint arXiv:2302.04843},
  year={2023}
}
```

## Distributed Solution 

This chapter includes two papers, and corresponding scripts are in ../Scripts/DistributedSolution repo.

 #### Clustering of the Blendshape Facial Model
 
 https://ieeexplore.ieee.org/abstract/document/9616061

 This paper clusters the blendashape face into a semantically meaningful segments, allowing for a distributed approach to solving the inverse rig problem. 
 
 ```bibtex
 @inproceedings{rackovic2021clustering,
  title={Clustering of the blendshape facial model},
  author={Rackovi{\'c}, Stevo and Soares, Cl{\'a}udia and Jakoveti{\'c}, Du{\v{s}}an and Desnica, Zoranka and Ljubobratovi{\'c}, Relja},
  booktitle={2021 29th European Signal Processing Conference (EUSIPCO)},
  pages={1556--1560},
  year={2021},
  organization={IEEE}
}
```

#### Distributed Solution of the Blendshape Rig Inversion Problem

https://dl.acm.org/doi/abs/10.1145/3610543.3626166

This is a further developement of the above paper, that incorporates the ADMM paradigm to improve the estimates of the overlapping segment components.

```bibtex
@incollection{rackovic2023distributed,
  title={Distributed Solution of the Blendshape Rig Inversion Problem},
  author={Rackovi{\'c}, Stevo and Soares, Cl{\'a}udia and Jakoveti{\'c}, Du{\v{s}}an},
  booktitle={SIGGRAPH Asia 2023 Technical Communications},
  pages={1--4},
  year={2023}
}
```
