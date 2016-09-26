# ACC-2017
Code to generate figures from the paper "Symmetry reduction for dynamic programming with application to MRI" by John Maidens, Axel Barrau, Silvère Bonnabel, and Murat Arcak, submitted to the 2017 American Control Conference

Two Jupyter notebooks are included. The notebook DP_MR_fingerprinting.ipynb computes the optimal control input using dynamic programming on a six-dimensional grid (in the standard coordinates). The notebook DP_MR_fingerprinting_reduced.ipynb exploits symmetry to compute the same optimal control using dynamic programming on a reduced five-dimensional grid. The files DynamicProgramming.jl and DynamicProgrammingReduced.jl contain Julia code that implements the dynamic programming algorithm in each case. In addition, the data file J_full.jld contains the optimal cost-to-go function evaluated at all the grid points for the six-dimensional case (since it takes a few hours to run). 

Tested in Julia 0.4.0 with the following packages: 
- Interpolations version 0.3.5
- HDF5 version 0.6.1
- JLD version 0.6.0
- Gadfly version 0.4.2
- Colors version 0.5.4