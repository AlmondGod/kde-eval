# File Map
[shell-adaptive]() contains the final new adaptive sampling algorithm (see [here](../c++-impls/adaptshell_kde.cpp) for the C++ implementation)
- `alg_projections_gauss.py` tests the accuracy of k-dimensional projections to estimate the kernel squared on the gaussian kernel
- `alg_projections.py` tests the accuracy of k-dimensional projections to estimate the kernel squared on the student-t kernel
- `shell_adaptive.py` is the final new adaptive shell algorithm
- `shell_alg.py` was v1 using predefined number of shells (50) for estimation

[hbe](hbe) contains the [hashing-based estimation](https://github.com/kexinrong/rehashing/tree/master/demo) algorithm written by Kexin Rong and Paris Simenilakis

[tsne](tsne) contains the python version of [t-Stochastic Neighbor Embeddings](https://github.com/lvdmaaten/bhtsne/) by Laurens van der Maaten and Geoffrey Hinton

[var-tests]() contains files used to evaluate the variances and kernels squared of the listed datasets to verify the feasibility of the new adaptive algorithm

[initial-tests]() contains:
- `optimal_T.py`: used to evaluate the true quantity $T = \frac{\sigma^2}{\epsilon^2 k^2}$ on various datasets
- `dataset download`: download listed datasets
- `jk.py`: evaluate the student and gaussian true kernel values on datasets to calculate the feasibility of the original T estimation method $T = \frac{1}{\delta \epsilon^2}$
- `kd_weights.py`: evaluate the entropy of datasets
- `random_sampling.py` and `matthew_random_sampling`: first adaptive random sampling algorithms

[param-tests]() containts files used to find the ideal number of dimensions k for dimension reduction (`shell_alg_ktest.py` and `testing_k`) and ideal method and number of shells to use on the KDTree to estimate the kernel squared (`testing_spheres`)

[taylor]() contains the attempted taylor-expansion variance estimation  algorithm


