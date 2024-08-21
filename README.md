Written by Anand Majmudar and Matthew Tsui, advsied by Professor Erik Waingarten during the University of Pennsylvania PURM program.

# Installation (`macos`)

```bash
git clone https://github.com/AlmondGod/kde-eval
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

see [src/README.md](src/README.md) for a filemap.



# Shell-Adaptive Random Sampling

## Introduction
Estimation of sums (or averages), linear time algorithms, worst-case complexity, and (1+ε)-approximations. We'll conclude with one of the main questions: when can one use random sampling to obtain practical, worst-case complexity results in the context of KDE?

### Problem
In kernel density computing, we receive:
- Dataset of P vectors P₁, P₂, ..., Pₙ ∈ ℝᵈ
- A kernel function k : ℝᵈ × ℝᵈ → ℝ

Given two vectors, the kernel function outputs a scalar value based on the L2 norm between the two vectors. The Gaussian and Student-t kernels are two commonly used kernel functions:

- Gaussian Kernel: k(p,q) = e^(-‖p-q‖²₂/2)
- Student-t kernel: k(p,q) = 1 / (‖p-q‖²₂ + 1)

Our goal is to output the average value over the dataset of the kernel function between the query and each datapoint. Specifically, given a query q ∈ ℝᵈ, we want to output: (1/|P|) ∑(p∈P) k(p,q).

### Naive Method
1. Store the dataset in a matrix.
2. Iterate through the entire dataset, evaluating the kernel function of the query and each datapoint and keeping a running average until we cover the entire dataset.

- Space complexity: N*d, where N is the number of vectors, and d is the number of dimensions.
- Time complexity: O(N), assuming that the time to evaluate k(p,q) for fixed p, q is constant.

When a kernel density is used as a subprocess for another algorithm, this O(N) time complexity becomes taxing. How can we make kernel density computation more time-efficient?

### Approximations
We now define Kernel Density Estimation (KDE), for which our goal is to estimate the kernel density rather than compute it directly. We have two extra constraints:

1. Our kernel density estimate must be inside the range specified by (1 ± ε) multiplied by the true kernel density, where Epsilon (ε) is a new problem parameter.
2. The probability that our kernel density estimate is outside the range specified by epsilon must be less than δ, where Delta (δ) is a new problem parameter.

Specifically, we want:
Pr[(1-ε)(1/|P|)∑(p∈P)k(p,q) ≤ ζ ≤ (1+ε)(1/|P|)∑(p∈P)k(p,q)] ≥ 1 - δ, where ζ is our estimator's algorithm.

### Random Sampling
Random sampling involves:
1. Sample T random points, p_{i₁}, p_{i₂}, ..., p_{iₜ} ∈ P where I = i₁, i₂, ..., iₜ ~ [n].
2. Output (1/T) ∑(i∈I) k(pᵢ, q).

The question is: how many samples do we need?

Consider the distribution of the variance of our algorithm:
E[(ζ - E[ζ])²] = (1/T²) ∑(l=1 to T) E_{iₗ}[(k(q,p_{iₗ}) - E[k(q, p_{iₗ})])²] ≤ 1/T

By Markov's inequality:
Pr[|ζ - μ| ≥ α] ≤ 1/(T·α²)

For failure probability, set α = ε, and then:
T = 1/(μ·ε²) → Pr[|ζ - μ| ≥ ε] ≤ δ, as needed

### When RS Excels
Random sampling is potentially highly efficient when our variance is low. The problem to be solved, however, is how to notify our random sampling algorithm when variance is low (included in our estimate for number of samples, so our indirect problem is computing the number of samples needed as described above).

## First Adaptive Sampling
We take one sample at a time, maintain a potential output, ζ, and keep a threshold value that we update every time we obtain a sample.

To prevent outputting incorrect values, we hope that ζ ≤ 100μ. Setting the threshold for the number of samples to be 1/(ε²δ(ζ/100)), we know that the failure probability will be ≤ δ. That is T = 1/(δε²) → Pr[|ζ - μ| ≥ ε] ≤ δ, as needed.

To further reduce the number of samples needed, we can take the median of some k iterations of this algorithm. Specifically, note that for the median to fail, it must be that more than half of the k iterations fail. So, this probability is (k choose k/2) · (1/10)^(k/2) = (2/√10)^(k/2) ≤ (2/3)^k. Finally, setting k = log_{2/3}δ, we have that the failure rate is ≤ δ. This means we need a total of (10/(ε²μ)) · log_{2/3}δ samples.

However, this also was not great empirically, because sometimes the estimates μ are quite small, making it impractical. We needed a better way to estimate our required number of samples.

## Toward Practical Algorithms...
We use a new formula for T which factors in variance of the kernel value in relation to the dataset and query. The number of samples T needed to guarantee our result is within $(1 \pm \epsilon)$ multiplicative error from true kernel value k is $T = \frac{\sigma^2}{\epsilon^2 k^2}$ where $\sigma^2 =$ variance.

To verify this idea would work, we verified the query-dataset variances and kernels squared on the following datasets:

covtype, HIGGS, HT_Sensor, mnist2500, poker hand training, Sensorless drive diagnosis, shuttle, skin nonskin, SUSY, Color Histogram

And found that in certain datasets, a significant number of queries had variances/kernels squared on the order of 1e-3, which would allow for efficient random samling given an efficient enough variance estimation method.

We can still use the adaptive algorithm with respect to k, but now we have a new problem: we don't know true variance $\sigma^2$. How do we estimate it? We will do so by an estimation of the kernel squared, which will always upper bound true variance since true variance is $E[k^2] - E[k]^2$. Which is the most efficient method to estimate this kernel squared? It seems we're back at our original KDE problem (except our kernel is now our kernel squared).


## Second Adaptive Sampling Procedure:
#### Kernel Squared Estimation via dimension reduction and KDTrees
We experimented with different variance estimation techniques (hashing, taylor) and arrived at our final dimension-reduction kdtree estimation algorithm below.

During our preprocessing phase, we initialize 1 gaussian vector (with random entries sampled from the gaussian distribution). We then take the dot product of this gaussian vector with every datapoint vector, and use the results as our 1-dimensional dataset. We then place each of 1-dimension reduced datapoints into our KDTree, a data structure commonly used to find the number of neighbors within a radius of a datapoint. (If one wants to do a 2+-dimensional projection, simply initialize k gaussian vectors and use the dot product of each data vector with the kth gaussian divided by $\sqrt{k}$ as the kth entry in the projected vector).

When we recieve the query, we also take its dot product with the gaussian vector. Then, doubling our current distance every time, we find the number of datapoints within this current distance radius from our query point (using the kdtree we can do this is $O(log n)$ time). We multiply the kernel squared value of this distance by the number of newly discovered points (within this current distance radius but not in any of the smaller radii), and we add this value to the running variance estimate total. We end once our total number of points covered within the current distance radius is the size of the dataset.

Now, using this variance estimate, we run the adaptive sampling algorithm using our new formula for number of samples $T = \frac{\sigma^2}{\epsilon^2 k^2}$.

## Informal Results
- The often defeated hashing-based estimation in both time to execute and accuracy on certain datasets, although it also was defeated by a large gap in both time and accuracy on other datasets
- This characteristic of the algorithm was suspected from the start: it can only do better than regular random sampling when the query-dataset variance on the kernel is low enough to allow for significant speedup, which for some datasets is simply not the case
- There were some subtleties in proving that the algorithm will always overestimate the kernel squared to overestimate the variance and ensure our number of samples ensures that our final estimate is within the bounds defined by $\epsilon$ and $\delta$ as described above
- One needed to be careful in upper bounding how much the variance changes after a projection (although we saw empirically that it almost always increases).

## Other Approaches
Other approaches for estimating upper bounds on the variance were attempted and were discovered to be unideal:

Tried to use the taylor approximation of the variance to separate out and precompute certain terms towards efficient variance estimation. However, the bound of this method was not tight eough (often 100x off)
