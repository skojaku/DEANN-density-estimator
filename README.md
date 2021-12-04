# A Python package for a fast kernel density estimator using DEANN
[![Unit Test & Deploy](https://github.com/skojaku/DEANN-density-estimator/actions/workflows/main.yml/badge.svg)](https://github.com/skojaku/DEANN-density-estimator/actions/workflows/main.yml)

This package is an implementation of DEANN algorithm. Please cite the following paper if you use this package (and if you find this package useful, give me a star!):

```
Karppa, Matti, Martin Aumüller, and Rasmus Pagh. 
"DEANN: Speeding up Kernel-Density Estimation using Approximate Nearest Neighbor Search." 
arXiv preprint arXiv:2107.02736 (2021).
```

# Install

Requirements: Python 3.7 or later

Clone the repository, and execute the following command at the root directory:
```bash
pip install -e .
```

### For users without GPUs

Although the package is tested in multiple environments, it is still possible that you come across issues related to `faiss`, the most common problem being the one related to GPUs. If you don't have GPUs and get some troubles, try install faiss-cpu instead:

```bash
conda install -c conda-forge faiss-cpu
```

or *with pip*:
```
pip install faiss-cpu
```

### For users with GPUs

This package uses CPUs by default but if you have GPUs, congratulations! You can get a substantial speed up!! To enable the GPUs, specify `gpu_id` in the input argument.


# Example

```python
X = np.random.randn(10000, 30) # (number of samples, dimension)

model = deann.DEANN(k=20, m=30, metric="euclidean", bandwidth=None, exact=True, gpu_id=None)
model.fit(X) # Fit the kernel density estimator
log_density = model.log_density(X) # Log density
percentile = model.percentile(X) # Percentile
```
- `k`: Number of the nearest data points. A large `k` improves the accuracy at the expense of computation time.
- `m`: Number of randomly sampled data points. A large `m` improves the accuracy at the expense of computation time.
- `metric`: Distance metrics. Set `euclidean` or `cosine`.
- `bandwidth`: Bandwidth of the kernel function. Set `bandwidth=None` will set the bandwidth based on a common hueristics. See the code for detail.
- `exact`: `exact=False` will use approximated nearest neighbor algorithms.
- `gpu_id`: ID of the GPU device. If `gpu_id=None`, CPUs will be used.


