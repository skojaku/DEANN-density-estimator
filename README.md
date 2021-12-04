# A Python Package for a fast kernel density estimator using DEANN

This package is an implementation of DEANN algorithm. Please cite the following paper if you use this package (and if you find this package useful, give me a star!):

```
Karppa, Matti, Martin Aumüller, and Rasmus Pagh. "DEANN: Speeding up Kernel-Density Estimation using Approximate Nearest Neighbor Search." arXiv preprint arXiv:2107.02736 (2021).
```

# Install

```bash
pip install -e .
```

# Example

```python
X = np.random.randn(10000, 30)

model = deann.DEANN(k=20, m=30, metric="euclidean", bandwidth=None, exact=True, gpu_id=None)
model.fit(X) # Fit the kernel density estimator
log_density = model.log_density(X) # Log density
percentile = model.percentile(X) # Percentile
```
- `k`: Number of the nearest data points. A large `k` improves the accuracy at expense of computation time.
- `m`: Number of randomly sampled data points. A large `m` improves the accuracy at expense of computation time.
- `metric`: Distance metrics. Set `euclidean` or `cosine`.
- `bandwidth`: Bandwidth of the kernel function. Set `bandwidth=None` will set the bandwidth based on a common hueristics. See the code for detail.
- `exact`: `exact=False` will use approximated nearest neighbor algorithms.
- `gpu_id`: ID of the GPU device. If `gpu_id=None`, CPUs will be used.





