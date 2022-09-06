"""k-nearest neighbor"""
import numpy as np
from scipy import sparse
import faiss
import numba
from tqdm import tqdm
from .adam import ADAM
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse.csgraph import connected_components


class DEANN:
    def __init__(
        self, k=20, m=10, metric="cosine", bandwidth=None, exact=False, gpu_id=None
    ):
        self.k = k
        self.m = m
        self.metric = metric
        self.gpu_id = gpu_id
        self.exact = exact
        self.bandwidth = bandwidth

        self.precomputed_density = None

    # fit knn model
    def fit(self, X):
        """Fit the model using X as training data and Y as target values

        :param X: training data
        :type X: numpy.ndarray
        :param Y: target values
        :type Y: numpy.ndarray
        :return: self
        :rtype: object
        """
        # make knn graph
        X = self._homogenize(X)
        self.n_indexed_samples = X.shape[0]
        self._make_faiss_index(X)
        self.k = int(np.minimum(self.n_indexed_samples, self.k))

        if self.bandwidth is None:
            self.bandwidth = self._estimate_bandwidth(X)
        self.precomputed_density = np.sort(self.log_density(X, is_train_data=True))
        return self

    def percentile(self, X):
        """Calculate the percentile of the density for a given data

        :param X: data to predict
        :type X: numpy.ndarray
        :return: predicted class labels
        :rtype: numpy.ndarray
        """
        X = self._homogenize(X)
        log_density = self.log_density(X)

        return np.searchsorted(self.precomputed_density, log_density) / len(
            self.precomputed_density
        )

    def gradient(self, X, is_train_data=False):
        """Calculate the log density of the data

        :param X: data to predict
        :type X: numpy.ndarray
        :return: predicted class labels
        :rtype: numpy.ndarray
        """
        if self.metric != "cosine":
            raise NotImplementedError("Only cosine metric is supported")

        X = self._homogenize(X)

        A = self._make_knn_graph(
            X, k=self.k + 1 if is_train_data else self.k, exclude_selfloop=is_train_data
        )

        # Calculate Z1
        A.data = A.data * np.exp(-A.data ** 2 / (2 * self.bandwidth ** 2))
        grad1 = -A @ self.X

        # Calculate Z2
        if self.k == self.n_samples:
            grad2 = np.zeros_like(X)
        else:
            dist, indices = calc_distance_to_non_neighbors(
                X, self.X, A, num_samples=self.m, metric=self.metric
            )
            rows = (
                np.arange(X.shape[0]).reshape((X.shape[0], 1))
                @ np.ones((1, indices.shape[1]))
            ).reshape(-1)
            dist, indices = dist.reshape(-1), indices.reshape(-1)

            B = sparse.csr_matrix(
                (dist, (rows, indices)), shape=(X.shape[0], self.X.shape[0])
            )

            B.data = B.data * np.exp(-B.data ** 2 / (2 * self.bandwidth ** 2))
            grad2 = -B @ self.X
        grad = grad1 + grad2 * (self.n_samples - self.k) / self.m

        grad = np.einsum("ij,i->ij", -grad, 1 / np.linalg.norm(grad, axis=1))
        return grad

    def log_density(self, X, is_train_data=False):
        """Calculate the log density of the data

        :param X: data to predict
        :type X: numpy.ndarray
        :return: predicted class labels
        :rtype: numpy.ndarray
        """

        X = self._homogenize(X)

        denom = np.power(2 * np.pi * self.bandwidth ** 2, self.n_features / 2)

        A = self._make_knn_graph(
            X, k=self.k + 1 if is_train_data else self.k, exclude_selfloop=is_train_data
        )

        # Calculate Z1
        A.data = np.exp(-A.data ** 2 / (2 * self.bandwidth ** 2))  #
        Z1 = np.array(A.sum(axis=1)).reshape(-1)

        # Calculate Z2
        if self.k == self.n_samples:
            Z2 = np.zeros(X.shape[0])
        else:
            dist, indices = calc_distance_to_non_neighbors(
                X, self.X, A, num_samples=self.m, metric=self.metric
            )
            dist = np.exp(-(dist ** 2) / (2 * self.bandwidth ** 2))
            Z2 = np.array(np.sum(dist, axis=1)).reshape(-1)

        density = Z1 + Z2 * (self.n_samples - self.k) / self.m
        log_density = (
            np.log(np.maximum(density, 1e-64)) - np.log(self.n_samples) - np.log(denom)
        )
        return log_density

    def _estimate_bandwidth(self, X):
        """A heuristic bandwidth estimator
        Jaakkola, Tommi S., Mark Diekhans, and David Haussler. "Using the Fisher kernel method to detect remote protein homologies." ISMB. Vol. 99. 1999.

        :param X: input data
        :type X: numpy.ndarray
        """
        d, _ = self.index.search(
            X.astype("float32"), 2
        )  # this set to 2 because 1 is taken for the indexed sample itself

        if self.metric == "cosine":
            d = 1 - d
        dh = np.maximum(np.quantile(np.array(d[:, 1]).reshape(-1), 0.95), 1e-12)
        # dh = np.maximum(np.median(np.array(d[:, 1]).reshape(-1)), 1e-12)
        return dh

    def _make_faiss_index(self, X):
        """Create an index for the provided data

        :param X: data to index
        :type X: numpy.ndarray
        :raises NotImplementedError: if the metric is not implemented
        :return: faiss index
        :rtype: faiss.Index
        """
        n_samples, n_features = X.shape[0], X.shape[1]
        self.n_features, self.n_samples = n_features, n_samples
        X = X.astype("float32")
        if n_samples < 1000:
            self.exact = True

        index = (
            faiss.IndexFlatL2(n_features)
            if self.metric == "euclidean"
            else faiss.IndexFlatIP(n_features)
        )

        if not self.exact:
            # code_size = 32
            train_sample_num = np.minimum(100000, X.shape[0])
            nlist = int(np.ceil(np.sqrt(train_sample_num)))
            faiss_metric = (
                faiss.METRIC_L2
                if self.metric == "euclidean"
                else faiss.METRIC_INNER_PRODUCT
            )
            # index = faiss.IndexIVFPQ(
            #    index, n_features, nlist, code_size, 8, faiss_metric
            # )
            index = faiss.IndexIVFFlat(index, n_features, nlist, faiss_metric)
            # index.nprobe = 5

        if self.gpu_id is not None:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, self.gpu_id, index)

        if not index.is_trained:
            Xtrain = X[
                np.random.choice(X.shape[0], train_sample_num, replace=False), :
            ].copy(order="C")
            index.train(Xtrain)
        index.add(X)
        self.index = index
        self.X = X

    def _make_knn_graph(self, X, k, exclude_selfloop=True):
        """Construct the k-nearest neighbor graph

        :param X: data to construct the graph
        :type X: numpy.ndarray
        :param k: number of neighbors
        :type k: int
        :param exclude_selfloop: whether to exclude self-loops, defaults to True
        :type exclude_selfloop: bool, optional
        :return: k-nearest neighbor graph
        :rtype: numpy.ndarray
        """
        # get the number of samples and features
        n_samples, n_features = X.shape

        # create a list of k nearest neighbors for each vector
        dist, indices = self.index.search(X.astype("float32"), k)

        if self.metric == "cosine":
            dist = 1 - dist

        rows = np.arange(n_samples).reshape((-1, 1)) @ np.ones((1, k))

        # create the knn graph
        rows, indices, dist = rows.ravel(), indices.ravel(), dist.ravel()
        if exclude_selfloop:
            s = rows != indices
            rows, indices, dist = rows[s], indices[s], dist[s]

        s = indices >= 0
        rows, indices, dist = rows[s], indices[s], dist[s]

        A = sparse.csr_matrix(
            (dist, (rows, indices)),
            shape=(n_samples, self.n_indexed_samples),
        )
        return A

    def _homogenize(self, X, Y=None):
        if self.metric == "cosine":
            X = np.einsum(
                "ij,i->ij", X, 1 / np.maximum(np.linalg.norm(X, axis=1), 1e-12)
            )
        X = X.astype("float32")

        if X.flags["C_CONTIGUOUS"]:
            X = X.copy(order="C")

        if Y is not None:
            if sparse.issparse(Y):
                if not sparse.isspmatrix_csr(Y):
                    Y = sparse.csr_matrix(Y)
            elif isinstance(Y, np.ndarray):
                Y = sparse.csr_matrix(Y)
            else:
                raise ValueError("Y must be a scipy sparse matrix or a numpy array")
            Y.data[Y.data != 1] = 1
            return X, Y
        else:
            return X

    def grad_ascent(
        self,
        initial_cluster_size=100,
        min_dist=0.1,
        tol=1e-4,
        eta=1e-4,
        max_iter=100,
        k=None,
        use_kde_grad=True,
    ):
        """
        Mean-Shift Algorithm
        """
        if k is None:
            k = self.k

        n, d = self.X.shape
        K = np.minimum(initial_cluster_size, n)
        kmeans = MiniBatchKMeans(n_clusters=K, random_state=0)
        kmeans.fit(self.X)
        xt = kmeans.cluster_centers_
        # ids = np.random.choice(self.X.shape[0], K, replace=False)
        # xt = self.X[ids, :].copy().astype("float32")
        optimizer = ADAM()
        optimizer.eta = eta
        pbar = tqdm(range(max_iter))
        for i in pbar:
            if use_kde_grad:
                dx = self.gradient(xt, is_train_data=False)
                xnew = optimizer.update(xt, dx, lasso_penalty=0)
            else:
                _, indices = self.index.search(xt.astype("float32"), self.k)
                xnew = []
                for j in range(indices.shape[0]):
                    xnew.append(
                        np.mean(self.X[indices[j, :], :], axis=0).astype("float32")
                    )
                xnew = np.vstack(xnew)

            # Early stop
            if np.linalg.norm(xnew - xt) / np.linalg.norm(xt) < tol:
                xt = xnew
                break
            if i % 10 == 0:
                p = self.percentile(xt)
                p = np.max(p)
                pbar.set_description(f"Ascending - max percentile={p:.2f}")
            xt = xnew.copy()

        # Merge very close cluster centers
        # The center of the mergerd cluster will be that with the highest density
        assert self.metric == "cosine"  # euclidean not implemented
        xemb = np.einsum("ij,i->ij", xt, 1 / np.linalg.norm(xt, axis=1))
        D = 1 - xemb @ xemb.T
        n_components, labels = connected_components(
            csgraph=sparse.csr_matrix(D <= min_dist), directed=False, return_labels=True
        )
        density = self.log_density(xt)
        merged_xt = []
        for i in range(n_components):
            clus_ids = np.where(i == labels)[0]
            i_max = np.argmax(density[clus_ids])
            merged_xt.append(xt[clus_ids[i_max], :])
        xt = np.vstack(merged_xt)
        return xt


def calc_distance_to_non_neighbors(X, Xref, A, num_samples, metric):
    indices = _random_sample_non_neighbors(
        A.indptr, A.indices, A.shape[0], A.shape[1], num_samples
    )

    dist = np.zeros(indices.shape)
    if metric == "euclidean":
        for k in range(indices.shape[1]):
            dist[:, k] = np.linalg.norm(X - Xref[indices[:, k]], axis=1)
    elif metric == "cosine":
        # nX = np.einsum("ij,i->ij", X, 1 / np.linalg.norm(X, axis=1))
        for k in range(indices.shape[1]):
            dist[:, k] = np.sum(X * Xref[indices[:, k], :], axis=1)
        dist = 1 - dist
    else:
        raise ValueError("Metric {} not implemented".format(metric))
    return dist, indices


@numba.njit(nogil=True)
def _isin_sorted(a, x):
    return a[np.searchsorted(a, x)] == x


@numba.njit(nogil=True)
def _random_sample_non_neighbors(A_indptr, A_indices, nrows, ncols, num_samples):
    retval = -np.ones((nrows, num_samples), dtype=np.int64)
    for i in range(nrows):
        nei = A_indices[A_indptr[i] : A_indptr[i + 1]]
        sampled = 0
        while sampled < num_samples:
            r = np.random.randint(0, ncols)
            if not _isin_sorted(nei, r):
                retval[i, sampled] = r
                sampled += 1
    return retval
