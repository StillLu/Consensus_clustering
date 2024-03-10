"eigenvector extraction + ckmeans"
import numpy as np
import pandas as pd
from itertools import combinations
import time
import warnings
from scipy.linalg import LinAlgError, qr, svd
from scipy.sparse import csc_matrix
from scipy.cluster import hierarchy
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.manifold import spectral_embedding
from sklearn.metrics.pairwise import KERNEL_PARAMS, pairwise_kernels
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils import as_float_array, check_random_state
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.cluster import KMeans

from numbers import Integral, Real
from typing import Any, Dict

from util import *

class CKmeansResult:
    def __init__(self,consensus_matrix,cluster_membership,k,names = None,km_cls = None, order = None):
        self.cmatrix = consensus_matrix
        self.cl = cluster_membership
        self.k = k
        self.names = np.arange(consensus_matrix.shape[0]).astype(str)
        self.km_cls = km_cls
        self.order = order

class CKmeans2:

    def __init__(self, k, n_rep, p_samp, p_feat, **kwargs: Dict[str, Any]):
        self.k = k
        self.n_rep = n_rep
        self.p_samp = p_samp
        self.p_feat = p_feat
        self.kmeans = None
        self.centers = None
        self.sel_feat = None
        self.sel_samp = None

        self._kmeans_kwargs = {'n_init': 10}
        self._kmeans_kwargs.update(kwargs)

    def fit(self,x):

        if isinstance(x, pd.DataFrame):
            x = x.values

        # _fit is called here to be able to extend later on.
        self._fit(x)

    def predict(self, x, return_cls = False,progress_callback = None) -> CKmeansResult:

        names = None

        if isinstance(x, pd.DataFrame):
            names = np.array(x.index).astype(str)
            x = x.values

        cmatrix = np.zeros((x.shape[0], x.shape[0]))

        km_cls = None
        if return_cls:
            km_cls = np.zeros((self.n_rep, x.shape[0]), dtype=int)

        "implementation based on the tight clustering"
        "reference: Tight Clustering: A Resampling-Based Approach for Identifying Stable and Tight Patterns in Data"
        for i, km in enumerate(self.kmeans):
            cl = km.predict(x[:, self.sel_feat[i]])
            if return_cls:
                km_cls[i] = cl
            a, b = np.meshgrid(cl, cl)
            cmatrix += a == b

            if progress_callback:
                progress_callback()

        cmatrix /= self.n_rep

        '''
        "implementation based on consensus clustering"
        "reference: Consensus Clustering: A Resampling-Based Method for Class Discovery and Visualization of Gene Expression Microarray Data"
        cmatrix = np.zeros((x.shape[0], x.shape[0]))
        cmcount = np.zeros((x.shape[0], x.shape[0])) ## count times of pairwise elements occurring in the same sampling

        for i, km in enumerate(self.kmeans):

            cmatrix_subset = np.zeros((len(self.sel_samp[i]),len(self.sel_samp[i])))
            
            cl1 = km.predict(x[self.sel_samp[i]][:, self.sel_feat[i]])
            cl2 = np.isin(range(x.shape[0]), self.sel_samp[i]).astype(int)

            a, b = np.meshgrid(cl1, cl1)
            cmatrix_subset = a == b

            c, d = np.meshgrid(cl2, cl2)
            cmcount += c == d

            cmatrix[np.ix_(self.sel_samp[i], self.sel_samp[i])] += cmatrix_subset

        cmatrix = np.divide(cmatrix, cmcount, where=cmcount != 0)
        cmatrix[tmpCount == 0] = 0
        np.fill_diagonal(cmatrix, 1)

        '''

        dist_m = 1 - cmatrix
        conden_dist = (dist_m)[np.triu_indices_from(dist_m, k=1)]
        linkage = hierarchy.linkage(conden_dist, method="average")
        linkage[np.abs(linkage) < 1e-8] = 0

        #linkage_mat = hierarchy.optimal_leaf_ordering(linkage, conden_dist)
        linkage_mat = reorder_linkage_gw(linkage,1 - cmatrix)
        #order = hierarchy.leaves_list(linkage_mat)

        cl = hierarchy.fcluster(linkage, self.k, criterion='maxclust') - 1

        return CKmeansResult(
            cmatrix,
            cl,
            k=self.k,
            order = linkage_mat,
            names=names,
            km_cls=km_cls,
        )

    def _fit(self, x):

        self._reset()

        self.kmeans = []

        n_samp = np.ceil(self.p_samp * x.shape[0]).astype(int)
        n_feat = np.ceil(self.p_feat * x.shape[1]).astype(int)

        self.sel_feat = np.zeros((self.n_rep, n_feat), dtype=int)
        self.sel_samp = np.zeros((self.n_rep, n_samp), dtype=int)
        self.centers = np.zeros((self.n_rep, self.k, n_feat))

        for i in range(self.n_rep):
            samp_idcs = np.random.choice(x.shape[0], size=n_samp,replace=False)
            feat_idcs = np.random.choice(x.shape[1], size=n_feat,replace=False)
            self.sel_feat[i] = feat_idcs
            self.sel_samp[i] = samp_idcs

            x_subset = x[samp_idcs][:, feat_idcs]

            km = KMeans(self.k, **self._kmeans_kwargs)
            km.fit(x_subset)
            self.kmeans.append(km)
            self.centers[i] = km.cluster_centers_


    def _reset(self):

        self.centers = None
        self.kmeans = None
        self.sel_feat = None
        self.sel_samp = None



""" ref SC eigenmap package
note: 
if norm_laplacian:
    # recover u = D^-1/2 x from the eigenvector output x
    embedding = embedding / dd
"""

class gen_eigmap:
    """Apply clustering to a projection of the normalized Laplacian.

    Parameters
    ----------


    eigen_solver : {'arpack', 'lobpcg', 'amg'}, default=None
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities. If None, then ``'arpack'`` is
        used. See [4]_ for more details regarding `'lobpcg'`.

    n_components : int, default=None
        Number of eigenvectors to use for the spectral embedding. If None,
        defaults to `n_clusters`.

    random_state : int, RandomState instance, default=None
        A pseudo random number generator used for the initialization
        of the lobpcg eigenvectors decomposition when `eigen_solver ==
        'amg'`, and for the K-Means initialization. Use an int to make
        the results deterministic across calls (See
        :term:`Glossary <random_state>`).

        .. note::
            When using `eigen_solver == 'amg'`,
            it is necessary to also fix the global numpy seed with
            `np.random.seed(int)` to get deterministic results. See
            https://github.com/pyamg/pyamg/issues/139 for further
            information.


    gamma : float, default=1.0
        Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels.
        Ignored for ``affinity='nearest_neighbors'``.

    affinity : str or callable, default='rbf'
        How to construct the affinity matrix.
         - 'nearest_neighbors': construct the affinity matrix by computing a
           graph of nearest neighbors.
         - 'rbf': construct the affinity matrix using a radial basis function
           (RBF) kernel.
         - 'precomputed': interpret ``X`` as a precomputed affinity matrix,
           where larger values indicate greater similarity between instances.
         - 'precomputed_nearest_neighbors': interpret ``X`` as a sparse graph
           of precomputed distances, and construct a binary affinity matrix
           from the ``n_neighbors`` nearest neighbors of each instance.
         - one of the kernels supported by
           :func:`~sklearn.metrics.pairwise.pairwise_kernels`.

        Only kernels that produce similarity scores (non-negative values that
        increase with similarity) should be used. This property is not checked
        by the clustering algorithm.

    n_neighbors : int, default=10
        Number of neighbors to use when constructing the affinity matrix using
        the nearest neighbors method. Ignored for ``affinity='rbf'``.

    eigen_tol : float, default="auto"
        Stopping criterion for eigen decomposition of the Laplacian matrix.
        If `eigen_tol="auto"` then the passed tolerance will depend on the
        `eigen_solver`:

        - If `eigen_solver="arpack"`, then `eigen_tol=0.0`;
        - If `eigen_solver="lobpcg"` or `eigen_solver="amg"`, then
          `eigen_tol=None` which configures the underlying `lobpcg` solver to
          automatically resolve the value according to their heuristics. See,
          :func:`scipy.sparse.linalg.lobpcg` for details.

        Note that when using `eigen_solver="lobpcg"` or `eigen_solver="amg"`
        values of `tol<1e-5` may lead to convergence issues and should be
        avoided.

        .. versionadded:: 1.2
           Added 'auto' option.

    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : dict of str to any, default=None
        Parameters (keyword arguments) and values for kernel passed as
        callable object. Ignored by other kernels.

    n_jobs : int, default=None
        The number of parallel jobs to run when `affinity='nearest_neighbors'`
        or `affinity='precomputed_nearest_neighbors'`. The neighbors search
        will be done in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.


    Attributes
    ----------
    affinity_matrix_ : array-like of shape (n_samples, n_samples)
        Affinity matrix used for clustering. Available only after calling
        ``fit``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    """

    _parameter_constraints: dict = {
        "eigen_solver": [StrOptions({"arpack", "lobpcg", "amg"}), None],
        "n_components": [Interval(Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "gamma": [Interval(Real, 0, None, closed="left")],
        "affinity": [
            callable,
            StrOptions(
                set(KERNEL_PARAMS)
                | {"nearest_neighbors", "precomputed", "precomputed_nearest_neighbors"}
            ),
        ],
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
        "eigen_tol": [
            Interval(Real, 0.0, None, closed="left"),
            StrOptions({"auto"}),
        ],
        "degree": [Interval(Real, 0, None, closed="left")],
        "coef0": [Interval(Real, None, None, closed="neither")],
        "kernel_params": [dict, None],
        "n_jobs": [Integral, None],
    }

    def __init__(
        self,
        eigen_solver=None,
        n_components=None,
        random_state=None,
        n_init=10,
        gamma=0.5,
        affinity="rbf",
        n_neighbors=2,
        eigen_tol="auto",
        degree=3,
        coef0=1,
        kernel_params=None,
        n_jobs=None,
    ):

        self.eigen_solver = eigen_solver
        self.n_components = n_components
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs

    def gen_eig(self, X):
        """Perform spectral clustering from features, or affinity matrix.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Training instances to cluster, similarities / affinities between
            instances if ``affinity='precomputed'``, or distances between
            instances if ``affinity='precomputed_nearest_neighbors``. If a
            sparse matrix is provided in a format other than ``csr_matrix``,
            ``csc_matrix``, or ``coo_matrix``, it will be converted into a
            sparse ``csr_matrix``.

        Returns
        -------

        """

        allow_squared = self.affinity in [
            "precomputed",
            "precomputed_nearest_neighbors",
        ]
        if X.shape[0] == X.shape[1] and not allow_squared:
            warnings.warn(
                "The spectral clustering API has changed. ``fit``"
                "now constructs an affinity matrix from data. To use"
                " a custom affinity matrix, "
                "set ``affinity=precomputed``."
            )

        if self.affinity == "nearest_neighbors":
            connectivity = kneighbors_graph(
                X, n_neighbors=self.n_neighbors, include_self=True, n_jobs=self.n_jobs
            )
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
        elif self.affinity == "precomputed_nearest_neighbors":
            estimator = NearestNeighbors(
                n_neighbors=self.n_neighbors, n_jobs=self.n_jobs, metric="precomputed"
            ).fit(X)
            connectivity = estimator.kneighbors_graph(X=X, mode="connectivity")
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
        elif self.affinity == "precomputed":
            self.affinity_matrix_ = X
        else:
            params = self.kernel_params
            if params is None:
                params = {}
            if not callable(self.affinity):
                params["gamma"] = self.gamma
                params["degree"] = self.degree
                params["coef0"] = self.coef0
            self.affinity_matrix_ = pairwise_kernels(
                X, metric=self.affinity, filter_params=True, **params
            )

        random_state = check_random_state(self.random_state)
        '''
        n_components = (
            self.n_clusters if self.n_components is None else self.n_components
        )
        '''
        n_components = self.n_components


        # We now obtain the real valued solution matrix to the
        # relaxed Ncut problem, solving the eigenvalue problem
        # L_sym x = lambda x  and recovering u = D^-1/2 x.
        # The first eigenvector is constant only for fully connected graphs
        # and should be kept for spectral clustering (drop_first = False)
        # See spectral_embedding documentation.
        maps = spectral_embedding(
            self.affinity_matrix_,
            n_components=n_components,
            eigen_solver=self.eigen_solver,
            random_state=random_state,
            eigen_tol=self.eigen_tol,
            drop_first=False,
        )


        return maps









