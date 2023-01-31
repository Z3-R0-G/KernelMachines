import numpy as np
from sklearn.preprocessing import StandardScaler

from nystrompca import KernelMachine, Transformation
from nystrompca.utils import (get_eigendecomposition, get_kappa,
                              IdentityScaler, demean_matrix, flip_dimensions)


class KernelPCA(KernelMachine, Transformation):
  
    def __init__(self, scale:  bool = True,
                       demean: bool = True,
                       **kwargs           ):

        super().__init__(**kwargs)

        if scale:
            self.scaler = StandardScaler()
        else:
            self.scaler = IdentityScaler()

        self.demean = demean

        self.explained_variance_: np.ndarray = None
        self.all_variances:       np.ndarray = None
        self.components_:         np.ndarray = None
        self.errors_:             np.ndarray = None
        self.K_p:                 np.ndarray = None
        self.Q:                   np.ndarray = None


    def _setup(self, X: np.ndarray) -> None:

        self.n = X.shape[0]

        self.X = self.scaler.fit_transform(X)

        if self.n_components is None:
            self.n_components = self.n


    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the Kernel PCA for the supplied data matrix.
        Calculates the principal components, scores and values
        (explained variance).
        """
        self._setup(X)

        K = self.kernel.matrix(self.X, demean=False)

        if self.demean:
            K_p = demean_matrix(K)
        else:
            K_p = K

        L, Q = get_eigendecomposition(K_p / self.n)

        explained_variance_ = L[:self.n_components]

        Lambda = np.diag(self.n * explained_variance_)

        Q = Q[:,:self.n_components]
        scores_ = Q @ np.sqrt(Lambda)

        scores_, Q = flip_dimensions(scores_, Q)

        # Save calculated variables
        self.K                   = K
        self.K_p                 = K_p
        self.scores_             = scores_
        self.explained_variance_ = explained_variance_
        self.Q                   = Q

        return scores_


    def transform(self, X_new: np.ndarray) -> np.ndarray:
        """
        Transform data into the coordinate system defined by the
        principal components including demeaning the data.
        Raises
        ------
        ValueError
            If the 'fit_transform' method has not been called yet
        """
        if self.scores_ is None:
            raise ValueError("Call 'fit_transform' before this function.")

        X_new = self.scaler.transform(X_new)

        n_new = X_new.shape[0]

        kappa_p = self.kernel.matrix(self.X, X_new, demean=False)

        kappa_p = get_kappa(kappa_p, self.K, self.demean)

        j = np.where(self.explained_variance_ > 0)[0][-1]
        Lambda_inv = np.diag(1 / (self.n * self.explained_variance_[:j+1]))

        X_transformed = kappa_p.T @ self.Q[:,:j+1] @ np.sqrt(Lambda_inv)

        # Add zeros for dimensions with zero explained variance
        X_transformed = np.c_[X_transformed,
                              np.zeros((n_new, self.n_components-(j+1)))]

        return X_transformed
