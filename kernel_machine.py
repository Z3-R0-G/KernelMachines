from abc import ABC

import numpy as np

from nystrompca.base import Kernel


class KernelMachine(ABC):

    """
    Abstract base class for kernel machines
    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel function
    sigma : float, default=10
        Scale parameter for rbf, laplace and cauchy kernels
    degree: int, default=2
        Degree for the polynomial and rbf kernels
    coef0 : float, default=1
        Intercept for polynomial kernel
    normalize : bool, default=True
        Whether to normalize the polynomial kernel
    Attributes
    ----------
    kernel : nystrompca.algorithms.base.Kernel
        Kernel function instance
    K : numpy.ndarray, 2d
        Kernel matrix
    """
    def __init__(self, kernel:       str           = 'rbf',
                       sigma:        float         = 3,
                       degree:       int           = 2,
                       coef0:        float         = 1,
                       normalize:    bool          = True,
                       **kwargs                           ):

        super().__init__(**kwargs) # type: ignore[call-arg]

        self.kernel = Kernel(kernel, sigma, degree, coef0, normalize)

        self.K: np.ndarray = None
