from abc import ABC

import numpy as np

from nystrompca.base import Kernel


class KernelMachine(ABC):

    def __init__(self, kernel:       str           = 'rbf',
                       sigma:        float         = 3,
                       degree:       int           = 2,
                       coef0:        float         = 1,
                       normalize:    bool          = True,
                       **kwargs                           ):

        super().__init__(**kwargs) # type: ignore[call-arg]

        self.kernel = Kernel(kernel, sigma, degree, coef0, normalize)

        self.K: np.ndarray = None
