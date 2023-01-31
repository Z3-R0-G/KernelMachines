from abc import ABC, abstractmethod

import numpy as np


class Transformation(ABC):

    def __init__(self, n_components: int = None, **kwargs):

        super().__init__(**kwargs) # type: ignore[call-arg]

        self.n_components:   int = n_components

        self.X:       np.ndarray = None

        self.n:              int = None

        self.scores_: np.ndarray = None


    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray:


    @abstractmethod
    def transform(self, X_new: np.ndarray) -> np.ndarray:



    def tot_variance(self, X: np.ndarray = None):

        if X is None:
            X = self.X

        if X is None:
            raise ValueError("No dataset supplied")

        total_variance = np.sum(X.var(0))

        return 
