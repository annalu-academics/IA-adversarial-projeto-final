from __future__ import annotations

import numpy as np
from sklearn.svm import SVR

from models.base_regressor import BaseRegressor


import numpy as np
from sklearn.kernel_ridge import KernelRidge

from models.base_regressor import BaseRegressor


class KernelRidgeRegressorModel(BaseRegressor):
    """Kernel Ridge Regression: classical non-linear baseline using the kernel trick."""

    def __init__(
        self,
        kernel: str = "rbf",
        alpha: float = 1.0,
        gamma: float | None = None,
    ) -> None:
        self.model = KernelRidge(
            kernel=kernel,
            alpha=alpha,
            gamma=gamma,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_representation(self, X: np.ndarray) -> np.ndarray:
        # Assim como na regressão linear pura, se não houver um espaço oculto 
        # linear tradicional exposto, podemos retornar o próprio espaço de entrada 
        # (ou o mapeamento de kernel, dependendo de como os ataques utilizam).
        return X
