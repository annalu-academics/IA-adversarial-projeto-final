from __future__ import annotations

import numpy as np
from sklearn.svm import SVR

from models.base_regressor import BaseRegressor


class SVMRegressorModel(BaseRegressor):
    """Support Vector Regression (SVR) model wrapping scikit-learn's SVR."""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1,
        gamma: str | float = "scale",
    ) -> None:
        self.model = SVR(
            kernel=kernel,
            C=C,
            epsilon=epsilon,
            gamma=gamma,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_representation(self, X: np.ndarray) -> np.ndarray:
        # O SVM não possui camadas ocultas intermediárias como uma MLP. 
        # Seguindo a lógica de modelos sem camadas (como a LinearRegressionModel), 
        # a representação padrão retorna o próprio espaço de entrada X.
        return X
