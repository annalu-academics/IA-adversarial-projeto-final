from __future__ import annotations

import math
from typing import Protocol

import numpy as np
import torch
from sklearn.decomposition import PCA

from models.torch_vqc_sim import VQCSimulator

Params = list[torch.Tensor]


class Surrogate(Protocol):
    """Functional, differentiable stand-in for a model, used to craft attacks."""

    def init_params(self, seed: int) -> Params: ...

    def forward(self, params: Params, X: torch.Tensor) -> torch.Tensor: ...

    def representation(self, params: Params, X: torch.Tensor) -> torch.Tensor: ...


class MLPSurrogate:
    def __init__(self, in_dim: int, hidden_layer_sizes: tuple[int, ...], activation: str = "relu") -> None:
        self.sizes = [in_dim, *hidden_layer_sizes, 1]
        self.activation = {"relu": torch.relu, "tanh": torch.tanh, "logistic": torch.sigmoid}.get(
            activation, lambda x: x
        )

    def init_params(self, seed: int) -> Params:
        gen = torch.Generator().manual_seed(seed)
        params: Params = []
        for fan_in, fan_out in zip(self.sizes[:-1], self.sizes[1:]):
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            w = (torch.rand(fan_in, fan_out, generator=gen, dtype=torch.float64) * 2 - 1) * bound
            params += [w.requires_grad_(), torch.zeros(fan_out, dtype=torch.float64, requires_grad=True)]
        return params

    def representation(self, params: Params, X: torch.Tensor) -> torch.Tensor:
        h = X
        for w, b in zip(params[:-2:2], params[1:-2:2]):
            h = self.activation(h @ w + b)
        return h

    def forward(self, params: Params, X: torch.Tensor) -> torch.Tensor:
        return (self.representation(params, X) @ params[-2] + params[-1]).reshape(-1)


class LinearSurrogate:
    """Linear regression y = Xw + b; the representation is the input itself (no hidden layer)."""

    def __init__(self, in_dim: int) -> None:
        self.in_dim = in_dim

    def init_params(self, seed: int) -> Params:
        gen = torch.Generator().manual_seed(seed)
        bound = math.sqrt(6.0 / (self.in_dim + 1))
        w = (torch.rand(self.in_dim, 1, generator=gen, dtype=torch.float64) * 2 - 1) * bound
        return [w.requires_grad_(), torch.zeros(1, dtype=torch.float64, requires_grad=True)]

    def representation(self, params: Params, X: torch.Tensor) -> torch.Tensor:
        return X

    def forward(self, params: Params, X: torch.Tensor) -> torch.Tensor:
        return (X @ params[0] + params[1]).reshape(-1)

class KernelRidgeSurrogate:
    """Kernel Ridge Regression surrogate com subamostragem de X_ref para evitar explosão de memória no autograd."""

    def __init__(self, X_ref: np.ndarray, gamma: float = 1.0, max_ref_samples: int = 300, seed: int = 42) -> None:
        # Se o dataset de treino for muito grande, fazemos uma amostragem para estabilizar o autograd do bilevel
        if len(X_ref) > max_ref_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X_ref), size=max_ref_samples, replace=False)
            X_ref = X_ref[idx]
            
        self.X_ref = torch.tensor(X_ref, dtype=torch.float64)
        self.gamma = gamma

    def init_params(self, seed: int) -> Params:
        gen = torch.Generator().manual_seed(seed)
        n_ref = len(self.X_ref)
        bound = math.sqrt(6.0 / (n_ref + 1))
        w = (torch.rand(n_ref, 1, generator=gen, dtype=torch.float64) * 2 - 1) * bound
        return [w.requires_grad_(), torch.zeros(1, dtype=torch.float64, requires_grad=True)]

    def _kernel(self, X: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(X, self.X_ref, p=2) ** 2
        return torch.exp(-self.gamma * dists)

    def representation(self, params: Params, X: torch.Tensor) -> torch.Tensor:
        return self._kernel(X)

    def forward(self, params: Params, X: torch.Tensor) -> torch.Tensor:
        K = self._kernel(X)
        return (K @ params[0] + params[1]).reshape(-1)

class VQCSurrogate:
    """Quantum surrogate: fixed linear PCA projection followed by the differentiable VQC simulator."""

    def __init__(
        self, X_ref: np.ndarray, num_qubits: int, ansatz_reps: int, random_state: int = 42, input_scale: float = 1.0,
        encoding: str = "ry",
        output_scale: float = 1.0,
    ) -> None:
        self.output_scale = output_scale
        pca = PCA(n_components=num_qubits, whiten=True, random_state=random_state).fit(X_ref)
        self.mean = torch.tensor(pca.mean_, dtype=torch.float64)
        scale = input_scale / np.sqrt(pca.explained_variance_)
        self.components = torch.tensor(pca.components_ * scale[:, None], dtype=torch.float64)
        self.sim = VQCSimulator(num_qubits, ansatz_reps, encoding=encoding)

    def init_params(self, seed: int) -> Params:
        gen = torch.Generator().manual_seed(seed)
        theta = torch.randn(self.sim.num_weights, generator=gen, dtype=torch.float64)
        return [theta.requires_grad_()]

    def _project(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) @ self.components.T

    def forward(self, params: Params, X: torch.Tensor) -> torch.Tensor:
        return self.output_scale * self.sim.forward(params[0], self._project(X))

    def representation(self, params: Params, X: torch.Tensor) -> torch.Tensor:
        return self.sim.representation(params[0], self._project(X))
