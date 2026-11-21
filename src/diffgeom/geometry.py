"""Differential Geometry Operators for Neural Network Parameter and Data Manifolds.

Implements exact, vectorized higher-order differential geometric quantities
(pullback metrics, volume forms, Christoffel symbols, Riemann/Gaussian curvature,
Data Information Matrix, Parameter Fisher Information Matrix, and purely geometric
loss functions) using composable PyTorch automatic differentiation (torch.func).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import (
    make_blobs,
    make_circles,
    make_classification,
    make_gaussian_quantiles,
    make_moons,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.func import functional_call, jacfwd, jacrev, vmap

MetricMode = Literal["euclidean", "graph", "conformal_gaussian"]
GeometricTask = Literal["curvature", "volume", "anisotropy"]


def get_synthetic_dataset(
    dataset_name: str = "moons",
    n_train: int = 600,
    n_test: int = 300,
    n_classes: int = 2,
    noise: float = 0.10,
    input_dim: int = 2,
    seed: int = 42,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    **dataset_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate standardized train/test splits from scikit-learn synthetic classification datasets.

    Supported dataset_name values:
      - 'moons' / 'make_moons'
      - 'circles' / 'make_circles'
      - 'blobs' / 'make_blobs'
      - 'classification' / 'make_classification'
      - 'gaussian_quantiles' / 'make_gaussian_quantiles'
    """
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")

    n_total = n_train + n_test
    key = dataset_name.lower().strip()
    key = key.removeprefix("make_")

    rng = np.random.RandomState(seed)

    if key == "moons":
        if n_classes == 2 and input_dim == 2:
            X, y = make_moons(n_samples=n_total, noise=noise, random_state=seed)
        else:
            # Generalized multi-class crescents using make_moons building blocks
            per_pair = math.ceil(n_total / n_classes)
            xs, ys = [], []
            for c in range(n_classes):
                x_pair, y_pair = make_moons(n_samples=2 * per_pair, noise=noise, random_state=seed + c)
                x_c = x_pair[y_pair == (c % 2)][:per_pair].copy()
                angle = 2.0 * math.pi * c / n_classes
                shift = np.array([1.35 * math.cos(angle), 1.35 * math.sin(angle)])
                x_c = x_c + shift
                if input_dim > 2:
                    pad = rng.normal(scale=max(noise, 1e-3), size=(len(x_c), input_dim - 2))
                    x_c = np.hstack([x_c, pad])
                xs.append(x_c)
                ys.append(np.full(len(x_c), c, dtype=np.int64))
            X = np.vstack(xs)[:n_total]
            y = np.concatenate(ys)[:n_total]
    elif key == "circles":
        factor = float(dataset_kwargs.get("factor", 0.5))
        if n_classes == 2 and input_dim == 2:
            X, y = make_circles(n_samples=n_total, noise=noise, factor=factor, random_state=seed)
        else:
            radii = np.linspace(max(factor, 0.25), 1.0 + 0.45 * (n_classes - 2), n_classes)
            counts = [n_total // n_classes + (1 if i < (n_total % n_classes) else 0) for i in range(n_classes)]
            xs, ys = [], []
            for c, (r_c, cnt) in enumerate(zip(radii, counts)):
                theta_c = rng.uniform(0.0, 2.0 * math.pi, size=cnt)
                pts = np.stack([r_c * np.cos(theta_c), r_c * np.sin(theta_c)], axis=-1)
                if noise > 0.0:
                    pts = pts + rng.normal(scale=noise, size=pts.shape)
                if input_dim > 2:
                    pts = np.hstack([pts, rng.normal(scale=max(noise, 1e-3), size=(cnt, input_dim - 2))])
                xs.append(pts)
                ys.append(np.full(cnt, c, dtype=np.int64))
            X = np.vstack(xs)
            y = np.concatenate(ys)
    elif key == "blobs":
        cluster_std = float(dataset_kwargs.get("cluster_std", max(noise * 4.0, 0.65)))
        raw_centers = dataset_kwargs.get("centers", n_classes)
        centers = n_classes if (isinstance(raw_centers, int) and n_classes != 2) else raw_centers
        center_box = dataset_kwargs.get("center_box", (-10.0, 10.0))
        X, y = make_blobs(
            n_samples=n_total,
            n_features=input_dim,
            centers=centers,
            cluster_std=cluster_std,
            center_box=center_box,
            random_state=seed,
        )
    elif key == "classification":
        n_informative = int(dataset_kwargs.get("n_informative", input_dim))
        n_redundant = int(dataset_kwargs.get("n_redundant", 0))
        n_repeated = int(dataset_kwargs.get("n_repeated", 0))
        n_clusters_per_class = int(
            dataset_kwargs.get(
                "n_clusters_per_class",
                1 if n_classes * 2 > 2**n_informative else min(2, max(1, (2**n_informative) // n_classes)),
            )
        )
        class_sep = float(dataset_kwargs.get("class_sep", 1.25))
        flip_y = float(dataset_kwargs.get("flip_y", min(max(noise, 0.0), 0.4)))
        X, y = make_classification(
            n_samples=n_total,
            n_features=input_dim,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            n_classes=n_classes,
            n_clusters_per_class=n_clusters_per_class,
            class_sep=class_sep,
            flip_y=flip_y,
            random_state=seed,
        )
    elif key == "gaussian_quantiles":
        cov = float(dataset_kwargs.get("cov", 1.0))
        X, y = make_gaussian_quantiles(
            n_samples=n_total,
            n_features=input_dim,
            n_classes=n_classes,
            cov=cov,
            random_state=seed,
        )
        if noise > 0.0:
            X = X + rng.normal(scale=noise, size=X.shape)
    else:
        raise ValueError(
            f"Unsupported dataset_name '{dataset_name}'. Choose from: "
            "'moons', 'circles', 'blobs', 'classification', 'gaussian_quantiles'."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=seed, stratify=y
    )
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return (
        torch.tensor(X_train, dtype=dtype, device=device),
        torch.tensor(y_train, dtype=torch.long, device=device),
        torch.tensor(X_test, dtype=dtype, device=device),
        torch.tensor(y_test, dtype=torch.long, device=device),
    )


def get_moons_dataset(
    n_train: int = 600,
    n_test: int = 300,
    noise: float = 0.10,
    seed: int = 42,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate standardized make_moons train and test datasets."""
    return get_synthetic_dataset(
        dataset_name="moons",
        n_train=n_train,
        n_test=n_test,
        n_classes=2,
        noise=noise,
        seed=seed,
        dtype=dtype,
        device=device,
    )


def make_evaluation_grid(
    X: torch.Tensor,
    n_grid: int = 35,
    padding: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Create a 2D meshgrid covering the input domain of X."""
    x_np = X.detach().cpu().numpy()
    x1_min, x1_max = x_np[:, 0].min() - padding, x_np[:, 0].max() + padding
    x2_min, x2_max = x_np[:, 1].min() - padding, x_np[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x1_min, x1_max, n_grid),
        np.linspace(x2_min, x2_max, n_grid),
    )
    grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    grid_tensor = torch.tensor(grid_pts, dtype=X.dtype, device=X.device)
    return xx, yy, grid_tensor


class MoonsMLP(nn.Module):
    """Configurable MLP supporting arbitrary layer_dims, hidden activation, and output activation."""

    def __init__(
        self,
        layer_dims: Sequence[int] = (2, 5, 10, 15, 10, 2),
        activation: str = "tanh",
        output_activation: str | None = "identity",
        beta: float = 3.0,
        seed: int = 42,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if len(layer_dims) < 2:
            raise ValueError(f"layer_dims must have at least 2 entries [in_dim, out_dim], got {layer_dims}")
        torch.manual_seed(seed)
        self.layer_dims = [int(d) for d in layer_dims]
        self.activation_name = activation.lower().strip()
        self.output_activation_name = (
            "identity" if output_activation is None else output_activation.lower().strip()
        )
        self.beta = beta
        self.num_layers = len(self.layer_dims) - 1

        layers = []
        for i in range(self.num_layers):
            lin = nn.Linear(self.layer_dims[i], self.layer_dims[i + 1], dtype=dtype)
            nn.init.xavier_normal_(lin.weight, gain=1.1)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
        self.layers = nn.ModuleList(layers)

    def _apply_activation(self, u: torch.Tensor, act_name: str) -> torch.Tensor:
        if act_name in ("identity", "linear", "none", ""):
            return u
        if act_name == "tanh":
            return torch.tanh(u)
        if act_name == "gelu":
            return F.gelu(u)
        if act_name in ("silu", "swish"):
            return F.silu(u)
        if act_name == "softplus":
            return F.softplus(u, beta=self.beta)
        if act_name == "elu":
            return F.elu(u)
        if act_name == "sigmoid":
            return torch.sigmoid(u)
        if act_name == "sin":
            return torch.sin(u)
        if act_name == "softmax":
            return F.softmax(u, dim=-1)
        if act_name == "log_softmax":
            return F.log_softmax(u, dim=-1)
        if act_name == "relu":
            return F.relu(u)
        raise ValueError(f"Unsupported activation: '{act_name}'")

    def _activate(self, u: torch.Tensor) -> torch.Tensor:
        return self._apply_activation(u, self.activation_name)

    def _activate_output(self, u: torch.Tensor) -> torch.Tensor:
        return self._apply_activation(u, self.output_activation_name)

    def forward_up_to_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Compute representation Phi^(layer_idx)(x) for layer_idx in {0, 1, ..., L}."""
        if layer_idx == 0:
            return x
        if not (1 <= layer_idx <= self.num_layers):
            raise ValueError(f"layer_idx must be in [0, {self.num_layers}], got {layer_idx}")

        h = x
        for idx in range(layer_idx):
            h = self.layers[idx](h)
            if idx < self.num_layers - 1:
                h = self._activate(h)
            else:
                h = self._activate_output(h)
        return h

    def forward_all_layers(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return representations [x, h^(1), ..., z^(L)]."""
        reps = [x]
        h = x
        for idx, layer in enumerate(self.layers):
            h = layer(h)
            if idx < self.num_layers - 1:
                h = self._activate(h)
            else:
                h = self._activate_output(h)
            reps.append(h)
        return reps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_up_to_layer(x, self.num_layers)


def _make_single_point_layer_fn(
    model: MoonsMLP,
    layer_idx: int,
    params: dict[str, torch.Tensor] | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a function phi(x_single) -> h_single for a single input point x_single."""
    if params is None:
        params = dict(model.named_parameters())

    def phi_single(x_single: torch.Tensor) -> torch.Tensor:
        if layer_idx == 0:
            return x_single
        h = x_single
        for idx in range(layer_idx):
            w = params[f"layers.{idx}.weight"]
            b = params[f"layers.{idx}.bias"]
            h = F.linear(h, w, b)
            if idx < model.num_layers - 1:
                h = model._activate(h)
            else:
                h = model._activate_output(h)
        return h

    return phi_single


def _metric_single_from_phi(
    phi_single: Callable[[torch.Tensor], torch.Tensor],
    metric_mode: MetricMode = "euclidean",
    eps: float = 1e-6,
    sigma: float = 2.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Construct single-point metric function g(x) in R^{2x2} from map phi: R^2 -> R^d."""
    jac_fn = jacrev(phi_single)

    def g_single(x_single: torch.Tensor) -> torch.Tensor:
        J = jac_fn(x_single)  # shape: (d_l, 2)
        jtj = J.T @ J  # shape: (2, 2)
        eye2 = torch.eye(x_single.shape[0], dtype=x_single.dtype, device=x_single.device)
        if metric_mode == "euclidean":
            return jtj + eps * eye2
        if metric_mode == "graph":
            # Graph immersion (x, phi(x)) in R^{2 + d_l}: g = I_2 + J^T J
            return eye2 + jtj
        if metric_mode == "conformal_gaussian":
            z = phi_single(x_single)
            weight = torch.exp(-torch.sum(z * z) / (sigma * sigma))
            return weight * jtj + eps * eye2
        raise ValueError(f"Unknown metric_mode: {metric_mode}")

    return g_single


def compute_layer_jacobian(
    model: MoonsMLP,
    X: torch.Tensor,
    layer_idx: int,
    params: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute layerwise Jacobian J^(l)(x) of shape (N, d_l, 2)."""
    phi_single = _make_single_point_layer_fn(model, layer_idx, params=params)
    return vmap(jacrev(phi_single))(X)


def compute_layer_pullback_metric(
    model: MoonsMLP,
    X: torch.Tensor,
    layer_idx: int,
    metric_mode: MetricMode = "euclidean",
    eps: float = 1e-6,
    sigma: float = 2.0,
    params: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute pullback metric tensor g^(l)_{ij}(x) of shape (N, 2, 2)."""
    phi_single = _make_single_point_layer_fn(model, layer_idx, params=params)
    g_single = _metric_single_from_phi(phi_single, metric_mode=metric_mode, eps=eps, sigma=sigma)
    return vmap(g_single)(X)


def compute_volume_element(g: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute Riemannian volume magnification sqrt(det g(x)) of shape (N,)."""
    det_g = torch.linalg.det(g)
    return torch.sqrt(torch.clamp(det_g, min=eps))


def compute_metric_spectrum(
    g: torch.Tensor, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return eigenvalues (N, 2), eigenvectors (N, 2, 2), and condition number lambda_max/lambda_min."""
    evals, evecs = torch.linalg.eigh(g)
    evals = torch.clamp(evals, min=eps)
    # Sort descending: lambda_1 >= lambda_2
    evals = evals.flip(-1)
    evecs = evecs.flip(-1)
    cond = evals[:, 0] / evals[:, 1]
    return evals, evecs, cond


def _christoffel_and_curvature_from_g_single(
    g_single: Callable[[torch.Tensor], torch.Tensor],
    eps: float = 1e-8,
) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return function computing (g, Gamma^k_{ij}, R_{1212}, K) at a single 2D point x."""
    dg_fn = jacrev(g_single)  # d g_{ij} / d x^k -> shape (2, 2, 2) indexed as [i, j, k]
    ddg_fn = jacrev(dg_fn)  # d^2 g_{ij} / (d x^k d x^l) -> shape (2, 2, 2, 2) indexed as [i, j, k, l]

    def geom_single(
        x_single: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        g = g_single(x_single)  # (2, 2)
        dg = dg_fn(x_single)  # dg[i, j, k] = \partial_k g_{ij}
        ddg = ddg_fn(x_single)  # ddg[i, j, k, l] = \partial_l \partial_k g_{ij}

        det_g = g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0]
        det_g_safe = torch.where(
            torch.abs(det_g) < eps,
            torch.sign(det_g + 1e-15) * eps,
            det_g,
        )
        # Analytic 2x2 inverse for fast higher-order autodiff
        g_inv = torch.stack(
            [
                torch.stack([g[1, 1], -g[0, 1]]),
                torch.stack([-g[1, 0], g[0, 0]]),
            ]
        ) / det_g_safe

        # Christoffel symbols of the first kind:
        # Gamma_lower[k, i, j] = 0.5 * (\partial_j g_{ki} + \partial_i g_{kj} - \partial_k g_{ij})
        gamma_lower = 0.5 * (
            dg.permute(0, 1, 2)  # dg[k, i, j] = \partial_j g_{ki}
            + dg.permute(0, 2, 1)  # dg[k, j, i] = \partial_i g_{kj}
            - dg.permute(2, 0, 1)  # dg[i, j, k] -> permuted to [k, i, j] = \partial_k g_{ij}
        )

        # Christoffel symbols of the second kind: Gamma^k_{ij} = g^{km} Gamma_{mij}
        gamma_upper = torch.einsum("km,mij->kij", g_inv, gamma_lower)

        # Riemann curvature tensor component R_{1212} for 2D metric:
        # R_{1212} = -0.5 * (\partial_1 \partial_1 g_{22} - 2 \partial_1 \partial_2 g_{12} + \partial_2 \partial_2 g_{11})
        #            + g^{km} (Gamma_{k12} Gamma_{m12} - Gamma_{k11} Gamma_{m22})
        d11_g22 = ddg[1, 1, 0, 0]
        d12_g12 = ddg[0, 1, 0, 1]
        d22_g11 = ddg[0, 0, 1, 1]

        second_deriv_term = -0.5 * (d11_g22 - 2.0 * d12_g12 + d22_g11)
        gamma_11 = gamma_lower[:, 0, 0]
        gamma_22 = gamma_lower[:, 1, 1]
        gamma_12 = gamma_lower[:, 0, 1]
        quad_term = torch.dot(gamma_12, g_inv @ gamma_12) - torch.dot(
            gamma_11, g_inv @ gamma_22
        )

        R_1212 = second_deriv_term + quad_term
        K = R_1212 / det_g_safe
        return g, gamma_upper, R_1212, K

    return geom_single


def compute_christoffel_symbols(
    model: MoonsMLP,
    X: torch.Tensor,
    layer_idx: int,
    metric_mode: MetricMode = "euclidean",
    eps: float = 1e-6,
    sigma: float = 2.0,
    params: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute Christoffel symbols Gamma^(l)k_{ij}(x) of shape (N, 2, 2, 2)."""
    phi_single = _make_single_point_layer_fn(model, layer_idx, params=params)
    g_single = _metric_single_from_phi(phi_single, metric_mode=metric_mode, eps=eps, sigma=sigma)
    geom_single = _christoffel_and_curvature_from_g_single(g_single)
    _, gamma_upper, _, _ = vmap(geom_single)(X)
    return gamma_upper


def compute_gaussian_curvature(
    model: MoonsMLP,
    X: torch.Tensor,
    layer_idx: int,
    metric_mode: MetricMode = "euclidean",
    eps: float = 1e-6,
    sigma: float = 2.0,
    params: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute Gaussian curvature K^(l)(x) of shape (N,)."""
    phi_single = _make_single_point_layer_fn(model, layer_idx, params=params)
    g_single = _metric_single_from_phi(phi_single, metric_mode=metric_mode, eps=eps, sigma=sigma)
    geom_single = _christoffel_and_curvature_from_g_single(g_single)
    _, _, _, K = vmap(geom_single)(X)
    return K


def compute_layer_geometry(
    model: MoonsMLP,
    X: torch.Tensor,
    layer_idx: int,
    metric_mode: MetricMode = "euclidean",
    eps: float = 1e-6,
    sigma: float = 2.0,
    params: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute metric tensor g, volume element, Christoffel symbols, R_1212, and Gaussian curvature K."""
    phi_single = _make_single_point_layer_fn(model, layer_idx, params=params)
    g_single = _metric_single_from_phi(phi_single, metric_mode=metric_mode, eps=eps, sigma=sigma)
    geom_single = _christoffel_and_curvature_from_g_single(g_single)
    g, gamma_upper, R_1212, K = vmap(geom_single)(X)
    vol = compute_volume_element(g)
    return {
        "g": g,
        "volume": vol,
        "christoffel": gamma_upper,
        "R_1212": R_1212,
        "K": K,
        "scalar_curvature": 2.0 * K,
    }


def flatten_params(param_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a dictionary of parameter tensors into a single 1D vector in R^P."""
    return torch.cat([p.reshape(-1) for p in param_dict.values()], dim=0)


def compute_parameter_fim(
    model: MoonsMLP,
    X: torch.Tensor,
    damping: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the P x P Parameter Fisher Information Matrix F(theta) and per-sample score vectors.

    F_{mu,nu}(theta) = E_x [ sum_{c=0}^{C-1} p(y=c|x; theta) d_mu log p(y=c|x) d_nu log p(y=c|x) ]
    """
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = dict(model.named_buffers())
    param_keys = list(params.keys())

    def log_probs_single(p_dict: dict[str, torch.Tensor], x_single: torch.Tensor) -> torch.Tensor:
        logits = functional_call(model, (p_dict, buffers), (x_single.unsqueeze(0),)).squeeze(0)
        return F.log_softmax(logits, dim=-1)

    with torch.no_grad():
        probs = F.softmax(model(X), dim=-1)  # (N, C)
    N, n_out = probs.shape

    # jac_params has keys matching params, each of shape (N, C, *param.shape)
    jac_fn = vmap(jacrev(log_probs_single, argnums=0), in_dims=(None, 0))
    jac_dict = jac_fn(params, X)

    J_flat = torch.cat(
        [jac_dict[k].reshape(N, n_out, -1) for k in param_keys], dim=-1
    )  # (N, C, P)

    # Weight each class derivative by sqrt(p(y=c|x))
    weighted_J = J_flat * torch.sqrt(probs).unsqueeze(-1)  # (N, C, P)
    flat_scores = weighted_J.reshape(N * n_out, -1)  # (N*C, P)
    P = flat_scores.shape[1]
    fim = (flat_scores.T @ flat_scores) / float(N) + damping * torch.eye(
        P, dtype=X.dtype, device=X.device
    )
    return fim, J_flat


def compute_data_information_matrix(
    model: MoonsMLP,
    X: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute the d_0 x d_0 Data Information Matrix DIM_{ij}(x; theta) for each input point x in X.

    DIM_{ij}(x) = sum_{c=0}^{C-1} p(y=c|x; theta) (d_i log p(y=c|x)) (d_j log p(y=c|x))
    """
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def log_probs_x(x_single: torch.Tensor) -> torch.Tensor:
        logits = functional_call(model, (params, buffers), (x_single.unsqueeze(0),)).squeeze(0)
        return F.log_softmax(logits, dim=-1)

    J_x = vmap(jacrev(log_probs_x))(X)  # (N, C, d_in)
    probs = torch.exp(vmap(log_probs_x)(X))  # (N, C)
    weighted_Jx = J_x * torch.sqrt(probs).unsqueeze(-1)  # (N, C, d_in)
    dim_matrix = torch.einsum("nci,ncj->nij", weighted_Jx, weighted_Jx)
    d_in = X.shape[-1]
    eye_d = torch.eye(d_in, dtype=X.dtype, device=X.device).unsqueeze(0)
    return dim_matrix + eps * eye_d


def scalar_geometric_predict(
    psi: torch.Tensor,
    n_classes: int = 2,
    margin: float = 0.5,
) -> torch.Tensor:
    """Classify points from a scalar geometric invariant psi(x) into {0, ..., n_classes - 1}.

    For n_classes == 2, reduces identically to I[psi(x) > 0].
    For n_classes > 2, partitions R into n_classes ordered intervals separated at midpoints
    between class target levels t_c = 2 * margin * (c - (n_classes - 1) / 2).
    """
    if n_classes <= 2:
        return (psi > 0.0).long()
    idx = torch.arange(n_classes - 1, dtype=psi.dtype, device=psi.device)
    boundaries = 2.0 * margin * (idx + 0.5 - 0.5 * (n_classes - 1))
    return torch.bucketize(psi.contiguous(), boundaries)


def _scalar_geometric_margin_loss_and_weights(
    psi: torch.Tensor,
    y: torch.Tensor,
    margin: float,
    beta_hinge: float,
    n_classes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute scalar geometric margin loss, predicted labels, and GGN residual weights."""
    inferred_c = int(y.max().item()) + 1 if y.numel() > 0 else 2
    c_total = max(2, inferred_c if n_classes is None else max(n_classes, inferred_c))

    if c_total == 2:
        signs = 2.0 * y.to(psi.dtype) - 1.0
        violations = margin - signs * psi
        loss = F.softplus(violations, beta=beta_hinge).mean()
        with torch.no_grad():
            preds = (psi > 0.0).long()
            sig = torch.sigmoid(beta_hinge * violations)
            w_psi = torch.sqrt(beta_hinge * sig * (1.0 - sig) + 0.10)
        return loss, preds, w_psi

    # Multi-class ordered geometric bands: t_c = 2 * margin * (c - (C - 1)/2)
    targets = 2.0 * margin * (y.to(psi.dtype) - 0.5 * (c_total - 1))
    half_band = 0.5 * margin
    lower_v = (targets - half_band) - psi
    upper_v = psi - (targets + half_band)
    # Outer classes are one-sided rays (<= t_0 or >= t_{C-1}); interior classes are bounded intervals
    pen_low = torch.where(y == 0, torch.zeros_like(psi), F.softplus(lower_v, beta=beta_hinge))
    pen_high = torch.where(y == (c_total - 1), torch.zeros_like(psi), F.softplus(upper_v, beta=beta_hinge))
    # Outer classes enforce psi <= t_0 + half_band (y=0) and psi >= t_{C-1} - half_band (y=C-1)
    pen_outer_0 = torch.where(y == 0, F.softplus(psi - (targets + half_band), beta=beta_hinge), torch.zeros_like(psi))
    pen_outer_top = torch.where(
        y == (c_total - 1), F.softplus((targets - half_band) - psi, beta=beta_hinge), torch.zeros_like(psi)
    )
    interior_mask = (y > 0) & (y < (c_total - 1))
    per_sample = torch.where(interior_mask, pen_low + pen_high, pen_outer_0 + pen_outer_top)
    loss = per_sample.mean()
    with torch.no_grad():
        preds = scalar_geometric_predict(psi, n_classes=c_total, margin=margin)
        sig_l = torch.sigmoid(beta_hinge * lower_v)
        sig_u = torch.sigmoid(beta_hinge * upper_v)
        curv_w = beta_hinge * (sig_l * (1.0 - sig_l) + sig_u * (1.0 - sig_u))
        w_psi = torch.sqrt(curv_w + 0.15)
    return loss, preds, w_psi


def curvature_margin_loss(
    K: torch.Tensor,
    y: torch.Tensor,
    g: torch.Tensor,
    margin: float = 0.5,
    beta_hinge: float = 4.0,
    lambda_vol: float = 0.05,
    lambda_iso: float = 0.01,
    k_clip: float = 15.0,
    n_classes: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Purely geometric loss separating classes by intrinsic Gaussian curvature K(x).

    Includes Riemannian volume stabilization (log det g)^2 and metric scale stabilization
    to prevent trivial metric collapse without using any non-geometric logit targets.
    """
    K_bounded = k_clip * torch.tanh(K / k_clip)
    loss_curv, preds_b, _ = _scalar_geometric_margin_loss_and_weights(
        K_bounded, y, margin=margin, beta_hinge=beta_hinge, n_classes=n_classes
    )

    det_g = torch.clamp(torch.linalg.det(g), min=1e-8)
    loss_vol = (torch.log(det_g) ** 2).mean()

    eye2 = torch.eye(g.shape[-1], dtype=g.dtype, device=g.device).unsqueeze(0)
    loss_iso = ((g - eye2) ** 2).mean()

    total_loss = loss_curv + lambda_vol * loss_vol + lambda_iso * loss_iso
    with torch.no_grad():
        c_eval = max(2, int(y.max().item()) + 1 if n_classes is None else n_classes)
        preds = (K > 0.0).long() if c_eval == 2 else preds_b
        acc = (preds == y).to(K.dtype).mean().item()
        k_c0 = K[y == 0].mean().item() if (y == 0).any() else 0.0
        k_c1 = K[y == 1].mean().item() if (y == 1).any() else 0.0

    metrics = {
        "loss_total": total_loss.item(),
        "loss_curv": loss_curv.item(),
        "loss_vol": loss_vol.item(),
        "loss_iso": loss_iso.item(),
        "geom_acc": acc,
        "mean_K_class0": k_c0,
        "mean_K_class1": k_c1,
    }
    return total_loss, metrics


def volume_contrastive_loss(
    g: torch.Tensor,
    y: torch.Tensor,
    margin: float = 0.6,
    beta_hinge: float = 4.0,
    lambda_iso: float = 0.02,
    n_classes: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Purely geometric loss separating classes by Riemannian log-volume element log(sqrt(det g))."""
    log_vol = 0.5 * torch.log(torch.clamp(torch.linalg.det(g), min=1e-8))
    loss_vol_sep, preds, _ = _scalar_geometric_margin_loss_and_weights(
        log_vol, y, margin=margin, beta_hinge=beta_hinge, n_classes=n_classes
    )
    loss_reg = (log_vol ** 2).mean() * lambda_iso
    total = loss_vol_sep + loss_reg

    with torch.no_grad():
        acc = (preds == y).to(g.dtype).mean().item()

    return total, {
        "loss_total": total.item(),
        "geom_acc": acc,
        "mean_log_vol_c0": log_vol[y == 0].mean().item() if (y == 0).any() else 0.0,
        "mean_log_vol_c1": log_vol[y == 1].mean().item() if (y == 1).any() else 0.0,
    }


def anisotropy_margin_loss(
    g: torch.Tensor,
    y: torch.Tensor,
    margin: float = 0.6,
    beta_hinge: float = 4.0,
    lambda_vol: float = 0.05,
    n_classes: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Purely geometric loss separating classes by directional metric polarization log(g_11 / g_22)."""
    log_ratio = torch.log(torch.clamp(g[:, 0, 0], min=1e-8) / torch.clamp(g[:, 1, 1], min=1e-8))
    loss_aniso, preds, _ = _scalar_geometric_margin_loss_and_weights(
        log_ratio, y, margin=margin, beta_hinge=beta_hinge, n_classes=n_classes
    )
    det_g = torch.clamp(torch.linalg.det(g), min=1e-8)
    loss_vol = (torch.log(det_g) ** 2).mean()
    total = loss_aniso + lambda_vol * loss_vol

    with torch.no_grad():
        acc = (preds == y).to(g.dtype).mean().item()

    return total, {
        "loss_total": total.item(),
        "geom_acc": acc,
        "mean_aniso_c0": log_ratio[y == 0].mean().item() if (y == 0).any() else 0.0,
        "mean_aniso_c1": log_ratio[y == 1].mean().item() if (y == 1).any() else 0.0,
    }


def _sym2_inv_sqrt(g: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Compute the symmetric matrix inverse square root g^{-1/2} for (N, 2, 2) SPD tensors."""
    evals, evecs = torch.linalg.eigh(g)
    inv_sqrt_evals = torch.rsqrt(torch.clamp(evals, min=eps))
    return torch.einsum("nij,nj,nkj->nik", evecs, inv_sqrt_evals, evecs)


def compute_fast_graph_geometry(
    model: MoonsMLP,
    X: torch.Tensor,
    layer_idx: int | None = None,
    params: dict[str, torch.Tensor] | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact graph pullback metric g = I_2 + J^T J and Gaussian curvature K via Gauss's equation.

    Uses 2nd-order forward-mode spatial derivatives jacfwd(jacfwd(phi)) rather than 3rd-order
    reverse-mode metric derivatives, matching compute_layer_geometry(..., metric_mode='graph')
    to float64 machine precision while accelerating parameter-space curvature Jacobians ~45x.
    """
    if layer_idx is None:
        layer_idx = model.num_layers
    phi_single = _make_single_point_layer_fn(model, layer_idx, params=params)
    jac_fn = jacfwd(phi_single)
    hess_fn = jacfwd(jac_fn)

    def single_geom(x_single: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        J = jac_fn(x_single)  # (d_l, 2)
        H = hess_fn(x_single)  # (d_l, 2, 2)
        eye2 = torch.eye(2, dtype=x_single.dtype, device=x_single.device)
        g = eye2 + J.T @ J

        det_g = g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0]
        det_g_safe = torch.where(
            torch.abs(det_g) < eps,
            torch.sign(det_g + 1e-15) * eps,
            det_g,
        )
        g_inv = torch.stack(
            [
                torch.stack([g[1, 1], -g[0, 1]]),
                torch.stack([-g[1, 0], g[0, 0]]),
            ]
        ) / det_g_safe

        gamma_11 = J.T @ H[:, 0, 0]
        gamma_22 = J.T @ H[:, 1, 1]
        gamma_12 = J.T @ H[:, 0, 1]

        r_extrinsic = torch.dot(H[:, 0, 0], H[:, 1, 1]) - torch.dot(H[:, 0, 1], H[:, 0, 1])
        r_connection = torch.dot(gamma_12, g_inv @ gamma_12) - torch.dot(gamma_11, g_inv @ gamma_22)
        K = (r_extrinsic + r_connection) / det_g_safe
        return g, K

    return vmap(single_geom)(X)


def compute_uncoupled_coord_metric(
    model: MoonsMLP,
    X: torch.Tensor,
    damping: float = 1e-2,
) -> torch.Tensor:
    """Compute the uncoupled P x P output-coordinate pullback metric E_x[(J_theta z)^T (J_theta z)] + damping * I."""
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = dict(model.named_buffers())
    param_keys = list(params.keys())
    N = X.shape[0]

    def z_flat(p_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return functional_call(model, (p_dict, buffers), (X,)).reshape(-1)

    J_dict = jacfwd(z_flat)(params)
    J_flat = torch.cat([J_dict[k].reshape(-1, params[k].numel()) for k in param_keys], dim=-1)
    P = J_flat.shape[1]
    return (J_flat.T @ J_flat) / float(N) + damping * torch.eye(P, dtype=X.dtype, device=X.device)


def compute_coupled_dewitt_metric(
    model: MoonsMLP,
    X: torch.Tensor,
    metric_mode: MetricMode = "graph",
    eps: float = 1e-6,
    inv_eps: float = 1e-3,
    volume_weighted: bool = False,
    damping: float = 1e-2,
) -> torch.Tensor:
    """Compute the P x P affine-invariant DeWitt / Ebin parameter metric coupled to data-space metric g(x; theta).

    Pulls back the canonical Riemannian metric on Sym^+_2:
        G^{Theta,DeWitt}_{mu,nu}(theta) = E_x[ w(x) Tr( g(x)^{-1} d_mu g(x) g(x)^{-1} d_nu g(x) ) ] + damping * I
    Automatically quotients out rigid output motions z -> Q z + b (E(2) = O(2) x R^2).
    """
    params = {k: v.detach() for k, v in model.named_parameters()}
    param_keys = list(params.keys())
    N = X.shape[0]

    with torch.no_grad():
        g0 = compute_layer_pullback_metric(
            model, X, layer_idx=model.num_layers, metric_mode=metric_mode, eps=eps, params=params
        )
        g0_inv_sqrt = _sym2_inv_sqrt(g0, eps=inv_eps)
        if volume_weighted:
            vol = torch.sqrt(torch.clamp(torch.linalg.det(g0), min=1e-12))
            w_vol = torch.sqrt(vol / vol.mean())
        else:
            w_vol = torch.ones(N, dtype=X.dtype, device=X.device)

    def whitened_metric_flat(p_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        g = compute_layer_pullback_metric(
            model, X, layer_idx=model.num_layers, metric_mode=metric_mode, eps=eps, params=p_dict
        )
        M = torch.einsum("nij,njk,nkl->nil", g0_inv_sqrt, g, g0_inv_sqrt) * w_vol[:, None, None]
        return M.reshape(-1)

    J_dict = jacfwd(whitened_metric_flat)(params)
    J_flat = torch.cat([J_dict[k].reshape(4 * N, -1) for k in param_keys], dim=-1)
    P = J_flat.shape[1]
    return (J_flat.T @ J_flat) / float(N) + damping * torch.eye(P, dtype=X.dtype, device=X.device)


def compute_coupled_geometric_metric(
    model: MoonsMLP,
    X: torch.Tensor,
    y: torch.Tensor,
    task: GeometricTask,
    margin: float = 0.5,
    beta_hinge: float = 4.0,
    dewitt_weight: float = 0.20,
    inv_eps: float = 1e-3,
    k_clip: float = 10.0,
    lambda_vol: float | None = None,
    lambda_iso: float = 0.005,
    damping: float = 2e-2,
    n_classes: int | None = None,
) -> torch.Tensor:
    """Compute the task-coupled P x P parameter metric induced by data-space geometry and target invariant.

    Combines the affine-invariant DeWitt metric pullback on Met(M) with the Generalized Gauss-Newton
    pullback of the target geometric field psi(x; theta) in {K, log sqrt(det g), log(g_11 / g_22)}.
    """
    params = {k: v.detach() for k, v in model.named_parameters()}
    param_keys = list(params.keys())
    N = X.shape[0]

    if lambda_vol is None:
        lambda_vol = {"curvature": 0.02, "volume": 0.02, "anisotropy": 0.05}[task]

    def eval_geom_components(
        p_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if task == "curvature":
            g, K = compute_fast_graph_geometry(model, X, params=p_dict)
            psi = k_clip * torch.tanh(K / k_clip)
            det_g = torch.clamp(torch.linalg.det(g), min=1e-8)
            r_vol = math.sqrt(2.0 * lambda_vol) * torch.log(det_g).unsqueeze(-1)
            eye2 = torch.eye(2, dtype=X.dtype, device=X.device).unsqueeze(0)
            r_iso = math.sqrt(2.0 * lambda_iso) * (g - eye2).reshape(N, 4)
            return psi, g, torch.cat([r_vol, r_iso], dim=-1)
        if task == "volume":
            g = compute_layer_pullback_metric(
                model, X, layer_idx=model.num_layers, metric_mode="euclidean", eps=1e-6, params=p_dict
            )
            psi = 0.5 * torch.log(torch.clamp(torch.linalg.det(g), min=1e-8))
            r_reg = math.sqrt(2.0 * lambda_vol) * psi.unsqueeze(-1)
            return psi, g, r_reg
        if task == "anisotropy":
            g = compute_layer_pullback_metric(
                model, X, layer_idx=model.num_layers, metric_mode="graph", params=p_dict
            )
            psi = torch.log(torch.clamp(g[:, 0, 0], min=1e-8) / torch.clamp(g[:, 1, 1], min=1e-8))
            det_g = torch.clamp(torch.linalg.det(g), min=1e-8)
            r_reg = math.sqrt(2.0 * lambda_vol) * torch.log(det_g).unsqueeze(-1)
            return psi, g, r_reg
        raise ValueError(f"Unsupported geometric task: {task}")

    with torch.no_grad():
        psi0, g0, _ = eval_geom_components(params)
        _, _, w_psi = _scalar_geometric_margin_loss_and_weights(
            psi0, y, margin=margin, beta_hinge=beta_hinge, n_classes=n_classes
        )
        g0_inv_sqrt = _sym2_inv_sqrt(g0, eps=inv_eps)
        w_dw = math.sqrt(dewitt_weight)

    def coupled_feature_vector(p_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        psi, g, extra_reg = eval_geom_components(p_dict)
        f_psi = (psi * w_psi).unsqueeze(-1)
        M = torch.einsum("nij,njk,nkl->nil", g0_inv_sqrt, g, g0_inv_sqrt) * w_dw
        return torch.cat([f_psi, M.reshape(N, 4), extra_reg], dim=-1).reshape(-1)

    J_dict = jacfwd(coupled_feature_vector)(params)
    J_flat = torch.cat([J_dict[k].reshape(-1, params[k].numel()) for k in param_keys], dim=-1)
    P = J_flat.shape[1]
    return (J_flat.T @ J_flat) / float(N) + damping * torch.eye(P, dtype=X.dtype, device=X.device)


# =============================================================================
# First-Order Efficient Approximations for Spatial & Coupled Geometric Training
# =============================================================================


def _eval_model_batch(
    model: MoonsMLP,
    X_batch: torch.Tensor,
    params: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if params is None:
        return model(X_batch)
    buffers = dict(model.named_buffers())
    return functional_call(model, (params, buffers), (X_batch,))


def compute_fd_pullback_metric(
    model: MoonsMLP,
    X: torch.Tensor,
    h: float = 1e-3,
    metric_mode: MetricMode = "euclidean",
    eps: float = 1e-6,
    params: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute d_0 x d_0 pullback metric via a single batched forward-difference pass ((d_0 + 1)*N).

    Eliminates symbolic spatial autodiff (jacrev/jacfwd) so loss.backward() wrt theta is purely 1st-order.
    """
    N, d0 = X.shape
    basis = torch.eye(d0, dtype=X.dtype, device=X.device) * h
    X_shifted = (X[:, None, :] + basis[None, :, :]).reshape(N * d0, d0)
    Z_all = _eval_model_batch(model, torch.cat([X, X_shifted], dim=0), params=params)
    z0 = Z_all[:N]
    z_shift = Z_all[N:].reshape(N, d0, -1)
    J = ((z_shift - z0[:, None, :]) / h).transpose(-1, -2)  # (N, d_out, d0)
    jtj = torch.einsum("nai,naj->nij", J, J)
    eye = torch.eye(d0, dtype=X.dtype, device=X.device).unsqueeze(0)
    if metric_mode == "graph":
        return eye + jtj
    if metric_mode == "euclidean":
        return jtj + eps * eye
    raise ValueError(f"Unsupported metric_mode for FD stencil: {metric_mode}")


def _plane_7pt_curvature(
    model: MoonsMLP,
    X: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    h: float = 2e-3,
    eps: float = 1e-8,
    params: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate 2D graph metric and Gaussian curvature on a 2-plane spanned by orthonormal (v1, v2) via 7-pt FD."""
    N = X.shape[0]
    s1 = h * v1
    s2 = h * v2
    s12 = s1 + s2
    X_all = torch.cat([X, X + s1, X - s1, X + s2, X - s2, X + s12, X - s12], dim=0)
    Z_all = _eval_model_batch(model, X_all, params=params).reshape(7, N, -1)
    z0, zp1, zm1, zp2, zm2, zp12, zm12 = (
        Z_all[0],
        Z_all[1],
        Z_all[2],
        Z_all[3],
        Z_all[4],
        Z_all[5],
        Z_all[6],
    )

    d1 = (zp1 - zm1) / (2.0 * h)
    d2 = (zp2 - zm2) / (2.0 * h)
    h11 = (zp1 - 2.0 * z0 + zm1) / (h * h)
    h22 = (zp2 - 2.0 * z0 + zm2) / (h * h)
    h12 = (zp12 - zp1 - zp2 + 2.0 * z0 - zm1 - zm2 + zm12) / (2.0 * h * h)

    J = torch.stack([d1, d2], dim=-1)  # (N, d_out, 2)
    eye2 = torch.eye(2, dtype=X.dtype, device=X.device).unsqueeze(0)
    g = eye2 + torch.einsum("nai,naj->nij", J, J)

    det_g = torch.clamp(g[:, 0, 0] * g[:, 1, 1] - g[:, 0, 1] * g[:, 1, 0], min=eps)
    g_inv = (
        torch.stack(
            [
                torch.stack([g[:, 1, 1], -g[:, 0, 1]], dim=-1),
                torch.stack([-g[:, 1, 0], g[:, 0, 0]], dim=-1),
            ],
            dim=-2,
        )
        / det_g[:, None, None]
    )

    gamma_11 = torch.einsum("nai,na->ni", J, h11)
    gamma_22 = torch.einsum("nai,na->ni", J, h22)
    gamma_12 = torch.einsum("nai,na->ni", J, h12)

    r_ext = (h11 * h22).sum(-1) - (h12 * h12).sum(-1)
    r_conn = torch.einsum("ni,nij,nj->n", gamma_12, g_inv, gamma_12) - torch.einsum(
        "ni,nij,nj->n", gamma_11, g_inv, gamma_22
    )
    K = (r_ext + r_conn) / det_g
    return g, K


def compute_fd_graph_curvature(
    model: MoonsMLP,
    X: torch.Tensor,
    h: float = 2e-3,
    eps: float = 1e-8,
    params: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 2D graph metric and Gaussian curvature K via a 7-point central stencil (pure 1st-order autodiff)."""
    if X.shape[1] != 2:
        raise ValueError(f"Coordinate 7-pt curvature stencil expects 2D inputs, got d_0={X.shape[1]}")
    e1 = torch.tensor([1.0, 0.0], dtype=X.dtype, device=X.device)
    e2 = torch.tensor([0.0, 1.0], dtype=X.dtype, device=X.device)
    return _plane_7pt_curvature(model, X, e1, e2, h=h, eps=eps, params=params)


def build_knn_jet_operator(
    X: torch.Tensor,
    k_neighbors: int = 14,
    order: Literal[1, 2] = 2,
    ridge: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute Gaussian-weighted local Taylor least-squares operators on a point cloud X (once at t = 0).

    - order = 1: returns (knn_idx, lin_op) of shape ((N, k), (N, d_0, k)) for 1st-order Jacobians.
    - order = 2: (for d_0 = 2) returns (knn_idx, jet_op) of shape ((N, k), (N, 5, k)) for
      [d_1, d_2, d_11, d_12, d_22].
    """
    dist_sq = torch.cdist(X, X) ** 2
    _, knn_idx = torch.topk(dist_sq, k=k_neighbors + 1, largest=False, dim=-1)
    knn_idx = knn_idx[:, 1:]  # (N, k)
    dX = X[knn_idx] - X[:, None, :]  # (N, k, d0)

    sigma2 = dX.pow(2).sum(-1).median(dim=-1, keepdim=True).values + 1e-6
    w = torch.exp(-dX.pow(2).sum(-1) / (2.0 * sigma2))  # (N, k)

    if order == 1:
        Phi = dX  # (N, k, d0)
    elif order == 2:
        if X.shape[1] != 2:
            raise ValueError("Quadratic order=2 jet operator expects 2D inputs.")
        dx1, dx2 = dX[:, :, 0], dX[:, :, 1]
        Phi = torch.stack([dx1, dx2, 0.5 * dx1**2, dx1 * dx2, 0.5 * dx2**2], dim=-1)  # (N, k, 5)
    else:
        raise ValueError(f"order must be 1 or 2, got {order}")

    m_dim = Phi.shape[-1]
    W_Phi = Phi * w.unsqueeze(-1)
    gram = torch.einsum("nka,nkb->nab", Phi, W_Phi) + ridge * torch.eye(
        m_dim, dtype=X.dtype, device=X.device
    ).unsqueeze(0)
    jet_op = torch.linalg.solve(gram, W_Phi.transpose(-1, -2))  # (N, m_dim, k)
    return knn_idx, jet_op


def compute_knn_jet_geometry(
    model: MoonsMLP,
    X: torch.Tensor,
    knn_idx: torch.Tensor,
    jet_op: torch.Tensor,
    metric_mode: MetricMode = "graph",
    eps: float = 1e-6,
    params: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute pullback metric g (and Gaussian curvature K if jet_op is order 2) from a SINGLE 1N forward pass."""
    Z = _eval_model_batch(model, X, params=params)  # (N, d_out) - exact 1N batch size!
    dZ = Z[knn_idx] - Z[:, None, :]  # (N, k, d_out)
    jets = torch.einsum("nmk,nka->nma", jet_op, dZ)  # (N, m_dim, d_out)

    d0 = X.shape[1]
    eye = torch.eye(d0, dtype=X.dtype, device=X.device).unsqueeze(0)

    if jet_op.shape[1] == d0:
        # Order-1 linear Jacobian operator: jets has shape (N, d0, d_out)
        J = jets.transpose(-1, -2)  # (N, d_out, d0)
        jtj = torch.einsum("nai,naj->nij", J, J)
        g = (eye + jtj) if metric_mode == "graph" else (jtj + eps * eye)
        return g, None

    # Order-2 quadratic 2D jet: [d1, d2, h11, h12, h22]
    d1, d2, h11, h12, h22 = jets[:, 0], jets[:, 1], jets[:, 2], jets[:, 3], jets[:, 4]
    J = torch.stack([d1, d2], dim=-1)  # (N, d_out, 2)
    jtj = torch.einsum("nai,naj->nij", J, J)
    g = (eye + jtj) if metric_mode == "graph" else (jtj + eps * eye)

    det_g = torch.clamp(g[:, 0, 0] * g[:, 1, 1] - g[:, 0, 1] * g[:, 1, 0], min=1e-8)
    g_inv = (
        torch.stack(
            [
                torch.stack([g[:, 1, 1], -g[:, 0, 1]], dim=-1),
                torch.stack([-g[:, 1, 0], g[:, 0, 0]], dim=-1),
            ],
            dim=-2,
        )
        / det_g[:, None, None]
    )

    gamma_11 = torch.einsum("nai,na->ni", J, h11)
    gamma_22 = torch.einsum("nai,na->ni", J, h22)
    gamma_12 = torch.einsum("nai,na->ni", J, h12)

    r_ext = (h11 * h22).sum(-1) - (h12 * h12).sum(-1)
    r_conn = torch.einsum("ni,nij,nj->n", gamma_12, g_inv, gamma_12) - torch.einsum(
        "ni,nij,nj->n", gamma_11, g_inv, gamma_22
    )
    K = (r_ext + r_conn) / det_g
    return g, K


def compute_sketched_pullback_metric(
    model: MoonsMLP,
    X: torch.Tensor,
    n_probes: int = 1,
    h: float = 1e-3,
    metric_mode: MetricMode = "euclidean",
    eps: float = 1e-6,
    generator: torch.Generator | None = None,
    params: dict[str, torch.Tensor] | None = None,
    return_ambient: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate pullback metric via m random orthonormal directional probes ((m + 1)*N forward evaluations).

    Returns (g_ambient_est, g_subspace), keeping compute O(m) independent of ambient input dimension d_0.
    Set return_ambient=False when only g_subspace is needed to avoid O(N * d_0^2) ambient tensor memory.
    """
    N, d0 = X.shape
    raw = torch.randn(d0, n_probes, generator=generator, dtype=X.dtype, device=X.device)
    Q, _ = torch.linalg.qr(raw)  # (d0, n_probes)
    X_shifts = (X[:, None, :] + h * Q.T[None, :, :]).reshape(N * n_probes, d0)
    Z_all = _eval_model_batch(model, torch.cat([X, X_shifts], dim=0), params=params)
    z0 = Z_all[:N]
    dz = (Z_all[N:].reshape(N, n_probes, -1) - z0[:, None, :]) / h  # (N, n_probes, d_out)

    g_sub = torch.einsum("npa,nqa->npq", dz, dz)  # (N, n_probes, n_probes)
    if not return_ambient:
        return g_sub, g_sub
    g_est = (float(d0) / float(n_probes)) * torch.einsum("ip,npq,jq->nij", Q, g_sub, Q)
    eye = torch.eye(d0, dtype=X.dtype, device=X.device).unsqueeze(0)
    g_out = (eye + g_est) if metric_mode == "graph" else (g_est + eps * eye)
    return g_out, g_sub


def compute_sketched_plane_curvature(
    model: MoonsMLP,
    X: torch.Tensor,
    h: float = 2e-3,
    eps: float = 1e-8,
    generator: torch.Generator | None = None,
    params: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate 2-plane sectional/Gaussian curvature along a random orthonormal 2-frame in R^{d_0} (7N evals, O(1) in d_0)."""
    d0 = X.shape[1]
    raw = torch.randn(d0, 2, generator=generator, dtype=X.dtype, device=X.device)
    Q, _ = torch.linalg.qr(raw)
    return _plane_7pt_curvature(model, X, Q[:, 0], Q[:, 1], h=h, eps=eps, params=params)


def woodbury_coupled_ngd_direction(
    model: MoonsMLP,
    X_sub: torch.Tensor,
    y_sub: torch.Tensor,
    grad_flat: torch.Tensor,
    task: GeometricTask = "curvature",
    margin: float = 0.40,
    beta_hinge: float = 4.0,
    dewitt_weight: float = 0.20,
    damping: float = 3e-2,
    n_classes: int | None = None,
) -> tuple[torch.Tensor, float]:
    """Matrix-free O(R^2 P) Woodbury Natural Gradient direction using 1st-order FD stencils on a subsample X_sub.

    Avoids ever allocating a P x P parameter metric matrix by solving an R x R system (R << P) via:
        (damping * I_P + (1/S) U^T U)^{-1} g = (1/damping) * [g - U^T (damping * S * I_R + U U^T)^{-1} (U g)]
    Returns (nat_dir, riem_norm).
    """
    params = {k: v.detach() for k, v in model.named_parameters()}
    param_keys = list(params.keys())
    S = X_sub.shape[0]

    def fd_sub_terms(p_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if task == "curvature":
            g, K = compute_fd_graph_curvature(model, X_sub, params=p_dict)
            psi = 10.0 * torch.tanh(K / 10.0)
            return psi, g
        if task == "volume":
            g = compute_fd_pullback_metric(model, X_sub, metric_mode="euclidean", params=p_dict)
            psi = 0.5 * torch.log(torch.clamp(torch.linalg.det(g), min=1e-8))
            return psi, g
        if task == "anisotropy":
            g = compute_fd_pullback_metric(model, X_sub, metric_mode="graph", params=p_dict)
            psi = torch.log(torch.clamp(g[:, 0, 0], min=1e-8) / torch.clamp(g[:, 1, 1], min=1e-8))
            return psi, g
        raise ValueError(task)

    with torch.no_grad():
        psi0, g0 = fd_sub_terms(params)
        _, _, w_psi = _scalar_geometric_margin_loss_and_weights(
            psi0, y_sub, margin=margin, beta_hinge=beta_hinge, n_classes=n_classes
        )
        g0_inv_sqrt = _sym2_inv_sqrt(g0, eps=1e-3)
        w_dw = math.sqrt(dewitt_weight)

    def sketch_features(p_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        psi, g = fd_sub_terms(p_dict)
        f_psi = (psi * w_psi).unsqueeze(-1)  # (S, 1)
        M = torch.einsum("nij,njk,nkl->nil", g0_inv_sqrt, g, g0_inv_sqrt) * w_dw
        # Symmetric 2x2 has 3 independent components: M_00, sqrt(2)*M_01, M_11
        f_dw = torch.stack([M[:, 0, 0], math.sqrt(2.0) * M[:, 0, 1], M[:, 1, 1]], dim=-1)  # (S, 3)
        return torch.cat([f_psi, f_dw], dim=-1).reshape(-1)  # (4*S,)

    J_dict = jacfwd(sketch_features)(params)
    U = torch.cat([J_dict[k].reshape(-1, params[k].numel()) for k in param_keys], dim=-1)  # (R, P), R = 4*S
    R_dim = U.shape[0]

    # Woodbury solve: O(R^2 P + R^3) with zero P x P matrix allocation
    Ug = U @ grad_flat  # (R,)
    inner = (damping * float(S)) * torch.eye(R_dim, dtype=X_sub.dtype, device=X_sub.device) + (U @ U.T)
    v = torch.linalg.solve(inner, Ug)
    nat_dir = (grad_flat - U.T @ v) / damping

    # Riemannian norm sqrt(nat_dir^T G nat_dir) computed matrix-free:
    U_nd = U @ nat_dir
    quad = damping * (nat_dir @ nat_dir) + (U_nd @ U_nd) / float(S)
    riem_norm = torch.sqrt(torch.clamp(quad, min=1e-12)).item()
    return nat_dir, riem_norm


# =============================================================================
# S_C-Symmetric Parameter-Data Coupled Multiclass Geometric Losses (Order-Free)
# =============================================================================


def compute_dewitt_dist_sq(
    g: torch.Tensor,
    proto_inv_sqrt: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Closed-form squared affine-invariant (DeWitt) geodesic distance on Sym^+_2.

    Given sample metrics g of shape (N, 2, 2) and prototype inverse square roots
    (G^*_c)^{-1/2} of shape (C, 2, 2), returns D2 of shape (N, C) with entries:
        D2[n, c] = || log( (G^*_c)^{-1/2} g[n] (G^*_c)^{-1/2} ) ||_F^2.
    """
    M = torch.einsum("cij,njk,ckl->ncil", proto_inv_sqrt, g, proto_inv_sqrt)
    tr = M[..., 0, 0] + M[..., 1, 1]
    det = torch.clamp(M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0], min=eps * eps)
    half_tr = 0.5 * torch.clamp(tr, min=2.0 * eps)
    disc = torch.sqrt(torch.clamp(half_tr * half_tr - det, min=0.0) + 1e-12)
    lam1 = torch.clamp(half_tr + disc, min=eps)
    lam2 = torch.clamp(det / lam1, min=eps)
    return torch.log(lam1) ** 2 + torch.log(lam2) ** 2


def build_dewitt_simplex_prototypes(
    n_classes: int = 3,
    tau: float = 0.75,
    graph_margin: float = 0.35,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct C S_C-symmetric prototype metrics G^*_c in {G > I_2} subset Sym^+_2.

    For C in {2, 3, 4}, all C*(C-1)/2 pairwise affine-invariant DeWitt geodesic distances
    d_AI(G^*_a, G^*_b) are strictly equal to float64 machine precision:
      - C = 2: Orthogonal Tissot ellipses (0 deg, 90 deg).
      - C = 3: Equilateral hyperbolic triangle on the Poincare anisotropy disk (0, 60, 120 deg).
      - C = 4: Regular 3D Riemannian tetrahedron in Sym^+_2 = R_vol x H^2.
    Setting graph_margin > 0 ensures min_eig(G^*_c) = exp(graph_margin) > 1 so every prototype
    is exactly realizable by the graph pullback metric g(x) = I_2 + J(x)^T J(x).
    Returns (prototypes, proto_inv_sqrt), each of shape (C, 2, 2).
    """
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")

    def rot_aniso(theta: float, r: float, log_scale: float) -> torch.Tensor:
        c_val, s_val = math.cos(theta), math.sin(theta)
        R = torch.tensor([[c_val, -s_val], [s_val, c_val]], dtype=dtype, device=device)
        D = torch.diag(
            torch.tensor([math.exp(log_scale + r), math.exp(log_scale - r)], dtype=dtype, device=device)
        )
        return R @ D @ R.T

    base_s = tau + graph_margin
    if n_classes == 2:
        protos = torch.stack([rot_aniso(0.0, tau, base_s), rot_aniso(0.5 * math.pi, tau, base_s)], dim=0)
    elif n_classes == 3:
        protos = torch.stack([rot_aniso(math.pi * c / 3.0, tau, base_s) for c in range(3)], dim=0)
    elif n_classes == 4:
        g0 = rot_aniso(0.0, tau, 0.0)
        g1 = rot_aniso(math.pi / 3.0, tau, 0.0)
        g0_isq = _sym2_inv_sqrt(g0.unsqueeze(0), eps=1e-14)
        evals_01 = torch.linalg.eigvalsh(g0_isq[0] @ g1 @ g0_isq[0])
        d_tri_sq = (torch.log(evals_01) ** 2).sum().item()
        ds = math.sqrt(max((d_tri_sq - 2.0 * tau * tau) / 2.0, 1e-14))
        protos = torch.stack(
            [
                rot_aniso(0.0, tau, base_s),
                rot_aniso(math.pi / 3.0, tau, base_s),
                rot_aniso(2.0 * math.pi / 3.0, tau, base_s),
                rot_aniso(0.0, 0.0, base_s + ds),
            ],
            dim=0,
        )
    else:
        protos = torch.stack(
            [
                rot_aniso(math.pi * c / n_classes, tau, base_s + (0.0 if c % 2 == 0 else tau))
                for c in range(n_classes)
            ],
            dim=0,
        )
    return protos, _sym2_inv_sqrt(protos, eps=1e-14)


def dewitt_simplex_predict(
    g: torch.Tensor,
    proto_inv_sqrt: torch.Tensor,
) -> torch.Tensor:
    """Classify points from their pullback metric g(x) by nearest DeWitt geodesic prototype."""
    return compute_dewitt_dist_sq(g, proto_inv_sqrt).argmin(dim=-1)


def dewitt_simplex_loss(
    g: torch.Tensor,
    y: torch.Tensor,
    proto_inv_sqrt: torch.Tensor,
    temperature: float = 2.5,
    anchor_weight: float = 0.12,
) -> tuple[torch.Tensor, dict[str, float]]:
    """S_C-equivariant Riemannian Voronoi + Frechet anchor loss on (Sym^+_2, d_AI).

    Eliminates 1D class ordering and Intermediate-Value Theorem phantom boundaries at
    identical O(N) compute and memory cost to scalar Volume/Anisotropy losses.
    """
    d2 = compute_dewitt_dist_sq(g, proto_inv_sqrt)
    loss_voronoi = F.cross_entropy(-0.5 * temperature * d2, y)
    loss_anchor = d2.gather(1, y.unsqueeze(1)).mean()
    total = loss_voronoi + anchor_weight * loss_anchor
    with torch.no_grad():
        preds = d2.argmin(dim=-1)
        acc = (preds == y).to(g.dtype).mean().item()
    return total, {
        "loss_total": total.item(),
        "loss_voronoi": loss_voronoi.item(),
        "mean_dewitt_d2": loss_anchor.item(),
        "geom_acc": acc,
    }


def build_metric_barrier_pairs(
    X: torch.Tensor,
    y: torch.Tensor,
    k_same: int = 8,
    k_diff: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute k-NN same-class and cross-class index pairs once at t = 0 for geodesic barrier training."""
    dist = torch.cdist(X, X)
    N = X.shape[0]
    same_mask = y[:, None] == y[None, :]
    same_mask.fill_diagonal_(False)
    diff_mask = ~(y[:, None] == y[None, :])

    inf = torch.tensor(1e9, dtype=X.dtype, device=X.device)
    d_same = torch.where(same_mask, dist, inf)
    d_diff = torch.where(diff_mask, dist, inf)

    ks = min(k_same, int(same_mask.sum(dim=-1).min().item()))
    kd = min(k_diff, int(diff_mask.sum(dim=-1).min().item()))
    idx_same = torch.topk(d_same, k=ks, largest=False, dim=-1).indices
    idx_diff = torch.topk(d_diff, k=kd, largest=False, dim=-1).indices
    src_same = torch.arange(N, device=X.device).unsqueeze(1).expand_as(idx_same).reshape(-1)
    src_diff = torch.arange(N, device=X.device).unsqueeze(1).expand_as(idx_diff).reshape(-1)
    return src_same, idx_same.reshape(-1), src_diff, idx_diff.reshape(-1)


def geodesic_barrier_loss(
    g: torch.Tensor,
    X: torch.Tensor,
    src_same: torch.Tensor,
    dst_same: torch.Tensor,
    src_diff: torch.Tensor,
    dst_diff: torch.Tensor,
    margin: float = 1.5,
    beta_hinge: float = 4.0,
    lambda_vol: float = 0.005,
) -> tuple[torch.Tensor, dict[str, float]]:
    """S_C-invariant relational geodesic barrier loss via trapezoidal endpoint quadrature.

    Contracts Riemannian path energy inside every class symmetrically while erecting high-metric
    walls between distinct classes, reusing the single 1N metric evaluation g(X).
    """
    dx_s = X[dst_same] - X[src_same]
    g_bar_s = 0.5 * (g[src_same] + g[dst_same])
    e_same = torch.einsum("pi,pij,pj->p", dx_s, g_bar_s, dx_s)

    dx_d = X[dst_diff] - X[src_diff]
    g_bar_d = 0.5 * (g[src_diff] + g[dst_diff])
    d_diff = torch.sqrt(torch.clamp(torch.einsum("pi,pij,pj->p", dx_d, g_bar_d, dx_d), min=1e-10))

    loss_pull = e_same.mean()
    loss_push = F.softplus(margin - d_diff, beta=beta_hinge).mean()
    det_g = torch.clamp(torch.linalg.det(g), min=1e-8)
    loss_reg = (torch.log(det_g) ** 2).mean()
    total = loss_pull + loss_push + lambda_vol * loss_reg
    return total, {
        "loss_total": total.item(),
        "mean_d_same": torch.sqrt(torch.clamp(e_same, min=1e-10)).mean().item(),
        "mean_d_diff": d_diff.mean().item(),
    }


def geodesic_knn_predict(
    model: MoonsMLP,
    X_eval: torch.Tensor,
    X_ref: torch.Tensor,
    y_ref: torch.Tensor,
    n_classes: int,
    k_eval: int = 9,
    n_quad: int = 4,
    metric_mode: MetricMode = "euclidean",
) -> torch.Tensor:
    """Classify points X_eval by inverse-geodesic-distance k-NN voting to training anchors X_ref."""
    with torch.no_grad():
        eucl_d = torch.cdist(X_eval, X_ref)
        cand_idx = torch.topk(eucl_d, k=min(24, X_ref.shape[0]), largest=False, dim=-1).indices
        M, Kc = cand_idx.shape
        t_vals = torch.linspace(0.0, 1.0, n_quad, dtype=X_eval.dtype, device=X_eval.device)
        dx = X_ref[cand_idx] - X_eval[:, None, :]
        pts = X_eval[:, None, None, :] + t_vals[None, None, :, None] * dx[:, :, None, :]
        g_pts = compute_layer_pullback_metric(
            model,
            pts.reshape(-1, X_eval.shape[1]),
            layer_idx=model.num_layers,
            metric_mode=metric_mode,
            eps=1e-6,
        ).reshape(M, Kc, n_quad, X_eval.shape[1], X_eval.shape[1])
        seg_len = torch.sqrt(
            torch.clamp(torch.einsum("mki,mkqij,mkj->mkq", dx, g_pts, dx), min=1e-10)
        ).mean(dim=-1)
        best_local = torch.topk(seg_len, k=min(k_eval, Kc), largest=False, dim=-1)
        nbr_labels = y_ref[cand_idx.gather(1, best_local.indices)]
        weights = 1.0 / (best_local.values + 1e-4)
        votes = torch.zeros(M, n_classes, dtype=X_eval.dtype, device=X_eval.device)
        votes.scatter_add_(1, nbr_labels, weights)
        return votes.argmax(dim=-1)


def compute_coupled_subspace_factors(
    model: MoonsMLP,
    X_batch: torch.Tensor,
    use_fd: bool = True,
    metric_mode: MetricMode = "euclidean",
    damping: float = 3e-2,
    whitener_inv_sqrt: torch.Tensor | None = None,
    plane_sketch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute G^Theta-whitened DeWitt sensitivity subspaces W_hat(x_i) in R^{N x 3 x P}.

    Each 3 x P block spans the parameter tangent directions that deform g(x_i; theta),
    orthonormalized against (G^{Theta,DeWitt}(theta) + damping * I_P)^{-1/2}.
    When plane_sketch=True (useful for high input dimension d_0 >> 2), evaluates a 2x2
    tangent-plane pullback metric in 3N forward passes before taking parameter derivatives.
    """
    params = dict(model.named_parameters())
    param_keys = list(params.keys())
    N, d0 = X_batch.shape
    eye2 = torch.eye(2, dtype=X_batch.dtype, device=X_batch.device).unsqueeze(0)

    if plane_sketch:
        gen = torch.Generator(device=X_batch.device).manual_seed(17)
        Q_plane, _ = torch.linalg.qr(
            torch.randn(d0, 2, generator=gen, dtype=X_batch.dtype, device=X_batch.device)
        )
    else:
        Q_plane = None

    def eval_metric_2x2(p_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        if plane_sketch and Q_plane is not None:
            h_step = 1e-3
            X_sh = (X_batch[:, None, :] + h_step * Q_plane.T[None, :, :]).reshape(2 * N, d0)
            Z_all = _eval_model_batch(model, torch.cat([X_batch, X_sh], dim=0), params=p_dict)
            dz = (Z_all[N:].reshape(N, 2, -1) - Z_all[:N, None, :]) / h_step
            g_sub = torch.einsum("npa,nqa->npq", dz, dz)
            return (eye2 + g_sub) if metric_mode == "graph" else (g_sub + 1e-6 * eye2)
        g_full = (
            compute_fd_pullback_metric(model, X_batch, metric_mode=metric_mode, params=p_dict)
            if use_fd
            else compute_layer_pullback_metric(
                model, X_batch, layer_idx=model.num_layers, metric_mode=metric_mode, params=p_dict
            )
        )
        return g_full[:, :2, :2]

    with torch.no_grad():
        params_det = {k: v.detach() for k, v in params.items()}
        g0 = eval_metric_2x2(params_det)
        g0_isq = _sym2_inv_sqrt(g0, eps=1e-3)

    def sym3_features(p_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        g = eval_metric_2x2(p_dict)
        M = torch.einsum("nij,njk,nkl->nil", g0_isq, g, g0_isq)
        return torch.stack([M[:, 0, 0], math.sqrt(2.0) * M[:, 0, 1], M[:, 1, 1]], dim=-1).reshape(-1)

    J_dict = jacfwd(sym3_features)(params)
    U = torch.cat([J_dict[k].reshape(N, 3, -1) for k in param_keys], dim=-1)  # (N, 3, P)

    if whitener_inv_sqrt is None:
        with torch.no_grad():
            U_det = U.detach().reshape(3 * N, -1)
            P = U_det.shape[1]
            G_dw = (U_det.T @ U_det) / float(N) + damping * torch.eye(
                P, dtype=X_batch.dtype, device=X_batch.device
            )
            eig_vals, eig_vecs = torch.linalg.eigh(G_dw)
            whitener_inv_sqrt = (
                eig_vecs @ torch.diag(torch.rsqrt(torch.clamp(eig_vals, min=1e-6))) @ eig_vecs.T
            )

    W = torch.einsum("nap,pq->naq", U, whitener_inv_sqrt)
    W_hat = W / torch.sqrt((W * W).sum(dim=(-2, -1), keepdim=True) + 1e-8)
    return W_hat, whitener_inv_sqrt


def coupled_subspace_orthogonality_loss(
    model: MoonsMLP,
    X_batch: torch.Tensor,
    y_batch: torch.Tensor,
    use_fd: bool = True,
    metric_mode: MetricMode = "euclidean",
    ortho_weight: float = 2.0,
    plane_sketch: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """S_C-symmetric parameter-data subspace alignment & orthogonality loss on (Theta, G^Theta).

    Aligns DeWitt metric-sensitivity subspaces within each class while driving distinct classes
    into mutually G^Theta-orthogonal subspaces of parameter space.
    """
    W_hat, _ = compute_coupled_subspace_factors(
        model, X_batch, use_fd=use_fd, metric_mode=metric_mode, plane_sketch=plane_sketch
    )
    cross = torch.einsum("nap,mbp->nmab", W_hat, W_hat)
    K_mat = (cross * cross).sum(dim=(-2, -1))

    N = X_batch.shape[0]
    same = (y_batch[:, None] == y_batch[None, :]).to(X_batch.dtype)
    eye = torch.eye(N, dtype=X_batch.dtype, device=X_batch.device)
    same_no_diag = same * (1.0 - eye)
    diff = 1.0 - same

    loss_align = ((1.0 - K_mat) * same_no_diag).sum() / torch.clamp(same_no_diag.sum(), min=1.0)
    loss_ortho = ((K_mat**2) * diff).sum() / torch.clamp(diff.sum(), min=1.0)
    total = loss_align + ortho_weight * loss_ortho

    with torch.no_grad():
        n_cls = int(y_batch.max().item()) + 1
        scores = []
        for c in range(n_cls):
            mask_c = (y_batch == c).to(X_batch.dtype).unsqueeze(0) * (1.0 - eye)
            scores.append((K_mat * mask_c).sum(dim=-1) / torch.clamp(mask_c.sum(dim=-1), min=1.0))
        preds = torch.stack(scores, dim=-1).argmax(dim=-1)
        acc = (preds == y_batch).to(X_batch.dtype).mean().item()
        mean_same = (K_mat * same_no_diag).sum().item() / max(same_no_diag.sum().item(), 1.0)
        mean_diff = (K_mat * diff).sum().item() / max(diff.sum().item(), 1.0)

    return total, {
        "loss_total": total.item(),
        "geom_acc": acc,
        "mean_same_overlap": mean_same,
        "mean_diff_overlap": mean_diff,
    }


def coupled_subspace_predict(
    model: MoonsMLP,
    X_eval: torch.Tensor,
    X_ref: torch.Tensor,
    y_ref: torch.Tensor,
    n_classes: int,
    use_fd: bool = False,
    metric_mode: MetricMode = "euclidean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify points X_eval by Grassmannian overlap of their DeWitt tangent subspace with class subspaces."""
    W_ref, whitener = compute_coupled_subspace_factors(
        model, X_ref, use_fd=use_fd, metric_mode=metric_mode
    )
    W_eval, _ = compute_coupled_subspace_factors(
        model, X_eval, use_fd=use_fd, metric_mode=metric_mode, whitener_inv_sqrt=whitener
    )
    with torch.no_grad():
        cross = torch.einsum("nap,mbp->nmab", W_eval.detach(), W_ref.detach())
        K_eval_ref = (cross * cross).sum(dim=(-2, -1))
        cls_scores = torch.stack(
            [K_eval_ref[:, y_ref == c].mean(dim=-1) for c in range(n_classes)],
            [
                K_eval_ref[:, y_ref == c].mean(dim=-1)
                if (y_ref == c).any()
                else torch.zeros(X_eval.shape[0], dtype=X_eval.dtype, device=X_eval.device)
                for c in range(n_classes)
            ],
            dim=-1,
        )
        return cls_scores.argmax(dim=-1), cls_scores


def fibered_dewitt_simplex_loss(
def _fibered_dewitt_prod_dist_sq(
    model: MoonsMLP,
    X: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    h: float = 1e-3,
    tau: float = 0.75,
    graph_margin: float = 0.35,
    temperature: float = 2.0,
    anchor_weight: float = 0.08,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Exact S_C-equidistant product-manifold DeWitt simplex loss on (Sym^+_2)^C in O(N C) time.

    Expects model output width d_L == 2 * n_classes (one 2D geometric fiber per class, invariant
    under E(2)^C translations and rotations of every fiber). Uses a single batched (d_0 + 1)*N
    FD pass (for d_0 = 2, 3N evaluations) and computes all C product-manifold DeWitt squared
    distances in O(N C) operations via S_rest(n) + (d^2_act(n, c) - d^2_rest(n, c)).
    """
    frame_2d: torch.Tensor | None = None,
) -> torch.Tensor:
    N, d0 = X.shape
    if d0 != 2:
        raise ValueError(f"fibered_dewitt_simplex_loss expects 2D fiber base (got d_0={d0})")
    basis = torch.eye(2, dtype=X.dtype, device=X.device) * h
    X_shifted = (X[:, None, :] + basis[None, :, :]).reshape(2 * N, 2)
    q_frame = compute_canonical_2frame(X) if frame_2d is None else frame_2d
    X_shifted = (X[:, None, :] + h * q_frame.T[None, :, :]).reshape(2 * N, d0)
    Z_all = model(torch.cat([X, X_shifted], dim=0))  # (3N, 2*C)
    z0 = Z_all[:N].reshape(N, n_classes, 2)
    z_shift = Z_all[N:].reshape(N, 2, n_classes, 2)
    # J_fibers: (N, C, 2_out, 2_in)
    J_f = ((z_shift - z0[:, None, :, :]) / h).permute(0, 2, 3, 1)
    eye2 = torch.eye(2, dtype=X.dtype, device=X.device)[None, None, :, :]
    g_fibers = eye2 + torch.einsum("ncai,ncaj->ncij", J_f, J_f)  # (N, C, 2, 2)

    # Active state diag(exp(m + 2*tau), exp(m)) vs Rest state diag(exp(m), exp(m))
    s_act = torch.tensor(
        [math.exp(-0.5 * (graph_margin + 2.0 * tau)), math.exp(-0.5 * graph_margin)],
        dtype=X.dtype,
        device=X.device,
    )
    s_rest = math.exp(-0.5 * graph_margin)
    ref_two = torch.stack([torch.diag(s_act), torch.eye(2, dtype=X.dtype, device=X.device) * s_rest], dim=0)
    d2_pair = compute_dewitt_dist_sq(g_fibers.reshape(N * n_classes, 2, 2), ref_two).reshape(N, n_classes, 2)
    d2_act = d2_pair[:, :, 0]   # (N, C)
    d2_rest = d2_pair[:, :, 1]  # (N, C)
    d2_act = d2_pair[:, :, 0]
    d2_rest = d2_pair[:, :, 1]
    return d2_rest.sum(dim=-1, keepdim=True) + (d2_act - d2_rest)

    # Product-manifold squared DeWitt distance to prototype c: sum_{k != c} d2_rest(k) + d2_act(c)
    d2_prod = d2_rest.sum(dim=-1, keepdim=True) + (d2_act - d2_rest)  # (N, C)

def fibered_dewitt_simplex_loss(
    model: MoonsMLP,
    X: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    h: float = 1e-3,
    tau: float = 0.75,
    graph_margin: float = 0.35,
    temperature: float = 2.0,
    anchor_weight: float = 0.08,
    frame_2d: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Exact S_C-equidistant product-manifold DeWitt simplex loss on (Sym^+_2)^C in O(N C) time."""
    d2_prod = _fibered_dewitt_prod_dist_sq(
        model, X, n_classes=n_classes, h=h, tau=tau, graph_margin=graph_margin, frame_2d=frame_2d
    )
    loss_voronoi = F.cross_entropy(-0.5 * temperature * d2_prod, y)
    loss_anchor = d2_prod.gather(1, y.unsqueeze(1)).mean()
    total = loss_voronoi + anchor_weight * loss_anchor
    with torch.no_grad():
        acc = (d2_prod.argmin(dim=-1) == y).to(X.dtype).mean().item()
    return total, {"loss_total": total.item(), "geom_acc": acc}


def fibered_dewitt_simplex_predict(
    model: MoonsMLP,
    X_eval: torch.Tensor,
    n_classes: int,
    h: float = 1e-3,
    tau: float = 0.75,
    graph_margin: float = 0.35,
    frame_2d: torch.Tensor | None = None,
) -> torch.Tensor:
    """Classify points X_eval by nearest exact S_C-equidistant product-manifold prototype on (Sym^+_2)^C."""
    with torch.no_grad():
        return _fibered_dewitt_prod_dist_sq(
            model, X_eval, n_classes=n_classes, h=h, tau=tau, graph_margin=graph_margin, frame_2d=frame_2d
        ).argmin(dim=-1)



# =============================================================================
# Memory-Scalable High-Dimensional (d_0 >> 2) & Many-Class (C >> 4) Coupled Losses
# =============================================================================


def sym2_to_hyperboloid(
    g: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map 2x2 SPD matrices g (..., 2, 2) via Sym^+_2 = R_vol x H^2 to (half_log_det, u_hyperboloid).

    Here half_log_det = 0.5 * log det(g) has shape (...) and u = (u_0, u_1, u_2) has shape (..., 3)
    satisfying the upper-sheet Minkowski hyperboloid identity u_0^2 - u_1^2 - u_2^2 = 1 (u_0 >= 1).
    """
    det = torch.clamp(g[..., 0, 0] * g[..., 1, 1] - g[..., 0, 1] * g[..., 1, 0], min=eps)
    inv_sqrt_det = torch.rsqrt(det)
    half_log_det = 0.5 * torch.log(det)
    g00 = g[..., 0, 0] * inv_sqrt_det
    g11 = g[..., 1, 1] * inv_sqrt_det
    g01 = g[..., 0, 1] * inv_sqrt_det
    u = torch.stack([0.5 * (g00 + g11), 0.5 * (g00 - g11), g01], dim=-1)
    return half_log_det, u


def hyperboloid_dewitt_dist_sq(
    s_g: torch.Tensor,
    u_g: torch.Tensor,
    s_ref: torch.Tensor,
    u_ref_mink: torch.Tensor,
) -> torch.Tensor:
    """Exact squared DeWitt geodesic distance on (Sym^+_2)^K via Minkowski inner products.

    Accepts broadcastable shapes (..., K) for half-log-determinants s_g, s_ref and (..., K, 3)
    for hyperboloid vectors u_g and Lorentz-signed prototype vectors u_ref_mink = (v_0, -v_1, -v_2),
    summing exact squared distances across the K fibers without (N, C, 2, 2) matrix contractions.
    """
    cosh_d = torch.clamp((u_g * u_ref_mink).sum(dim=-1), min=1.0)
    acosh_sq = torch.log(cosh_d + torch.sqrt(torch.clamp(cosh_d * cosh_d - 1.0, min=0.0) + 1e-12)) ** 2
    return (2.0 * (s_g - s_ref) ** 2 + 2.0 * acosh_sq).sum(dim=-1)


def compute_canonical_2frame(X_ref: torch.Tensor) -> torch.Tensor:
    """Compute a deterministic top-2 PCA Stiefel 2-frame Q_0 in R^{d_0 x 2} from training inputs X_ref.

    Fixing a canonical tangent 2-plane once at t = 0 avoids high-d_0 random-plane orientation washout
    while keeping pullback metric evaluation at 3N forward passes (O(1) in d_0) and E(2)^K-invariant.
    """
    d0 = X_ref.shape[1]
    if d0 == 2:
        return torch.eye(2, dtype=X_ref.dtype, device=X_ref.device)
    with torch.no_grad():
        Xc = X_ref - X_ref.mean(dim=0, keepdim=True)
        _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
        Q0 = Vh[:2].T.clone()  # (d_0, 2)
        for col in range(2):
            max_idx = int(Q0[:, col].abs().argmax().item())
            if float(Q0[max_idx, col].item()) < 0.0:
                Q0[:, col] = -Q0[:, col]
        return Q0.contiguous()


def build_log_fiber_prototypes(
    n_classes: int,
    tau: float = 0.80,
    graph_margin: float = 0.35,
    n_fibers: int | None = None,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Build C spherical-code metric prototypes on (Sym^+_2)^K with K = ceil(log2 C) fibers (d_L = 2K).

    Uses an exact centered regular (C-1)-simplex when C <= 3K + 1 and repulsive Riesz dispersion
    on S^{3K - 1} when C > 3K + 1, eliminating prototype crowding with O(log C) output width.
    Returns (K, s_proto, u_proto_mink) with shapes (C, K) and (C, K, 3).
    """
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")
    K = max(1, math.ceil(math.log2(max(n_classes, 2)))) if n_fibers is None else int(n_fibers)
    dim_tan = 3 * K

    if n_classes <= dim_tan + 1:
        # Exact regular (C - 1)-simplex inscribed in R^{3K} (all pairwise distances identical)
        centered = torch.eye(n_classes, dtype=dtype, device=device) - (1.0 / float(n_classes))
        U_svd, _, _ = torch.linalg.svd(centered, full_matrices=False)
        simp = U_svd[:, : n_classes - 1] * math.sqrt(float(n_classes) / float(n_classes - 1))
        V = torch.zeros(n_classes, dim_tan, dtype=dtype, device=device)
        # Rotate into a balanced basis across all 3K coordinates via deterministic orthogonal matrix
        gen_q = torch.Generator(device="cpu").manual_seed(seed)
        Q_bal, _ = torch.linalg.qr(torch.randn(dim_tan, dim_tan, generator=gen_q, dtype=dtype).to(device))
        V[:, : n_classes - 1] = simp
        V = V @ Q_bal.T
    else:
        with torch.no_grad():
            gen = torch.Generator(device="cpu").manual_seed(seed)
            work_dtype = torch.float32 if n_classes > 256 else dtype
            V = F.normalize(torch.randn(n_classes, dim_tan, generator=gen, dtype=work_dtype).to(device), dim=-1)
            n_iters = 100 if n_classes <= 256 else 40
            for _ in range(n_iters):
                dist_sq = torch.clamp(2.0 - 2.0 * (V @ V.T), min=1e-5)
                dist_sq.fill_diagonal_(100.0)
                inv_d4 = 1.0 / (dist_sq * dist_sq)
                inv_d4.fill_diagonal_(0.0)
                grad_v = inv_d4 @ V - inv_d4.sum(dim=-1, keepdim=True) * V
                step_dir = grad_v / torch.clamp(grad_v.norm(dim=-1, keepdim=True), min=1e-12)
                V = F.normalize(V - 0.08 * step_dir, dim=-1)
            V = V.to(dtype=dtype)

    scale = tau * math.sqrt(min(float(K), 2.5))
    V_fib = V.reshape(n_classes, K, 3) * scale
    base_s = tau + graph_margin
    s_proto = base_s + V_fib[..., 0]  # (C, K)
    r = torch.sqrt(V_fib[..., 1] ** 2 + V_fib[..., 2] ** 2 + 1e-16)
    sinh_over_r = torch.sinh(r) / r
    u_proto_mink = torch.stack(
        [torch.cosh(r), -sinh_over_r * V_fib[..., 1], -sinh_over_r * V_fib[..., 2]],
        dim=-1,
    )
    return K, s_proto, u_proto_mink


def fiber_hyperboloid_coords(
    model: MoonsMLP,
    X: torch.Tensor,
    n_fibers: int,
    frame_2d: torch.Tensor,
    h: float = 1e-3,
    metric_mode: MetricMode = "graph",
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate K 2x2 fiber pullback metrics along canonical 2-frame frame_2d in 3N forward passes."""
    N, d0 = X.shape
    X_sh = (X[:, None, :] + h * frame_2d.T[None, :, :]).reshape(2 * N, d0)
    Z_all = model(torch.cat([X, X_sh], dim=0))  # (3N, 2*K)
    z0 = Z_all[:N].reshape(N, n_fibers, 2)
    z_sh = Z_all[N:].reshape(N, 2, n_fibers, 2)
    dz = (z_sh - z0[:, None, :, :]) / h  # (N, 2_in, K, 2_out)
    g_raw = torch.einsum("npka,nqka->nkpq", dz, dz)
    eye2 = torch.eye(2, dtype=X.dtype, device=X.device)[None, None, :, :]
    g_fib = (eye2 + g_raw) if metric_mode == "graph" else (g_raw + eps * eye2)
    return sym2_to_hyperboloid(g_fib)


def log_fiber_dewitt_loss(
    model: MoonsMLP,
    X: torch.Tensor,
    y: torch.Tensor,
    n_fibers: int,
    frame_2d: torch.Tensor,
    s_proto: torch.Tensor,
    u_proto_mink: torch.Tensor,
    m_neg: int = 8,
    temperature: float = 1.5,
    anchor_weight: float = 0.12,
    h: float = 1e-3,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """O(1)-in-d_0, O(log C)-in-C Hyperboloid DeWitt loss with sampled negative prototypes.

    Evaluates K = ceil(log2 C) 2D output fibers along canonical 2-frame frame_2d (3N forward passes)
    and computes exact DeWitt distances only to the ground-truth prototype plus m_neg negatives.
    """
    N = X.shape[0]
    n_classes = s_proto.shape[0]
    s_g, u_g = fiber_hyperboloid_coords(model, X, n_fibers, frame_2d, h=h)
    d2_pos = hyperboloid_dewitt_dist_sq(s_g, u_g, s_proto[y], u_proto_mink[y])  # (N,)

    m_eff = min(m_neg, n_classes - 1)
    if m_eff == n_classes - 1:
        all_c = torch.arange(n_classes, device=X.device).unsqueeze(0).expand(N, n_classes)
        neg_idx = all_c[all_c != y.unsqueeze(1)].reshape(N, n_classes - 1)
    else:
        offsets = torch.randint(1, n_classes, (N, m_eff), generator=generator, device=X.device)
        neg_idx = (y[:, None] + offsets) % n_classes

    d2_neg = hyperboloid_dewitt_dist_sq(
        s_g[:, None, :],
        u_g[:, None, :, :],
        s_proto[neg_idx],
        u_proto_mink[neg_idx],
    )  # (N, m_eff)

    logits = -temperature * torch.cat([d2_pos[:, None], d2_neg], dim=1)
    targets = torch.zeros(N, dtype=torch.long, device=X.device)
    loss_vor = F.cross_entropy(logits, targets)
    loss_anc = d2_pos.mean()
    total = loss_vor + anchor_weight * loss_anc

    with torch.no_grad():
        sampled_acc = (d2_pos[:, None] < d2_neg).all(dim=-1).to(X.dtype).mean().item()
    return total, {
        "loss_total": total.item(),
        "mean_dewitt_d2": loss_anc.item(),
        "sampled_margin_acc": sampled_acc,
    }


def log_fiber_dewitt_predict(
    model: MoonsMLP,
    X_eval: torch.Tensor,
    n_fibers: int,
    frame_2d: torch.Tensor,
    s_proto: torch.Tensor,
    u_proto_mink: torch.Tensor,
    h: float = 1e-3,
    class_chunk: int = 64,
) -> torch.Tensor:
    """Memory-bounded nearest-prototype DeWitt inference on (Sym^+_2)^K in O(N K) working memory."""
    with torch.no_grad():
        s_g, u_g = fiber_hyperboloid_coords(model, X_eval, n_fibers, frame_2d, h=h)
        n_classes = s_proto.shape[0]
        best_d2 = torch.full((X_eval.shape[0],), float("inf"), dtype=X_eval.dtype, device=X_eval.device)
        best_cls = torch.zeros(X_eval.shape[0], dtype=torch.long, device=X_eval.device)
        for c_start in range(0, n_classes, class_chunk):
            c_end = min(c_start + class_chunk, n_classes)
            d2_blk = hyperboloid_dewitt_dist_sq(
                s_g[:, None, :],
                u_g[:, None, :, :],
                s_proto[None, c_start:c_end, :],
                u_proto_mink[None, c_start:c_end, :, :],
            )
            blk_min, blk_idx = d2_blk.min(dim=-1)
            mask = blk_min < best_d2
            best_d2 = torch.where(mask, blk_min, best_d2)
            best_cls = torch.where(mask, blk_idx + c_start, best_cls)
        return best_cls


def midpoint_polygonal_barrier_loss(
    model: MoonsMLP,
    X: torch.Tensor,
    src_same: torch.Tensor,
    dst_same: torch.Tensor,
    src_diff: torch.Tensor,
    dst_diff: torch.Tensor,
    margin: float = 1.5,
    beta_hinge: float = 4.0,
    lambda_vol: float = 0.01,
    n_sample_pairs: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """O(1)-in-d_0, O(1)-in-C 2-segment midpoint polygonal Riemannian barrier loss (3N forward passes).

    Evaluates polygonal path energy and arc length directly in output space across sampled k-NN
    same-class and cross-class midpoints without materializing any (d_0 x d_0) metric tensors.
    """
    N = X.shape[0]
    m_pairs = N if n_sample_pairs is None else int(n_sample_pairs)
    idx_s = torch.randint(0, src_same.shape[0], (m_pairs,), generator=generator, device=X.device)
    idx_d = torch.randint(0, src_diff.shape[0], (m_pairs,), generator=generator, device=X.device)
    ss, ds = src_same[idx_s], dst_same[idx_s]
    sd, dd = src_diff[idx_d], dst_diff[idx_d]

    X_mid_s = 0.5 * (X[ss] + X[ds])
    X_mid_d = 0.5 * (X[sd] + X[dd])
    Z_all = model(torch.cat([X, X_mid_s, X_mid_d], dim=0))  # (N + 2*m_pairs, d_L)
    z0 = Z_all[:N]
    zm_s = Z_all[N : N + m_pairs]
    zm_d = Z_all[N + m_pairs :]

    e_same = 2.0 * ((zm_s - z0[ss]).pow(2).sum(dim=-1) + (z0[ds] - zm_s).pow(2).sum(dim=-1))
    l_diff = torch.sqrt((zm_d - z0[sd]).pow(2).sum(dim=-1) + 1e-8) + torch.sqrt(
        (z0[dd] - zm_d).pow(2).sum(dim=-1) + 1e-8
    )

    loss_pull = e_same.mean()
    loss_push = F.softplus(margin - l_diff, beta=beta_hinge).mean()
    z_centered = z0 - z0.mean(dim=0, keepdim=True)
    z_cov = (z_centered.T @ z_centered) / float(N) + 1e-4 * torch.eye(
        z0.shape[1], dtype=X.dtype, device=X.device
    )
    loss_reg = torch.log(torch.clamp(torch.linalg.det(z_cov), min=1e-8)) ** 2
    total = loss_pull + loss_push + lambda_vol * loss_reg
    return total, {
        "loss_total": total.item(),
        "mean_d_same": torch.sqrt(torch.clamp(e_same, min=1e-10)).mean().item(),
        "mean_d_diff": l_diff.mean().item(),
    }


def polygonal_barrier_knn_predict(
    model: MoonsMLP,
    X_eval: torch.Tensor,
    X_ref: torch.Tensor,
    y_ref: torch.Tensor,
    n_classes: int,
    k_eval: int = 7,
    k_cand: int = 16,
) -> torch.Tensor:
    """Classify points X_eval by 2-segment polygonal arc-length k-NN voting without (d_0 x d_0) metrics."""
    with torch.no_grad():
        z_ref = model(X_ref)
        z_eval = model(X_eval)
        kc = min(k_cand, X_ref.shape[0])
        cand_idx = torch.topk(torch.cdist(z_eval, z_ref), k=kc, largest=False, dim=-1).indices
        M = X_eval.shape[0]
        mids = 0.5 * (X_eval[:, None, :] + X_ref[cand_idx])  # (M, kc, d_0)
        z_mids = model(mids.reshape(M * kc, -1)).reshape(M, kc, -1)
        poly_len = torch.sqrt((z_mids - z_eval[:, None, :]).pow(2).sum(dim=-1) + 1e-10) + torch.sqrt(
            (z_ref[cand_idx] - z_mids).pow(2).sum(dim=-1) + 1e-10
        )
        best_local = torch.topk(poly_len, k=min(k_eval, kc), largest=False, dim=-1)
        nbr_labels = y_ref[cand_idx.gather(1, best_local.indices)]
        weights = 1.0 / (best_local.values + 1e-4)
        votes = torch.zeros(M, n_classes, dtype=X_eval.dtype, device=X_eval.device)
        votes.scatter_add_(1, nbr_labels, weights)
        return votes.argmax(dim=-1)


def gram_subspace_orthogonality_loss(
    W_hat: torch.Tensor,
    y: torch.Tensor,
    ortho_weight: float = 2.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """O(N r + B_c r^2) Grassmannian alignment + off-diagonal orthogonality via the r x r Gram matrix.

    Uses the exact trace identity sum_{a != b} ||W_a W_b^T||_F^2 = ||sum_c W_c^T W_c||_F^2 - sum_c ||W_c W_c^T||_F^2
    on normalized class-mean subspaces W_c in R^{3 x r}, eliminating both O(N^2) and O(C^2) tensors.
    """
    _, _, r_dim = W_hat.shape
    uniq_y, inv_idx = torch.unique(y, sorted=True, return_inverse=True)
    b_cls = uniq_y.shape[0]

    counts = torch.zeros(b_cls, dtype=W_hat.dtype, device=W_hat.device).scatter_add_(
        0, inv_idx, torch.ones_like(inv_idx, dtype=W_hat.dtype)
    )
    W_cls = torch.zeros(b_cls, 3, r_dim, dtype=W_hat.dtype, device=W_hat.device)
    W_cls.index_add_(0, inv_idx, W_hat)
    W_cls = W_cls / counts[:, None, None]
    W_cls = W_cls / torch.sqrt((W_cls * W_cls).sum(dim=(-2, -1), keepdim=True) + 1e-8)

    W_target = W_cls[inv_idx]
    overlap_self = (torch.einsum("nar,nbr->nab", W_hat, W_target) ** 2).sum(dim=(-2, -1))
    loss_align = (1.0 - overlap_self).mean()

    M_flat = W_cls.reshape(3 * b_cls, r_dim)
    gram_total = M_flat.T @ M_flat  # (r_dim, r_dim)
    total_sq = (gram_total * gram_total).sum()
    diag_sq = (torch.einsum("car,cbr->cab", W_cls, W_cls) ** 2).sum()
    denom = max(float(b_cls * (b_cls - 1)), 1.0)
    off_diag_mean = torch.clamp(total_sq - diag_sq, min=0.0) / denom
    total = loss_align + ortho_weight * off_diag_mean

    return total, {
        "loss_total": total.item(),
        "mean_self_overlap": overlap_self.mean().item(),
        "mean_cross_cls_sq": off_diag_mean.item(),
    }


def compute_analytic_head_subspaces(
    model: MoonsMLP,
    X: torch.Tensor,
    frame_2d: torch.Tensor,
    h: float = 1e-3,
    damping: float = 3e-2,
    whitener_inv_sqrt: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form analytic head-parameter DeWitt subspaces U_head(x) = d(g_0^{-1/2} g g_0^{-1/2})/dW_L.

    Requires only 3N standard 1st-order forward passes (zero jacfwd, zero 2nd-order autograd),
    with subspace width P_head = 2 * d_{L-1} strictly independent of ambient input dimension d_0.
    """
    N, d0 = X.shape
    X_3N = torch.cat([X, (X[:, None, :] + h * frame_2d.T[None, :, :]).reshape(2 * N, d0)], dim=0)
    feat = model.forward_up_to_layer(X_3N, model.num_layers - 1)
    W_L = model.layers[-1].weight  # (2, d_hid)
    if W_L.shape[0] != 2:
        raise ValueError(f"compute_analytic_head_subspaces expects d_L == 2, got {W_L.shape[0]}")

    H_j = (feat[N:].reshape(N, 2, -1) - feat[:N, None, :]) / h  # (N, 2_in, d_hid)
    J_z = torch.einsum("oa,npa->npo", W_L, H_j)  # (N, 2_in, 2_out)
    eye2 = torch.eye(2, dtype=X.dtype, device=X.device).unsqueeze(0)
    g = eye2 + torch.einsum("npo,nqo->npq", J_z, J_z)

    with torch.no_grad():
        g0_isq = _sym2_inv_sqrt(g.detach(), eps=1e-4)

    H_tilde = torch.einsum("nip,npa->nia", g0_isq, H_j)  # (N, 2, d_hid)
    Z_tilde = torch.einsum("nip,npo->nio", g0_isq, J_z)  # (N, 2, 2)
    u00 = 2.0 * Z_tilde[:, 0, :, None] * H_tilde[:, 0, None, :]
    u01 = math.sqrt(2.0) * (
        Z_tilde[:, 0, :, None] * H_tilde[:, 1, None, :] + Z_tilde[:, 1, :, None] * H_tilde[:, 0, None, :]
    )
    u11 = 2.0 * Z_tilde[:, 1, :, None] * H_tilde[:, 1, None, :]
    U = torch.stack([u00.reshape(N, -1), u01.reshape(N, -1), u11.reshape(N, -1)], dim=1)  # (N, 3, 2*d_hid)
    p_head = U.shape[-1]

    if whitener_inv_sqrt is None:
        with torch.no_grad():
            U_det = U.detach().reshape(3 * N, p_head)
            G_h = (U_det.T @ U_det) / float(N) + damping * torch.eye(p_head, dtype=X.dtype, device=X.device)
            ev, evec = torch.linalg.eigh(G_h)
            whitener_inv_sqrt = evec @ torch.diag(torch.rsqrt(torch.clamp(ev, min=1e-6))) @ evec.T

    W = torch.einsum("nap,pq->naq", U, whitener_inv_sqrt)
    W_hat = W / torch.sqrt((W * W).sum(dim=(-2, -1), keepdim=True) + 1e-8)
    return W_hat, whitener_inv_sqrt


def analytic_head_subspace_loss(
    model: MoonsMLP,
    X: torch.Tensor,
    y: torch.Tensor,
    frame_2d: torch.Tensor,
    ortho_weight: float = 2.5,
    h: float = 1e-3,
    damping: float = 3e-2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """O(1)-in-d_0, O(1)-in-C 1st-order Analytic Head-Block Coupled Subspace loss via Gram identity."""
    W_hat, _ = compute_analytic_head_subspaces(model, X, frame_2d, h=h, damping=damping)
    return gram_subspace_orthogonality_loss(W_hat, y, ortho_weight=ortho_weight)


def analytic_head_subspace_predict(
    model: MoonsMLP,
    X_eval: torch.Tensor,
    X_ref: torch.Tensor,
    y_ref: torch.Tensor,
    n_classes: int,
    frame_2d: torch.Tensor,
    h: float = 1e-3,
    damping: float = 3e-2,
) -> torch.Tensor:
    """Classify X_eval by alignment with class-mean Analytic Head DeWitt subspaces (O(N * P_head) memory)."""
    with torch.no_grad():
        W_ref, wh = compute_analytic_head_subspaces(model, X_ref, frame_2d, h=h, damping=damping)
        W_eval, _ = compute_analytic_head_subspaces(
            model, X_eval, frame_2d, h=h, damping=damping, whitener_inv_sqrt=wh
        )
        p_head = W_ref.shape[-1]
        counts = torch.zeros(n_classes, dtype=X_eval.dtype, device=X_eval.device).scatter_add_(
            0, y_ref, torch.ones_like(y_ref, dtype=X_eval.dtype)
        )
        W_cls = torch.zeros(n_classes, 3, p_head, dtype=X_eval.dtype, device=X_eval.device)
        W_cls.index_add_(0, y_ref, W_ref)
        W_cls = W_cls / torch.clamp(counts[:, None, None], min=1.0)
        W_cls = W_cls / torch.sqrt((W_cls * W_cls).sum(dim=(-2, -1), keepdim=True) + 1e-8)
        scores = (torch.einsum("nap,cbp->ncab", W_eval, W_cls) ** 2).sum(dim=(-2, -1))
        return scores.argmax(dim=-1)


def compute_sketched_param_subspaces(
    model: MoonsMLP,
    X: torch.Tensor,
    frame_2d: torch.Tensor,
    r_probes: int = 6,
    eps_theta: float = 1e-3,
    h: float = 1e-3,
    damping: float = 3e-2,
    v_orth: torch.Tensor | None = None,
    whitener_inv_sqrt: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full-parameter r-probe 1st-order FD DeWitt sensitivity subspaces W_hat in R^{N x 3 x r}."""
    N, d0 = X.shape
    params = dict(model.named_parameters())
    keys = list(params.keys())
    shapes = [params[k].shape for k in keys]
    numels = [params[k].numel() for k in keys]
    p_tot = sum(numels)

    if v_orth is None:
        v_raw = torch.randn(p_tot, r_probes, generator=generator, dtype=X.dtype, device=X.device)
        v_orth, _ = torch.linalg.qr(v_raw)

    X_3N = torch.cat([X, (X[:, None, :] + h * frame_2d.T[None, :, :]).reshape(2 * N, d0)], dim=0)
    eye2 = torch.eye(2, dtype=X.dtype, device=X.device).unsqueeze(0)

    def eval_g(p_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        z_all = functional_call(model, p_dict, (X_3N,))
        dz = (z_all[N:].reshape(N, 2, -1) - z_all[:N, None, :]) / h
        return eye2 + torch.einsum("npa,nqa->npq", dz, dz)

    with torch.no_grad():
        g0_isq = _sym2_inv_sqrt(eval_g({k: v.detach() for k, v in params.items()}), eps=1e-4)

    def sym3(p_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        g = eval_g(p_dict)
        M = torch.einsum("nij,njk,nkl->nil", g0_isq, g, g0_isq)
        return torch.stack([M[:, 0, 0], math.sqrt(2.0) * M[:, 0, 1], M[:, 1, 1]], dim=-1)

    m0 = sym3(params)
    u_cols = []
    for q in range(v_orth.shape[1]):
        splits = torch.split(v_orth[:, q], numels)
        p_shift = {
            k: params[k] + eps_theta * sc.reshape(sh)
            for k, sc, sh in zip(keys, splits, shapes, strict=True)
        }
        u_cols.append((sym3(p_shift) - m0) / eps_theta)
    U_r = torch.stack(u_cols, dim=-1)  # (N, 3, r)
    r_eff = U_r.shape[-1]

    if whitener_inv_sqrt is None:
        with torch.no_grad():
            U_det = U_r.detach().reshape(3 * N, r_eff)
            G_r = (U_det.T @ U_det) / float(N) + damping * torch.eye(r_eff, dtype=X.dtype, device=X.device)
            ev, evec = torch.linalg.eigh(G_r)
            whitener_inv_sqrt = evec @ torch.diag(torch.rsqrt(torch.clamp(ev, min=1e-6))) @ evec.T

    W = torch.einsum("nar,rq->naq", U_r, whitener_inv_sqrt)
    W_hat = W / torch.sqrt((W * W).sum(dim=(-2, -1), keepdim=True) + 1e-8)
    return W_hat, v_orth, whitener_inv_sqrt


def sketched_param_subspace_loss(
    model: MoonsMLP,
    X: torch.Tensor,
    y: torch.Tensor,
    frame_2d: torch.Tensor,
    r_probes: int = 6,
    ortho_weight: float = 2.5,
    eps_theta: float = 1e-3,
    h: float = 1e-3,
    damping: float = 3e-2,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Full-network r-probe 1st-order FD Coupled Subspace loss via r x r DeWitt whitener & Gram identity."""
    W_hat, _, _ = compute_sketched_param_subspaces(
        model,
        X,
        frame_2d,
        r_probes=r_probes,
        eps_theta=eps_theta,
        h=h,
        damping=damping,
        generator=generator,
    )
    return gram_subspace_orthogonality_loss(W_hat, y, ortho_weight=ortho_weight)


def sketched_param_subspace_predict(
    model: MoonsMLP,
    X_eval: torch.Tensor,
    X_ref: torch.Tensor,
    y_ref: torch.Tensor,
    n_classes: int,
    frame_2d: torch.Tensor,
    r_probes: int = 12,
    n_ensembles: int = 3,
    seed: int = 0,
) -> torch.Tensor:
    """Classify X_eval by averaging class-mean subspace overlap across n_ensembles r-probe sketches."""
    with torch.no_grad():
        gen = torch.Generator(device=X_eval.device).manual_seed(seed)
        scores_acc = torch.zeros(X_eval.shape[0], n_classes, dtype=X_eval.dtype, device=X_eval.device)
        counts = torch.zeros(n_classes, dtype=X_eval.dtype, device=X_eval.device).scatter_add_(
            0, y_ref, torch.ones_like(y_ref, dtype=X_eval.dtype)
        )
        for _ in range(n_ensembles):
            W_ref, v_orth, wh = compute_sketched_param_subspaces(
                model, X_ref, frame_2d, r_probes=r_probes, generator=gen
            )
            W_eval, _, _ = compute_sketched_param_subspaces(
                model, X_eval, frame_2d, r_probes=r_probes, v_orth=v_orth, whitener_inv_sqrt=wh
            )
            W_cls = torch.zeros(n_classes, 3, r_probes, dtype=X_eval.dtype, device=X_eval.device)
            W_cls.index_add_(0, y_ref, W_ref)
            W_cls = W_cls / torch.clamp(counts[:, None, None], min=1.0)
            W_cls = W_cls / torch.sqrt((W_cls * W_cls).sum(dim=(-2, -1), keepdim=True) + 1e-8)
            scores_acc += (torch.einsum("nar,cbr->ncab", W_eval, W_cls) ** 2).sum(dim=(-2, -1))
        return scores_acc.argmax(dim=-1)
