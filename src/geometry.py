"""Differential Geometry Operators for Neural Network Parameter and Data Manifolds.

Implements exact, vectorized higher-order differential geometric quantities
(pullback metrics, volume forms, Christoffel symbols, Riemann/Gaussian curvature,
Data Information Matrix, Parameter Fisher Information Matrix, and purely geometric
loss functions) using composable PyTorch automatic differentiation (torch.func).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from sklearn.datasets import (
    make_blobs,
    make_circles,
    make_classification,
    make_gaussian_quantiles,
    make_moons,
)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    if key.startswith("make_"):
        key = key[len("make_") :]

    rng = np.random.RandomState(seed)

    if key == "moons":
        if n_classes == 2 and input_dim == 2:
            X, y = make_moons(n_samples=n_total, noise=noise, random_state=seed)
        else:
            # Generalized multi-class crescents using make_moons building blocks
            per_pair = int(math.ceil(n_total / n_classes))
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
        cluster_std = float(dataset_kwargs.get("cluster_std", max(noise * 6.0, 0.20)))
        centers = dataset_kwargs.get("centers", n_classes)
        center_box = dataset_kwargs.get("center_box", (-3.0, 3.0))
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
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
        output_activation: Optional[str] = "identity",
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

    def forward_all_layers(self, x: torch.Tensor) -> List[torch.Tensor]:
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
    params: Optional[Dict[str, torch.Tensor]] = None,
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
    params: Optional[Dict[str, torch.Tensor]] = None,
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
    params: Optional[Dict[str, torch.Tensor]] = None,
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
) -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return function computing (g, Gamma^k_{ij}, R_{1212}, K) at a single 2D point x."""
    dg_fn = jacrev(g_single)  # d g_{ij} / d x^k -> shape (2, 2, 2) indexed as [i, j, k]
    ddg_fn = jacrev(dg_fn)  # d^2 g_{ij} / (d x^k d x^l) -> shape (2, 2, 2, 2) indexed as [i, j, k, l]

    def geom_single(
        x_single: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    params: Optional[Dict[str, torch.Tensor]] = None,
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
    params: Optional[Dict[str, torch.Tensor]] = None,
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
    params: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
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


def flatten_params(param_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a dictionary of parameter tensors into a single 1D vector in R^P."""
    return torch.cat([p.reshape(-1) for p in param_dict.values()], dim=0)


def compute_parameter_fim(
    model: MoonsMLP,
    X: torch.Tensor,
    damping: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the P x P Parameter Fisher Information Matrix F(theta) and per-sample score vectors.

    F_{mu,nu}(theta) = E_x [ sum_{c=0}^{C-1} p(y=c|x; theta) d_mu log p(y=c|x) d_nu log p(y=c|x) ]
    """
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = dict(model.named_buffers())
    param_keys = list(params.keys())

    def log_probs_single(p_dict: Dict[str, torch.Tensor], x_single: torch.Tensor) -> torch.Tensor:
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
    n_classes: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    n_classes: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
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
    n_classes: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
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
    n_classes: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
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
    layer_idx: Optional[int] = None,
    params: Optional[Dict[str, torch.Tensor]] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def single_geom(x_single: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def z_flat(p_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
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

    def whitened_metric_flat(p_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
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
    lambda_vol: Optional[float] = None,
    lambda_iso: float = 0.005,
    damping: float = 2e-2,
    n_classes: Optional[int] = None,
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
        p_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def coupled_feature_vector(p_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
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
    params: Optional[Dict[str, torch.Tensor]] = None,
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
    params: Optional[Dict[str, torch.Tensor]] = None,
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
    params: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    params: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    params: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
    generator: Optional[torch.Generator] = None,
    params: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate pullback metric via m random orthonormal directional probes ((m + 1)*N forward evaluations).

    Returns (g_ambient_est, g_subspace), keeping compute O(m) independent of ambient input dimension d_0.
    """
    N, d0 = X.shape
    raw = torch.randn(d0, n_probes, generator=generator, dtype=X.dtype, device=X.device)
    Q, _ = torch.linalg.qr(raw)  # (d0, n_probes)
    X_shifts = (X[:, None, :] + h * Q.T[None, :, :]).reshape(N * n_probes, d0)
    Z_all = _eval_model_batch(model, torch.cat([X, X_shifts], dim=0), params=params)
    z0 = Z_all[:N]
    dz = (Z_all[N:].reshape(N, n_probes, -1) - z0[:, None, :]) / h  # (N, n_probes, d_out)

    g_sub = torch.einsum("npa,nqa->npq", dz, dz)  # (N, n_probes, n_probes)
    g_est = (float(d0) / float(n_probes)) * torch.einsum("ip,npq,jq->nij", Q, g_sub, Q)
    eye = torch.eye(d0, dtype=X.dtype, device=X.device).unsqueeze(0)
    g_out = (eye + g_est) if metric_mode == "graph" else (g_est + eps * eye)
    return g_out, g_sub


def compute_sketched_plane_curvature(
    model: MoonsMLP,
    X: torch.Tensor,
    h: float = 2e-3,
    eps: float = 1e-8,
    generator: Optional[torch.Generator] = None,
    params: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    n_classes: Optional[int] = None,
) -> Tuple[torch.Tensor, float]:
    """Matrix-free O(R^2 P) Woodbury Natural Gradient direction using 1st-order FD stencils on a subsample X_sub.

    Avoids ever allocating a P x P parameter metric matrix by solving an R x R system (R << P) via:
        (damping * I_P + (1/S) U^T U)^{-1} g = (1/damping) * [g - U^T (damping * S * I_R + U U^T)^{-1} (U g)]
    Returns (nat_dir, riem_norm).
    """
    params = {k: v.detach() for k, v in model.named_parameters()}
    param_keys = list(params.keys())
    S = X_sub.shape[0]

    def fd_sub_terms(p_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def sketch_features(p_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
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
