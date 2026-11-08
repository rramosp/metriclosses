"""Differential Geometry Operators for Neural Network Parameter and Data Manifolds.

Implements exact, vectorized higher-order differential geometric quantities
(pullback metrics, volume forms, Christoffel symbols, Riemann/Gaussian curvature,
Data Information Matrix, Parameter Fisher Information Matrix, and purely geometric
loss functions) using composable PyTorch automatic differentiation (torch.func).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jacrev, vmap

MetricMode = Literal["euclidean", "graph", "conformal_gaussian"]


def get_moons_dataset(
    n_train: int = 600,
    n_test: int = 300,
    noise: float = 0.10,
    seed: int = 42,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate standardized make_moons train and test datasets."""
    X, y = make_moons(n_samples=n_train + n_test, noise=noise, random_state=seed)
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
    """MLP with architecture [2 -> 5 -> 10 -> 15 -> 10 -> 2] and smooth activations."""

    def __init__(
        self,
        layer_dims: Sequence[int] = (2, 5, 10, 15, 10, 2),
        activation: str = "tanh",
        beta: float = 3.0,
        seed: int = 42,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.layer_dims = list(layer_dims)
        self.activation_name = activation.lower()
        self.beta = beta
        self.num_layers = len(self.layer_dims) - 1

        layers = []
        for i in range(self.num_layers):
            lin = nn.Linear(self.layer_dims[i], self.layer_dims[i + 1], dtype=dtype)
            nn.init.xavier_normal_(lin.weight, gain=1.1)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
        self.layers = nn.ModuleList(layers)

    def _activate(self, u: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "tanh":
            return torch.tanh(u)
        if self.activation_name == "gelu":
            return F.gelu(u)
        if self.activation_name == "silu":
            return F.silu(u)
        if self.activation_name == "softplus":
            return F.softplus(u, beta=self.beta)
        if self.activation_name == "relu":
            return F.relu(u)
        raise ValueError(f"Unsupported activation: {self.activation_name}")

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
        return h

    def forward_all_layers(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return representations [x, h^(1), h^(2), h^(3), h^(4), z^(5)]."""
        reps = [x]
        h = x
        for idx, layer in enumerate(self.layers):
            h = layer(h)
            if idx < self.num_layers - 1:
                h = self._activate(h)
            reps.append(h)
        return reps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_up_to_layer(x, self.num_layers)


def _make_single_point_layer_fn(
    model: MoonsMLP,
    layer_idx: int,
    params: Optional[Dict[str, torch.Tensor]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a function phi(x_single) -> h_single for a single 2D point x_single."""
    if params is None:
        params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

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

    F_{mu,nu}(theta) = E_x [ sum_{c=0}^1 p(y=c|x; theta) d_mu log p(y=c|x) d_nu log p(y=c|x) ]
    """
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = dict(model.named_buffers())
    param_keys = list(params.keys())

    def log_probs_single(p_dict: Dict[str, torch.Tensor], x_single: torch.Tensor) -> torch.Tensor:
        logits = functional_call(model, (p_dict, buffers), (x_single.unsqueeze(0),)).squeeze(0)
        return F.log_softmax(logits, dim=-1)

    # jac_params has keys matching params, each of shape (N, 2, *param.shape)
    jac_fn = vmap(jacrev(log_probs_single, argnums=0), in_dims=(None, 0))
    jac_dict = jac_fn(params, X)

    N = X.shape[0]
    J_flat = torch.cat(
        [jac_dict[k].reshape(N, 2, -1) for k in param_keys], dim=-1
    )  # (N, 2, P)

    with torch.no_grad():
        probs = F.softmax(model(X), dim=-1)  # (N, 2)

    # Weight each class derivative by sqrt(p(y=c|x))
    weighted_J = J_flat * torch.sqrt(probs).unsqueeze(-1)  # (N, 2, P)
    flat_scores = weighted_J.reshape(N * 2, -1)  # (2N, P)
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
    """Compute the 2 x 2 Data Information Matrix DIM_{ij}(x; theta) for each input point x in X.

    DIM_{ij}(x) = sum_{c=0}^1 p(y=c|x; theta) (d_i log p(y=c|x)) (d_j log p(y=c|x))
    """
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def log_probs_x(x_single: torch.Tensor) -> torch.Tensor:
        logits = functional_call(model, (params, buffers), (x_single.unsqueeze(0),)).squeeze(0)
        return F.log_softmax(logits, dim=-1)

    J_x = vmap(jacrev(log_probs_x))(X)  # (N, 2_classes, 2_inputs)
    probs = torch.exp(vmap(log_probs_x)(X))  # (N, 2_classes)
    weighted_Jx = J_x * torch.sqrt(probs).unsqueeze(-1)  # (N, 2, 2)
    dim_matrix = torch.einsum("nci,ncj->nij", weighted_Jx, weighted_Jx)
    eye2 = torch.eye(2, dtype=X.dtype, device=X.device).unsqueeze(0)
    return dim_matrix + eps * eye2


def curvature_margin_loss(
    K: torch.Tensor,
    y: torch.Tensor,
    g: torch.Tensor,
    margin: float = 0.5,
    beta_hinge: float = 4.0,
    lambda_vol: float = 0.05,
    lambda_iso: float = 0.01,
    k_clip: float = 15.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Purely geometric loss placing Class 0 in K <= -margin and Class 1 in K >= +margin.

    Includes Riemannian volume stabilization (log det g)^2 and metric scale stabilization
    to prevent trivial metric collapse without using any non-geometric logit targets.
    """
    signs = (2.0 * y.to(K.dtype) - 1.0)  # -1 for Class 0, +1 for Class 1
    # Smoothly bound extreme curvature spikes during early training
    K_bounded = k_clip * torch.tanh(K / k_clip)
    margin_violations = margin - signs * K_bounded
    loss_curv = F.softplus(margin_violations, beta=beta_hinge).mean()

    det_g = torch.clamp(torch.linalg.det(g), min=1e-8)
    loss_vol = (torch.log(det_g) ** 2).mean()

    eye2 = torch.eye(2, dtype=g.dtype, device=g.device).unsqueeze(0)
    loss_iso = ((g - eye2) ** 2).mean()

    total_loss = loss_curv + lambda_vol * loss_vol + lambda_iso * loss_iso
    with torch.no_grad():
        preds = (K > 0.0).long()
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
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Purely geometric loss separating classes by Riemannian log-volume element log(sqrt(det g)).

    Class 0 -> volume contraction (log sqrt(det g) <= -margin)
    Class 1 -> volume expansion   (log sqrt(det g) >= +margin)
    """
    signs = (2.0 * y.to(g.dtype) - 1.0)
    log_vol = 0.5 * torch.log(torch.clamp(torch.linalg.det(g), min=1e-8))
    loss_vol_sep = F.softplus(margin - signs * log_vol, beta=beta_hinge).mean()
    loss_reg = (log_vol ** 2).mean() * lambda_iso
    total = loss_vol_sep + loss_reg

    with torch.no_grad():
        preds = (log_vol > 0.0).long()
        acc = (preds == y).to(g.dtype).mean().item()

    return total, {
        "loss_total": total.item(),
        "geom_acc": acc,
        "mean_log_vol_c0": log_vol[y == 0].mean().item(),
        "mean_log_vol_c1": log_vol[y == 1].mean().item(),
    }


def anisotropy_margin_loss(
    g: torch.Tensor,
    y: torch.Tensor,
    margin: float = 0.6,
    beta_hinge: float = 4.0,
    lambda_vol: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Purely geometric loss separating classes by directional metric polarization log(g_11 / g_22).

    Class 0 -> horizontal metric compression / vertical stretch (log(g_11 / g_22) <= -margin)
    Class 1 -> horizontal metric stretch / vertical compression (log(g_11 / g_22) >= +margin)
    """
    signs = (2.0 * y.to(g.dtype) - 1.0)
    log_ratio = torch.log(torch.clamp(g[:, 0, 0], min=1e-8) / torch.clamp(g[:, 1, 1], min=1e-8))
    loss_aniso = F.softplus(margin - signs * log_ratio, beta=beta_hinge).mean()
    det_g = torch.clamp(torch.linalg.det(g), min=1e-8)
    loss_vol = (torch.log(det_g) ** 2).mean()
    total = loss_aniso + lambda_vol * loss_vol

    with torch.no_grad():
        preds = (log_ratio > 0.0).long()
        acc = (preds == y).to(g.dtype).mean().item()

    return total, {
        "loss_total": total.item(),
        "geom_acc": acc,
        "mean_aniso_c0": log_ratio[y == 0].mean().item(),
        "mean_aniso_c1": log_ratio[y == 1].mean().item(),
    }
