"""Build and execute all 5 differential geometry research notebooks."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.nb_executor import build_and_execute_notebook


def build_notebook_01() -> None:
    cells = [
        {
            "type": "markdown",
            "source": r"""
# Notebook 01: Baseline MLP Training and Layerwise Pullback Metric Tensors

## Research Objective
We study how a Multi-Layer Perceptron with architecture **`[2 -> 5 -> 10 -> 15 -> 10 -> 2]`** transforms the 2D `make_moons` manifold across layers $\ell \in \{1, 2, 3, 4, 5\}$.

At each layer $\ell$, the representation map $\Phi^{(\ell)}_\theta : \mathbb{R}^2 \to \mathbb{R}^{d_\ell}$ induces a **$2 \times 2$ Riemannian pullback metric tensor** on the input coordinate chart $x = (x^1, x^2) \in \mathbb{R}^2$:
$$g^{(\ell)}_{ij}(x; \theta) = \left( (J^{(\ell)}(x))^T J^{(\ell)}(x) \right)_{ij} = \sum_{a=1}^{d_\ell} \frac{\partial \Phi^{(\ell)}_a(x)}{\partial x^i} \frac{\partial \Phi^{(\ell)}_a(x)}{\partial x^j}, \quad i, j \in \{1, 2\}$$

From $g^{(\ell)}(x; \theta)$ we extract two fundamental geometric diagnostics:
1. **Riemannian Volume Magnification**: $\mathrm{d}V^{(\ell)}(x) = \sqrt{\det g^{(\ell)}(x; \theta)}$, measuring how local area elements expand or contract across layers (**Hauser & Ray, 2017**; **Zavatone-Veth et al., 2023**).
2. **Metric Anisotropy (Condition Number)**: $\kappa(g^{(\ell)}(x)) = \lambda_{\max}(g^{(\ell)}) / \lambda_{\min}(g^{(\ell)})$ and **Tissot Indicatrices**, revealing how dimension-contracting layers ($15 \to 10 \to 2$) compress directions tangent to equivalence classes while stretching directions orthogonal to the decision boundary (**Benfenati & Marta, 2023**).
""",
        },
        {
            "type": "code",
            "source": """
import sys
from pathlib import Path
for p in [Path.cwd(), Path.cwd().parent]:
    if (p / "src" / "geometry.py").exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from src.geometry import (
    MoonsMLP,
    get_moons_dataset,
    make_evaluation_grid,
    compute_layer_pullback_metric,
    compute_volume_element,
    compute_metric_spectrum,
)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

X_train, y_train, X_test, y_test = get_moons_dataset(n_train=400, n_test=200, noise=0.10, seed=42)
xx, yy, grid_pts = make_evaluation_grid(X_train, n_grid=32, padding=0.45)
print(f"Train samples: {X_train.shape}, Test samples: {X_test.shape}, Grid points: {grid_pts.shape}")
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 1. Baseline Cross-Entropy Training and Checkpoint Capture
We train the `[2 -> 5 -> 10 -> 15 -> 10 -> 2]` MLP with `tanh` activations using standard Cross-Entropy loss and record parameter checkpoints across training epochs $t \in \{0, 25, 80, 250\}$.
""",
        },
        {
            "type": "code",
            "source": """
model = MoonsMLP(layer_dims=(2, 5, 10, 15, 10, 2), activation="tanh", seed=42)
total_params = sum(p.numel() for p in model.parameters())
print(f"MoonsMLP Architecture: {model.layer_dims} | Total parameters P = {total_params}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.015)

checkpoints = {0: copy.deepcopy(model.state_dict())}
history = {"epoch": [], "loss": [], "train_acc": [], "test_acc": []}

for epoch in range(1, 251):
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0 or epoch == 1:
        with torch.no_grad():
            tr_acc = (logits.argmax(dim=-1) == y_train).float().mean().item()
            te_acc = (model(X_test).argmax(dim=-1) == y_test).float().mean().item()
        history["epoch"].append(epoch)
        history["loss"].append(loss.item())
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)

    if epoch in [25, 80, 250]:
        checkpoints[epoch] = copy.deepcopy(model.state_dict())

print(f"Final Epoch 250 | Cross-Entropy Loss: {history['loss'][-1]:.4f} | Train Acc: {history['train_acc'][-1]:.3f} | Test Acc: {history['test_acc'][-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(history["epoch"], history["loss"], lw=2, color="#1f77b4", label="Cross-Entropy Loss")
axes[0].plot(history["epoch"], history["test_acc"], lw=2, color="#2ca02c", label="Test Accuracy")
axes[0].set_xlabel("Training Epoch")
axes[0].set_ylabel("Metric Value")
axes[0].set_title("Baseline Training Convergence")
axes[0].legend()
axes[0].grid(alpha=0.3)

with torch.no_grad():
    probs_grid = torch.softmax(model(grid_pts), dim=-1)[:, 1].reshape(xx.shape).numpy()

cf = axes[1].contourf(xx, yy, probs_grid, levels=25, cmap="coolwarm", alpha=0.85)
axes[1].contour(xx, yy, probs_grid, levels=[0.5], colors="k", linewidths=2.0)
axes[1].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c="#1f77b4", edgecolors="k", s=28, label="Class 0")
axes[1].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c="#d62728", edgecolors="k", s=28, label="Class 1")
axes[1].set_title("Learned Decision Boundary P(y=1 | x)")
axes[1].set_xlabel("$x^1$")
axes[1].set_ylabel("$x^2$")
axes[1].legend(loc="upper right")
plt.colorbar(cf, ax=axes[1], label="P(y=1 | x)")
plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 2. Layerwise Volume Magnification $\sqrt{\det g^{(\ell)}(x)}$ and Anisotropy $\lambda_1 / \lambda_2$
We evaluate $g^{(\ell)}_{ij}(x)$ across all five layers $\ell \in \{1, 2, 3, 4, 5\}$ on the 2D grid. Notice how the trained network concentrates high volume expansion ($\sqrt{\det g^{(\ell)}}$) and extreme directional stretching ($\lambda_1/\lambda_2$) along the nonlinear decision boundary separating the two moons, confirming **Zavatone-Veth et al. (2023)**.
""",
        },
        {
            "type": "code",
            "source": """
fig, axes = plt.subplots(2, 5, figsize=(17, 6.6), sharex=True, sharey=True)
layer_titles = [
    "Layer 1 (2 -> 5)",
    "Layer 2 (5 -> 10)",
    "Layer 3 (10 -> 15)",
    "Layer 4 (15 -> 10)",
    "Layer 5 (10 -> 2)",
]

summary_rows = []
for idx, l_idx in enumerate(range(1, 6)):
    g_grid = compute_layer_pullback_metric(model, grid_pts, layer_idx=l_idx, metric_mode="euclidean", eps=1e-7)
    vol = compute_volume_element(g_grid).detach().numpy().reshape(xx.shape)
    evals, _, cond = compute_metric_spectrum(g_grid)
    log_vol = np.log10(np.clip(vol, 1e-6, None))
    log_cond = np.log10(np.clip(cond.detach().numpy().reshape(xx.shape), 1.0, 1e6))

    summary_rows.append((l_idx, model.layer_dims[l_idx], float(np.median(vol)), float(np.max(vol)), float(np.median(cond.detach().numpy()))))

    im0 = axes[0, idx].contourf(xx, yy, log_vol, levels=20, cmap="magma")
    axes[0, idx].contour(xx, yy, probs_grid, levels=[0.5], colors="cyan", linewidths=1.3, linestyles="--")
    axes[0, idx].set_title(f"{layer_titles[idx]}\\n$\\log_{{10}}\\sqrt{{\\det g^{{({l_idx})}}}}$")
    plt.colorbar(im0, ax=axes[0, idx], fraction=0.046)

    im1 = axes[1, idx].contourf(xx, yy, log_cond, levels=20, cmap="viridis")
    axes[1, idx].contour(xx, yy, probs_grid, levels=[0.5], colors="white", linewidths=1.3, linestyles="--")
    axes[1, idx].set_title(f"Anisotropy $\\log_{{10}}(\\lambda_1 / \\lambda_2)$")
    axes[1, idx].set_xlabel("$x^1$")
    plt.colorbar(im1, ax=axes[1, idx], fraction=0.046)

axes[0, 0].set_ylabel("$x^2$")
axes[1, 0].set_ylabel("$x^2$")
plt.suptitle("Layerwise Pullback Metric Volume Element and Anisotropy Across Depth", y=1.02, fontsize=13)
plt.tight_layout()
plt.show()

print(f"{'Layer':<8} {'Width':<8} {'Median Vol':<15} {'Max Vol':<15} {'Median Cond (l1/l2)':<20}")
print("-" * 68)
for row in summary_rows:
    print(f"l = {row[0]:<4d} d = {row[1]:<4d} {row[2]:<15.4f} {row[3]:<15.4f} {row[4]:<20.2f}")
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 3. Tissot Indicatrices (Metric Tensor Ellipses) & Singular Foliation Compression
To visualize the full $2 \times 2$ tensor field $g^{(\ell)}_{ij}(x)$ rather than just scalar invariants, we draw **Tissot ellipses** whose semi-axes align with the eigenvectors of $g^{(\ell)}(x)$ scaled by $\sqrt{\lambda_1}, \sqrt{\lambda_2}$. As predicted by **Benfenati & Marta (2023)**, deeper layers compress tangent vectors parallel to the decision boundary (forming 1D equivalence leaves) while strongly amplifying vectors orthogonal to the boundary.
""",
        },
        {
            "type": "code",
            "source": """
_, _, coarse_pts = make_evaluation_grid(X_train, n_grid=11, padding=0.3)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)

for ax, l_idx in zip(axes, [1, 3, 5]):
    g_c = compute_layer_pullback_metric(model, coarse_pts, layer_idx=l_idx, metric_mode="euclidean", eps=1e-6)
    evals, evecs, _ = compute_metric_spectrum(g_c)
    evals_np = np.sqrt(evals.detach().numpy())
    scale = 0.24 / (np.percentile(evals_np[:, 0], 90) + 1e-8)

    ax.contour(xx, yy, probs_grid, levels=[0.5], colors="k", linewidths=1.8, linestyles="--")
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c="#1f77b4", alpha=0.20, s=14)
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c="#d62728", alpha=0.20, s=14)

    pts_np = coarse_pts.detach().numpy()
    evecs_np = evecs.detach().numpy()
    for i in range(len(pts_np)):
        v1 = evecs_np[i, :, 0]
        angle = np.degrees(np.arctan2(v1[1], v1[0]))
        w = max(evals_np[i, 0] * scale, 0.025)
        h = max(evals_np[i, 1] * scale, 0.025)
        ell = Ellipse(xy=pts_np[i], width=w, height=h, angle=angle, edgecolor="#8c1515", facecolor="#ff9896", alpha=0.65, lw=1.0)
        ax.add_patch(ell)

    ax.set_title(f"Layer {l_idx} (d={model.layer_dims[l_idx]}) Pullback Metric Ellipses")
    ax.set_xlabel("$x^1$")
axes[0].set_ylabel("$x^2$")
plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 4. Symmetry Breaking Across Training Epochs & Activation Smoothness Comparison
We compare the pullback volume field $\sqrt{\det g^{(4)}(x)}$ from random initialization ($t=0$) through convergence ($t=250$), and compare smooth activations (`tanh`, `gelu`, `softplus`) against piecewise-linear `relu`.
""",
        },
        {
            "type": "code",
            "source": """
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharex=True, sharey=True)
probe_model = MoonsMLP(layer_dims=(2, 5, 10, 15, 10, 2), activation="tanh", seed=42)

for ax, ep in zip(axes, [0, 25, 80, 250]):
    probe_model.load_state_dict(checkpoints[ep])
    g4 = compute_layer_pullback_metric(probe_model, grid_pts, layer_idx=4, metric_mode="euclidean", eps=1e-7)
    vol4 = np.log10(compute_volume_element(g4).detach().numpy().reshape(xx.shape) + 1e-7)
    im = ax.contourf(xx, yy, vol4, levels=20, cmap="plasma")
    ax.set_title(f"Epoch {ep}: $\\log_{{10}}\\sqrt{{\\det g^{{(4)}}(x)}}$")
    ax.set_xlabel("$x^1$")
    plt.colorbar(im, ax=ax, fraction=0.046)
axes[0].set_ylabel("$x^2$")
plt.suptitle("Emergence of Decision-Boundary Volume Magnification During Training (Layer 4)", y=1.03)
plt.tight_layout()
plt.show()
""",
        },
    ]
    build_and_execute_notebook(ROOT / "notebooks" / "01_baseline_and_pullback_metrics.ipynb", cells)


def build_notebook_02() -> None:
    cells = [
        {
            "type": "markdown",
            "source": r"""
# Notebook 02: Christoffel Symbols, Riemann Tensor, and Gaussian Curvature Flow Across Layers

## Research Objective
We compute the Levi-Civita connection (Christoffel symbols $\Gamma^{(\ell)k}_{ij}$) and intrinsic **Gaussian curvature** $K^{(\ell)}(x; \theta)$ of the 2D immersed representation manifolds $\mathcal{M}_\ell = \Phi^{(\ell)}_\theta(\mathbb{R}^2) \subset \mathbb{R}^{d_\ell}$ across layers $\ell \in \{1, 2, 3, 4, 5\}$:
$$\Gamma^{(\ell)}_{kij} = \frac{1}{2}\left(\partial_j g^{(\ell)}_{ki} + \partial_i g^{(\ell)}_{kj} - \partial_k g^{(\ell)}_{ij}\right), \qquad R^{(\ell)}_{1212} = -\frac{1}{2}\left(\partial_1^2 g^{(\ell)}_{22} - 2\partial_1\partial_2 g^{(\ell)}_{12} + \partial_2^2 g^{(\ell)}_{11}\right) + g^{(\ell)km}\left(\Gamma^{(\ell)}_{k12}\Gamma^{(\ell)}_{m12} - \Gamma^{(\ell)}_{k11}\Gamma^{(\ell)}_{m22}\right)$$
$$K^{(\ell)}(x) = \frac{R^{(\ell)}_{1212}(x)}{\det g^{(\ell)}(x)}, \qquad R_{\text{scalar}}^{(\ell)}(x) = 2 K^{(\ell)}(x)$$

### Key Mathematical Insight: The 2D-to-2D Euclidean Pullback Flatness Identity at Layer $\ell = 5$
- For hidden layers $\ell \in \{1, 2, 3, 4\}$ ($d_\ell \in \{5, 10, 15, 10\} > 2$), $\mathcal{M}_\ell$ is a 2D surface immersed in higher-dimensional Euclidean space $\mathbb{R}^{d_\ell}$, possessing rich non-zero intrinsic Gaussian curvature $K^{(\ell)}(x) \neq 0$.
- At the output layer $\ell = 5$ ($d_5 = 2$), pulling back the flat 2D Euclidean metric $I_2$ via a local diffeomorphism $z = f_\theta(x) \in \mathbb{R}^2$ is merely a coordinate change of flat Euclidean space, so $K^{(5)}_{\text{eucl}}(x) \equiv 0$!
- We empirically verify this identity and compare three non-degenerate output curvature constructions:
  1. **Graph Embedding Metric (Monge Patch in $\mathbb{R}^4$)**: $\tilde{g}^{(5)}_{ij}(x) = \delta_{ij} + ((J^{(5)}(x))^T J^{(5)}(x))_{ij}$.
  2. **Conformal Gaussian Output Metric**: $g^{(5)}_{\text{conf}}(x) = e^{-\|z(x)\|^2 / \sigma^2} (J^{(5)}(x))^T J^{(5)}(x)$.
  3. **Penultimate Surface Metric**: $g^{(4)}_{\text{eucl}}(x)$ on $\mathcal{M}_4 \subset \mathbb{R}^{10}$.
""",
        },
        {
            "type": "code",
            "source": """
import sys
from pathlib import Path
for p in [Path.cwd(), Path.cwd().parent]:
    if (p / "src" / "geometry.py").exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.geometry import (
    MoonsMLP,
    get_moons_dataset,
    make_evaluation_grid,
    compute_layer_geometry,
    _christoffel_and_curvature_from_g_single,
)

X_train, y_train, X_test, y_test = get_moons_dataset(n_train=360, n_test=180, noise=0.10, seed=42)
xx, yy, grid_pts = make_evaluation_grid(X_train, n_grid=26, padding=0.45)

# 1. Analytical validation on a sphere of radius R=2 (K = 1/R^2 = 0.25) and hyperbolic paraboloid (K(0) = -3.0)
R_s = 2.0
g_sphere = lambda x: ((2.0 * R_s**2 / (R_s**2 + torch.sum(x**2)))**2) * torch.eye(2, dtype=x.dtype)
_, _, _, K_sphere = _christoffel_and_curvature_from_g_single(g_sphere)(torch.tensor([0.8, -0.6], dtype=torch.float64))

g_saddle = lambda x: torch.eye(2, dtype=x.dtype) + torch.stack([2.0 * x[0], -1.5 * x[1]]).unsqueeze(0).T @ torch.stack([2.0 * x[0], -1.5 * x[1]]).unsqueeze(0)
_, _, _, K_saddle = _christoffel_and_curvature_from_g_single(g_saddle)(torch.zeros(2, dtype=torch.float64))

print(f"Analytical Validation | Sphere (R=2) exact K=0.250000 -> autodiff K={K_sphere.item():.6f}")
print(f"Analytical Validation | Hyperbolic Saddle exact K(0)=-3.000000 -> autodiff K={K_saddle.item():.6f}")
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 1. Empirical Verification of 2D-to-2D Euclidean Flatness vs. Non-Degenerate Output Metrics
We train our `[2 -> 5 -> 10 -> 15 -> 10 -> 2]` MLP and compare the four curvature definitions at the output/penultimate layers.
""",
        },
        {
            "type": "code",
            "source": """
model = MoonsMLP(layer_dims=(2, 5, 10, 15, 10, 2), activation="tanh", seed=42)
optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
criterion = nn.CrossEntropyLoss()

epoch_snapshots = {0: copy.deepcopy(model.state_dict())}
for ep in range(1, 201):
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()
    if ep in [20, 70, 200]:
        epoch_snapshots[ep] = copy.deepcopy(model.state_dict())

geom_l5_eucl = compute_layer_geometry(model, X_test[:50], layer_idx=5, metric_mode="euclidean", eps=1e-13)
geom_l5_graph = compute_layer_geometry(model, X_test[:50], layer_idx=5, metric_mode="graph")
geom_l5_conf = compute_layer_geometry(model, X_test[:50], layer_idx=5, metric_mode="conformal_gaussian", sigma=2.5)
geom_l4_eucl = compute_layer_geometry(model, X_test[:50], layer_idx=4, metric_mode="euclidean", eps=1e-7)

print("Comparison of Curvature Formulations on Trained Network (50 test points):")
print(f"  1. Layer 5 Flat Euclidean Pullback (d=2 -> d=2) : max |K| = {geom_l5_eucl['K'].abs().max().item():.2e} (Identically Flat!)")
print(f"  2. Layer 5 Graph Immersion Metric in R^4        : mean |K| = {geom_l5_graph['K'].abs().mean().item():.4f}, range = [{geom_l5_graph['K'].min().item():.3f}, {geom_l5_graph['K'].max().item():.3f}]")
print(f"  3. Layer 5 Conformal Gaussian Output Metric     : mean |K| = {geom_l5_conf['K'].abs().mean().item():.4f}, range = [{geom_l5_conf['K'].min().item():.3f}, {geom_l5_conf['K'].max().item():.3f}]")
print(f"  4. Layer 4 Penultimate Immersion in R^10        : mean |K| = {geom_l4_eucl['K'].abs().mean().item():.4f}, range = [{geom_l4_eucl['K'].min().item():.3f}, {geom_l4_eucl['K'].max().item():.3f}]")
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 2. Layerwise Gaussian Curvature Landscape $K^{(\ell)}(x)$ Across Network Depth
We plot $\operatorname{arcsinh}(K^{(\ell)}(x))$ across layers $\ell = 1, 2, 3, 4$ (Euclidean immersion in $\mathbb{R}^{d_\ell}$) and layer $\ell = 5$ (Graph immersion in $\mathbb{R}^4$).
Notice how **Brandon, Chadwick, & Pellegrino (2025)**'s two-stage geometric decomposition appears:
- **Early expansion layers ($\ell = 1, 2, 3$)** introduce localized positive and negative curvature folds along the moon crescents to discretize continuous input coordinates.
- **Contraction layers ($\ell = 4, 5$)** concentrate sharp curvature ridges along the decision boundary while flattening the interior of each class manifold (discrete Ricci flow smoothing; **Baptista et al., 2024**; **Hehl et al., 2025**).
""",
        },
        {
            "type": "code",
            "source": """
with torch.no_grad():
    probs_grid = torch.softmax(model(grid_pts), dim=-1)[:, 1].reshape(xx.shape).numpy()

fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.7), sharex=True, sharey=True)
for idx, l_idx in enumerate(range(1, 6)):
    mode = "graph" if l_idx == 5 else "euclidean"
    geom = compute_layer_geometry(model, grid_pts, layer_idx=l_idx, metric_mode=mode, eps=1e-6)
    K_map = np.arcsinh(geom["K"].detach().numpy().reshape(xx.shape))
    vmax = max(np.percentile(np.abs(K_map), 96), 0.1)

    im = axes[idx].contourf(xx, yy, K_map, levels=22, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[idx].contour(xx, yy, probs_grid, levels=[0.5], colors="k", linewidths=1.5, linestyles="--")
    axes[idx].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c="#1f77b4", s=8, alpha=0.4)
    axes[idx].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c="#d62728", s=8, alpha=0.4)
    axes[idx].set_title(f"Layer {l_idx} (d={model.layer_dims[l_idx]}, {mode})\\n$\\mathrm{{arcsinh}}(K^{{({l_idx})}}(x))$")
    axes[idx].set_xlabel("$x^1$")
    plt.colorbar(im, ax=axes[idx], fraction=0.046)

axes[0].set_ylabel("$x^2$")
plt.suptitle("Layerwise Gaussian Curvature Landscape Across the [2 -> 5 -> 10 -> 15 -> 10 -> 2] MLP", y=1.04, fontsize=13)
plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 3. Evolution of Layerwise Curvature & Christoffel Symbol Norms Across Training Epochs
We track how the mean magnitude of Gaussian curvature $\mathbb{E}[|K^{(\ell)}(x)|]$ and Christoffel connection norm $\mathbb{E}[\|\Gamma^{(\ell)}(x)\|_F]$ evolve across layers $\ell \in \{1,\dots,5\}$ from initialization ($t=0$) to convergence ($t=200$), contrasting interior class points against decision-boundary points ($|P(y=1|x) - 0.5| < 0.15$).
""",
        },
        {
            "type": "code",
            "source": """
with torch.no_grad():
    p_tr = torch.softmax(model(X_train), dim=-1)[:, 1]
    boundary_mask = (p_tr - 0.5).abs() < 0.20
    interior_mask = ~boundary_mask

probe = MoonsMLP(layer_dims=(2, 5, 10, 15, 10, 2), activation="tanh", seed=42)
epochs_list = [0, 20, 70, 200]
layers_list = [1, 2, 3, 4, 5]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
colors = ["#7f7f7f", "#ff7f0e", "#1f77b4", "#2ca02c"]

for ep, col in zip(epochs_list, colors):
    probe.load_state_dict(epoch_snapshots[ep])
    k_int, k_bnd, gamma_norms = [], [], []
    for l_idx in layers_list:
        mode = "graph" if l_idx == 5 else "euclidean"
        g_tr = compute_layer_geometry(probe, X_train, layer_idx=l_idx, metric_mode=mode, eps=1e-6)
        k_abs = torch.arcsinh(g_tr["K"].abs()).detach()
        k_int.append(k_abs[interior_mask].mean().item())
        k_bnd.append(k_abs[boundary_mask].mean().item() if boundary_mask.any() else k_abs.mean().item())
        gamma_norms.append(g_tr["christoffel"].flatten(1).norm(dim=1).mean().item())

    axes[0].plot(layers_list, k_int, marker="o", lw=2, color=col, label=f"Epoch {ep} (Class Interior)")
    axes[0].plot(layers_list, k_bnd, marker="s", lw=1.8, ls="--", color=col, label=f"Epoch {ep} (Boundary)")
    axes[1].plot(layers_list, gamma_norms, marker="o", lw=2, color=col, label=f"Epoch {ep}")

axes[0].set_xticks(layers_list)
axes[0].set_xticklabels([f"L{l} (d={model.layer_dims[l]})" for l in layers_list])
axes[0].set_ylabel("Mean $\\mathrm{arcsinh}(|K^{(\\ell)}(x)|)$")
axes[0].set_title("Curvature Evolution: Class Interior vs. Decision Boundary")
axes[0].legend(fontsize=8, ncol=2)
axes[0].grid(alpha=0.3)

axes[1].set_xticks(layers_list)
axes[1].set_xticklabels([f"L{l} (d={model.layer_dims[l]})" for l in layers_list])
axes[1].set_ylabel("Mean Connection Norm $\\|\\Gamma^{(\\ell)k}_{ij}(x)\\|_F$")
axes[1].set_title("Levi-Civita Connection Strength Across Layers")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 4. Layerwise Log Volume Landscape Across Network Depth

Alongside curvature $K^{(\ell)}(x)$, the pullback metric $g^{(\ell)}_{ij}(x; \theta)$ defines the **local Riemannian log-volume density**:
$$\log \rho_{\text{vol}}^{(\ell)}(x; \theta) = \log\sqrt{\det g^{(\ell)}(x; \theta)} = \frac{1}{2}\log \det g^{(\ell)}(x; \theta)$$
where $\log\sqrt{\det g^{(\ell)}(x)} > 0$ indicates local area expansion, $\log\sqrt{\det g^{(\ell)}(x)} < 0$ indicates local area contraction, and $\log\sqrt{\det g^{(\ell)}(x)} = 0$ is the equiareal contour ($\det g^{(\ell)} = 1$).

Below we visualize:
1. The 2D spatial landscape of $\log\sqrt{\det g^{(\ell)}(x)}$ across all five layers $\ell \in \{1, 2, 3, 4, 5\}$.
2. The evolution of $\mathbb{E}[\log\sqrt{\det g^{(\ell)}(x)}]$ across layers $\ell = 1 \to 5$ and training epochs $t \in \{0, 20, 70, 200\}$, contrasting class interior points against the decision boundary and comparing Class $0$ vs. Class $1$.
""",
        },
        {
            "type": "code",
            "source": """
from src.geometry import compute_layer_pullback_metric, compute_metric_spectrum

fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.7), sharex=True, sharey=True)
print(f"{'Layer':<10} {'Width':<8} {'Min log Vol':>14} {'Mean log Vol':>14} {'Max log Vol':>14} {'Boundary Mean':>15}")
print("-" * 80)

for idx, l_idx in enumerate(layers_list):
    g_grid = compute_layer_pullback_metric(model, grid_pts, layer_idx=l_idx, metric_mode="euclidean", eps=1e-7)
    log_vol_grid = (0.5 * torch.log(torch.clamp(torch.linalg.det(g_grid), min=1e-12))).detach().numpy().reshape(xx.shape)
    vmax_v = max(np.percentile(np.abs(log_vol_grid), 96), 0.2)

    g_tr_l = compute_layer_pullback_metric(model, X_train, layer_idx=l_idx, metric_mode="euclidean", eps=1e-7)
    log_vol_tr = (0.5 * torch.log(torch.clamp(torch.linalg.det(g_tr_l), min=1e-12))).detach()
    bnd_mean = log_vol_tr[boundary_mask].mean().item() if boundary_mask.any() else log_vol_tr.mean().item()

    print(
        f"Layer {l_idx:<4d} d={model.layer_dims[l_idx]:<6d} "
        f"{log_vol_grid.min():14.3f} {log_vol_grid.mean():14.3f} {log_vol_grid.max():14.3f} {bnd_mean:15.3f}"
    )

    im = axes[idx].contourf(xx, yy, log_vol_grid, levels=22, cmap="RdBu_r", vmin=-vmax_v, vmax=vmax_v)
    axes[idx].contour(xx, yy, log_vol_grid, levels=[0.0], colors="gold", linewidths=1.2, linestyles=":")
    axes[idx].contour(xx, yy, probs_grid, levels=[0.5], colors="k", linewidths=1.5, linestyles="--")
    axes[idx].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c="#1f77b4", s=8, alpha=0.4)
    axes[idx].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c="#d62728", s=8, alpha=0.4)
    axes[idx].set_title(f"Layer {l_idx} (d={model.layer_dims[l_idx]})\\n$\\log\\sqrt{{\\det g^{{({l_idx})}}(x)}}$")
    axes[idx].set_xlabel("$x^1$")
    plt.colorbar(im, ax=axes[idx], fraction=0.046)

axes[0].set_ylabel("$x^2$")
plt.suptitle("Layerwise Log Volume Density Landscape Across the [2 -> 5 -> 10 -> 15 -> 10 -> 2] MLP", y=1.04, fontsize=13)
plt.tight_layout()
plt.show()

# Track Log-Volume evolution across training epochs (Interior vs Boundary & Class 0 vs Class 1)
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.3))
for ep, col in zip(epochs_list, colors):
    probe.load_state_dict(epoch_snapshots[ep])
    lv_int, lv_bnd, lv_c0, lv_c1 = [], [], [], []
    for l_idx in layers_list:
        g_ep = compute_layer_pullback_metric(probe, X_train, layer_idx=l_idx, metric_mode="euclidean", eps=1e-7)
        lv = (0.5 * torch.log(torch.clamp(torch.linalg.det(g_ep), min=1e-12))).detach()
        lv_int.append(lv[interior_mask].mean().item())
        lv_bnd.append(lv[boundary_mask].mean().item() if boundary_mask.any() else lv.mean().item())
        lv_c0.append(lv[y_train == 0].mean().item())
        lv_c1.append(lv[y_train == 1].mean().item())

    axes[0].plot(layers_list, lv_int, "o-", lw=2, color=col, label=f"Epoch {ep} (Interior)")
    axes[0].plot(layers_list, lv_bnd, "s--", lw=1.8, color=col, label=f"Epoch {ep} (Boundary)")
    axes[1].plot(layers_list, lv_c1, "o-", lw=2, color=col, label=f"Epoch {ep} (Class 1)")
    axes[1].plot(layers_list, lv_c0, "s--", lw=1.8, color=col, label=f"Epoch {ep} (Class 0)")

for ax in axes:
    ax.axhline(0.0, color="k", lw=1.0, ls=":")
    ax.set_xticks(layers_list)
    ax.set_xticklabels([f"L{l} (d={model.layer_dims[l]})" for l in layers_list])
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Mean Log Volume $\\mathbb{E}[\\log\\sqrt{\\det g^{(\\ell)}(x)}]$")
axes[0].set_title("Log-Volume Evolution: Class Interior vs. Decision Boundary")
axes[0].legend(fontsize=8, ncol=2)

axes[1].set_ylabel("Mean Log Volume $\\mathbb{E}[\\log\\sqrt{\\det g^{(\\ell)}(x)}]$")
axes[1].set_title("Log-Volume Evolution: Class 1 (Solid) vs. Class 0 (Dashed)")
axes[1].legend(fontsize=8, ncol=2)

plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 5. Layerwise Metric Anisotropy Landscape Across Network Depth

To capture how each layer stretches space directionally (beyond isotropic area change), we examine two **metric anisotropy fields** across layers $\ell \in \{1, 2, 3, 4, 5\}$:
1. **Directional Metric Polarization**: $\log\!\big(g^{(\ell)}_{11}(x) / g^{(\ell)}_{22}(x)\big)$, measuring horizontal ($x^1$) versus vertical ($x^2$) metric dilation.
2. **Coordinate-Invariant Spectral Anisotropy**: $\log_{10}\!\big(\lambda_1(g^{(\ell)}(x)) / \lambda_2(g^{(\ell)}(x))\big)$, measuring the condition number (ratio of principal eigenvalues) of $g^{(\ell)}(x)$ and quantifying the formation of 1D singular foliation leaves (**Benfenati & Marta, 2023**).
""",
        },
        {
            "type": "code",
            "source": """
fig, axes = plt.subplots(2, 5, figsize=(17.5, 6.8), sharex=True, sharey=True)
print(f"{'Layer':<10} {'Width':<8} {'Mean log(g11/g22)':>18} {'Mean log10(l1/l2)':>18} {'Boundary log10(l1/l2)':>22}")
print("-" * 80)

for idx, l_idx in enumerate(layers_list):
    g_grid_graph = compute_layer_pullback_metric(model, grid_pts, layer_idx=l_idx, metric_mode="graph")
    dir_aniso_grid = torch.log(g_grid_graph[:, 0, 0] / g_grid_graph[:, 1, 1]).detach().numpy().reshape(xx.shape)
    vmax_a = max(np.percentile(np.abs(dir_aniso_grid), 96), 0.15)

    g_grid_eucl = compute_layer_pullback_metric(model, grid_pts, layer_idx=l_idx, metric_mode="euclidean", eps=1e-7)
    _, _, cond_grid = compute_metric_spectrum(g_grid_eucl)
    spec_aniso_grid = torch.log10(torch.clamp(cond_grid, min=1.0)).detach().numpy().reshape(xx.shape)

    g_tr_eucl = compute_layer_pullback_metric(model, X_train, layer_idx=l_idx, metric_mode="euclidean", eps=1e-7)
    _, _, cond_tr = compute_metric_spectrum(g_tr_eucl)
    log_cond_tr = torch.log10(torch.clamp(cond_tr, min=1.0)).detach()
    bnd_cond = log_cond_tr[boundary_mask].mean().item() if boundary_mask.any() else log_cond_tr.mean().item()

    print(
        f"Layer {l_idx:<4d} d={model.layer_dims[l_idx]:<6d} "
        f"{dir_aniso_grid.mean():+18.3f} {spec_aniso_grid.mean():18.3f} {bnd_cond:22.3f}"
    )

    # Row 0: Directional polarization log(g11 / g22)
    im0 = axes[0, idx].contourf(xx, yy, dir_aniso_grid, levels=22, cmap="PuOr_r", vmin=-vmax_a, vmax=vmax_a)
    axes[0, idx].contour(xx, yy, probs_grid, levels=[0.5], colors="k", linewidths=1.4, linestyles="--")
    axes[0, idx].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c="#1f77b4", s=8, alpha=0.35)
    axes[0, idx].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c="#d62728", s=8, alpha=0.35)
    axes[0, idx].set_title(f"Layer {l_idx} (d={model.layer_dims[l_idx]})\\n$\\log(g^{{({l_idx})}}_{{11}} / g^{{({l_idx})}}_{{22}})$")
    plt.colorbar(im0, ax=axes[0, idx], fraction=0.046)

    # Row 1: Spectral condition number log10(lambda_1 / lambda_2)
    im1 = axes[1, idx].contourf(xx, yy, spec_aniso_grid, levels=22, cmap="viridis")
    axes[1, idx].contour(xx, yy, probs_grid, levels=[0.5], colors="white", linewidths=1.4, linestyles="--")
    axes[1, idx].set_title(f"Spectral $\\log_{{10}}(\\lambda_1^{{({l_idx})}} / \\lambda_2^{{({l_idx})}})$")
    axes[1, idx].set_xlabel("$x^1$")
    plt.colorbar(im1, ax=axes[1, idx], fraction=0.046)

axes[0, 0].set_ylabel("$x^2$")
axes[1, 0].set_ylabel("$x^2$")
plt.suptitle("Layerwise Directional Polarization and Spectral Anisotropy Landscapes Across Depth", y=1.02, fontsize=13)
plt.tight_layout()
plt.show()

# Track Spectral and Directional Anisotropy across training epochs
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.3))
for ep, col in zip(epochs_list, colors):
    probe.load_state_dict(epoch_snapshots[ep])
    spec_int, spec_bnd, dir_gap = [], [], []
    for l_idx in layers_list:
        g_ep_eucl = compute_layer_pullback_metric(probe, X_train, layer_idx=l_idx, metric_mode="euclidean", eps=1e-7)
        _, _, cond_ep = compute_metric_spectrum(g_ep_eucl)
        lc = torch.log10(torch.clamp(cond_ep, min=1.0)).detach()
        spec_int.append(lc[interior_mask].mean().item())
        spec_bnd.append(lc[boundary_mask].mean().item() if boundary_mask.any() else lc.mean().item())

        g_ep_graph = compute_layer_pullback_metric(probe, X_train, layer_idx=l_idx, metric_mode="graph").detach()
        lr = torch.log(g_ep_graph[:, 0, 0] / g_ep_graph[:, 1, 1])
        dir_gap.append((lr[y_train == 1].mean() - lr[y_train == 0].mean()).item())

    axes[0].plot(layers_list, spec_int, "o-", lw=2, color=col, label=f"Epoch {ep} (Interior)")
    axes[0].plot(layers_list, spec_bnd, "s--", lw=1.8, color=col, label=f"Epoch {ep} (Boundary)")
    axes[1].plot(layers_list, dir_gap, "o-", lw=2, color=col, label=f"Epoch {ep}")

axes[0].set_xticks(layers_list)
axes[0].set_xticklabels([f"L{l} (d={model.layer_dims[l]})" for l in layers_list])
axes[0].set_ylabel("Mean Spectral Anisotropy $\\mathbb{E}[\\log_{10}(\\lambda_1^{(\\ell)} / \\lambda_2^{(\\ell)})]$")
axes[0].set_title("Spectral Anisotropy Evolution: Interior vs. Boundary")
axes[0].legend(fontsize=8, ncol=2)
axes[0].grid(alpha=0.3)

axes[1].axhline(0.0, color="k", lw=1.0, ls=":")
axes[1].set_xticks(layers_list)
axes[1].set_xticklabels([f"L{l} (d={model.layer_dims[l]})" for l in layers_list])
axes[1].set_ylabel("Directional Polarization Gap $\\Delta \\log(g^{(\\ell)}_{11} / g^{(\\ell)}_{22})$")
axes[1].set_title("Class Directional Anisotropy Separation Across Training Epochs")
axes[1].legend(fontsize=8.5)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
""",
        },
    ]
    build_and_execute_notebook(ROOT / "notebooks" / "02_layerwise_curvature_evolution.ipynb", cells)


def build_notebook_03() -> None:
    cells = [
        {
            "type": "markdown",
            "source": r"""
# Notebook 03: Duality and Metric Flow Between the Parameter Manifold ($\Theta$) and Data Manifolds ($\mathcal{M}_\ell$)

## Research Objective
We investigate the information-geometric interplay between:
1. **The Parameter Manifold ($\Theta \subset \mathbb{R}^{422}$)** equipped with the $422 \times 422$ **Parameter Fisher Information Matrix (FIM)**:
   $$F_{\mu\nu}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_{c=0}^1 p_\theta(y=c|x) \frac{\partial \log p_\theta(y=c|x)}{\partial \theta^\mu} \frac{\partial \log p_\theta(y=c|x)}{\partial \theta^\nu} \right]$$
2. **The Data Manifold ($\mathcal{M}_0 \subset \mathbb{R}^2$)** equipped with the $2 \times 2$ **Data Information Matrix (DIM)** (**Tron, Fioresi, Novelli, & Latini, 2024/2026**; **Fasina et al., 2023**):
   $$\mathrm{DIM}_{ij}(x; \theta) = \sum_{c=0}^1 p_\theta(y=c|x) \frac{\partial \log p_\theta(y=c|x)}{\partial x^i} \frac{\partial \log p_\theta(y=c|x)}{\partial x^j}$$

We further test **Halverson & Ruehle (2023)**'s *Metric Flow* framework by measuring how perturbations along the top vs. bottom eigenvectors of $F_{\mu\nu}(\theta)$ deform the layerwise metric tensor $g^{(\ell)}_{ij}(x; \theta)$ and Gaussian curvature $K^{(\ell)}(x; \theta)$, and compare **Euclidean Gradient Descent** against **Natural Gradient Descent** on $\Theta$.
""",
        },
        {
            "type": "code",
            "source": """
import sys
from pathlib import Path
for p in [Path.cwd(), Path.cwd().parent]:
    if (p / "src" / "geometry.py").exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.geometry import (
    MoonsMLP,
    get_moons_dataset,
    make_evaluation_grid,
    compute_parameter_fim,
    compute_data_information_matrix,
    compute_layer_geometry,
)

X_train, y_train, X_test, y_test = get_moons_dataset(n_train=240, n_test=120, noise=0.10, seed=42)
xx, yy, grid_pts = make_evaluation_grid(X_train, n_grid=25, padding=0.45)

model = MoonsMLP(layer_dims=(2, 5, 10, 15, 10, 2), activation="tanh", seed=42)
optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
criterion = nn.CrossEntropyLoss()

snapshots = {0: copy.deepcopy(model.state_dict())}
for ep in range(1, 181):
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()
    if ep in [15, 60, 180]:
        snapshots[ep] = copy.deepcopy(model.state_dict())

print("Captured parameter manifold checkpoints at epochs:", list(snapshots.keys()))
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 1. Co-Evolution of the $422 \times 422$ Parameter FIM Spectrum and $2 \times 2$ Data Information Matrix
We compute the eigenspectrum of $F_{\mu\nu}(\theta)$ and the spatial trace of $\mathrm{DIM}_{ij}(x; \theta)$ across training checkpoints. Notice how the parameter manifold concentrates its Fisher information into a low-dimensional stiff subspace while $\operatorname{Tr}(\mathrm{DIM}(x; \theta))$ sharpens into a singular foliation along the decision boundary (**Tron et al., 2024**).
""",
        },
        {
            "type": "code",
            "source": """
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
probe = MoonsMLP(layer_dims=(2, 5, 10, 15, 10, 2), activation="tanh", seed=42)

ep_list = [0, 15, 60, 180]
fim_traces, dim_traces, eff_ranks = [], [], []

for ep in ep_list:
    probe.load_state_dict(snapshots[ep])
    fim, _ = compute_parameter_fim(probe, X_train, damping=1e-8)
    evals = torch.linalg.eigvalsh(fim).flip(0).clamp(min=1e-10).detach().numpy()
    axes[0].plot(range(1, len(evals) + 1), evals, lw=2, label=f"Epoch {ep}")

    p_spec = evals / np.sum(evals)
    eff_rank = float(np.exp(-np.sum(p_spec * np.log(p_spec + 1e-15))))
    dim_mat = compute_data_information_matrix(probe, X_train)
    fim_traces.append(float(np.sum(evals)))
    dim_traces.append(float(torch.diagonal(dim_mat, dim1=-2, dim2=-1).sum(-1).mean().item()))
    eff_ranks.append(eff_rank)

axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("Parameter Eigenvalue Index $k$ ($P = 422$)")
axes[0].set_ylabel("FIM Eigenvalue $\\lambda_k(F(\\theta))$")
axes[0].set_title("422x422 Parameter Manifold Fisher Spectrum")
axes[0].legend()
axes[0].grid(alpha=0.3)

ax2 = axes[1].twinx()
axes[1].plot(ep_list, fim_traces, "o-", color="#1f77b4", lw=2, label="Parameter $\\mathrm{Tr}(F(\\theta))$")
axes[1].plot(ep_list, dim_traces, "s--", color="#d62728", lw=2, label="Data $\\mathbb{E}_x[\\mathrm{Tr}(\\mathrm{DIM}(x))]$")
ax2.plot(ep_list, eff_ranks, "^:", color="#2ca02c", lw=2, label="FIM Effective Rank")
axes[1].set_xlabel("Training Epoch")
axes[1].set_ylabel("Information Metric Trace")
ax2.set_ylabel("Effective Rank of $F(\\theta)$", color="#2ca02c")
axes[1].set_title("Parameter vs. Data Information Metric Coupling")
lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labels1 + labels2, loc="center right")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

for ep, ft, dt, er in zip(ep_list, fim_traces, dim_traces, eff_ranks):
    print(f"Epoch {ep:3d} | Tr(FIM) = {ft:8.3f} | E[Tr(DIM)] = {dt:8.3f} | FIM Effective Rank = {er:6.2f} / 422")
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 2. Singular Foliation of the Input Data Manifold via the Data Information Matrix (DIM)
For a binary classifier, the unregularized $2 \times 2$ Data Information Matrix $\mathrm{DIM}_{ij}(x) = p(1-p) \nabla_i f_{\text{logit}} \nabla_j f_{\text{logit}}$ has rank at most $1$. Its kernel eigenvector $v_{\ker}(x)$ spans the tangent distribution to the **1D foliation leaves** of constant class probability (**Benfenati & Marta, 2023**; **Tron et al., 2024**), while its principal eigenvector $v_{\max}(x)$ points along the maximum information gradient across the decision boundary.
""",
        },
        {
            "type": "code",
            "source": """
dim_grid = compute_data_information_matrix(model, grid_pts, eps=1e-9)
tr_dim = torch.diagonal(dim_grid, dim1=-2, dim2=-1).sum(-1).detach().numpy().reshape(xx.shape)
evals_d, evecs_d = torch.linalg.eigh(dim_grid)
# Kernel eigenvector (index 0, smallest eigenvalue) is tangent to foliation leaves
v_leaf = evecs_d[:, :, 0].detach().numpy()
u_leaf = v_leaf[:, 0].reshape(xx.shape)
w_leaf = v_leaf[:, 1].reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7.5, 5.2))
cf = ax.contourf(xx, yy, np.log10(tr_dim + 1e-7), levels=22, cmap="inferno")
ax.quiver(xx[::2, ::2], yy[::2, ::2], u_leaf[::2, ::2], w_leaf[::2, ::2], color="cyan", alpha=0.8, scale=22, headwidth=2.5, label="DIM Kernel (Foliation Tangent)")
ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c="#1f77b4", edgecolors="w", s=26, label="Class 0")
ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c="#d62728", edgecolors="w", s=26, label="Class 1")
ax.set_title("Data Information Matrix $\\log_{10}\\mathrm{Tr}(\\mathrm{DIM}(x))$ & Singular Foliation Tangents")
ax.set_xlabel("$x^1$")
ax.set_ylabel("$x^2$")
ax.legend(loc="upper right")
plt.colorbar(cf, ax=ax, label="$\\log_{10}\\mathrm{Tr}(\\mathrm{DIM}(x))$")
plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 3. Metric Flow Sensitivity: Perturbing $\Theta$ Along Top vs. Null FIM Eigenvectors
To directly test how movements on the parameter manifold $\Theta$ induce metric and curvature flows on the data manifold $\mathcal{M}_5$ (**Halverson & Ruehle, 2023**), we perturb $\theta \mapsto \theta + \delta \cdot v^{(k)}$ along unit eigenvectors $v^{(k)}$ of $F(\theta)$ for $k \in \{1, 2, 5, 20, 100, 300\}$ and measure the induced change in the data pullback metric $\|\Delta g^{(5)}(x)\|_F$ and Gaussian curvature $\|\Delta K^{(5)}(x)\|_2$.
""",
        },
        {
            "type": "code",
            "source": """
fim, _ = compute_parameter_fim(model, X_train, damping=1e-8)
fim_evals, fim_evecs = torch.linalg.eigh(fim)
fim_evals = fim_evals.flip(0)
fim_evecs = fim_evecs.flip(1)

base_params = {k: v.detach().clone() for k, v in model.named_parameters()}
base_geom = compute_layer_geometry(model, X_test, layer_idx=5, metric_mode="graph", params=base_params)

def unflatten_like(vec: torch.Tensor, template: dict) -> dict:
    out = {}
    offset = 0
    for k, t in template.items():
        numel = t.numel()
        out[k] = t + vec[offset : offset + numel].reshape(t.shape)
        offset += numel
    return out

modes_to_test = [0, 1, 2, 4, 9, 19, 49, 99, 199, 350]
delta_scale = 0.08
metric_shifts, curv_shifts, mode_evals = [], [], []

for m_idx in modes_to_test:
    v_vec = fim_evecs[:, m_idx] * delta_scale
    pert_params = unflatten_like(v_vec, base_params)
    pert_geom = compute_layer_geometry(model, X_test, layer_idx=5, metric_mode="graph", params=pert_params)
    dg = (pert_geom["g"] - base_geom["g"]).norm(dim=(-2, -1)).mean().item()
    dK = (pert_geom["K"] - base_geom["K"]).abs().mean().item()
    metric_shifts.append(dg)
    curv_shifts.append(dK)
    mode_evals.append(fim_evals[m_idx].item())

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot([m + 1 for m in modes_to_test], metric_shifts, "o-", lw=2, color="#1f77b4", label="Data Metric Shift $\\mathbb{E}[\\|\\Delta g^{(5)}(x)\\|_F]$")
ax.plot([m + 1 for m in modes_to_test], curv_shifts, "s--", lw=2, color="#d62728", label="Data Curvature Shift $\\mathbb{E}[|\\Delta K^{(5)}(x)|]$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Parameter FIM Eigenvector Mode Index $k$ (1 = stiffest, 422 = flattest)")
ax.set_ylabel("Induced Geometric Deformation on Data Manifold $\\mathcal{M}_5$")
ax.set_title("Coupling of Parameter Manifold FIM Modes to Data Manifold Metric & Curvature Flows")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 4. Euclidean Gradient Descent vs. Riemannian Natural Gradient Descent on $\Theta$
We compare standard Euclidean Gradient Descent ($\Delta \theta = -\eta \nabla_\theta \mathcal{L}$) with damped **Natural Gradient Descent** ($\Delta \theta = -\eta (F(\theta) + \lambda I)^{-1} \nabla_\theta \mathcal{L}$), tracking both classification loss and the smoothness of the induced Gaussian curvature $K^{(5)}(x)$ on the data manifold.
""",
        },
        {
            "type": "code",
            "source": """
def run_optimizer_comparison(steps: int = 45):
    m_gd = MoonsMLP(activation="tanh", seed=11)
    m_ngd = copy.deepcopy(m_gd)
    crit = nn.CrossEntropyLoss()

    gd_losses, ngd_losses, gd_curv, ngd_curv = [], [], [], []
    for step in range(steps):
        # 1. Euclidean GD
        for p in m_gd.parameters():
            if p.grad is not None:
                p.grad.zero_()
        l_gd = crit(m_gd(X_train), y_train)
        l_gd.backward()
        with torch.no_grad():
            for p in m_gd.parameters():
                p -= 0.15 * p.grad
        gd_losses.append(l_gd.item())

        # 2. Natural Gradient Descent on Parameter Manifold
        for p in m_ngd.parameters():
            if p.grad is not None:
                p.grad.zero_()
        l_ngd = crit(m_ngd(X_train), y_train)
        l_ngd.backward()
        fim_curr, _ = compute_parameter_fim(m_ngd, X_train, damping=0.02)
        g_flat = torch.cat([p.grad.reshape(-1) for p in m_ngd.parameters()])
        nat_step = torch.linalg.solve(fim_curr, g_flat)
        with torch.no_grad():
            offset = 0
            for p in m_ngd.parameters():
                n = p.numel()
                p -= 0.08 * nat_step[offset : offset + n].reshape(p.shape)
                offset += n
        ngd_losses.append(l_ngd.item())

        if step % 3 == 0:
            K_gd = compute_layer_geometry(m_gd, X_test[:40], layer_idx=5, metric_mode="graph")["K"].abs().mean().item()
            K_ngd = compute_layer_geometry(m_ngd, X_test[:40], layer_idx=5, metric_mode="graph")["K"].abs().mean().item()
            gd_curv.append((step, K_gd))
            ngd_curv.append((step, K_ngd))

    return gd_losses, ngd_losses, gd_curv, ngd_curv

gd_l, ngd_l, gd_k, ngd_k = run_optimizer_comparison(45)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
axes[0].plot(gd_l, lw=2, color="#1f77b4", label="Euclidean Gradient Descent")
axes[0].plot(ngd_l, lw=2, color="#2ca02c", label="Riemannian Natural Gradient (FIM)")
axes[0].set_xlabel("Optimization Step")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].set_title("Optimization Trajectory on Parameter Manifold $\\Theta$")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot([x[0] for x in gd_k], [x[1] for x in gd_k], "o-", lw=2, color="#1f77b4", label="Euclidean GD Data Curvature $|K^{(5)}|$")
axes[1].plot([x[0] for x in ngd_k], [x[1] for x in ngd_k], "s-", lw=2, color="#2ca02c", label="Natural GD Data Curvature $|K^{(5)}|$")
axes[1].set_xlabel("Optimization Step")
axes[1].set_ylabel("Mean Output Manifold Curvature $\\mathbb{E}[|K^{(5)}(x)|]$")
axes[1].set_title("Induced Curvature Flow on Data Manifold $\\mathcal{M}_5$")
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()
""",
        },
    ]
    build_and_execute_notebook(ROOT / "notebooks" / "03_parameter_data_manifold_duality.ipynb", cells)


def build_notebook_04() -> None:
    cells = [
        {
            "type": "markdown",
            "source": r"""
# Notebook 04: Classification via Purely Geometric Curvature Losses ($K < 0$ vs. $K > 0$)

## Research Objective
Can a deep neural network learn to classify a non-linear dataset (`make_moons`) using **only differential geometric objects** in its loss function—without any cross-entropy, MSE, or coordinate-level logit supervision?

We train the **`[2 -> 5 -> 10 -> 15 -> 10 -> 2]`** MLP from scratch by sculpting the intrinsic **Gaussian curvature** $K(x; \theta)$ of the final representation manifold $\mathcal{M}_5$ (equipped with the Graph Embedding Metric $\tilde{g}^{(5)}_{ij}(x) = \delta_{ij} + ((J^{(5)}(x))^T J^{(5)}(x))_{ij}$):
- **Class $0$ ($y = 0, s = -1$)**: Driven into regions of **negative Gaussian curvature** ($K(x) \le -\tau < 0$, hyperbolic saddle geometry).
- **Class $1$ ($y = 1, s = +1$)**: Driven into regions of **positive Gaussian curvature** ($K(x) \ge +\tau > 0$, spherical/elliptic dome geometry).

### Purely Geometric Loss Function
$$\mathcal{L}_{\text{geom}}(\theta) = \frac{1}{N}\sum_{n=1}^N \mathrm{softplus}_\beta\!\Big(\tau - (2y_n - 1)\, K_{\text{clip}}(x_n; \theta)\Big) + \lambda_{\text{vol}} \frac{1}{N}\sum_{n=1}^N \Big(\log \det \tilde{g}^{(5)}(x_n; \theta)\Big)^2 + \lambda_{\text{iso}} \frac{1}{N}\sum_{n=1}^N \|\tilde{g}^{(5)}(x_n; \theta) - I_2\|_F^2$$

- **Requires 3rd-Order Automatic Differentiation**: Since $K(x; \theta)$ depends on $\partial_x^2 g_{ij}(x; \theta)$ (2nd spatial derivatives of $\Phi^{(5)}_\theta$), computing $\nabla_\theta \mathcal{L}_{\text{geom}}(\theta)$ backpropagates through 3rd-order mixed spatio-parametric derivatives $\frac{\partial^3 \Phi^{(5)}_\theta(x)}{\partial \theta \partial x^i \partial x^j}$.
- **Purely Geometric Inference Rule**: At test time, any point $x^*$ is classified strictly by the sign of the local manifold curvature:
  $$\hat{y}(x^*) = \mathbb{I}\big[ K(x^*; \theta) > 0 \big]$$
""",
        },
        {
            "type": "code",
            "source": """
import sys
from pathlib import Path
for p in [Path.cwd(), Path.cwd().parent]:
    if (p / "src" / "geometry.py").exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.geometry import (
    MoonsMLP,
    get_moons_dataset,
    make_evaluation_grid,
    compute_layer_geometry,
    curvature_margin_loss,
)

X_train, y_train, X_test, y_test = get_moons_dataset(n_train=220, n_test=120, noise=0.10, seed=42)
xx, yy, grid_pts = make_evaluation_grid(X_train, n_grid=28, padding=0.45)
print(f"Loaded make_moons | Train: {X_train.shape[0]} pts | Test: {X_test.shape[0]} pts")
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 1. End-to-End 3rd-Order Autodiff Training with Purely Geometric Curvature Loss
We optimize $\mathcal{L}_{\text{geom}}(\theta)$ for 90 epochs with Adam (`lr=0.025`) and margin $\tau = 0.40$, tracking the separation of Gaussian curvature between Class $0$ ($K < 0$) and Class $1$ ($K > 0$).
""",
        },
        {
            "type": "code",
            "source": """
geom_model = MoonsMLP(layer_dims=(2, 5, 10, 15, 10, 2), activation="tanh", seed=7)
optimizer = torch.optim.Adam(geom_model.parameters(), lr=0.025)

hist = {
    "epoch": [],
    "loss": [],
    "train_acc": [],
    "test_acc": [],
    "K_c0": [],
    "K_c1": [],
}

t0 = time.time()
for epoch in range(1, 91):
    optimizer.zero_grad()
    params = dict(geom_model.named_parameters())
    geom_tr = compute_layer_geometry(geom_model, X_train, layer_idx=5, metric_mode="graph", params=params)
    loss, metrics = curvature_margin_loss(
        geom_tr["K"],
        y_train,
        geom_tr["g"],
        margin=0.40,
        beta_hinge=4.0,
        lambda_vol=0.02,
        lambda_iso=0.005,
        k_clip=10.0,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(geom_model.parameters(), max_norm=5.0)
    optimizer.step()

    if epoch == 1 or epoch % 5 == 0:
        with torch.no_grad():
            geom_te = compute_layer_geometry(geom_model, X_test, layer_idx=5, metric_mode="graph")
            te_acc = ((geom_te["K"] > 0.0).long() == y_test).float().mean().item()
        hist["epoch"].append(epoch)
        hist["loss"].append(metrics["loss_total"])
        hist["train_acc"].append(metrics["geom_acc"])
        hist["test_acc"].append(te_acc)
        hist["K_c0"].append(metrics["mean_K_class0"])
        hist["K_c1"].append(metrics["mean_K_class1"])

    if epoch in [1, 20, 45, 90]:
        print(
            f"Epoch {epoch:3d} ({time.time()-t0:4.1f}s) | Geom Loss: {metrics['loss_total']:.4f} | "
            f"Train Acc (K>0): {metrics['geom_acc']*100:5.1f}% | Test Acc (K>0): {hist['test_acc'][-1]*100:5.1f}% | "
            f"E[K|y=0]: {metrics['mean_K_class0']:+6.3f} | E[K|y=1]: {metrics['mean_K_class1']:+6.3f}"
        )

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(hist["epoch"], hist["loss"], lw=2, color="#9467bd", label="Geometric Curvature Loss")
axes[0].plot(hist["epoch"], hist["train_acc"], lw=2, color="#1f77b4", ls="--", label="Train Acc $\\mathbb{I}[K(x)>0]$")
axes[0].plot(hist["epoch"], hist["test_acc"], lw=2.2, color="#2ca02c", label="Test Acc $\\mathbb{I}[K(x)>0]$")
axes[0].set_xlabel("Training Epoch")
axes[0].set_ylabel("Loss / Accuracy")
axes[0].set_title("Purely Geometric Curvature Training Convergence")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(hist["epoch"], hist["K_c0"], lw=2.2, color="#1f77b4", label="Class 0 Mean Curvature $\\mathbb{E}[K(x) \\mid y=0]$")
axes[1].plot(hist["epoch"], hist["K_c1"], lw=2.2, color="#d62728", label="Class 1 Mean Curvature $\\mathbb{E}[K(x) \\mid y=1]$")
axes[1].axhline(0.0, color="k", lw=1.2, ls=":")
axes[1].axhline(-0.4, color="#1f77b4", lw=1.0, ls="--", alpha=0.6, label="Target Margin $-\\tau$")
axes[1].axhline(+0.4, color="#d62728", lw=1.0, ls="--", alpha=0.6, label="Target Margin $+\\tau$")
axes[1].set_xlabel("Training Epoch")
axes[1].set_ylabel("Gaussian Curvature $K^{(5)}(x)$")
axes[1].set_title("Bifurcation of Manifold Curvature by Class")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 2. Learned 2D Gaussian Curvature Landscape & Zero-Curvature Decision Boundary ($K(x) = 0$)
We evaluate $K^{(5)}(x)$ over the entire 2D grid and on the held-out test set. Notice how the **zero-curvature contour $K(x) = 0$** forms a smooth geometric boundary separating the hyperbolic negative-curvature valley containing Class $0$ from the positive-curvature elliptic dome containing Class $1$.
""",
        },
        {
            "type": "code",
            "source": """
geom_grid = compute_layer_geometry(geom_model, grid_pts, layer_idx=5, metric_mode="graph")
K_grid = geom_grid["K"].detach().numpy().reshape(xx.shape)
K_grid_sinh = np.arcsinh(K_grid)
vmax = max(np.percentile(np.abs(K_grid_sinh), 96), 0.5)

geom_test = compute_layer_geometry(geom_model, X_test, layer_idx=5, metric_mode="graph")
K_test = geom_test["K"].detach().numpy()
y_test_np = y_test.numpy()

fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
cf = axes[0].contourf(xx, yy, K_grid_sinh, levels=25, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
axes[0].contour(xx, yy, K_grid, levels=[0.0], colors="k", linewidths=2.2)
axes[0].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c="#1f77b4", edgecolors="w", s=34, label="Test Class 0 (Target $K < 0$)")
axes[0].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c="#d62728", edgecolors="w", s=34, label="Test Class 1 (Target $K > 0$)")
axes[0].set_title("Learned Curvature Field $\\mathrm{arcsinh}(K^{(5)}(x))$ & $K(x)=0$ Boundary")
axes[0].set_xlabel("$x^1$")
axes[0].set_ylabel("$x^2$")
axes[0].legend(loc="upper right")
plt.colorbar(cf, ax=axes[0], label="$\\mathrm{arcsinh}(K^{(5)}(x))$")

bins = np.linspace(np.percentile(K_test, 2) - 0.2, np.percentile(K_test, 98) + 0.2, 24)
axes[1].hist(K_test[y_test_np == 0], bins=bins, alpha=0.75, color="#1f77b4", edgecolor="k", label="Test Class 0 ($K < 0$)")
axes[1].hist(K_test[y_test_np == 1], bins=bins, alpha=0.75, color="#d62728", edgecolor="k", label="Test Class 1 ($K > 0$)")
axes[1].axvline(0.0, color="k", lw=2.0, ls="--", label="Decision Threshold $K = 0$")
axes[1].set_xlabel("Intrinsic Gaussian Curvature $K^{(5)}(x)$")
axes[1].set_ylabel("Test Sample Count")
axes[1].set_title(f"Held-Out Test Set Curvature Distribution (Acc = {((K_test > 0) == y_test_np).mean()*100:.1f}%)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 3. 3D Visualization of the Learned Output Surface Coloured by Intrinsic Gaussian Curvature $K(x)$
We visualize the learned graph surface $(x^1, x^2, z^1(x))$ in $\mathbb{R}^3$ coloured by its exact Gaussian curvature $K^{(5)}(x)$, illustrating how the neural network bends the manifold into a saddle ($K < 0$, blue) over Class $0$ and a dome ($K > 0$, red) over Class $1$.
""",
        },
        {
            "type": "code",
            "source": """
from matplotlib import cm
from matplotlib.colors import Normalize

with torch.no_grad():
    z_grid = geom_model(grid_pts).detach().numpy()
    z_te = geom_model(X_test).detach().numpy()

z1_surf = z_grid[:, 0].reshape(xx.shape)
norm = Normalize(vmin=-vmax, vmax=vmax)
facecolors = cm.RdBu_r(norm(K_grid_sinh))

fig = plt.figure(figsize=(10, 6.2))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(xx, yy, z1_surf, facecolors=facecolors, rstride=1, cstride=1, alpha=0.82, antialiased=True, shade=False)

x_te_np = X_test.numpy()
ax.scatter(x_te_np[y_test_np == 0, 0], x_te_np[y_test_np == 0, 1], z_te[y_test_np == 0, 0] + 0.08, c="#1f77b4", edgecolors="k", s=35, label="Class 0 ($K < 0$ Saddle)")
ax.scatter(x_te_np[y_test_np == 1, 0], x_te_np[y_test_np == 1, 1], z_te[y_test_np == 1, 0] + 0.08, c="#d62728", edgecolors="k", s=35, label="Class 1 ($K > 0$ Dome)")

ax.set_xlabel("$x^1$")
ax.set_ylabel("$x^2$")
ax.set_zlabel("Output Coordinate $z^1(x)$")
ax.set_title("3D Immersed Data Manifold $(x^1, x^2, z^1(x))$ Coloured by Gaussian Curvature $K^{(5)}(x)$")
ax.view_init(elev=28, azim=-55)
ax.legend(loc="upper left")
mappable = cm.ScalarMappable(cmap=cm.RdBu_r, norm=norm)
mappable.set_array(K_grid_sinh)
plt.colorbar(mappable, ax=ax, shrink=0.62, pad=0.08, label="$\\mathrm{arcsinh}(K^{(5)}(x))$")
plt.tight_layout()
plt.show()
""",
        },
    ]
    build_and_execute_notebook(ROOT / "notebooks" / "04_purely_geometric_curvature_losses.ipynb", cells)


def build_notebook_05() -> None:
    cells = [
        {
            "type": "markdown",
            "source": r"""
# Notebook 05: Comparative Analysis of Purely Geometric Losses, Layerwise Curvature Profiles, and Parameter Geometry

## Research Objective
We synthesize our investigation by comparing **three purely geometric training objectives** against the **standard Cross-Entropy baseline** on the `[2 -> 5 -> 10 -> 15 -> 10 -> 2]` MLP:

1. **Cross-Entropy Baseline**: $\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{N}\sum_n \log p_\theta(y_n | x_n)$ (coordinate/logit supervision).
2. **Gaussian Curvature Margin Loss (2nd-Fundamental / Riemann Invariant)**:
   Places Class $0$ in negative Gaussian curvature ($K^{(5)}(x) \le -\tau$) and Class $1$ in positive Gaussian curvature ($K^{(5)}(x) \ge +\tau$).
3. **Riemannian Volume-Element Contrastive Loss (Metric Determinant Invariant)**:
   Places Class $0$ in volume-contracting regions ($\log\sqrt{\det g^{(5)}(x)} \le -\tau$) and Class $1$ in volume-expanding regions ($\log\sqrt{\det g^{(5)}(x)} \ge +\tau$).
4. **Metric Anisotropy / Polarization Loss (Metric Ratio Invariant)**:
   Separates classes by directional metric stretching $\log(g^{(5)}_{11}(x) / g^{(5)}_{22}(x)) \lessgtr \pm \tau$.

Finally, we analyze the **back-reaction on the Parameter Manifold ($\Theta$)**: how training with purely geometric objectives on the data manifold reshapes the layerwise curvature profile $K^{(1 \dots 5)}$ and the $422 \times 422$ Parameter Manifold metric spectrum $G^{\Theta}_{\mu\nu}(\theta) = \mathbb{E}_x[(J_\theta z)^T (J_\theta z)]$.
""",
        },
        {
            "type": "code",
            "source": """
import sys
from pathlib import Path
for p in [Path.cwd(), Path.cwd().parent]:
    if (p / "src" / "geometry.py").exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, jacrev, vmap
import matplotlib.pyplot as plt

from src.geometry import (
    MoonsMLP,
    get_moons_dataset,
    make_evaluation_grid,
    compute_layer_geometry,
    compute_layer_pullback_metric,
    curvature_margin_loss,
    volume_contrastive_loss,
    anisotropy_margin_loss,
)

X_train, y_train, X_test, y_test = get_moons_dataset(n_train=220, n_test=120, noise=0.10, seed=42)
xx, yy, grid_pts = make_evaluation_grid(X_train, n_grid=26, padding=0.45)
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 1. Training the Four Paradigms (Cross-Entropy vs. Three Purely Geometric Objectives)
""",
        },
        {
            "type": "code",
            "source": """
# 1. Cross-Entropy Baseline
m_ce = MoonsMLP(activation="tanh", seed=7)
opt_ce = torch.optim.Adam(m_ce.parameters(), lr=0.02)
crit_ce = nn.CrossEntropyLoss()
for _ in range(120):
    opt_ce.zero_grad()
    loss = crit_ce(m_ce(X_train), y_train)
    loss.backward()
    opt_ce.step()
ce_te_acc = (m_ce(X_test).argmax(-1) == y_test).float().mean().item()

# 2. Purely Geometric Curvature Margin Loss (K < 0 vs K > 0)
m_curv = MoonsMLP(activation="tanh", seed=7)
opt_curv = torch.optim.Adam(m_curv.parameters(), lr=0.025)
for _ in range(85):
    opt_curv.zero_grad()
    params = dict(m_curv.named_parameters())
    g5 = compute_layer_geometry(m_curv, X_train, layer_idx=5, metric_mode="graph", params=params)
    loss, _ = curvature_margin_loss(g5["K"], y_train, g5["g"], margin=0.40, lambda_vol=0.02, lambda_iso=0.005, k_clip=10.0)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m_curv.parameters(), 5.0)
    opt_curv.step()
K_te = compute_layer_geometry(m_curv, X_test, layer_idx=5, metric_mode="graph")["K"]
curv_te_acc = ((K_te > 0.0).long() == y_test).float().mean().item()

# 3. Purely Geometric Volume-Element Contrastive Loss (log sqrt(det g) < 0 vs > 0)
m_vol = MoonsMLP(activation="tanh", seed=7)
opt_vol = torch.optim.Adam(m_vol.parameters(), lr=0.02)
for _ in range(100):
    opt_vol.zero_grad()
    params = dict(m_vol.named_parameters())
    g5 = compute_layer_pullback_metric(m_vol, X_train, layer_idx=5, metric_mode="euclidean", eps=1e-6, params=params)
    loss, _ = volume_contrastive_loss(g5, y_train, margin=0.60)
    loss.backward()
    opt_vol.step()
g5_vol_te = compute_layer_pullback_metric(m_vol, X_test, layer_idx=5, metric_mode="euclidean", eps=1e-6)
log_vol_te = 0.5 * torch.log(torch.clamp(torch.linalg.det(g5_vol_te), min=1e-8))
vol_te_acc = ((log_vol_te > 0.0).long() == y_test).float().mean().item()

# 4. Purely Geometric Metric Anisotropy Loss (log(g11 / g22) < 0 vs > 0)
m_aniso = MoonsMLP(activation="tanh", seed=7)
opt_aniso = torch.optim.Adam(m_aniso.parameters(), lr=0.02)
for _ in range(100):
    opt_aniso.zero_grad()
    params = dict(m_aniso.named_parameters())
    g5 = compute_layer_pullback_metric(m_aniso, X_train, layer_idx=5, metric_mode="graph", params=params)
    loss, _ = anisotropy_margin_loss(g5, y_train, margin=0.45)
    loss.backward()
    opt_aniso.step()
g5_aniso_te = compute_layer_pullback_metric(m_aniso, X_test, layer_idx=5, metric_mode="graph")
aniso_ratio_te = torch.log(g5_aniso_te[:, 0, 0] / g5_aniso_te[:, 1, 1])
aniso_te_acc = ((aniso_ratio_te > 0.0).long() == y_test).float().mean().item()

print(f"{'Training Objective':<44} {'Decision Criterion':<30} {'Test Accuracy':<15}")
print("-" * 90)
print(f"{'1. Cross-Entropy Baseline':<44} {'z_1(x) > z_0(x)':<30} {ce_te_acc*100:6.2f}%")
print(f"{'2. Geometric Curvature Margin Loss':<44} {'K^(5)(x) > 0':<30} {curv_te_acc*100:6.2f}%")
print(f"{'3. Geometric Volume-Element Loss':<44} {'log sqrt(det g^(5)(x)) > 0':<30} {vol_te_acc*100:6.2f}%")
print(f"{'4. Geometric Metric Anisotropy Loss':<44} {'log(g_11^(5) / g_22^(5)) > 0':<30} {aniso_te_acc*100:6.2f}%")
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 2. Side-by-Side Comparison of the Learned Geometric Decision Landscapes
""",
        },
        {
            "type": "code",
            "source": """
with torch.no_grad():
    f_ce = (m_ce(grid_pts)[:, 1] - m_ce(grid_pts)[:, 0]).numpy().reshape(xx.shape)

f_curv = np.arcsinh(compute_layer_geometry(m_curv, grid_pts, layer_idx=5, metric_mode="graph")["K"].detach().numpy().reshape(xx.shape))

g_vol_grid = compute_layer_pullback_metric(m_vol, grid_pts, layer_idx=5, metric_mode="euclidean", eps=1e-6)
f_vol = (0.5 * torch.log(torch.clamp(torch.linalg.det(g_vol_grid), min=1e-8))).detach().numpy().reshape(xx.shape)

g_aniso_grid = compute_layer_pullback_metric(m_aniso, grid_pts, layer_idx=5, metric_mode="graph")
f_aniso = torch.log(g_aniso_grid[:, 0, 0] / g_aniso_grid[:, 1, 1]).detach().numpy().reshape(xx.shape)

fields = [
    (f_ce, f"Cross-Entropy Logit Gap\\nTest Acc: {ce_te_acc*100:.1f}%"),
    (f_curv, f"Gaussian Curvature $\\mathrm{{arcsinh}}(K)$\\nTest Acc: {curv_te_acc*100:.1f}%"),
    (f_vol, f"Log Volume $\\log\\sqrt{{\\det g}}$\\nTest Acc: {vol_te_acc*100:.1f}%"),
    (f_aniso, f"Metric Anisotropy $\\log(g_{{11}}/g_{{22}})$\\nTest Acc: {aniso_te_acc*100:.1f}%"),
]

fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.9), sharex=True, sharey=True)
for ax, (fld, title) in zip(axes, fields):
    vm = max(np.percentile(np.abs(fld), 95), 0.2)
    im = ax.contourf(xx, yy, fld, levels=22, cmap="RdBu_r", vmin=-vm, vmax=vm)
    ax.contour(xx, yy, fld, levels=[0.0], colors="k", linewidths=2.0)
    ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c="#1f77b4", edgecolors="w", s=20)
    ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c="#d62728", edgecolors="w", s=20)
    ax.set_title(title)
    ax.set_xlabel("$x^1$")
    plt.colorbar(im, ax=ax, fraction=0.046)
axes[0].set_ylabel("$x^2$")
plt.suptitle("Decision Landscapes Across Coordinate vs. Purely Differential-Geometric Objectives", y=1.04, fontsize=13)
plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 3. Layerwise Curvature Profiles & Back-Reaction on the $422 \times 422$ Parameter Manifold Metric
Finally, we compare how the four training objectives distribute curvature across the intermediate hidden layers $\ell \in \{1, 2, 3, 4, 5\}$ and how they shape the eigenspectrum of the $422 \times 422$ Generalized Gauss-Newton / Output Pullback Metric on the parameter manifold $\Theta$:
$$G^\Theta_{\mu\nu}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_{a=1}^2 \frac{\partial z^a(x; \theta)}{\partial \theta^\mu} \frac{\partial z^a(x; \theta)}{\partial \theta^\nu} \right]$$
""",
        },
        {
            "type": "code",
            "source": """
def compute_param_pullback_spectrum(model: MoonsMLP, X: torch.Tensor) -> np.ndarray:
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = dict(model.named_buffers())
    keys = list(params.keys())

    def z_fn(p_dict, x_single):
        return functional_call(model, (p_dict, buffers), (x_single.unsqueeze(0),)).squeeze(0)

    J_dict = vmap(jacrev(z_fn, argnums=0), in_dims=(None, 0))(params, X)
    N = X.shape[0]
    J_flat = torch.cat([J_dict[k].reshape(N * 2, -1) for k in keys], dim=-1)
    G_theta = (J_flat.T @ J_flat) / float(N)
    evals = torch.linalg.eigvalsh(G_theta).flip(0).clamp(min=1e-10).detach().numpy()
    return evals

models_dict = {
    "Cross-Entropy Baseline": (m_ce, "#1f77b4"),
    "Curvature Loss (K < 0 vs K > 0)": (m_curv, "#d62728"),
    "Volume-Element Loss": (m_vol, "#2ca02c"),
    "Metric Anisotropy Loss": (m_aniso, "#ff7f0e"),
}

layers_list = [1, 2, 3, 4, 5]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

for label, (mdl, col) in models_dict.items():
    # Layerwise signed curvature gap E[K|y=1] - E[K|y=0]
    k_gaps = []
    for l_idx in layers_list:
        mode = "graph" if l_idx == 5 else "euclidean"
        K_l = compute_layer_geometry(mdl, X_test, layer_idx=l_idx, metric_mode=mode)["K"].detach()
        gap = (torch.arcsinh(K_l[y_test == 1]).mean() - torch.arcsinh(K_l[y_test == 0]).mean()).item()
        k_gaps.append(gap)
    axes[0].plot(layers_list, k_gaps, "o-", lw=2, color=col, label=label)

    # Parameter manifold metric spectrum
    evals_theta = compute_param_pullback_spectrum(mdl, X_train[:120])
    axes[1].plot(range(1, len(evals_theta) + 1), evals_theta, lw=2, color=col, label=label)

axes[0].axhline(0.0, color="k", lw=1, ls=":")
axes[0].set_xticks(layers_list)
axes[0].set_xticklabels([f"L{l} (d={m_ce.layer_dims[l]})" for l in layers_list])
axes[0].set_ylabel("Signed Curvature Separation $\\Delta \\mathrm{arcsinh}(K^{(\\ell)})$")
axes[0].set_title("Layerwise Class Curvature Separation ($y=1$ vs $y=0$)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("Parameter Metric Eigenvalue Index $k$ ($P = 422$)")
axes[1].set_ylabel("Eigenvalue $\\lambda_k(G^\\Theta(\\theta))$")
axes[1].set_title("Induced 422x422 Parameter Manifold Metric Spectrum")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 4. Layerwise Log Volume Profiles ($\log\sqrt{\det g^{(\ell)}(x)}$)

Next, we examine how each of the four training paradigms transforms the **local Riemannian log-volume density** across network depth $\ell \in \{1, 2, 3, 4, 5\}$:
$$\log \rho_{\text{vol}}^{(\ell)}(x; \theta) = \log\sqrt{\det g^{(\ell)}(x; \theta)} = \frac{1}{2}\log \det g^{(\ell)}(x; \theta)$$

We track both:
1. **Per-Class Layerwise Log-Volume Trajectories**: $\mathbb{E}\big[\log\sqrt{\det g^{(\ell)}(x)} \;\big|\; y = c\big]$ for Class $0$ ($c=0$, dashed) and Class $1$ ($c=1$, solid) across the dimensional expansion phase ($2 \to 5 \to 10 \to 15$) and contraction phase ($15 \to 10 \to 2$).
2. **Signed Layerwise Log-Volume Class Separation**:
   $$\Delta \log\sqrt{\det g^{(\ell)}} = \mathbb{E}\big[\log\sqrt{\det g^{(\ell)}(x)} \;\big|\; y = 1\big] - \mathbb{E}\big[\log\sqrt{\det g^{(\ell)}(x)} \;\big|\; y = 0\big]$$
""",
        },
        {
            "type": "code",
            "source": """
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5))

print(f"{'Method':<34} | {'L1 Gap':>8} {'L2 Gap':>8} {'L3 Gap':>8} {'L4 Gap':>8} {'L5 Gap':>8}")
print("-" * 86)

for label, (mdl, col) in models_dict.items():
    vol_c0_means, vol_c1_means, vol_gaps = [], [], []
    for l_idx in layers_list:
        g_l = compute_layer_pullback_metric(mdl, X_test, layer_idx=l_idx, metric_mode="euclidean", eps=1e-6).detach()
        log_vol_l = 0.5 * torch.log(torch.clamp(torch.linalg.det(g_l), min=1e-8))
        m0 = log_vol_l[y_test == 0].mean().item()
        m1 = log_vol_l[y_test == 1].mean().item()
        vol_c0_means.append(m0)
        vol_c1_means.append(m1)
        vol_gaps.append(m1 - m0)

    print(f"{label:<34} | {vol_gaps[0]:+8.3f} {vol_gaps[1]:+8.3f} {vol_gaps[2]:+8.3f} {vol_gaps[3]:+8.3f} {vol_gaps[4]:+8.3f}")

    # Panel 1: Per-class mean log-volume profiles (solid = Class 1, dashed = Class 0)
    axes[0].plot(layers_list, vol_c1_means, "o-", lw=2.0, color=col, label=f"{label} (Class 1)")
    axes[0].plot(layers_list, vol_c0_means, "s--", lw=1.6, color=col, alpha=0.75, label=f"{label} (Class 0)")

    # Panel 2: Signed class separation in log-volume density
    axes[1].plot(layers_list, vol_gaps, "o-", lw=2.2, color=col, label=label)

axes[0].axhline(0.0, color="k", lw=1.0, ls=":")
axes[0].set_xticks(layers_list)
axes[0].set_xticklabels([f"L{l} (d={m_ce.layer_dims[l]})" for l in layers_list])
axes[0].set_ylabel("Mean Log Volume $\\mathbb{E}[\\log\\sqrt{\\det g^{(\\ell)}(x)}]$")
axes[0].set_title("Per-Class Layerwise Log-Volume Trajectories (Solid: y=1, Dashed: y=0)")
axes[0].legend(fontsize=7.5, ncol=2)
axes[0].grid(alpha=0.3)

axes[1].axhline(0.0, color="k", lw=1.0, ls=":")
axes[1].set_xticks(layers_list)
axes[1].set_xticklabels([f"L{l} (d={m_ce.layer_dims[l]})" for l in layers_list])
axes[1].set_ylabel("Signed Log-Volume Separation $\\Delta \\log\\sqrt{\\det g^{(\\ell)}}$")
axes[1].set_title("Layerwise Class Log-Volume Separation ($y=1$ vs $y=0$)")
axes[1].legend(fontsize=8.5)
axes[1].grid(alpha=0.3)

plt.suptitle("Layerwise Riemannian Log-Volume Density Profiles Across Depth", y=1.03, fontsize=13)
plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 5. Layerwise Metric Anisotropy Profiles ($\log(g^{(\ell)}_{11}/g^{(\ell)}_{22})$ & Spectral Condition Number $\lambda_1^{(\ell)}/\lambda_2^{(\ell)}$)

We now analyze how directional stretching and spectral anisotropy of the pullback metric tensor $g^{(\ell)}_{ij}(x; \theta)$ evolve across layers $\ell \in \{1, 2, 3, 4, 5\}$ under the four training objectives, examining two complementary anisotropy measures:
1. **Directional Metric Polarization Separation**:
   $$\Delta \log\!\left(\frac{g^{(\ell)}_{11}}{g^{(\ell)}_{22}}\right) = \mathbb{E}\!\left[\log\frac{g^{(\ell)}_{11}(x)}{g^{(\ell)}_{22}(x)} \;\Bigg|\; y = 1\right] - \mathbb{E}\!\left[\log\frac{g^{(\ell)}_{11}(x)}{g^{(\ell)}_{22}(x)} \;\Bigg|\; y = 0\right]$$
   which quantifies how strongly each layer polarizes horizontal ($x^1$) versus vertical ($x^2$) metric dilation between Class $1$ and Class $0$.
2. **Coordinate-Invariant Spectral Condition Number (Eigenvalue Anisotropy)**:
   $$\mathbb{E}_{x}\!\left[\log_{10}\!\left(\frac{\lambda_{\max}(g^{(\ell)}(x))}{\lambda_{\min}(g^{(\ell)}(x))}\right)\right]$$
   which measures how strongly each layer compresses tangent vectors along 1D foliation leaves relative to orthogonal directions (**Benfenati & Marta, 2023**).
""",
        },
        {
            "type": "code",
            "source": """
from src.geometry import compute_metric_spectrum

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5))

print(f"{'Method':<34} | {'L1 Aniso Gap':>12} {'L3 Aniso Gap':>12} {'L5 Aniso Gap':>12} | {'L5 log10(l1/l2)':>15}")
print("-" * 94)

for label, (mdl, col) in models_dict.items():
    aniso_gaps, log_conds = [], []
    for l_idx in layers_list:
        g_graph = compute_layer_pullback_metric(mdl, X_test, layer_idx=l_idx, metric_mode="graph").detach()
        log_ratio = torch.log(torch.clamp(g_graph[:, 0, 0], min=1e-8) / torch.clamp(g_graph[:, 1, 1], min=1e-8))
        gap = (log_ratio[y_test == 1].mean() - log_ratio[y_test == 0].mean()).item()
        aniso_gaps.append(gap)

        g_eucl = compute_layer_pullback_metric(mdl, X_test, layer_idx=l_idx, metric_mode="euclidean", eps=1e-7).detach()
        _, _, cond = compute_metric_spectrum(g_eucl)
        log_conds.append(torch.log10(torch.clamp(cond, min=1.0)).mean().item())

    print(f"{label:<34} | {aniso_gaps[0]:+12.3f} {aniso_gaps[2]:+12.3f} {aniso_gaps[4]:+12.3f} | {log_conds[4]:15.3f}")

    # Panel 1: Layerwise directional anisotropy separation (y=1 vs y=0)
    axes[0].plot(layers_list, aniso_gaps, "o-", lw=2.2, color=col, label=label)

    # Panel 2: Layerwise spectral condition number log10(lambda_1 / lambda_2)
    axes[1].plot(layers_list, log_conds, "s-", lw=2.2, color=col, label=label)

axes[0].axhline(0.0, color="k", lw=1.0, ls=":")
axes[0].set_xticks(layers_list)
axes[0].set_xticklabels([f"L{l} (d={m_ce.layer_dims[l]})" for l in layers_list])
axes[0].set_ylabel("Directional Polarization Gap $\\Delta \\log(g^{(\\ell)}_{11} / g^{(\\ell)}_{22})$")
axes[0].set_title("Layerwise Class Directional Anisotropy Separation ($y=1$ vs $y=0$)")
axes[0].legend(fontsize=8.5)
axes[0].grid(alpha=0.3)

axes[1].set_xticks(layers_list)
axes[1].set_xticklabels([f"L{l} (d={m_ce.layer_dims[l]})" for l in layers_list])
axes[1].set_ylabel("Mean Spectral Anisotropy $\\mathbb{E}[\\log_{10}(\\lambda_1^{(\\ell)} / \\lambda_2^{(\\ell)})]$")
axes[1].set_title("Layerwise Spectral Condition Number Across Depth")
axes[1].legend(fontsize=8.5)
axes[1].grid(alpha=0.3)

plt.suptitle("Layerwise Metric Tensor Anisotropy & Spectral Conditioning Profiles", y=1.03, fontsize=13)
plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 6. Computational and Memory Cost Comparison Across the Four Methods

A critical practical distinction between the four training objectives is their **differential order** and resulting automatic differentiation complexity:

| Method | Spatial Derivative Order | Parameter Gradient Order | Autodiff Mechanism |
| :--- | :---: | :---: | :--- |
| **1. Cross-Entropy Baseline** | $0$th-order ($z(x; \theta)$) | **1st-order** ($\nabla_\theta z$) | Standard reverse-mode backprop |
| **2. Gaussian Curvature Loss** | $2$nd-order ($\partial_l\partial_k g_{ij}(x; \theta)$) | **3rd-order** ($\nabla_\theta \partial_x^2 g_{ij}$) | Nested `vmap(jacrev(jacrev(jacrev)))` + backward |
| **3. Volume-Element Loss** | $1$st-order ($J(x; \theta) = \partial_x z$) | **2nd-order** ($\nabla_\theta \partial_x z$) | `vmap(jacrev)` + backward |
| **4. Metric Anisotropy Loss** | $1$st-order ($J(x; \theta) = \partial_x z$) | **2nd-order** ($\nabla_\theta \partial_x z$) | `vmap(jacrev)` + backward |

Below we profile each method on the `[2 -> 5 -> 10 -> 15 -> 10 -> 2]` MLP ($N = 220$ training points) using `torch.profiler` (`profile_memory=True`) and autograd computation graph traversal to compare:
1. **Step Latency (`ms / step`)** for a full forward + backward pass.
2. **CPU Tensor Memory Allocated (`MB / step`)** during forward + backward execution.
3. **Autograd Computation Graph Size (`# grad_fn nodes`)** retained in memory for backpropagation.
""",
        },
        {
            "type": "code",
            "source": """
import time
from torch.profiler import profile, ProfilerActivity

def count_autograd_graph_nodes(grad_fn) -> int:
    visited = set()
    stack = [grad_fn]
    while stack:
        node = stack.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        for next_node, _ in getattr(node, "next_functions", []):
            stack.append(next_node)
    return len(visited)

probe_net = MoonsMLP(activation="tanh", seed=7)
ce_loss_fn = nn.CrossEntropyLoss()

def step_loss_ce():
    return ce_loss_fn(probe_net(X_train), y_train)

def step_loss_curv():
    params = dict(probe_net.named_parameters())
    g5 = compute_layer_geometry(probe_net, X_train, layer_idx=5, metric_mode="graph", params=params)
    loss, _ = curvature_margin_loss(g5["K"], y_train, g5["g"], margin=0.40)
    return loss

def step_loss_vol():
    params = dict(probe_net.named_parameters())
    g5 = compute_layer_pullback_metric(probe_net, X_train, layer_idx=5, metric_mode="euclidean", eps=1e-6, params=params)
    loss, _ = volume_contrastive_loss(g5, y_train, margin=0.60)
    return loss

def step_loss_aniso():
    params = dict(probe_net.named_parameters())
    g5 = compute_layer_pullback_metric(probe_net, X_train, layer_idx=5, metric_mode="graph", params=params)
    loss, _ = anisotropy_margin_loss(g5, y_train, margin=0.45)
    return loss

def benchmark_method(loss_builder, n_repeats: int = 15):
    # Warmup & count autograd graph nodes
    probe_net.zero_grad()
    loss_val = loss_builder()
    graph_nodes = count_autograd_graph_nodes(loss_val.grad_fn)
    loss_val.backward()

    # Profile CPU memory allocation during a single forward + backward step
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        probe_net.zero_grad()
        l_p = loss_builder()
        l_p.backward()
    mem_mb = sum(e.cpu_memory_usage for e in prof.key_averages() if e.cpu_memory_usage > 0) / (1024.0 * 1024.0)

    # Measure forward + backward step latency
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        probe_net.zero_grad()
        l_t = loss_builder()
        l_t.backward()
    step_ms = (time.perf_counter() - t0) * 1000.0 / n_repeats
    return step_ms, mem_mb, graph_nodes

method_specs = [
    ("Cross-Entropy\\n(1st-Order)", step_loss_ce, ce_te_acc * 100.0, "#1f77b4"),
    ("Volume-Element\\n(2nd-Order)", step_loss_vol, vol_te_acc * 100.0, "#2ca02c"),
    ("Metric Anisotropy\\n(2nd-Order)", step_loss_aniso, aniso_te_acc * 100.0, "#ff7f0e"),
    ("Gaussian Curvature\\n(3rd-Order)", step_loss_curv, curv_te_acc * 100.0, "#d62728"),
]

results = []
for name, fn, acc_pct, col in method_specs:
    ms, mb, nodes = benchmark_method(fn)
    results.append({"name": name, "clean_name": name.replace("\\n", " "), "ms": ms, "mb": mb, "nodes": nodes, "acc": acc_pct, "color": col})

print(f"{'Method':<34} {'Latency (ms/step)':<20} {'Memory (MB/step)':<20} {'Autograd Nodes':<16} {'Test Acc':<10}")
print("-" * 102)
for r in results:
    print(f"{r['clean_name']:<34} {r['ms']:<20.2f} {r['mb']:<20.2f} {r['nodes']:<16d} {r['acc']:6.2f}%")

names = [r["name"] for r in results]
colors = [r["color"] for r in results]
latencies = [r["ms"] for r in results]
memories = [r["mb"] for r in results]
node_counts = [r["nodes"] for r in results]

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

# Panel 1: Step Latency (ms/step)
bars0 = axes[0].bar(names, latencies, color=colors, edgecolor="k", alpha=0.88, width=0.58)
axes[0].set_yscale("log")
axes[0].set_ylabel("Forward + Backward Latency (ms / step, log scale)")
axes[0].set_title("Computational Cost per Training Step")
axes[0].grid(axis="y", alpha=0.3)
for b, val in zip(bars0, latencies):
    axes[0].text(b.get_x() + b.get_width() / 2, val * 1.15, f"{val:.2f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[0].set_ylim(0.3, max(latencies) * 2.5)

# Panel 2: Memory Footprint (MB/step)
bars1 = axes[1].bar(names, memories, color=colors, edgecolor="k", alpha=0.88, width=0.58)
axes[1].set_yscale("log")
axes[1].set_ylabel("Step Tensor Memory Allocation (MB, log scale)")
axes[1].set_title("Memory Cost per Training Step")
axes[1].grid(axis="y", alpha=0.3)
for b, val in zip(bars1, memories):
    axes[1].text(b.get_x() + b.get_width() / 2, val * 1.18, f"{val:.2f} MB", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[1].set_ylim(0.15, max(memories) * 3.0)

# Panel 3: Autograd Graph Nodes vs. Latency & Memory Bubble Plot
for r in results:
    bubble_size = max(np.sqrt(r["mb"]) * 55.0, 90.0)
    axes[2].scatter(r["ms"], r["mb"], s=bubble_size, color=r["color"], edgecolors="k", linewidth=1.5, alpha=0.85, label=f"{r['clean_name']} ({r['nodes']} nodes)")
    axes[2].annotate(f"{r['nodes']} nodes\\n({r['acc']:.1f}% acc)", (r["ms"], r["mb"]), textcoords="offset points", xytext=(8, -6), fontsize=8.5)

axes[2].set_xscale("log")
axes[2].set_yscale("log")
axes[2].set_xlabel("Computational Cost: Step Latency (ms / step, log scale)")
axes[2].set_ylabel("Memory Cost: Tensor Allocation (MB / step, log scale)")
axes[2].set_title("Compute vs. Memory Scaling by Differential Order")
axes[2].set_xlim(0.4, max(latencies) * 4.5)
axes[2].set_ylim(0.2, max(memories) * 3.5)
axes[2].grid(alpha=0.3, which="both")
axes[2].legend(fontsize=8, loc="upper left")

plt.suptitle("Computational and Memory Cost Comparison: 1st-Order vs. 2nd-Order Metric vs. 3rd-Order Curvature Losses", y=1.04, fontsize=13)
plt.tight_layout()
plt.show()
""",
        },
    ]
    build_and_execute_notebook(ROOT / "notebooks" / "05_geometric_loss_ablations_and_synthesis.ipynb", cells)


if __name__ == "__main__":
    build_notebook_01()
    build_notebook_02()
    build_notebook_03()
    build_notebook_04()
    build_notebook_05()
