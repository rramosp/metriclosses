"""Build and execute all 7 differential geometry research notebooks."""

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
We study how a configurable Multi-Layer Perceptron (default **`[2 -> 5 -> 10 -> 15 -> 10 -> 2]`**) transforms a 2D synthetic classification manifold across layers $\ell \in \{1, \dots, L\}$.

At each layer $\ell$, the representation map $\Phi^{(\ell)}_\theta : \mathbb{R}^2 \to \mathbb{R}^{d_\ell}$ induces a **$2 \times 2$ Riemannian pullback metric tensor** on the input coordinate chart $x = (x^1, x^2) \in \mathbb{R}^2$:
$$g^{(\ell)}_{ij}(x; \theta) = \left( (J^{(\ell)}(x))^T J^{(\ell)}(x) \right)_{ij} = \sum_{a=1}^{d_\ell} \frac{\partial \Phi^{(\ell)}_a(x)}{\partial x^i} \frac{\partial \Phi^{(\ell)}_a(x)}{\partial x^j}, \quad i, j \in \{1, 2\}$$

From $g^{(\ell)}(x; \theta)$ we extract two fundamental geometric diagnostics:
1. **Riemannian Volume Magnification**: $\mathrm{d}V^{(\ell)}(x) = \sqrt{\det g^{(\ell)}(x; \theta)}$, measuring how local area elements expand or contract across layers (**Hauser & Ray, 2017**; **Zavatone-Veth et al., 2023**).
2. **Metric Anisotropy (Condition Number)**: $\kappa(g^{(\ell)}(x)) = \lambda_{\max}(g^{(\ell)}) / \lambda_{\min}(g^{(\ell)})$ and **Tissot Indicatrices**, revealing how dimension-contracting layers compress directions tangent to equivalence classes while stretching directions orthogonal to the decision boundary (**Benfenati & Marta, 2023**).
""",
        },
        {
            "type": "code",
            "source": r"""
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path.cwd()
for p in [Path.cwd(), Path.cwd().parent, Path.cwd() / Path("projects") / Path("diffgeom")]:
    if (p / "src" / "geometry.py").exists():
        PROJECT_ROOT = p
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

DATASET_NAME = CFG["dataset"]["name"]
N_CLASSES = CFG["dataset"].get("n_classes", 2)
DATASET_PARAMS = dict(CFG["dataset"].get("params", {}))
LAYER_DIMS = list(CFG["model"]["layer_dims"])
HIDDEN_ACTIVATION = CFG["model"].get("hidden_activation", "tanh")
OUTPUT_ACTIVATION = CFG["model"].get("output_activation", "identity")

import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from src.geometry import (
    MoonsMLP,
    get_synthetic_dataset,
    make_evaluation_grid,
    compute_layer_pullback_metric,
    compute_volume_element,
    compute_metric_spectrum,
)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

CE_LAYER_DIMS = [*LAYER_DIMS[:-1], max(LAYER_DIMS[-1], N_CLASSES)]
CLASS_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

X_train, y_train, X_test, y_test = get_synthetic_dataset(
    dataset_name=DATASET_NAME,
    n_classes=N_CLASSES,
    input_dim=CE_LAYER_DIMS[0],
    **DATASET_PARAMS,
)
xx, yy, grid_pts = make_evaluation_grid(X_train, n_grid=32, padding=0.45)
print(
    f"Dataset: {DATASET_NAME} (classes={N_CLASSES}) | Train: {X_train.shape}, Test: {X_test.shape} | "
    f"MLP: {CE_LAYER_DIMS} (hidden={HIDDEN_ACTIVATION}, out={OUTPUT_ACTIVATION})"
)
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 1. Baseline Cross-Entropy Training and Checkpoint Capture
We train the configured MLP using standard Cross-Entropy loss and record parameter checkpoints across training epochs $t \in \{0, 25, 80, 250\}$.
""",
        },
        {
            "type": "code",
            "source": r"""
model = MoonsMLP(
    layer_dims=CE_LAYER_DIMS,
    activation=HIDDEN_ACTIVATION,
    output_activation=OUTPUT_ACTIVATION,
    seed=42,
)
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
    probs_all = torch.softmax(model(grid_pts), dim=-1)
    if N_CLASSES == 2:
        decision_field = probs_all[:, 1].reshape(xx.shape).numpy()
        bnd_levels = [0.5]
    else:
        decision_field = probs_all.argmax(dim=-1).reshape(xx.shape).numpy().astype(float)
        bnd_levels = list(np.arange(0.5, N_CLASSES - 0.5, 1.0))

cf = axes[1].contourf(xx, yy, decision_field, levels=25 if N_CLASSES == 2 else N_CLASSES, cmap="coolwarm", alpha=0.85)
if decision_field.min() < bnd_levels[0] and decision_field.max() > bnd_levels[0]:
    axes[1].contour(xx, yy, decision_field, levels=bnd_levels, colors="k", linewidths=2.0)
for c in range(N_CLASSES):
    mask_c = y_test == c
    axes[1].scatter(X_test[mask_c, 0], X_test[mask_c, 1], c=CLASS_COLORS[c % len(CLASS_COLORS)], edgecolors="k", s=28, label=f"Class {c}")
axes[1].set_title("Learned Decision Regions")
axes[1].set_xlabel("$x^1$")
axes[1].set_ylabel("$x^2$")
axes[1].legend(loc="upper right")
plt.colorbar(cf, ax=axes[1], label="P(y=1 | x)" if N_CLASSES == 2 else "Predicted Class")
plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 2. Layerwise Volume Magnification $\sqrt{\det g^{(\ell)}(x)}$ and Anisotropy $\lambda_1 / \lambda_2$
We evaluate $g^{(\ell)}_{ij}(x)$ across all layers $\ell \in \{1, \dots, L\}$ on the 2D grid. Notice how the trained network concentrates high volume expansion ($\sqrt{\det g^{(\ell)}}$) and extreme directional stretching ($\lambda_1/\lambda_2$) along the nonlinear decision boundary, confirming **Zavatone-Veth et al. (2023)**.
""",
        },
        {
            "type": "code",
            "source": r"""
L = model.num_layers
fig, axes = plt.subplots(2, L, figsize=(3.4 * L, 6.6), sharex=True, sharey=True, squeeze=False)

summary_rows = []
for idx, l_idx in enumerate(range(1, L + 1)):
    d_prev, d_curr = model.layer_dims[l_idx - 1], model.layer_dims[l_idx]
    g_grid = compute_layer_pullback_metric(model, grid_pts, layer_idx=l_idx, metric_mode="euclidean", eps=1e-7)
    vol = compute_volume_element(g_grid).detach().numpy().reshape(xx.shape)
    evals, _, cond = compute_metric_spectrum(g_grid)
    log_vol = np.log10(np.clip(vol, 1e-6, None))
    log_cond = np.log10(np.clip(cond.detach().numpy().reshape(xx.shape), 1.0, 1e6))

    summary_rows.append((l_idx, d_curr, float(np.median(vol)), float(np.max(vol)), float(np.median(cond.detach().numpy()))))

    im0 = axes[0, idx].contourf(xx, yy, log_vol, levels=20, cmap="magma")
    if decision_field.min() < bnd_levels[0] and decision_field.max() > bnd_levels[0]:
        axes[0, idx].contour(xx, yy, decision_field, levels=bnd_levels, colors="cyan", linewidths=1.3, linestyles="--")
    axes[0, idx].set_title(f"Layer {l_idx} ({d_prev} -> {d_curr})\n" + rf"$\log_{{10}}\sqrt{{\det g^{{({l_idx})}}}}$")
    plt.colorbar(im0, ax=axes[0, idx], fraction=0.046)

    im1 = axes[1, idx].contourf(xx, yy, log_cond, levels=20, cmap="viridis")
    if decision_field.min() < bnd_levels[0] and decision_field.max() > bnd_levels[0]:
        axes[1, idx].contour(xx, yy, decision_field, levels=bnd_levels, colors="white", linewidths=1.3, linestyles="--")
    axes[1, idx].set_title(r"Anisotropy $\log_{10}(\lambda_1 / \lambda_2)$")
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
            "source": r"""
_, _, coarse_pts = make_evaluation_grid(X_train, n_grid=11, padding=0.3)
sample_layers = sorted(set([1, max(1, (L + 1) // 2), L]))
fig, axes = plt.subplots(1, len(sample_layers), figsize=(5.0 * len(sample_layers), 4.6), sharex=True, sharey=True, squeeze=False)

for ax, l_idx in zip(axes[0], sample_layers):
    g_c = compute_layer_pullback_metric(model, coarse_pts, layer_idx=l_idx, metric_mode="euclidean", eps=1e-6)
    evals, evecs, _ = compute_metric_spectrum(g_c)
    evals_np = np.sqrt(evals.detach().numpy())
    scale = 0.24 / (np.percentile(evals_np[:, 0], 90) + 1e-8)

    if decision_field.min() < bnd_levels[0] and decision_field.max() > bnd_levels[0]:
        ax.contour(xx, yy, decision_field, levels=bnd_levels, colors="k", linewidths=1.8, linestyles="--")
    for c in range(N_CLASSES):
        mc = y_train == c
        ax.scatter(X_train[mc, 0], X_train[mc, 1], c=CLASS_COLORS[c % len(CLASS_COLORS)], alpha=0.20, s=14)

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
axes[0, 0].set_ylabel("$x^2$")
plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 4. Symmetry Breaking Across Training Epochs
We track the pullback volume field $\sqrt{\det g^{(\ell_{\text{penult}})}(x)}$ from random initialization ($t=0$) through convergence ($t=250$).
""",
        },
        {
            "type": "code",
            "source": r"""
penult_idx = max(1, L - 1)
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharex=True, sharey=True)
probe_model = MoonsMLP(
    layer_dims=CE_LAYER_DIMS,
    activation=HIDDEN_ACTIVATION,
    output_activation=OUTPUT_ACTIVATION,
    seed=42,
)

for ax, ep in zip(axes, [0, 25, 80, 250]):
    probe_model.load_state_dict(checkpoints[ep])
    g_pen = compute_layer_pullback_metric(probe_model, grid_pts, layer_idx=penult_idx, metric_mode="euclidean", eps=1e-7)
    vol_pen = np.log10(compute_volume_element(g_pen).detach().numpy().reshape(xx.shape) + 1e-7)
    im = ax.contourf(xx, yy, vol_pen, levels=20, cmap="plasma")
    ax.set_title(f"Epoch {ep}: " + rf"$\log_{{10}}\sqrt{{\det g^{{({penult_idx})}}(x)}}$")
    ax.set_xlabel("$x^1$")
    plt.colorbar(im, ax=ax, fraction=0.046)
axes[0].set_ylabel("$x^2$")
plt.suptitle(f"Emergence of Decision-Boundary Volume Magnification During Training (Layer {penult_idx})", y=1.03)
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
We compute the Levi-Civita connection (Christoffel symbols $\Gamma^{(\ell)k}_{ij}$) and intrinsic **Gaussian curvature** $K^{(\ell)}(x; \theta)$ of the 2D immersed representation manifolds $\mathcal{M}_\ell = \Phi^{(\ell)}_\theta(\mathbb{R}^2) \subset \mathbb{R}^{d_\ell}$ across layers $\ell \in \{1, \dots, L\}$:
$$\Gamma^{(\ell)}_{kij} = \frac{1}{2}\left(\partial_j g^{(\ell)}_{ki} + \partial_i g^{(\ell)}_{kj} - \partial_k g^{(\ell)}_{ij}\right), \qquad R^{(\ell)}_{1212} = -\frac{1}{2}\left(\partial_1^2 g^{(\ell)}_{22} - 2\partial_1\partial_2 g^{(\ell)}_{12} + \partial_2^2 g^{(\ell)}_{11}\right) + g^{(\ell)km}\left(\Gamma^{(\ell)}_{k12}\Gamma^{(\ell)}_{m12} - \Gamma^{(\ell)}_{k11}\Gamma^{(\ell)}_{m22}\right)$$
$$K^{(\ell)}(x) = \frac{R^{(\ell)}_{1212}(x)}{\det g^{(\ell)}(x)}, \qquad R_{\text{scalar}}^{(\ell)}(x) = 2 K^{(\ell)}(x)$$

### Key Mathematical Insight: The 2D-to-2D Euclidean Pullback Flatness Identity at Output Width $d_L = 2$
- For hidden layers with $d_\ell > 2$, $\mathcal{M}_\ell$ is a 2D surface immersed in higher-dimensional Euclidean space $\mathbb{R}^{d_\ell}$, possessing rich non-zero intrinsic Gaussian curvature $K^{(\ell)}(x) \neq 0$.
- When $d_L = 2$, pulling back the flat 2D Euclidean metric $I_2$ via a local diffeomorphism $z = f_\theta(x) \in \mathbb{R}^2$ is merely a coordinate change of flat Euclidean space, so $K^{(L)}_{\text{eucl}}(x) \equiv 0$!
- We empirically verify this identity and compare three non-degenerate output curvature constructions:
  1. **Graph Embedding Metric (Monge Patch in $\mathbb{R}^{2+d_L}$)**: $\tilde{g}^{(L)}_{ij}(x) = \delta_{ij} + ((J^{(L)}(x))^T J^{(L)}(x))_{ij}$.
  2. **Conformal Gaussian Output Metric**: $g^{(L)}_{\text{conf}}(x) = e^{-\|z(x)\|^2 / \sigma^2} (J^{(L)}(x))^T J^{(L)}(x)$.
  3. **Penultimate Surface Metric**: $g^{(L-1)}_{\text{eucl}}(x)$ on $\mathcal{M}_{L-1}$.
""",
        },
        {
            "type": "code",
            "source": r"""
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path.cwd()
for p in [Path.cwd(), Path.cwd().parent, Path.cwd() / Path("projects") / Path("diffgeom")]:
    if (p / "src" / "geometry.py").exists():
        PROJECT_ROOT = p
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

DATASET_NAME = CFG["dataset"]["name"]
N_CLASSES = CFG["dataset"].get("n_classes", 2)
DATASET_PARAMS = dict(CFG["dataset"].get("params", {}))
LAYER_DIMS = list(CFG["model"]["layer_dims"])
HIDDEN_ACTIVATION = CFG["model"].get("hidden_activation", "tanh")
OUTPUT_ACTIVATION = CFG["model"].get("output_activation", "identity")

import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.geometry import (
    MoonsMLP,
    get_synthetic_dataset,
    make_evaluation_grid,
    compute_layer_geometry,
    compute_layer_pullback_metric,
    compute_metric_spectrum,
    _christoffel_and_curvature_from_g_single,
)

CE_LAYER_DIMS = [*LAYER_DIMS[:-1], max(LAYER_DIMS[-1], N_CLASSES)]
CLASS_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

X_train, y_train, X_test, y_test = get_synthetic_dataset(
    dataset_name=DATASET_NAME,
    n_classes=N_CLASSES,
    input_dim=CE_LAYER_DIMS[0],
    **DATASET_PARAMS,
)
xx, yy, grid_pts = make_evaluation_grid(X_train, n_grid=26, padding=0.45)

# 1. Analytical validation on a sphere of radius R=2 (K = 1/R^2 = 0.25) and hyperbolic paraboloid (K(0) = -3.0)
R_s = 2.0
g_sphere = lambda x: ((2.0 * R_s**2 / (R_s**2 + torch.sum(x**2)))**2) * torch.eye(2, dtype=x.dtype)
_, _, _, K_sphere = _christoffel_and_curvature_from_g_single(g_sphere)(torch.tensor([0.8, -0.6], dtype=torch.float64))

g_saddle = lambda x: torch.eye(2, dtype=x.dtype) + torch.stack([2.0 * x[0], -1.5 * x[1]]).unsqueeze(0).T @ torch.stack([2.0 * x[0], -1.5 * x[1]]).unsqueeze(0)
_, _, _, K_saddle = _christoffel_and_curvature_from_g_single(g_saddle)(torch.zeros(2, dtype=torch.float64))

print(f"Dataset: {DATASET_NAME} (classes={N_CLASSES}) | MLP: {CE_LAYER_DIMS} (hidden={HIDDEN_ACTIVATION}, out={OUTPUT_ACTIVATION})")
print(f"Analytical Validation | Sphere (R=2) exact K=0.250000 -> autodiff K={K_sphere.item():.6f}")
print(f"Analytical Validation | Hyperbolic Saddle exact K(0)=-3.000000 -> autodiff K={K_saddle.item():.6f}")
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 1. Empirical Verification of 2D-to-2D Euclidean Flatness vs. Non-Degenerate Output Metrics
We train our configured MLP and compare the four curvature definitions at the output and penultimate layers.
""",
        },
        {
            "type": "code",
            "source": r"""
model = MoonsMLP(
    layer_dims=CE_LAYER_DIMS,
    activation=HIDDEN_ACTIVATION,
    output_activation=OUTPUT_ACTIVATION,
    seed=42,
)
L = model.num_layers
penult_idx = max(1, L - 1)
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

geom_lL_eucl = compute_layer_geometry(model, X_test[:50], layer_idx=L, metric_mode="euclidean", eps=1e-13)
geom_lL_graph = compute_layer_geometry(model, X_test[:50], layer_idx=L, metric_mode="graph")
geom_lL_conf = compute_layer_geometry(model, X_test[:50], layer_idx=L, metric_mode="conformal_gaussian", sigma=2.5)
geom_pen_eucl = compute_layer_geometry(model, X_test[:50], layer_idx=penult_idx, metric_mode="euclidean", eps=1e-7)

print("Comparison of Curvature Formulations on Trained Network (50 test points):")
print(f"  1. Layer {L} Euclidean Pullback (d={model.layer_dims[L]}) : max |K| = {geom_lL_eucl['K'].abs().max().item():.2e}")
print(f"  2. Layer {L} Graph Immersion Metric                  : mean |K| = {geom_lL_graph['K'].abs().mean().item():.4f}, range = [{geom_lL_graph['K'].min().item():.3f}, {geom_lL_graph['K'].max().item():.3f}]")
print(f"  3. Layer {L} Conformal Gaussian Output Metric        : mean |K| = {geom_lL_conf['K'].abs().mean().item():.4f}, range = [{geom_lL_conf['K'].min().item():.3f}, {geom_lL_conf['K'].max().item():.3f}]")
print(f"  4. Layer {penult_idx} Penultimate Immersion (d={model.layer_dims[penult_idx]}) : mean |K| = {geom_pen_eucl['K'].abs().mean().item():.4f}, range = [{geom_pen_eucl['K'].min().item():.3f}, {geom_pen_eucl['K'].max().item():.3f}]")
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 2. Layerwise Gaussian Curvature Landscape $K^{(\ell)}(x)$ Across Network Depth
We plot $\mathrm{arcsinh}(K^{(\ell)}(x))$ across all layers $\ell \in \{1, \dots, L\}$ (using Euclidean immersion whenever $d_\ell > 2$ and Graph immersion when $d_\ell \le 2$).
""",
        },
        {
            "type": "code",
            "source": r"""
with torch.no_grad():
    probs_all = torch.softmax(model(grid_pts), dim=-1)
    if N_CLASSES == 2:
        decision_field = probs_all[:, 1].reshape(xx.shape).numpy()
        bnd_levels = [0.5]
    else:
        decision_field = probs_all.argmax(dim=-1).reshape(xx.shape).numpy().astype(float)
        bnd_levels = list(np.arange(0.5, N_CLASSES - 0.5, 1.0))

layers_list = list(range(1, L + 1))
fig, axes = plt.subplots(1, L, figsize=(3.5 * L, 3.7), sharex=True, sharey=True, squeeze=False)
for idx, l_idx in enumerate(layers_list):
    mode = "graph" if model.layer_dims[l_idx] <= 2 else "euclidean"
    geom = compute_layer_geometry(model, grid_pts, layer_idx=l_idx, metric_mode=mode, eps=1e-6)
    K_map = np.arcsinh(geom["K"].detach().numpy().reshape(xx.shape))
    vmax = max(np.percentile(np.abs(K_map), 96), 0.1)

    im = axes[0, idx].contourf(xx, yy, K_map, levels=22, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    if decision_field.min() < bnd_levels[0] and decision_field.max() > bnd_levels[0]:
        axes[0, idx].contour(xx, yy, decision_field, levels=bnd_levels, colors="k", linewidths=1.5, linestyles="--")
    for c in range(N_CLASSES):
        mc = y_test == c
        axes[0, idx].scatter(X_test[mc, 0], X_test[mc, 1], c=CLASS_COLORS[c % len(CLASS_COLORS)], s=8, alpha=0.4)
    axes[0, idx].set_title(f"Layer {l_idx} (d={model.layer_dims[l_idx]}, {mode})\n" + rf"$\mathrm{{arcsinh}}(K^{{({l_idx})}}(x))$")
    axes[0, idx].set_xlabel("$x^1$")
    plt.colorbar(im, ax=axes[0, idx], fraction=0.046)

axes[0, 0].set_ylabel("$x^2$")
plt.suptitle(f"Layerwise Gaussian Curvature Landscape Across {model.layer_dims} MLP", y=1.04, fontsize=13)
plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 3. Evolution of Layerwise Curvature & Christoffel Symbol Norms Across Training Epochs
We track how the mean magnitude of Gaussian curvature $\mathbb{E}[|K^{(\ell)}(x)|]$ and Christoffel connection norm $\mathbb{E}[\|\Gamma^{(\ell)}(x)\|_F]$ evolve across layers $\ell \in \{1,\dots,L\}$ from initialization ($t=0$) to convergence ($t=200$), contrasting interior class points against decision-boundary points.
""",
        },
        {
            "type": "code",
            "source": r"""
with torch.no_grad():
    p_tr_all = torch.softmax(model(X_train), dim=-1)
    max_p = p_tr_all.max(dim=-1).values
    boundary_mask = max_p < (1.0 / N_CLASSES + 0.20)
    interior_mask = ~boundary_mask
    if not interior_mask.any():
        interior_mask = torch.ones_like(boundary_mask)

probe = MoonsMLP(
    layer_dims=CE_LAYER_DIMS,
    activation=HIDDEN_ACTIVATION,
    output_activation=OUTPUT_ACTIVATION,
    seed=42,
)
epochs_list = [0, 20, 70, 200]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
colors = ["#7f7f7f", "#ff7f0e", "#1f77b4", "#2ca02c"]

for ep, col in zip(epochs_list, colors):
    probe.load_state_dict(epoch_snapshots[ep])
    k_int, k_bnd, gamma_norms = [], [], []
    for l_idx in layers_list:
        mode = "graph" if model.layer_dims[l_idx] <= 2 else "euclidean"
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
axes[0].set_ylabel(r"Mean $\mathrm{arcsinh}(|K^{(\ell)}(x)|)$")
axes[0].set_title("Curvature Evolution: Class Interior vs. Decision Boundary")
axes[0].legend(fontsize=8, ncol=2)
axes[0].grid(alpha=0.3)

axes[1].set_xticks(layers_list)
axes[1].set_xticklabels([f"L{l} (d={model.layer_dims[l]})" for l in layers_list])
axes[1].set_ylabel(r"Mean Connection Norm $\|\Gamma^{(\ell)k}_{ij}(x)\|_F$")
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
""",
        },
        {
            "type": "code",
            "source": r"""
fig, axes = plt.subplots(1, L, figsize=(3.5 * L, 3.7), sharex=True, sharey=True, squeeze=False)
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

    im = axes[0, idx].contourf(xx, yy, log_vol_grid, levels=22, cmap="RdBu_r", vmin=-vmax_v, vmax=vmax_v)
    if log_vol_grid.min() < 0.0 < log_vol_grid.max():
        axes[0, idx].contour(xx, yy, log_vol_grid, levels=[0.0], colors="gold", linewidths=1.2, linestyles=":")
    if decision_field.min() < bnd_levels[0] and decision_field.max() > bnd_levels[0]:
        axes[0, idx].contour(xx, yy, decision_field, levels=bnd_levels, colors="k", linewidths=1.5, linestyles="--")
    for c in range(N_CLASSES):
        mc = y_test == c
        axes[0, idx].scatter(X_test[mc, 0], X_test[mc, 1], c=CLASS_COLORS[c % len(CLASS_COLORS)], s=8, alpha=0.4)
    axes[0, idx].set_title(f"Layer {l_idx} (d={model.layer_dims[l_idx]})\n" + rf"$\log\sqrt{{\det g^{{({l_idx})}}(x)}}$")
    axes[0, idx].set_xlabel("$x^1$")
    plt.colorbar(im, ax=axes[0, idx], fraction=0.046)

axes[0, 0].set_ylabel("$x^2$")
plt.suptitle(f"Layerwise Log Volume Density Landscape Across {model.layer_dims} MLP", y=1.04, fontsize=13)
plt.tight_layout()
plt.show()

# Track Log-Volume evolution across training epochs
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
        lv_c1.append(lv[y_train == (N_CLASSES - 1)].mean().item())

    axes[0].plot(layers_list, lv_int, "o-", lw=2, color=col, label=f"Epoch {ep} (Interior)")
    axes[0].plot(layers_list, lv_bnd, "s--", lw=1.8, color=col, label=f"Epoch {ep} (Boundary)")
    axes[1].plot(layers_list, lv_c1, "o-", lw=2, color=col, label=f"Epoch {ep} (Class {N_CLASSES - 1})")
    axes[1].plot(layers_list, lv_c0, "s--", lw=1.8, color=col, label=f"Epoch {ep} (Class 0)")

for ax in axes:
    ax.axhline(0.0, color="k", lw=1.0, ls=":")
    ax.set_xticks(layers_list)
    ax.set_xticklabels([f"L{l} (d={model.layer_dims[l]})" for l in layers_list])
    ax.grid(alpha=0.3)

axes[0].set_ylabel(r"Mean Log Volume $\mathbb{E}[\log\sqrt{\det g^{(\ell)}(x)}]$")
axes[0].set_title("Log-Volume Evolution: Class Interior vs. Decision Boundary")
axes[0].legend(fontsize=8, ncol=2)

axes[1].set_ylabel(r"Mean Log Volume $\mathbb{E}[\log\sqrt{\det g^{(\ell)}(x)}]$")
axes[1].set_title(f"Log-Volume Evolution: Class {N_CLASSES - 1} (Solid) vs. Class 0 (Dashed)")
axes[1].legend(fontsize=8, ncol=2)

plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 5. Layerwise Metric Anisotropy Landscape Across Network Depth

To capture how each layer stretches space directionally (beyond isotropic area change), we examine **Directional Metric Polarization** $\log(g^{(\ell)}_{11} / g^{(\ell)}_{22})$ and **Spectral Anisotropy** $\log_{10}(\lambda_1 / \lambda_2)$ across layers $\ell \in \{1, \dots, L\}$.
""",
        },
        {
            "type": "code",
            "source": r"""
fig, axes = plt.subplots(2, L, figsize=(3.5 * L, 6.8), sharex=True, sharey=True, squeeze=False)
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

    im0 = axes[0, idx].contourf(xx, yy, dir_aniso_grid, levels=22, cmap="PuOr_r", vmin=-vmax_a, vmax=vmax_a)
    if decision_field.min() < bnd_levels[0] and decision_field.max() > bnd_levels[0]:
        axes[0, idx].contour(xx, yy, decision_field, levels=bnd_levels, colors="k", linewidths=1.4, linestyles="--")
    for c in range(N_CLASSES):
        mc = y_test == c
        axes[0, idx].scatter(X_test[mc, 0], X_test[mc, 1], c=CLASS_COLORS[c % len(CLASS_COLORS)], s=8, alpha=0.35)
    axes[0, idx].set_title(f"Layer {l_idx} (d={model.layer_dims[l_idx]})\n" + rf"$\log(g^{{({l_idx})}}_{{11}} / g^{{({l_idx})}}_{{22}})$")
    plt.colorbar(im0, ax=axes[0, idx], fraction=0.046)

    im1 = axes[1, idx].contourf(xx, yy, spec_aniso_grid, levels=22, cmap="viridis")
    if decision_field.min() < bnd_levels[0] and decision_field.max() > bnd_levels[0]:
        axes[1, idx].contour(xx, yy, decision_field, levels=bnd_levels, colors="white", linewidths=1.4, linestyles="--")
    axes[1, idx].set_title(rf"Spectral $\log_{{10}}(\lambda_1^{{({l_idx})}} / \lambda_2^{{({l_idx})}})$")
    axes[1, idx].set_xlabel("$x^1$")
    plt.colorbar(im1, ax=axes[1, idx], fraction=0.046)

axes[0, 0].set_ylabel("$x^2$")
axes[1, 0].set_ylabel("$x^2$")
plt.suptitle("Layerwise Directional Polarization and Spectral Anisotropy Landscapes Across Depth", y=1.02, fontsize=13)
plt.tight_layout()
plt.show()

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
        dir_gap.append((lr[y_train == (N_CLASSES - 1)].mean() - lr[y_train == 0].mean()).item())

    axes[0].plot(layers_list, spec_int, "o-", lw=2, color=col, label=f"Epoch {ep} (Interior)")
    axes[0].plot(layers_list, spec_bnd, "s--", lw=1.8, color=col, label=f"Epoch {ep} (Boundary)")
    axes[1].plot(layers_list, dir_gap, "o-", lw=2, color=col, label=f"Epoch {ep}")

axes[0].set_xticks(layers_list)
axes[0].set_xticklabels([f"L{l} (d={model.layer_dims[l]})" for l in layers_list])
axes[0].set_ylabel(r"Mean Spectral Anisotropy $\mathbb{E}[\log_{10}(\lambda_1^{(\ell)} / \lambda_2^{(\ell)})]$")
axes[0].set_title("Spectral Anisotropy Evolution: Interior vs. Boundary")
axes[0].legend(fontsize=8, ncol=2)
axes[0].grid(alpha=0.3)

axes[1].axhline(0.0, color="k", lw=1.0, ls=":")
axes[1].set_xticks(layers_list)
axes[1].set_xticklabels([f"L{l} (d={model.layer_dims[l]})" for l in layers_list])
axes[1].set_ylabel(r"Directional Polarization Gap $\Delta \log(g^{(\ell)}_{11} / g^{(\ell)}_{22})$")
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
1. **The Parameter Manifold ($\Theta \subset \mathbb{R}^P$)** equipped with the $P \times P$ **Parameter Fisher Information Matrix (FIM)**:
   $$F_{\mu\nu}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_{c=0}^{C-1} p_\theta(y=c|x) \frac{\partial \log p_\theta(y=c|x)}{\partial \theta^\mu} \frac{\partial \log p_\theta(y=c|x)}{\partial \theta^\nu} \right]$$
2. **The Data Manifold ($\mathcal{M}_0 \subset \mathbb{R}^2$)** equipped with the $2 \times 2$ **Data Information Matrix (DIM)** (**Tron, Fioresi, Novelli, & Latini, 2024/2026**; **Fasina et al., 2023**):
   $$\mathrm{DIM}_{ij}(x; \theta) = \sum_{c=0}^{C-1} p_\theta(y=c|x) \frac{\partial \log p_\theta(y=c|x)}{\partial x^i} \frac{\partial \log p_\theta(y=c|x)}{\partial x^j}$$

We further test **Halverson & Ruehle (2023)**'s *Metric Flow* framework by measuring how perturbations along the top vs. bottom eigenvectors of $F_{\mu\nu}(\theta)$ deform the layerwise metric tensor $g^{(\ell)}_{ij}(x; \theta)$ and Gaussian curvature $K^{(\ell)}(x; \theta)$, and compare **Euclidean Gradient Descent** against **Natural Gradient Descent** on $\Theta$.
""",
        },
        {
            "type": "code",
            "source": r"""
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path.cwd()
for p in [Path.cwd(), Path.cwd().parent, Path.cwd() / Path("projects") / Path("diffgeom")]:
    if (p / "src" / "geometry.py").exists():
        PROJECT_ROOT = p
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

DATASET_NAME = CFG["dataset"]["name"]
N_CLASSES = CFG["dataset"].get("n_classes", 2)
DATASET_PARAMS = dict(CFG["dataset"].get("params", {}))
LAYER_DIMS = list(CFG["model"]["layer_dims"])
HIDDEN_ACTIVATION = CFG["model"].get("hidden_activation", "tanh")
OUTPUT_ACTIVATION = CFG["model"].get("output_activation", "identity")

import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.geometry import (
    MoonsMLP,
    get_synthetic_dataset,
    make_evaluation_grid,
    compute_parameter_fim,
    compute_data_information_matrix,
    compute_layer_geometry,
)

CE_LAYER_DIMS = [*LAYER_DIMS[:-1], max(LAYER_DIMS[-1], N_CLASSES)]
CLASS_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

X_train, y_train, X_test, y_test = get_synthetic_dataset(
    dataset_name=DATASET_NAME,
    n_classes=N_CLASSES,
    input_dim=CE_LAYER_DIMS[0],
    **DATASET_PARAMS,
)
xx, yy, grid_pts = make_evaluation_grid(X_train, n_grid=25, padding=0.45)

model = MoonsMLP(
    layer_dims=CE_LAYER_DIMS,
    activation=HIDDEN_ACTIVATION,
    output_activation=OUTPUT_ACTIVATION,
    seed=42,
)
L = model.num_layers
P_total = sum(p.numel() for p in model.parameters())
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

print(
    f"Dataset: {DATASET_NAME} (classes={N_CLASSES}) | MLP: {model.layer_dims} (P={P_total}, "
    f"hidden={HIDDEN_ACTIVATION}, out={OUTPUT_ACTIVATION}) | Checkpoints: {list(snapshots.keys())}"
)
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 1. Co-Evolution of the $P \times P$ Parameter FIM Spectrum and $2 \times 2$ Data Information Matrix
We compute the eigenspectrum of $F_{\mu\nu}(\theta)$ and the spatial trace of $\mathrm{DIM}_{ij}(x; \theta)$ across training checkpoints. Notice how the parameter manifold concentrates its Fisher information into a low-dimensional stiff subspace while $\mathrm{Tr}(\mathrm{DIM}(x; \theta))$ sharpens into a singular foliation along the decision boundary (**Tron et al., 2024**).
""",
        },
        {
            "type": "code",
            "source": r"""
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
probe = MoonsMLP(
    layer_dims=CE_LAYER_DIMS,
    activation=HIDDEN_ACTIVATION,
    output_activation=OUTPUT_ACTIVATION,
    seed=42,
)

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
axes[0].set_xlabel(f"Parameter Eigenvalue Index $k$ ($P = {P_total}$)")
axes[0].set_ylabel(r"FIM Eigenvalue $\lambda_k(F(\theta))$")
axes[0].set_title(f"{P_total}x{P_total} Parameter Manifold Fisher Spectrum")
axes[0].legend()
axes[0].grid(alpha=0.3)

ax2 = axes[1].twinx()
axes[1].plot(ep_list, fim_traces, "o-", color="#1f77b4", lw=2, label=r"Parameter $\mathrm{Tr}(F(\theta))$")
axes[1].plot(ep_list, dim_traces, "s--", color="#d62728", lw=2, label=r"Data $\mathbb{E}_x[\mathrm{Tr}(\mathrm{DIM}(x))]$")
ax2.plot(ep_list, eff_ranks, "^:", color="#2ca02c", lw=2, label="FIM Effective Rank")
axes[1].set_xlabel("Training Epoch")
axes[1].set_ylabel("Information Metric Trace")
ax2.set_ylabel(r"Effective Rank of $F(\theta)$", color="#2ca02c")
axes[1].set_title("Parameter vs. Data Information Metric Coupling")
lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labels1 + labels2, loc="center right")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

for ep, ft, dt, er in zip(ep_list, fim_traces, dim_traces, eff_ranks):
    print(f"Epoch {ep:3d} | Tr(FIM) = {ft:8.3f} | E[Tr(DIM)] = {dt:8.3f} | FIM Effective Rank = {er:6.2f} / {P_total}")
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 2. Singular Foliation of the Input Data Manifold via the Data Information Matrix (DIM)
The kernel eigenvector $v_{\ker}(x)$ of the $2 \times 2$ Data Information Matrix spans the tangent distribution to the **1D foliation leaves** of constant class probability (**Benfenati & Marta, 2023**; **Tron et al., 2024**), while its principal eigenvector points along the maximum information gradient across the decision boundary.
""",
        },
        {
            "type": "code",
            "source": r"""
dim_grid = compute_data_information_matrix(model, grid_pts, eps=1e-9)
tr_dim = torch.diagonal(dim_grid, dim1=-2, dim2=-1).sum(-1).detach().numpy().reshape(xx.shape)
evals_d, evecs_d = torch.linalg.eigh(dim_grid)
v_leaf = evecs_d[:, :, 0].detach().numpy()
u_leaf = v_leaf[:, 0].reshape(xx.shape)
w_leaf = v_leaf[:, 1].reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7.5, 5.2))
cf = ax.contourf(xx, yy, np.log10(tr_dim + 1e-7), levels=22, cmap="inferno")
ax.quiver(xx[::2, ::2], yy[::2, ::2], u_leaf[::2, ::2], w_leaf[::2, ::2], color="cyan", alpha=0.8, scale=22, headwidth=2.5, label="DIM Kernel (Foliation Tangent)")
for c in range(N_CLASSES):
    mc = y_test == c
    ax.scatter(X_test[mc, 0], X_test[mc, 1], c=CLASS_COLORS[c % len(CLASS_COLORS)], edgecolors="w", s=26, label=f"Class {c}")
ax.set_title(r"Data Information Matrix $\log_{10}\mathrm{Tr}(\mathrm{DIM}(x))$ & Singular Foliation Tangents")
ax.set_xlabel("$x^1$")
ax.set_ylabel("$x^2$")
ax.legend(loc="upper right")
plt.colorbar(cf, ax=ax, label=r"$\log_{10}\mathrm{Tr}(\mathrm{DIM}(x))$")
plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 3. Metric Flow Sensitivity: Perturbing $\Theta$ Along Top vs. Null FIM Eigenvectors
To directly test how movements on the parameter manifold $\Theta$ induce metric and curvature flows on the output data manifold $\mathcal{M}_L$ (**Halverson & Ruehle, 2023**), we perturb $\theta \mapsto \theta + \delta \cdot v^{(k)}$ along unit eigenvectors $v^{(k)}$ of $F(\theta)$ and measure the induced change in the data pullback metric $\|\Delta g^{(L)}(x)\|_F$ and Gaussian curvature $|\Delta K^{(L)}(x)|$.
""",
        },
        {
            "type": "code",
            "source": r"""
fim, _ = compute_parameter_fim(model, X_train, damping=1e-8)
fim_evals, fim_evecs = torch.linalg.eigh(fim)
fim_evals = fim_evals.flip(0)
fim_evecs = fim_evecs.flip(1)

base_params = {k: v.detach().clone() for k, v in model.named_parameters()}
base_geom = compute_layer_geometry(model, X_test, layer_idx=L, metric_mode="graph", params=base_params)

def unflatten_like(vec: torch.Tensor, template: dict) -> dict:
    out = {}
    offset = 0
    for k, t in template.items():
        numel = t.numel()
        out[k] = t + vec[offset : offset + numel].reshape(t.shape)
        offset += numel
    return out

modes_to_test = sorted(set(int(round(m)) for m in (np.geomspace(1, P_total, num=min(10, P_total)) - 1)))
delta_scale = 0.08
metric_shifts, curv_shifts, mode_evals = [], [], []

for m_idx in modes_to_test:
    v_vec = fim_evecs[:, m_idx] * delta_scale
    pert_params = unflatten_like(v_vec, base_params)
    pert_geom = compute_layer_geometry(model, X_test, layer_idx=L, metric_mode="graph", params=pert_params)
    dg = (pert_geom["g"] - base_geom["g"]).norm(dim=(-2, -1)).mean().item()
    dK = (pert_geom["K"] - base_geom["K"]).abs().mean().item()
    metric_shifts.append(dg)
    curv_shifts.append(dK)
    mode_evals.append(fim_evals[m_idx].item())

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot([m + 1 for m in modes_to_test], metric_shifts, "o-", lw=2, color="#1f77b4", label=rf"Data Metric Shift $\mathbb{{E}}[\|\Delta g^{{({L})}}(x)\|_F]$")
ax.plot([m + 1 for m in modes_to_test], curv_shifts, "s--", lw=2, color="#d62728", label=rf"Data Curvature Shift $\mathbb{{E}}[|\Delta K^{{({L})}}(x)|]$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(f"Parameter FIM Eigenvector Mode Index $k$ (1 = stiffest, {P_total} = flattest)")
ax.set_ylabel(rf"Induced Geometric Deformation on Data Manifold $\mathcal{{M}}_{{{L}}}$")
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
We compare standard Euclidean Gradient Descent ($\Delta \theta = -\eta \nabla_\theta \mathcal{L}$) with damped **Natural Gradient Descent** ($\Delta \theta = -\eta (F(\theta) + \lambda I)^{-1} \nabla_\theta \mathcal{L}$), tracking both classification loss and the smoothness of the induced Gaussian curvature $K^{(L)}(x)$ on the data manifold.
""",
        },
        {
            "type": "code",
            "source": r"""
def run_optimizer_comparison(steps: int = 45):
    m_gd = MoonsMLP(
        layer_dims=CE_LAYER_DIMS,
        activation=HIDDEN_ACTIVATION,
        output_activation=OUTPUT_ACTIVATION,
        seed=11,
    )
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
                if p.grad is not None:
                    p -= 0.15 * p.grad
        gd_losses.append(l_gd.item())

        # 2. Natural Gradient Descent on Parameter Manifold
        for p in m_ngd.parameters():
            if p.grad is not None:
                p.grad.zero_()
        l_ngd = crit(m_ngd(X_train), y_train)
        l_ngd.backward()
        fim_curr, _ = compute_parameter_fim(m_ngd, X_train, damping=0.02)
        g_flat = torch.cat([
            p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel(), dtype=X_train.dtype)
            for p in m_ngd.parameters()
        ])
        nat_step = torch.linalg.solve(fim_curr, g_flat)
        with torch.no_grad():
            offset = 0
            for p in m_ngd.parameters():
                n = p.numel()
                p -= 0.08 * nat_step[offset : offset + n].reshape(p.shape)
                offset += n
        ngd_losses.append(l_ngd.item())

        if step % 3 == 0:
            K_gd = compute_layer_geometry(m_gd, X_test[:40], layer_idx=L, metric_mode="graph")["K"].abs().mean().item()
            K_ngd = compute_layer_geometry(m_ngd, X_test[:40], layer_idx=L, metric_mode="graph")["K"].abs().mean().item()
            gd_curv.append((step, K_gd))
            ngd_curv.append((step, K_ngd))

    return gd_losses, ngd_losses, gd_curv, ngd_curv

gd_l, ngd_l, gd_k, ngd_k = run_optimizer_comparison(45)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
axes[0].plot(gd_l, lw=2, color="#1f77b4", label="Euclidean Gradient Descent")
axes[0].plot(ngd_l, lw=2, color="#2ca02c", label="Riemannian Natural Gradient (FIM)")
axes[0].set_xlabel("Optimization Step")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].set_title(r"Optimization Trajectory on Parameter Manifold $\Theta$")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot([x[0] for x in gd_k], [x[1] for x in gd_k], "o-", lw=2, color="#1f77b4", label=rf"Euclidean GD Data Curvature $|K^{{({L})}}|$")
axes[1].plot([x[0] for x in ngd_k], [x[1] for x in ngd_k], "s-", lw=2, color="#2ca02c", label=rf"Natural GD Data Curvature $|K^{{({L})}}|$")
axes[1].set_xlabel("Optimization Step")
axes[1].set_ylabel(rf"Mean Output Manifold Curvature $\mathbb{{E}}[|K^{{({L})}}(x)|]$")
axes[1].set_title(rf"Induced Curvature Flow on Data Manifold $\mathcal{{M}}_{{{L}}}$")
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
Can a deep neural network learn to classify a non-linear dataset using **only differential geometric objects** in its loss function—without any cross-entropy, MSE, or coordinate-level logit supervision?

We train our configured MLP from scratch by sculpting the intrinsic **Gaussian curvature** $K(x; \theta)$ of the final representation manifold $\mathcal{M}_L$ (equipped with the Graph Embedding Metric $\tilde{g}^{(L)}_{ij}(x) = \delta_{ij} + ((J^{(L)}(x))^T J^{(L)}(x))_{ij}$):
- **Class $0$ ($y = 0, s = -1$)**: Driven into regions of **negative Gaussian curvature** ($K(x) \le -\tau < 0$, hyperbolic saddle geometry).
- **Class $1$ ($y = 1, s = +1$)**: Driven into regions of **positive Gaussian curvature** ($K(x) \ge +\tau > 0$, spherical/elliptic dome geometry).

### Purely Geometric Loss Function
$$\mathcal{L}_{\text{geom}}(\theta) = \frac{1}{N}\sum_{n=1}^N \mathrm{softplus}_\beta\!\Big(\tau - (2y_n - 1)\, K_{\text{clip}}(x_n; \theta)\Big) + \lambda_{\text{vol}} \frac{1}{N}\sum_{n=1}^N \Big(\log \det \tilde{g}^{(L)}(x_n; \theta)\Big)^2 + \lambda_{\text{iso}} \frac{1}{N}\sum_{n=1}^N \|\tilde{g}^{(L)}(x_n; \theta) - I_2\|_F^2$$

- **Requires 3rd-Order Automatic Differentiation**: Computing $\nabla_\theta \mathcal{L}_{\text{geom}}(\theta)$ backpropagates through 3rd-order mixed spatio-parametric derivatives $\frac{\partial^3 \Phi^{(L)}_\theta(x)}{\partial \theta \partial x^i \partial x^j}$.
- **Purely Geometric Inference Rule**: At test time, any point $x^*$ is classified strictly by local manifold curvature ($\hat{y}(x^*) = \mathbb{I}[K(x^*; \theta) > 0]$ for $C=2$).
""",
        },
        {
            "type": "code",
            "source": r"""
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path.cwd()
for p in [Path.cwd(), Path.cwd().parent, Path.cwd() / Path("projects") / Path("diffgeom")]:
    if (p / "src" / "geometry.py").exists():
        PROJECT_ROOT = p
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

DATASET_NAME = CFG["dataset"]["name"]
N_CLASSES = CFG["dataset"].get("n_classes", 2)
DATASET_PARAMS = dict(CFG["dataset"].get("params", {}))
LAYER_DIMS = list(CFG["model"]["layer_dims"])
HIDDEN_ACTIVATION = CFG["model"].get("hidden_activation", "tanh")
OUTPUT_ACTIVATION = CFG["model"].get("output_activation", "identity")

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.geometry import (
    MoonsMLP,
    get_synthetic_dataset,
    make_evaluation_grid,
    compute_layer_geometry,
    curvature_margin_loss,
    scalar_geometric_predict,
)

CLASS_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

X_train, y_train, X_test, y_test = get_synthetic_dataset(
    dataset_name=DATASET_NAME,
    n_classes=N_CLASSES,
    input_dim=LAYER_DIMS[0],
    **DATASET_PARAMS,
)
xx, yy, grid_pts = make_evaluation_grid(X_train, n_grid=28, padding=0.45)
print(
    f"Loaded {DATASET_NAME} (classes={N_CLASSES}) | Train: {X_train.shape[0]} pts | Test: {X_test.shape[0]} pts | "
    f"MLP: {LAYER_DIMS} (hidden={HIDDEN_ACTIVATION}, out={OUTPUT_ACTIVATION})"
)
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 1. End-to-End 3rd-Order Autodiff Training with Purely Geometric Curvature Loss
We optimize $\mathcal{L}_{\text{geom}}(\theta)$ for 90 epochs with Adam (`lr=0.025`) and margin $\tau = 0.40$, tracking the separation of Gaussian curvature across classes.
""",
        },
        {
            "type": "code",
            "source": r"""
geom_model = MoonsMLP(
    layer_dims=LAYER_DIMS,
    activation=HIDDEN_ACTIVATION,
    output_activation=OUTPUT_ACTIVATION,
    seed=7,
)
L = geom_model.num_layers
MARGIN_CURV = 0.40
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
    geom_tr = compute_layer_geometry(geom_model, X_train, layer_idx=L, metric_mode="graph", params=params)
    loss, metrics = curvature_margin_loss(
        geom_tr["K"],
        y_train,
        geom_tr["g"],
        margin=MARGIN_CURV,
        beta_hinge=4.0,
        lambda_vol=0.02,
        lambda_iso=0.005,
        k_clip=10.0,
        n_classes=N_CLASSES,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(geom_model.parameters(), max_norm=5.0)
    optimizer.step()

    if epoch == 1 or epoch % 5 == 0:
        with torch.no_grad():
            geom_te = compute_layer_geometry(geom_model, X_test, layer_idx=L, metric_mode="graph")
            preds_te = scalar_geometric_predict(geom_te["K"], n_classes=N_CLASSES, margin=MARGIN_CURV)
            te_acc = (preds_te == y_test).float().mean().item()
        hist["epoch"].append(epoch)
        hist["loss"].append(metrics["loss_total"])
        hist["train_acc"].append(metrics["geom_acc"])
        hist["test_acc"].append(te_acc)
        hist["K_c0"].append(metrics["mean_K_class0"])
        hist["K_c1"].append(metrics["mean_K_class1"])

    if epoch in [1, 20, 45, 90]:
        print(
            f"Epoch {epoch:3d} ({time.time()-t0:4.1f}s) | Geom Loss: {metrics['loss_total']:.4f} | "
            f"Train Acc: {metrics['geom_acc']*100:5.1f}% | Test Acc: {hist['test_acc'][-1]*100:5.1f}% | "
            f"E[K|y=0]: {metrics['mean_K_class0']:+6.3f} | E[K|y=1]: {metrics['mean_K_class1']:+6.3f}"
        )

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(hist["epoch"], hist["loss"], lw=2, color="#9467bd", label="Geometric Curvature Loss")
axes[0].plot(hist["epoch"], hist["train_acc"], lw=2, color="#1f77b4", ls="--", label="Train Acc")
axes[0].plot(hist["epoch"], hist["test_acc"], lw=2.2, color="#2ca02c", label="Test Acc")
axes[0].set_xlabel("Training Epoch")
axes[0].set_ylabel("Loss / Accuracy")
axes[0].set_title("Purely Geometric Curvature Training Convergence")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(hist["epoch"], hist["K_c0"], lw=2.2, color="#1f77b4", label=r"Class 0 Mean Curvature $\mathbb{E}[K(x) \mid y=0]$")
axes[1].plot(hist["epoch"], hist["K_c1"], lw=2.2, color="#d62728", label=r"Class 1 Mean Curvature $\mathbb{E}[K(x) \mid y=1]$")
axes[1].axhline(0.0, color="k", lw=1.2, ls=":")
axes[1].axhline(-MARGIN_CURV, color="#1f77b4", lw=1.0, ls="--", alpha=0.6, label=r"Target Margin $-\tau$")
axes[1].axhline(+MARGIN_CURV, color="#d62728", lw=1.0, ls="--", alpha=0.6, label=r"Target Margin $+\tau$")
axes[1].set_xlabel("Training Epoch")
axes[1].set_ylabel(rf"Gaussian Curvature $K^{{({L})}}(x)$")
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
We evaluate $K^{(L)}(x)$ over the entire 2D grid and on the held-out test set.
""",
        },
        {
            "type": "code",
            "source": r"""
geom_grid = compute_layer_geometry(geom_model, grid_pts, layer_idx=L, metric_mode="graph")
K_grid = geom_grid["K"].detach().numpy().reshape(xx.shape)
K_grid_sinh = np.arcsinh(K_grid)
vmax = max(np.percentile(np.abs(K_grid_sinh), 96), 0.5)

geom_test = compute_layer_geometry(geom_model, X_test, layer_idx=L, metric_mode="graph")
K_test_t = geom_test["K"].detach()
K_test = K_test_t.numpy()
y_test_np = y_test.numpy()
preds_test_np = scalar_geometric_predict(K_test_t, n_classes=N_CLASSES, margin=MARGIN_CURV).numpy()

bnd_thresholds = [0.0] if N_CLASSES == 2 else list(2.0 * MARGIN_CURV * (np.arange(N_CLASSES - 1) + 0.5 - 0.5 * (N_CLASSES - 1)))

fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
cf = axes[0].contourf(xx, yy, K_grid_sinh, levels=25, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
valid_bnds = [b for b in bnd_thresholds if K_grid.min() < b < K_grid.max()]
if valid_bnds:
    axes[0].contour(xx, yy, K_grid, levels=valid_bnds, colors="k", linewidths=2.2)
for c in range(N_CLASSES):
    mc = y_test == c
    axes[0].scatter(X_test[mc, 0], X_test[mc, 1], c=CLASS_COLORS[c % len(CLASS_COLORS)], edgecolors="w", s=34, label=f"Test Class {c}")
axes[0].set_title(rf"Learned Curvature Field $\mathrm{{arcsinh}}(K^{{({L})}}(x))$ & Decision Boundary")
axes[0].set_xlabel("$x^1$")
axes[0].set_ylabel("$x^2$")
axes[0].legend(loc="upper right")
plt.colorbar(cf, ax=axes[0], label=rf"$\mathrm{{arcsinh}}(K^{{({L})}}(x))$")

bins = np.linspace(np.percentile(K_test, 2) - 0.2, np.percentile(K_test, 98) + 0.2, 24)
for c in range(N_CLASSES):
    axes[1].hist(K_test[y_test_np == c], bins=bins, alpha=0.72, color=CLASS_COLORS[c % len(CLASS_COLORS)], edgecolor="k", label=f"Test Class {c}")
for b_idx, b_val in enumerate(bnd_thresholds):
    axes[1].axvline(b_val, color="k", lw=2.0, ls="--", label="Decision Threshold" if b_idx == 0 else None)
axes[1].set_xlabel(rf"Intrinsic Gaussian Curvature $K^{{({L})}}(x)$")
axes[1].set_ylabel("Test Sample Count")
axes[1].set_title(f"Held-Out Test Set Curvature Distribution (Acc = {(preds_test_np == y_test_np).mean()*100:.1f}%)")
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
We visualize the learned graph surface $(x^1, x^2, z^1(x))$ in $\mathbb{R}^3$ coloured by its exact Gaussian curvature $K^{(L)}(x)$, illustrating how the neural network bends the manifold into hyperbolic saddles ($K < 0$, blue) vs. elliptic domes ($K > 0$, red).
""",
        },
        {
            "type": "code",
            "source": r"""
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
for c in range(N_CLASSES):
    mc = y_test_np == c
    ax.scatter(x_te_np[mc, 0], x_te_np[mc, 1], z_te[mc, 0] + 0.08, c=CLASS_COLORS[c % len(CLASS_COLORS)], edgecolors="k", s=35, label=f"Class {c}")

ax.set_xlabel("$x^1$")
ax.set_ylabel("$x^2$")
ax.set_zlabel("Output Coordinate $z^1(x)$")
ax.set_title(rf"3D Immersed Data Manifold $(x^1, x^2, z^1(x))$ Coloured by Gaussian Curvature $K^{{({L})}}(x)$")
ax.view_init(elev=28, azim=-55)
ax.legend(loc="upper left")
mappable = cm.ScalarMappable(cmap=cm.RdBu_r, norm=norm)
mappable.set_array(K_grid_sinh)
plt.colorbar(mappable, ax=ax, shrink=0.62, pad=0.08, label=rf"$\mathrm{{arcsinh}}(K^{{({L})}}(x))$")
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
We synthesize our investigation by comparing **three purely geometric training objectives** against the **standard Cross-Entropy baseline** on the configured MLP:

1. **Cross-Entropy Baseline**: $\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{N}\sum_n \log p_\theta(y_n | x_n)$ (coordinate/logit supervision).
2. **Gaussian Curvature Margin Loss (2nd-Fundamental / Riemann Invariant)**:
   Separates classes by intrinsic Gaussian curvature $K^{(L)}(x)$.
3. **Riemannian Volume-Element Contrastive Loss (Metric Determinant Invariant)**:
   Separates classes by local log-volume density $\log\sqrt{\det g^{(L)}(x)}$.
4. **Metric Anisotropy / Polarization Loss (Metric Ratio Invariant)**:
   Separates classes by directional metric stretching $\log(g^{(L)}_{11}(x) / g^{(L)}_{22}(x))$.

Finally, we analyze the **back-reaction on the Parameter Manifold ($\Theta$)**: how training with purely geometric objectives on the data manifold reshapes the layerwise curvature profile $K^{(1 \dots L)}$ and the $P \times P$ Parameter Manifold metric spectrum $G^{\Theta}_{\mu\nu}(\theta) = \mathbb{E}_x[(J_\theta z)^T (J_\theta z)]$.
""",
        },
        {
            "type": "code",
            "source": r"""
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path.cwd()
for p in [Path.cwd(), Path.cwd().parent, Path.cwd() / Path("projects") / Path("diffgeom")]:
    if (p / "src" / "geometry.py").exists():
        PROJECT_ROOT = p
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

DATASET_NAME = CFG["dataset"]["name"]
N_CLASSES = CFG["dataset"].get("n_classes", 2)
DATASET_PARAMS = dict(CFG["dataset"].get("params", {}))
LAYER_DIMS = list(CFG["model"]["layer_dims"])
HIDDEN_ACTIVATION = CFG["model"].get("hidden_activation", "tanh")
OUTPUT_ACTIVATION = CFG["model"].get("output_activation", "identity")

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, jacrev, vmap
import matplotlib.pyplot as plt

from src.geometry import (
    MoonsMLP,
    get_synthetic_dataset,
    make_evaluation_grid,
    compute_layer_geometry,
    compute_layer_pullback_metric,
    compute_metric_spectrum,
    curvature_margin_loss,
    volume_contrastive_loss,
    anisotropy_margin_loss,
    scalar_geometric_predict,
)

CE_LAYER_DIMS = [*LAYER_DIMS[:-1], max(LAYER_DIMS[-1], N_CLASSES)]
CLASS_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

X_train, y_train, X_test, y_test = get_synthetic_dataset(
    dataset_name=DATASET_NAME,
    n_classes=N_CLASSES,
    input_dim=LAYER_DIMS[0],
    **DATASET_PARAMS,
)
xx, yy, grid_pts = make_evaluation_grid(X_train, n_grid=26, padding=0.45)
print(
    f"Dataset: {DATASET_NAME} (classes={N_CLASSES}) | Train: {X_train.shape}, Test: {X_test.shape} | "
    f"MLP: {LAYER_DIMS} (hidden={HIDDEN_ACTIVATION}, out={OUTPUT_ACTIVATION})"
)
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
            "source": r"""
# 1. Cross-Entropy Baseline
m_ce = MoonsMLP(layer_dims=CE_LAYER_DIMS, activation=HIDDEN_ACTIVATION, output_activation=OUTPUT_ACTIVATION, seed=7)
opt_ce = torch.optim.Adam(m_ce.parameters(), lr=0.02)
crit_ce = nn.CrossEntropyLoss()
for _ in range(120):
    opt_ce.zero_grad()
    loss = crit_ce(m_ce(X_train), y_train)
    loss.backward()
    opt_ce.step()
ce_te_acc = (m_ce(X_test).argmax(-1) == y_test).float().mean().item()

# 2. Purely Geometric Curvature Margin Loss
m_curv = MoonsMLP(layer_dims=LAYER_DIMS, activation=HIDDEN_ACTIVATION, output_activation=OUTPUT_ACTIVATION, seed=7)
L = m_curv.num_layers
opt_curv = torch.optim.Adam(m_curv.parameters(), lr=0.025)
for _ in range(85):
    opt_curv.zero_grad()
    params = dict(m_curv.named_parameters())
    gL = compute_layer_geometry(m_curv, X_train, layer_idx=L, metric_mode="graph", params=params)
    loss, _ = curvature_margin_loss(gL["K"], y_train, gL["g"], margin=0.40, lambda_vol=0.02, lambda_iso=0.005, k_clip=10.0, n_classes=N_CLASSES)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m_curv.parameters(), 5.0)
    opt_curv.step()
K_te = compute_layer_geometry(m_curv, X_test, layer_idx=L, metric_mode="graph")["K"]
curv_te_acc = (scalar_geometric_predict(K_te, n_classes=N_CLASSES, margin=0.40) == y_test).float().mean().item()

# 3. Purely Geometric Volume-Element Contrastive Loss
m_vol = MoonsMLP(layer_dims=LAYER_DIMS, activation=HIDDEN_ACTIVATION, output_activation=OUTPUT_ACTIVATION, seed=7)
opt_vol = torch.optim.Adam(m_vol.parameters(), lr=0.02)
for _ in range(100):
    opt_vol.zero_grad()
    params = dict(m_vol.named_parameters())
    gL = compute_layer_pullback_metric(m_vol, X_train, layer_idx=L, metric_mode="euclidean", eps=1e-6, params=params)
    loss, _ = volume_contrastive_loss(gL, y_train, margin=0.60, n_classes=N_CLASSES)
    loss.backward()
    opt_vol.step()
gL_vol_te = compute_layer_pullback_metric(m_vol, X_test, layer_idx=L, metric_mode="euclidean", eps=1e-6)
log_vol_te = 0.5 * torch.log(torch.clamp(torch.linalg.det(gL_vol_te), min=1e-8))
vol_te_acc = (scalar_geometric_predict(log_vol_te, n_classes=N_CLASSES, margin=0.60) == y_test).float().mean().item()

# 4. Purely Geometric Metric Anisotropy Loss
m_aniso = MoonsMLP(layer_dims=LAYER_DIMS, activation=HIDDEN_ACTIVATION, output_activation=OUTPUT_ACTIVATION, seed=7)
opt_aniso = torch.optim.Adam(m_aniso.parameters(), lr=0.02)
for _ in range(100):
    opt_aniso.zero_grad()
    params = dict(m_aniso.named_parameters())
    gL = compute_layer_pullback_metric(m_aniso, X_train, layer_idx=L, metric_mode="graph", params=params)
    loss, _ = anisotropy_margin_loss(gL, y_train, margin=0.45, n_classes=N_CLASSES)
    loss.backward()
    opt_aniso.step()
gL_aniso_te = compute_layer_pullback_metric(m_aniso, X_test, layer_idx=L, metric_mode="graph")
aniso_ratio_te = torch.log(gL_aniso_te[:, 0, 0] / gL_aniso_te[:, 1, 1])
aniso_te_acc = (scalar_geometric_predict(aniso_ratio_te, n_classes=N_CLASSES, margin=0.45) == y_test).float().mean().item()

print(f"{'Training Objective':<44} {'Decision Criterion':<30} {'Test Accuracy':<15}")
print("-" * 90)
print(f"{'1. Cross-Entropy Baseline':<44} {'argmax_c z_c(x)':<30} {ce_te_acc*100:6.2f}%")
print(f"{'2. Geometric Curvature Margin Loss':<44} {f'K^({L})(x) bands':<30} {curv_te_acc*100:6.2f}%")
print(f"{'3. Geometric Volume-Element Loss':<44} {f'log sqrt(det g^({L})(x))':<30} {vol_te_acc*100:6.2f}%")
print(f"{'4. Geometric Metric Anisotropy Loss':<44} {f'log(g_11^({L}) / g_22^({L}))':<30} {aniso_te_acc*100:6.2f}%")
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
            "source": r"""
with torch.no_grad():
    logits_grid = m_ce(grid_pts)
    if N_CLASSES == 2:
        f_ce = (logits_grid[:, 1] - logits_grid[:, 0]).numpy().reshape(xx.shape)
    else:
        f_ce = (logits_grid.argmax(dim=-1).float() - 0.5 * (N_CLASSES - 1)).numpy().reshape(xx.shape)

f_curv = np.arcsinh(compute_layer_geometry(m_curv, grid_pts, layer_idx=L, metric_mode="graph")["K"].detach().numpy().reshape(xx.shape))

g_vol_grid = compute_layer_pullback_metric(m_vol, grid_pts, layer_idx=L, metric_mode="euclidean", eps=1e-6)
f_vol = (0.5 * torch.log(torch.clamp(torch.linalg.det(g_vol_grid), min=1e-8))).detach().numpy().reshape(xx.shape)

g_aniso_grid = compute_layer_pullback_metric(m_aniso, grid_pts, layer_idx=L, metric_mode="graph")
f_aniso = torch.log(g_aniso_grid[:, 0, 0] / g_aniso_grid[:, 1, 1]).detach().numpy().reshape(xx.shape)

fields = [
    (f_ce, f"Cross-Entropy Logit Gap\nTest Acc: {ce_te_acc*100:.1f}%"),
    (f_curv, "Gaussian Curvature " + r"$\mathrm{arcsinh}(K)$" + f"\nTest Acc: {curv_te_acc*100:.1f}%"),
    (f_vol, "Log Volume " + r"$\log\sqrt{\det g}$" + f"\nTest Acc: {vol_te_acc*100:.1f}%"),
    (f_aniso, "Metric Anisotropy " + r"$\log(g_{11}/g_{22})$" + f"\nTest Acc: {aniso_te_acc*100:.1f}%"),
]

fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.9), sharex=True, sharey=True)
for ax, (fld, title) in zip(axes, fields):
    vm = max(np.percentile(np.abs(fld), 95), 0.2)
    im = ax.contourf(xx, yy, fld, levels=22, cmap="RdBu_r", vmin=-vm, vmax=vm)
    if fld.min() < 0.0 < fld.max():
        ax.contour(xx, yy, fld, levels=[0.0], colors="k", linewidths=2.0)
    for c in range(N_CLASSES):
        mc = y_test == c
        ax.scatter(X_test[mc, 0], X_test[mc, 1], c=CLASS_COLORS[c % len(CLASS_COLORS)], edgecolors="w", s=20)
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
## 3. Layerwise Curvature Profiles & Back-Reaction on the $P \times P$ Parameter Manifold Metric
Finally, we compare how the four training objectives distribute curvature across layers $\ell \in \{1, \dots, L\}$ and how they shape the eigenspectrum of the $P \times P$ Generalized Gauss-Newton / Output Pullback Metric on the parameter manifold $\Theta$:
$$G^\Theta_{\mu\nu}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_{a=1}^{d_L} \frac{\partial z^a(x; \theta)}{\partial \theta^\mu} \frac{\partial z^a(x; \theta)}{\partial \theta^\nu} \right]$$
""",
        },
        {
            "type": "code",
            "source": r"""
def compute_param_pullback_spectrum(model: MoonsMLP, X: torch.Tensor) -> np.ndarray:
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = dict(model.named_buffers())
    keys = list(params.keys())

    def z_fn(p_dict, x_single):
        return functional_call(model, (p_dict, buffers), (x_single.unsqueeze(0),)).squeeze(0)

    J_dict = vmap(jacrev(z_fn, argnums=0), in_dims=(None, 0))(params, X)
    N = X.shape[0]
    J_flat = torch.cat([J_dict[k].reshape(-1, params[k].numel()) for k in keys], dim=-1)
    G_theta = (J_flat.T @ J_flat) / float(N)
    evals = torch.linalg.eigvalsh(G_theta).flip(0).clamp(min=1e-10).detach().numpy()
    return evals

models_dict = {
    "Cross-Entropy Baseline": (m_ce, "#1f77b4"),
    "Curvature Loss (K < 0 vs K > 0)": (m_curv, "#d62728"),
    "Volume-Element Loss": (m_vol, "#2ca02c"),
    "Metric Anisotropy Loss": (m_aniso, "#ff7f0e"),
}

layers_list = list(range(1, L + 1))
P_geom = sum(p.numel() for p in m_curv.parameters())
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

for label, (mdl, col) in models_dict.items():
    k_gaps = []
    for l_idx in layers_list:
        mode = "graph" if mdl.layer_dims[l_idx] <= 2 else "euclidean"
        K_l = compute_layer_geometry(mdl, X_test, layer_idx=l_idx, metric_mode=mode)["K"].detach()
        gap = (torch.arcsinh(K_l[y_test == (N_CLASSES - 1)]).mean() - torch.arcsinh(K_l[y_test == 0]).mean()).item()
        k_gaps.append(gap)
    axes[0].plot(layers_list, k_gaps, "o-", lw=2, color=col, label=label)

    evals_theta = compute_param_pullback_spectrum(mdl, X_train[:120])
    axes[1].plot(range(1, len(evals_theta) + 1), evals_theta, lw=2, color=col, label=label)

axes[0].axhline(0.0, color="k", lw=1, ls=":")
axes[0].set_xticks(layers_list)
axes[0].set_xticklabels([f"L{l} (d={m_curv.layer_dims[l]})" for l in layers_list])
axes[0].set_ylabel(r"Signed Curvature Separation $\Delta \mathrm{arcsinh}(K^{(\ell)})$")
axes[0].set_title(f"Layerwise Class Curvature Separation ($y={N_CLASSES - 1}$ vs $y=0$)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel(f"Parameter Metric Eigenvalue Index $k$ ($P = {P_geom}$)")
axes[1].set_ylabel(r"Eigenvalue $\lambda_k(G^\Theta(\theta))$")
axes[1].set_title(f"Induced {P_geom}x{P_geom} Parameter Manifold Metric Spectrum")
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

Next, we examine how each of the four training paradigms transforms the **local Riemannian log-volume density** across network depth $\ell \in \{1, \dots, L\}$:
$$\log \rho_{\text{vol}}^{(\ell)}(x; \theta) = \log\sqrt{\det g^{(\ell)}(x; \theta)} = \frac{1}{2}\log \det g^{(\ell)}(x; \theta)$$
""",
        },
        {
            "type": "code",
            "source": r"""
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5))

hdr_cols = " ".join(f"{f'L{l} Gap':>8}" for l in layers_list)
print(f"{'Method':<34} | {hdr_cols}")
print("-" * (37 + 9 * len(layers_list)))

for label, (mdl, col) in models_dict.items():
    vol_c0_means, vol_c1_means, vol_gaps = [], [], []
    for l_idx in layers_list:
        g_l = compute_layer_pullback_metric(mdl, X_test, layer_idx=l_idx, metric_mode="euclidean", eps=1e-6).detach()
        log_vol_l = 0.5 * torch.log(torch.clamp(torch.linalg.det(g_l), min=1e-8))
        m0 = log_vol_l[y_test == 0].mean().item()
        m1 = log_vol_l[y_test == (N_CLASSES - 1)].mean().item()
        vol_c0_means.append(m0)
        vol_c1_means.append(m1)
        vol_gaps.append(m1 - m0)

    vals_str = " ".join(f"{vg:+8.3f}" for vg in vol_gaps)
    print(f"{label:<34} | {vals_str}")

    axes[0].plot(layers_list, vol_c1_means, "o-", lw=2.0, color=col, label=f"{label} (Class {N_CLASSES - 1})")
    axes[0].plot(layers_list, vol_c0_means, "s--", lw=1.6, color=col, alpha=0.75, label=f"{label} (Class 0)")
    axes[1].plot(layers_list, vol_gaps, "o-", lw=2.2, color=col, label=label)

axes[0].axhline(0.0, color="k", lw=1.0, ls=":")
axes[0].set_xticks(layers_list)
axes[0].set_xticklabels([f"L{l} (d={m_curv.layer_dims[l]})" for l in layers_list])
axes[0].set_ylabel(r"Mean Log Volume $\mathbb{E}[\log\sqrt{\det g^{(\ell)}(x)}]$")
axes[0].set_title(f"Per-Class Layerwise Log-Volume Trajectories (Solid: y={N_CLASSES - 1}, Dashed: y=0)")
axes[0].legend(fontsize=7.5, ncol=2)
axes[0].grid(alpha=0.3)

axes[1].axhline(0.0, color="k", lw=1.0, ls=":")
axes[1].set_xticks(layers_list)
axes[1].set_xticklabels([f"L{l} (d={m_curv.layer_dims[l]})" for l in layers_list])
axes[1].set_ylabel(r"Signed Log-Volume Separation $\Delta \log\sqrt{\det g^{(\ell)}}$")
axes[1].set_title(f"Layerwise Class Log-Volume Separation ($y={N_CLASSES - 1}$ vs $y=0$)")
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

We analyze how directional stretching and spectral anisotropy of the pullback metric tensor $g^{(\ell)}_{ij}(x; \theta)$ evolve across layers $\ell \in \{1, \dots, L\}$ under the four training objectives.
""",
        },
        {
            "type": "code",
            "source": r"""
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5))
mid_idx = len(layers_list) // 2
mid_l = layers_list[mid_idx]

print(f"{'Method':<34} | {'L1 Aniso Gap':>12} {f'L{mid_l} Aniso Gap':>12} {f'L{L} Aniso Gap':>12} | {f'L{L} log10(l1/l2)':>15}")
print("-" * 94)

for label, (mdl, col) in models_dict.items():
    aniso_gaps, log_conds = [], []
    for l_idx in layers_list:
        g_graph = compute_layer_pullback_metric(mdl, X_test, layer_idx=l_idx, metric_mode="graph").detach()
        log_ratio = torch.log(torch.clamp(g_graph[:, 0, 0], min=1e-8) / torch.clamp(g_graph[:, 1, 1], min=1e-8))
        gap = (log_ratio[y_test == (N_CLASSES - 1)].mean() - log_ratio[y_test == 0].mean()).item()
        aniso_gaps.append(gap)

        g_eucl = compute_layer_pullback_metric(mdl, X_test, layer_idx=l_idx, metric_mode="euclidean", eps=1e-7).detach()
        _, _, cond = compute_metric_spectrum(g_eucl)
        log_conds.append(torch.log10(torch.clamp(cond, min=1.0)).mean().item())

    print(f"{label:<34} | {aniso_gaps[0]:+12.3f} {aniso_gaps[mid_idx]:+12.3f} {aniso_gaps[-1]:+12.3f} | {log_conds[-1]:15.3f}")

    axes[0].plot(layers_list, aniso_gaps, "o-", lw=2.2, color=col, label=label)
    axes[1].plot(layers_list, log_conds, "s-", lw=2.2, color=col, label=label)

axes[0].axhline(0.0, color="k", lw=1.0, ls=":")
axes[0].set_xticks(layers_list)
axes[0].set_xticklabels([f"L{l} (d={m_curv.layer_dims[l]})" for l in layers_list])
axes[0].set_ylabel(r"Directional Polarization Gap $\Delta \log(g^{(\ell)}_{11} / g^{(\ell)}_{22})$")
axes[0].set_title(f"Layerwise Class Directional Anisotropy Separation ($y={N_CLASSES - 1}$ vs $y=0$)")
axes[0].legend(fontsize=8.5)
axes[0].grid(alpha=0.3)

axes[1].set_xticks(layers_list)
axes[1].set_xticklabels([f"L{l} (d={m_curv.layer_dims[l]})" for l in layers_list])
axes[1].set_ylabel(r"Mean Spectral Anisotropy $\mathbb{E}[\log_{10}(\lambda_1^{(\ell)} / \lambda_2^{(\ell)})]$")
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
""",
        },
        {
            "type": "code",
            "source": r"""
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

probe_net_ce = MoonsMLP(layer_dims=CE_LAYER_DIMS, activation=HIDDEN_ACTIVATION, output_activation=OUTPUT_ACTIVATION, seed=7)
probe_net = MoonsMLP(layer_dims=LAYER_DIMS, activation=HIDDEN_ACTIVATION, output_activation=OUTPUT_ACTIVATION, seed=7)
ce_loss_fn = nn.CrossEntropyLoss()

def step_loss_ce():
    return ce_loss_fn(probe_net_ce(X_train), y_train)

def step_loss_curv():
    params = dict(probe_net.named_parameters())
    gL = compute_layer_geometry(probe_net, X_train, layer_idx=L, metric_mode="graph", params=params)
    loss, _ = curvature_margin_loss(gL["K"], y_train, gL["g"], margin=0.40, n_classes=N_CLASSES)
    return loss

def step_loss_vol():
    params = dict(probe_net.named_parameters())
    gL = compute_layer_pullback_metric(probe_net, X_train, layer_idx=L, metric_mode="euclidean", eps=1e-6, params=params)
    loss, _ = volume_contrastive_loss(gL, y_train, margin=0.60, n_classes=N_CLASSES)
    return loss

def step_loss_aniso():
    params = dict(probe_net.named_parameters())
    gL = compute_layer_pullback_metric(probe_net, X_train, layer_idx=L, metric_mode="graph", params=params)
    loss, _ = anisotropy_margin_loss(gL, y_train, margin=0.45, n_classes=N_CLASSES)
    return loss

def benchmark_method(loss_builder, net_ref, n_repeats: int = 15):
    net_ref.zero_grad()
    loss_val = loss_builder()
    graph_nodes = count_autograd_graph_nodes(loss_val.grad_fn)
    loss_val.backward()

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        net_ref.zero_grad()
        l_p = loss_builder()
        l_p.backward()
    mem_mb = sum(e.cpu_memory_usage for e in prof.key_averages() if e.cpu_memory_usage > 0) / (1024.0 * 1024.0)

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        net_ref.zero_grad()
        l_t = loss_builder()
        l_t.backward()
    step_ms = (time.perf_counter() - t0) * 1000.0 / n_repeats
    return step_ms, mem_mb, graph_nodes

method_specs = [
    ("Cross-Entropy\n(1st-Order)", step_loss_ce, probe_net_ce, ce_te_acc * 100.0, "#1f77b4"),
    ("Volume-Element\n(2nd-Order)", step_loss_vol, probe_net, vol_te_acc * 100.0, "#2ca02c"),
    ("Metric Anisotropy\n(2nd-Order)", step_loss_aniso, probe_net, aniso_te_acc * 100.0, "#ff7f0e"),
    ("Gaussian Curvature\n(3rd-Order)", step_loss_curv, probe_net, curv_te_acc * 100.0, "#d62728"),
]

results = []
for name, fn, net_r, acc_pct, col in method_specs:
    ms, mb, nodes = benchmark_method(fn, net_r)
    results.append({"name": name, "clean_name": name.replace("\n", " "), "ms": ms, "mb": mb, "nodes": nodes, "acc": acc_pct, "color": col})

print(f"{'Method':<34} {'Latency (ms/step)':<20} {'Memory (MB/step)':<20} {'Autograd Nodes':<16} {'Test Acc':<10}")
print("-" * 102)
for r in results:
    print(f"{r['clean_name']:<34} {r['ms']:<20.2f} {r['mb']:<20.2f} {r['nodes']:<16d} {r['acc']:6.2f}%")

names = [r["name"] for r in results]
colors = [r["color"] for r in results]
latencies = [r["ms"] for r in results]
memories = [r["mb"] for r in results]

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

bars0 = axes[0].bar(names, latencies, color=colors, edgecolor="k", alpha=0.88, width=0.58)
axes[0].set_yscale("log")
axes[0].set_ylabel("Forward + Backward Latency (ms / step, log scale)")
axes[0].set_title("Computational Cost per Training Step")
axes[0].grid(axis="y", alpha=0.3)
for b, val in zip(bars0, latencies):
    axes[0].text(b.get_x() + b.get_width() / 2, val * 1.15, f"{val:.2f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[0].set_ylim(0.3, max(latencies) * 2.5)

bars1 = axes[1].bar(names, memories, color=colors, edgecolor="k", alpha=0.88, width=0.58)
axes[1].set_yscale("log")
axes[1].set_ylabel("Step Tensor Memory Allocation (MB, log scale)")
axes[1].set_title("Memory Cost per Training Step")
axes[1].grid(axis="y", alpha=0.3)
for b, val in zip(bars1, memories):
    axes[1].text(b.get_x() + b.get_width() / 2, val * 1.18, f"{val:.2f} MB", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[1].set_ylim(0.15, max(memories) * 3.0)

for r in results:
    bubble_size = max(np.sqrt(r["mb"]) * 55.0, 90.0)
    axes[2].scatter(r["ms"], r["mb"], s=bubble_size, color=r["color"], edgecolors="k", linewidth=1.5, alpha=0.85, label=f"{r['clean_name']} ({r['nodes']} nodes)")
    axes[2].annotate(f"{r['nodes']} nodes\n({r['acc']:.1f}% acc)", (r["ms"], r["mb"]), textcoords="offset points", xytext=(8, -6), fontsize=8.5)

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
        {
            "type": "markdown",
            "source": r"""
## 7. $3 \times 4$ Cross-Evaluation Matrix of Final-Layer Geometric Fields Across All Experiments

Finally, we evaluate **all three differential-geometric measures** at the final layer ($\ell = L$) across **all four trained networks** in the 2D input space $(x^1, x^2)$.
""",
        },
        {
            "type": "code",
            "source": r"""
from matplotlib.colors import TwoSlopeNorm

exp_columns = [
    ("Cross-Entropy Baseline", m_ce),
    ("Curvature Loss", m_curv),
    ("Volume-Element Loss", m_vol),
    ("Metric Anisotropy Loss", m_aniso),
]

row_labels = [
    "Gaussian Curvature\n" + rf"$\mathrm{{arcsinh}}(K^{{({L})}}(x))$",
    "Log Volume Density\n" + rf"$\log\sqrt{{\det g^{{({L})}}(x)}}$",
    "Metric Anisotropy\n" + rf"$\log(g^{{({L})}}_{{11}}(x) / g^{{({L})}}_{{22}}(x))$",
]
row_margins = [0.40, 0.60, 0.45]

def evaluate_final_layer_fields(mdl: MoonsMLP, pts: torch.Tensor):
    l_final = mdl.num_layers
    geom_graph = compute_layer_geometry(mdl, pts, layer_idx=l_final, metric_mode="graph")
    curv_field = np.arcsinh(geom_graph["K"].detach().numpy())

    g_eucl = compute_layer_pullback_metric(mdl, pts, layer_idx=l_final, metric_mode="euclidean", eps=1e-6)
    log_vol_field = (0.5 * torch.log(torch.clamp(torch.linalg.det(g_eucl), min=1e-8))).detach().numpy()

    g_gr = geom_graph["g"].detach()
    aniso_field = torch.log(torch.clamp(g_gr[:, 0, 0], min=1e-8) / torch.clamp(g_gr[:, 1, 1], min=1e-8)).numpy()

    return [curv_field, log_vol_field, aniso_field], [geom_graph["K"].detach(), 0.5 * torch.log(torch.clamp(torch.linalg.det(g_eucl), min=1e-8)).detach(), torch.log(torch.clamp(g_gr[:, 0, 0], min=1e-8) / torch.clamp(g_gr[:, 1, 1], min=1e-8)).detach()]

grid_matrix = [[None for _ in range(4)] for _ in range(3)]
test_acc_matrix = [[0.0 for _ in range(4)] for _ in range(3)]

for col_idx, (_, mdl) in enumerate(exp_columns):
    grid_fields, _ = evaluate_final_layer_fields(mdl, grid_pts)
    _, test_raw_tensors = evaluate_final_layer_fields(mdl, X_test)
    for row_idx in range(3):
        grid_matrix[row_idx][col_idx] = grid_fields[row_idx].reshape(xx.shape)
        preds_pos = scalar_geometric_predict(test_raw_tensors[row_idx], n_classes=N_CLASSES, margin=row_margins[row_idx])
        test_acc_matrix[row_idx][col_idx] = float((preds_pos == y_test).float().mean().item() * 100.0)

fig, axes = plt.subplots(3, 4, figsize=(17.5, 11.5), sharex=True, sharey=True)

for row_idx in range(3):
    for col_idx, (col_title, _) in enumerate(exp_columns):
        ax = axes[row_idx, col_idx]
        fld = grid_matrix[row_idx][col_idx]
        vm = float(max(np.percentile(np.abs(fld), 96), 0.25))
        levels = np.linspace(-vm, vm, 25)
        norm = TwoSlopeNorm(vmin=-vm, vcenter=0.0, vmax=vm)

        im = ax.contourf(xx, yy, fld, levels=levels, cmap="RdBu_r", norm=norm, extend="both")
        if fld.min() < 0.0 < fld.max():
            ax.contour(xx, yy, fld, levels=[0.0], colors="k", linewidths=1.8)

        for c in range(N_CLASSES):
            mc = y_test == c
            ax.scatter(X_test[mc, 0], X_test[mc, 1], c=CLASS_COLORS[c % len(CLASS_COLORS)], edgecolors="w", s=18, alpha=0.85)

        acc_val = test_acc_matrix[row_idx][col_idx]
        is_target_pair = (row_idx == 0 and col_idx == 1) or (row_idx == 1 and col_idx == 2) or (row_idx == 2 and col_idx == 3)
        tag = f"★ Target (Acc: {acc_val:.1f}%)" if is_target_pair else f"Geom Acc: {acc_val:.1f}%"

        if row_idx == 0:
            ax.set_title(f"{col_title}\n({tag})", fontsize=10, fontweight="bold" if is_target_pair else "normal")
        else:
            ax.set_title(f"({tag})", fontsize=9.5, fontweight="bold" if is_target_pair else "normal")

        if col_idx == 0:
            ax.set_ylabel(f"{row_labels[row_idx]}\n$x^2$", fontsize=10)
        if row_idx == 2:
            ax.set_xlabel("$x^1$", fontsize=10)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

plt.suptitle(
    f"3x4 Matrix of Final-Layer (l={L}) Geometric Fields Across Experiments (RdBu_r: Blue < 0, White = 0, Red > 0)",
    y=1.01,
    fontsize=13.5,
)
plt.tight_layout()
plt.show()
""",
        },
    ]
    build_and_execute_notebook(ROOT / "notebooks" / "05_geometric_loss_ablations_and_synthesis.ipynb", cells)


def build_notebook_06() -> None:
    cells = [
        {
            "type": "markdown",
            "source": r"""
# Notebook 06: Coupled Parameter–Data Manifold Optimization Across Purely Geometric Losses

## Research Objective
When a neural network is trained using **purely differential-geometric losses** on its representation manifold—such as **Gaussian curvature** $K^{(L)}(x; \theta)$, **Riemannian log-volume density** $\log\sqrt{\det g^{(L)}(x; \theta)}$, or **directional metric anisotropy** $\log(\tilde{g}^{(L)}_{11}(x; \theta) / \tilde{g}^{(L)}_{22}(x; \theta))$—a parameter vector $\theta \in \Theta \subset \mathbb{R}^P$ parameterizes an **immersed Riemannian metric field** on the data manifold:
$$\mathcal{G} : \Theta \longrightarrow \mathrm{Met}(\mathcal{M}_0), \qquad \theta \longmapsto g(\cdot\,; \theta)$$

Consequently, the Riemannian metric on parameter space $\Theta$ should **depend directly on the induced geometry of the data space**, rather than treating weights as flat Euclidean coordinates ($I_P$) or pulling back flat output coordinates ($G^{\Theta,\text{coord}} = \mathbb{E}_x[J_\theta^T J_\theta]$).

### Four Parameter-Space Regimes Compared
1. **Uncoupled Euclidean Gradient Descent ($G^\Theta = I_P$)**: Standard steepest descent $\Delta\theta = -\eta \nabla_\theta \mathcal{L}_{\text{geom}}$ in flat weight coordinates.
2. **Uncoupled Adam (Diagonal Coordinate Preconditioning)**: Empirical per-weight second-moment scaling in $\mathbb{R}^P$ (the uncoupled baseline used in Notebooks 04 and 05).
3. **Uncoupled Output-Coordinate Pullback NGD ($G^{\Theta,\text{coord}}$)**: Pulls back the flat Euclidean identity $h^{\text{out}} = I_{d_L}$ on raw outputs $z(x; \theta) \in \mathbb{R}^{d_L}$:
   $$G^{\Theta,\text{coord}}_{\mu\nu}(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\!\left[ \sum_{a=1}^{d_L} \frac{\partial z^a(x; \theta)}{\partial \theta^\mu} \frac{\partial z^a(x; \theta)}{\partial \theta^\nu} \right] + \lambda I_P$$
4. **Coupled Data-Geometry Pullback NGD ($G^{\Theta,\text{DeWitt}}$ & $G^{\Theta,\text{coupled}}$)**: Pulls back the **affine-invariant DeWitt / Ebin metric** on $2 \times 2$ SPD metric tensors $\mathrm{Sym}^+_2$ alongside the Generalized Gauss-Newton (GGN) metric of the target geometric scalar field $\psi(x; \theta)$:
   $$G^{\Theta,\text{DeWitt}}_{\mu\nu}(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\!\left[ g^{ik}(x; \theta)\, g^{jl}(x; \theta)\, \frac{\partial g_{ij}(x; \theta)}{\partial \theta^\mu} \frac{\partial g_{kl}(x; \theta)}{\partial \theta^\nu} \right]$$
   $$G^{\Theta,\text{coupled}}_{\mu\nu}(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\!\left[ w_\psi(x; \theta) \frac{\partial \psi(x; \theta)}{\partial \theta^\mu} \frac{\partial \psi(x; \theta)}{\partial \theta^\nu} \right] + \gamma\, G^{\Theta,\text{DeWitt}}_{\mu\nu}(\theta) + G^{\Theta,\text{reg}}_{\mu\nu}(\theta) + \lambda I_P$$
""",
        },
        {
            "type": "code",
            "source": r"""
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path.cwd()
for p in [Path.cwd(), Path.cwd().parent, Path.cwd() / Path("projects") / Path("diffgeom")]:
    if (p / "src" / "geometry.py").exists():
        PROJECT_ROOT = p
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

DATASET_NAME = CFG["dataset"]["name"]
N_CLASSES = CFG["dataset"].get("n_classes", 2)
DATASET_PARAMS = dict(CFG["dataset"].get("params", {}))
LAYER_DIMS = list(CFG["model"]["layer_dims"])
HIDDEN_ACTIVATION = CFG["model"].get("hidden_activation", "tanh")
OUTPUT_ACTIVATION = CFG["model"].get("output_activation", "identity")

import copy
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from src.geometry import (
    MoonsMLP,
    get_synthetic_dataset,
    make_evaluation_grid,
    compute_layer_geometry,
    compute_layer_pullback_metric,
    compute_fast_graph_geometry,
    compute_uncoupled_coord_metric,
    compute_coupled_dewitt_metric,
    compute_coupled_geometric_metric,
    curvature_margin_loss,
    volume_contrastive_loss,
    anisotropy_margin_loss,
    scalar_geometric_predict,
)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

CLASS_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

X_train, y_train, X_test, y_test = get_synthetic_dataset(
    dataset_name=DATASET_NAME,
    n_classes=N_CLASSES,
    input_dim=LAYER_DIMS[0],
    **DATASET_PARAMS,
)
xx, yy, grid_pts = make_evaluation_grid(X_train, n_grid=28, padding=0.45)
print(
    f"Dataset: {DATASET_NAME} (classes={N_CLASSES}) | Train: {X_train.shape}, Test: {X_test.shape} | "
    f"MLP: {LAYER_DIMS} (hidden={HIDDEN_ACTIVATION}, out={OUTPUT_ACTIVATION})"
)
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 1. Exact $E(2)$ Output Symmetry Quotienting & Spectral Structure on $\Theta$

When `OUTPUT_ACTIVATION = "identity"`, all three geometric objectives depend exclusively on $J_x(x)^T J_x(x)$ (or $I_2 + J_x(x)^T J_x(x)$), which is **strictly invariant under rigid Euclidean motions acting on output coordinates** $z \mapsto Qz + b$:
- **Output Translations**: Shifting the final bias $b^{(L)} \mapsto b^{(L)} + \delta b$.
- **Output Rotation ($\mathfrak{so}(d_L)$)**: Rotating the final layer $(W^{(L)}, b^{(L)}) \mapsto (e^{\epsilon \Omega} W^{(L)}, e^{\epsilon \Omega} b^{(L)})$.

Below we verify that `compute_fast_graph_geometry` matches `compute_layer_geometry` to `float64` machine precision and evaluate $v^T G^\Theta v$ along these output symmetry generators.
""",
        },
        {
            "type": "code",
            "source": r"""
probe_model = MoonsMLP(
    layer_dims=LAYER_DIMS,
    activation=HIDDEN_ACTIVATION,
    output_activation=OUTPUT_ACTIVATION,
    seed=7,
)
L = probe_model.num_layers
last_idx = L - 1
d_out = probe_model.layer_dims[-1]
P_total = sum(p.numel() for p in probe_model.parameters())
params_probe = {k: v.detach() for k, v in probe_model.named_parameters()}

# 1. Verify Gauss-equation curvature equivalence to float64 machine precision
ref_geom = compute_layer_geometry(probe_model, X_train[:64], layer_idx=L, metric_mode="graph", params=params_probe)
g_fast, K_fast = compute_fast_graph_geometry(probe_model, X_train[:64], params=params_probe)
print(f"Gauss Equation Consistency | max |g_fast - g_ref| = {(g_fast - ref_geom['g']).abs().max().item():.2e}")
print(f"Gauss Equation Consistency | max |K_fast - K_ref| = {(K_fast - ref_geom['K']).abs().max().item():.2e}\n")

# 2. Construct unit parameter vectors for output symmetry generators + a generic hidden direction
def build_unit_param_direction(overrides: dict, seed_random: int | None = None) -> torch.Tensor:
    if seed_random is not None:
        gen = torch.Generator().manual_seed(seed_random)
        raw = torch.randn(P_total, generator=gen, dtype=torch.float64)
        return raw / raw.norm()
    chunks = []
    for k, p in params_probe.items():
        if k in overrides:
            chunks.append(overrides[k].reshape(-1))
        else:
            chunks.append(torch.zeros_like(p).reshape(-1))
    vec = torch.cat(chunks)
    return vec / vec.norm()

b_e1 = torch.zeros(d_out, dtype=torch.float64)
b_e1[0] = 1.0
b_e2 = torch.zeros(d_out, dtype=torch.float64)
b_e2[min(1, d_out - 1)] = 1.0
omega = torch.zeros((d_out, d_out), dtype=torch.float64)
if d_out >= 2:
    omega[0, 1] = -1.0
    omega[1, 0] = 1.0

directions = [
    ("Trans. z^1 (bL_0)", build_unit_param_direction({f"layers.{last_idx}.bias": b_e1})),
    ("Trans. z^2 (bL_1)", build_unit_param_direction({f"layers.{last_idx}.bias": b_e2})),
    ("Rot. so(2) (WL,bL)", build_unit_param_direction({
        f"layers.{last_idx}.weight": omega @ params_probe[f"layers.{last_idx}.weight"],
        f"layers.{last_idx}.bias": omega @ params_probe[f"layers.{last_idx}.bias"],
    })),
    ("Generic Hidden Dir.", build_unit_param_direction({}, seed_random=123)),
]

G_eucl = torch.eye(P_total, dtype=torch.float64)
G_coord_0 = compute_uncoupled_coord_metric(probe_model, X_train, damping=0.0)
G_dewitt_0 = compute_coupled_dewitt_metric(probe_model, X_train, metric_mode="graph", damping=0.0)
G_coup_vol_0 = compute_coupled_geometric_metric(probe_model, X_train, y_train, task="volume", margin=0.60, damping=0.0, n_classes=N_CLASSES)
G_coup_curv_0 = compute_coupled_geometric_metric(probe_model, X_train, y_train, task="curvature", margin=0.40, damping=0.0, n_classes=N_CLASSES)

metric_mats = [
    ("Uncoupled I_P", G_eucl, "#7f7f7f"),
    ("Uncoupled Coord-GGN", G_coord_0, "#ff7f0e"),
    ("Coupled DeWitt", G_dewitt_0, "#1f77b4"),
    ("Coupled Curvature-GGN", G_coup_curv_0, "#2ca02c"),
]

print(f"{'Parameter Direction':<22} | {'Uncoupled I_P':>14} | {'Uncoupled Coord':>15} | {'Coupled DeWitt':>15} | {'Coupled Curv-GGN':>16}")
print("-" * 94)
quad_table = {m_name: [] for m_name, _, _ in metric_mats}
for d_name, v in directions:
    vals = []
    for m_name, G_mat, _ in metric_mats:
        q = abs(float(v @ (G_mat @ v)))
        quad_table[m_name].append(max(q, 1e-18))
        vals.append(q)
    print(f"{d_name:<22} | {vals[0]:14.3e} | {vals[1]:15.3e} | {vals[2]:15.3e} | {vals[3]:16.3e}")

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))

x_pos = np.arange(len(directions))
bar_w = 0.19
for idx, (m_name, _, col) in enumerate(metric_mats):
    axes[0].bar(x_pos + (idx - 1.5) * bar_w, quad_table[m_name], width=bar_w, color=col, edgecolor="k", alpha=0.88, label=m_name)
axes[0].set_yscale("log")
axes[0].set_ylim(1e-19, 5e1)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels([d[0] for d in directions], rotation=12)
axes[0].set_ylabel(r"Metric Norm Squared $|v^T G^\Theta v|$ (log scale)")
axes[0].set_title(r"Exact Quotienting of $E(2)$ Output Symmetries")
axes[0].legend(fontsize=8.5, loc="upper left")
axes[0].grid(axis="y", alpha=0.3)

spec_items = [
    ("Uncoupled Coord-GGN", G_coord_0, "#ff7f0e", "--"),
    ("Coupled DeWitt (Graph)", G_dewitt_0, "#1f77b4", "-"),
    ("Coupled Volume-GGN", G_coup_vol_0, "#9467bd", "-."),
    ("Coupled Curvature-GGN", G_coup_curv_0, "#2ca02c", "-"),
]
for label, G_m, col, ls in spec_items:
    ev = torch.linalg.eigvalsh(G_m).flip(0).clamp(min=1e-16).numpy()
    axes[1].plot(range(1, len(ev) + 1), ev, lw=2.0, color=col, ls=ls, label=label)

axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel(f"Eigenvalue Index $k$ ($P = {P_total}$)")
axes[1].set_ylabel(r"Eigenvalue $\lambda_k(G^\Theta(\theta))$")
axes[1].set_title(f"{P_total}x{P_total} Parameter Metric Spectra at Initialization")
axes[1].legend(fontsize=8.5)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 2. Coupled vs. Uncoupled Optimization Across All Three Purely Geometric Losses

We now train identical configured MLPs (`seed = 7`) for **45 optimization steps** across all three purely geometric classification paradigms, comparing **Uncoupled Euclidean GD**, **Uncoupled Adam**, **Uncoupled Coord-NGD** ($G^{\Theta,\text{coord}}$), and **Coupled Geom-NGD** ($G^{\Theta,\text{coupled}}$).
""",
        },
        {
            "type": "code",
            "source": r"""
TASK_CONFIGS = {
    "volume": {
        "title": rf"Volume-Element Loss ($\log\sqrt{{\det g^{{({L})}}}}$)",
        "margin": 0.60,
        "gd_lr": 0.08,
        "adam_lr": 0.02,
        "coord_lr": 0.22,
        "coupled_lr": 0.32,
        "dewitt_weight": 0.30,
        "damping": 0.025,
    },
    "anisotropy": {
        "title": r"Metric Anisotropy Loss ($\log(\tilde{g}_{11}/\tilde{g}_{22})$)",
        "margin": 0.45,
        "gd_lr": 0.08,
        "adam_lr": 0.02,
        "coord_lr": 0.22,
        "coupled_lr": 0.32,
        "dewitt_weight": 0.25,
        "damping": 0.020,
    },
    "curvature": {
        "title": rf"Gaussian Curvature Loss ($K^{{({L})}} \lessgtr 0$)",
        "margin": 0.40,
        "gd_lr": 0.04,
        "adam_lr": 0.02,
        "coord_lr": 0.12,
        "coupled_lr": 0.28,
        "dewitt_weight": 0.20,
        "damping": 0.025,
    },
}


def evaluate_geometric_task(model: MoonsMLP, X_tr: torch.Tensor, y_tr: torch.Tensor, X_te: torch.Tensor, y_te: torch.Tensor, task: str, margin: float):
    params = dict(model.named_parameters())
    l_out = model.num_layers
    if task == "volume":
        g_tr = compute_layer_pullback_metric(model, X_tr, layer_idx=l_out, metric_mode="euclidean", eps=1e-6, params=params)
        loss, metrics = volume_contrastive_loss(g_tr, y_tr, margin=margin, n_classes=N_CLASSES)
        with torch.no_grad():
            g_te = compute_layer_pullback_metric(model, X_te, layer_idx=l_out, metric_mode="euclidean", eps=1e-6)
            psi_te = 0.5 * torch.log(torch.clamp(torch.linalg.det(g_te), min=1e-8))
            te_acc = (scalar_geometric_predict(psi_te, n_classes=N_CLASSES, margin=margin) == y_te).float().mean().item()
        return loss, metrics["geom_acc"], te_acc

    if task == "anisotropy":
        g_tr = compute_layer_pullback_metric(model, X_tr, layer_idx=l_out, metric_mode="graph", params=params)
        loss, metrics = anisotropy_margin_loss(g_tr, y_tr, margin=margin, n_classes=N_CLASSES)
        with torch.no_grad():
            g_te = compute_layer_pullback_metric(model, X_te, layer_idx=l_out, metric_mode="graph")
            psi_te = torch.log(torch.clamp(g_te[:, 0, 0], min=1e-8) / torch.clamp(g_te[:, 1, 1], min=1e-8))
            te_acc = (scalar_geometric_predict(psi_te, n_classes=N_CLASSES, margin=margin) == y_te).float().mean().item()
        return loss, metrics["geom_acc"], te_acc

    if task == "curvature":
        g_tr, K_tr = compute_fast_graph_geometry(model, X_tr, params=params)
        loss, metrics = curvature_margin_loss(K_tr, y_tr, g_tr, margin=margin, lambda_vol=0.02, lambda_iso=0.005, k_clip=10.0, n_classes=N_CLASSES)
        with torch.no_grad():
            _, K_te = compute_fast_graph_geometry(model, X_te)
            te_acc = (scalar_geometric_predict(K_te, n_classes=N_CLASSES, margin=margin) == y_te).float().mean().item()
        return loss, metrics["geom_acc"], te_acc

    raise ValueError(task)


def train_geometric_optimizer(task: str, opt_kind: str, n_steps: int = 45, seed: int = 7):
    cfg = TASK_CONFIGS[task]
    margin = cfg["margin"]
    mdl = MoonsMLP(
        layer_dims=LAYER_DIMS,
        activation=HIDDEN_ACTIVATION,
        output_activation=OUTPUT_ACTIVATION,
        seed=seed,
    )
    bL_init = mdl.layers[-1].bias.detach().clone()

    adam_opt = torch.optim.Adam(mdl.parameters(), lr=cfg["adam_lr"]) if opt_kind == "adam" else None
    hist = {"loss": [], "train_acc": [], "test_acc": [], "b5_drift": [], "time_s": []}
    t_start = time.perf_counter()

    for step in range(n_steps):
        mdl.zero_grad()
        loss, tr_acc, te_acc = evaluate_geometric_task(mdl, X_train, y_train, X_test, y_test, task, margin)
        b_drift = (mdl.layers[-1].bias.detach() - bL_init).norm().item()

        hist["loss"].append(loss.item())
        hist["train_acc"].append(tr_acc * 100.0)
        hist["test_acc"].append(te_acc * 100.0)
        hist["b5_drift"].append(b_drift)
        hist["time_s"].append(time.perf_counter() - t_start)

        loss.backward()

        if opt_kind == "gd":
            if task == "curvature":
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=5.0)
            with torch.no_grad():
                for p in mdl.parameters():
                    if p.grad is not None:
                        p -= cfg["gd_lr"] * p.grad
        elif opt_kind == "adam":
            if task == "curvature":
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=5.0)
            adam_opt.step()
        elif opt_kind in ("coord_ngd", "coupled_ngd"):
            grad_flat = torch.cat([
                p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel(), dtype=X_train.dtype, device=X_train.device)
                for p in mdl.parameters()
            ])
            if opt_kind == "coord_ngd":
                G_mat = compute_uncoupled_coord_metric(mdl, X_train, damping=cfg["damping"])
                lr = cfg["coord_lr"]
            else:
                G_mat = compute_coupled_geometric_metric(
                    mdl,
                    X_train,
                    y_train,
                    task=task,
                    margin=margin,
                    dewitt_weight=cfg["dewitt_weight"],
                    damping=cfg["damping"],
                    n_classes=N_CLASSES,
                )
                lr = cfg["coupled_lr"]

            nat_dir = torch.linalg.solve(G_mat, grad_flat)
            riem_norm = torch.sqrt(torch.clamp(nat_dir @ (G_mat @ nat_dir), min=1e-12)).item()
            trust_scale = min(1.0, 0.85 / riem_norm)
            with torch.no_grad():
                offset = 0
                for p in mdl.parameters():
                    n = p.numel()
                    p -= (lr * trust_scale) * nat_dir[offset : offset + n].reshape(p.shape)
                    offset += n
        else:
            raise ValueError(opt_kind)

    loss_f, tr_f, te_f = evaluate_geometric_task(mdl, X_train, y_train, X_test, y_test, task, margin)
    hist["loss"].append(loss_f.item())
    hist["train_acc"].append(tr_f * 100.0)
    hist["test_acc"].append(te_f * 100.0)
    hist["b5_drift"].append((mdl.layers[-1].bias.detach() - bL_init).norm().item())
    hist["time_s"].append(time.perf_counter() - t_start)

    return mdl, hist


OPTIMIZER_SPECS = [
    ("gd", "Uncoupled GD (I_P)", "#7f7f7f", ":"),
    ("adam", "Uncoupled Adam", "#1f77b4", "--"),
    ("coord_ngd", "Uncoupled Coord-NGD", "#ff7f0e", "-."),
    ("coupled_ngd", "Coupled Geom-NGD", "#2ca02c", "-"),
]

N_STEPS = 45
all_runs = {t: {} for t in ["volume", "anisotropy", "curvature"]}

print(f"{'Geometric Task':<13} | {'Optimizer':<21} | {'Step 10 Acc':>11} | {'Step 25 Acc':>11} | {'Final Loss':>10} | {'Final Test Acc':>14} | {'bL Gauge Drift':>14}")
print("-" * 112)

for task_name in ["volume", "anisotropy", "curvature"]:
    for opt_key, opt_label, _, _ in OPTIMIZER_SPECS:
        mdl_trained, h = train_geometric_optimizer(task_name, opt_key, n_steps=N_STEPS, seed=7)
        all_runs[task_name][opt_key] = (mdl_trained, h)
        print(
            f"{task_name:<13} | {opt_label:<21} | {h['test_acc'][10]:10.2f}% | {h['test_acc'][25]:10.2f}% | "
            f"{h['loss'][-1]:10.4f} | {h['test_acc'][-1]:13.2f}% | {h['b5_drift'][-1]:14.3e}"
        )
    print("-" * 112)

fig, axes = plt.subplots(2, 3, figsize=(16.8, 7.6), sharex=True)
steps_axis = np.arange(N_STEPS + 1)

for col_idx, task_name in enumerate(["volume", "anisotropy", "curvature"]):
    for opt_key, opt_label, col, ls in OPTIMIZER_SPECS:
        _, h = all_runs[task_name][opt_key]
        lw = 2.6 if opt_key == "coupled_ngd" else 1.9
        axes[0, col_idx].plot(steps_axis, h["loss"], color=col, ls=ls, lw=lw, label=opt_label)
        axes[1, col_idx].plot(steps_axis, h["test_acc"], color=col, ls=ls, lw=lw, label=opt_label)

    axes[0, col_idx].set_yscale("log")
    axes[0, col_idx].set_title(TASK_CONFIGS[task_name]["title"])
    axes[0, col_idx].grid(alpha=0.3, which="both")

    axes[1, col_idx].set_ylim(min(38.0, 100.0 / N_CLASSES - 10.0), 102.5)
    axes[1, col_idx].set_xlabel("Optimization Step")
    axes[1, col_idx].grid(alpha=0.3)

axes[0, 0].set_ylabel("Geometric Loss (log scale)")
axes[1, 0].set_ylabel("Held-Out Test Accuracy (%)")
axes[0, 0].legend(fontsize=8.5, loc="upper right")
axes[1, 2].legend(fontsize=8.5, loc="lower right")

plt.suptitle("Coupled Data-Geometry Natural Gradient vs. Uncoupled Optimizers Across All Three Geometric Losses", y=1.02, fontsize=13)
plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 3. $3 \times 4$ Spatial Decision Landscapes: Coupled vs. Uncoupled Learned Manifolds

To inspect how parameter-data metric coupling shapes the learned representation manifold in ambient input space $(x^1, x^2)$, we plot the learned target geometric field $\psi(x^1, x^2)$ after 45 steps across all **3 geometric objectives (rows)** and **4 optimizers (columns)**.
""",
        },
        {
            "type": "code",
            "source": r"""
def eval_target_field_on_points(mdl: MoonsMLP, pts: torch.Tensor, task: str) -> np.ndarray:
    l_out = mdl.num_layers
    if task == "volume":
        g = compute_layer_pullback_metric(mdl, pts, layer_idx=l_out, metric_mode="euclidean", eps=1e-6)
        return (0.5 * torch.log(torch.clamp(torch.linalg.det(g), min=1e-8))).detach().numpy()
    if task == "anisotropy":
        g = compute_layer_pullback_metric(mdl, pts, layer_idx=l_out, metric_mode="graph")
        return torch.log(torch.clamp(g[:, 0, 0], min=1e-8) / torch.clamp(g[:, 1, 1], min=1e-8)).detach().numpy()
    if task == "curvature":
        _, K = compute_fast_graph_geometry(mdl, pts)
        return np.arcsinh(K.detach().numpy())
    raise ValueError(task)

row_specs = [
    ("volume", "Log Volume\n" + rf"$\log\sqrt{{\det g^{{({L})}}(x)}}$"),
    ("anisotropy", "Anisotropy\n" + rf"$\log(\tilde{{g}}^{{({L})}}_{{11}}/\tilde{{g}}^{{({L})}}_{{22}})$"),
    ("curvature", "Curvature\n" + rf"$\mathrm{{arcsinh}}(K^{{({L})}}(x))$"),
]

fig, axes = plt.subplots(3, 4, figsize=(17.2, 11.0), sharex=True, sharey=True)

for r_idx, (task_name, row_lbl) in enumerate(row_specs):
    for c_idx, (opt_key, opt_label, _, _) in enumerate(OPTIMIZER_SPECS):
        ax = axes[r_idx, c_idx]
        mdl, h = all_runs[task_name][opt_key]
        fld = eval_target_field_on_points(mdl, grid_pts, task_name).reshape(xx.shape)
        vm = float(max(np.percentile(np.abs(fld), 96), 0.25))
        norm = TwoSlopeNorm(vmin=-vm, vcenter=0.0, vmax=vm)

        im = ax.contourf(xx, yy, fld, levels=np.linspace(-vm, vm, 25), cmap="RdBu_r", norm=norm, extend="both")
        if fld.min() < 0.0 < fld.max():
            ax.contour(xx, yy, fld, levels=[0.0], colors="k", linewidths=1.9)

        for c in range(N_CLASSES):
            mc = y_test == c
            ax.scatter(X_test[mc, 0], X_test[mc, 1], c=CLASS_COLORS[c % len(CLASS_COLORS)], edgecolors="w", s=18, alpha=0.85)

        acc_final = h["test_acc"][-1]
        loss_final = h["loss"][-1]
        is_coupled = opt_key == "coupled_ngd"
        header = f"{opt_label}\n" if r_idx == 0 else ""
        ax.set_title(
            f"{header}Acc: {acc_final:.1f}% | Loss: {loss_final:.3f}",
            fontsize=9.5,
            fontweight="bold" if is_coupled else "normal",
        )
        if c_idx == 0:
            ax.set_ylabel(f"{row_lbl}\n$x^2$", fontsize=10)
        if r_idx == 2:
            ax.set_xlabel("$x^1$", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

plt.suptitle("Learned Geometric Decision Landscapes at Step 45: Uncoupled Baselines vs. Coupled Geometric NGD", y=1.01, fontsize=13.5)
plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 4. Mechanistic Diagnostics: Spurious Gauge Drift & Unclipped Curvature Stiffness

Comparing the optimization trajectories exposes **two fundamental geometric mechanisms** distinguishing coupled from uncoupled parameter metrics:
1. **Elimination of Cross-Covariance Gauge Drift (`Uncoupled Coord-NGD` vs. `Coupled Geom-NGD`)**.
2. **Intrinsic Stiffness Preconditioning Without Gradient Clipping**.
""",
        },
        {
            "type": "code",
            "source": r"""
def run_unclipped_curvature_stiffness_test(steps: int = 35):
    results_stiff = {}
    for mode_name, lr in [("Uncoupled GD (Unclipped, lr=0.07)", 0.07), ("Coupled Geom-NGD (Unclipped, lr=0.28)", 0.28)]:
        mdl = MoonsMLP(
            layer_dims=LAYER_DIMS,
            activation=HIDDEN_ACTIVATION,
            output_activation=OUTPUT_ACTIVATION,
            seed=7,
        )
        losses, grad_norms, test_accs = [], [], []

        for _ in range(steps):
            mdl.zero_grad()
            params = dict(mdl.named_parameters())
            g_tr, K_tr = compute_fast_graph_geometry(mdl, X_train, params=params)
            loss, _ = curvature_margin_loss(K_tr, y_train, g_tr, margin=0.65, lambda_vol=0.02, lambda_iso=0.005, k_clip=25.0, n_classes=N_CLASSES)
            loss.backward()

            g_flat = torch.cat([
                p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel(), dtype=X_train.dtype)
                for p in mdl.parameters()
            ])
            grad_norms.append(g_flat.norm().item())
            losses.append(loss.item())

            with torch.no_grad():
                _, K_te = compute_fast_graph_geometry(mdl, X_test)
                preds_k = scalar_geometric_predict(K_te, n_classes=N_CLASSES, margin=0.65)
                test_accs.append((preds_k == y_test).float().mean().item() * 100.0)

            if "Uncoupled" in mode_name:
                with torch.no_grad():
                    for p in mdl.parameters():
                        if p.grad is not None:
                            p -= lr * p.grad
            else:
                G_c = compute_coupled_geometric_metric(
                    mdl, X_train, y_train, task="curvature", margin=0.65, k_clip=25.0, dewitt_weight=0.20, damping=0.025, n_classes=N_CLASSES
                )
                nat_step = torch.linalg.solve(G_c, g_flat)
                r_norm = torch.sqrt(torch.clamp(nat_step @ (G_c @ nat_step), min=1e-12)).item()
                scale = min(1.0, 0.85 / r_norm)
                with torch.no_grad():
                    offset = 0
                    for p in mdl.parameters():
                        n = p.numel()
                        p -= (lr * scale) * nat_step[offset : offset + n].reshape(p.shape)
                        offset += n

        results_stiff[mode_name] = {"loss": losses, "grad_norm": grad_norms, "test_acc": test_accs}
    return results_stiff

stiff_res = run_unclipped_curvature_stiffness_test(steps=35)

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.3))

task_colors = {"volume": "#9467bd", "anisotropy": "#ff7f0e", "curvature": "#d62728"}
for t_name, t_col in task_colors.items():
    axes[0].plot(all_runs[t_name]["coord_ngd"][1]["b5_drift"], ls="-.", lw=2.0, color=t_col, label=f"Uncoupled Coord-NGD ({t_name})")
    axes[0].plot(all_runs[t_name]["coupled_ngd"][1]["b5_drift"], ls="-", lw=2.2, color=t_col, alpha=0.85, label=f"Coupled Geom-NGD ({t_name})")

axes[0].set_xlabel("Optimization Step")
axes[0].set_ylabel(rf"Output Bias Drift $\|b^{{({L})}}(t) - b^{{({L})}}(0)\|_2$")
axes[0].set_title(r"Spurious Translation Gauge Drift on $\Theta$")
axes[0].legend(fontsize=7.5, ncol=1, loc="upper left")
axes[0].grid(alpha=0.3)

for label, col, ls in [
    ("Uncoupled GD (Unclipped, lr=0.07)", "#d62728", "--"),
    ("Coupled Geom-NGD (Unclipped, lr=0.28)", "#2ca02c", "-"),
]:
    axes[1].plot(stiff_res[label]["loss"], color=col, ls=ls, lw=2.2, label=label)
    axes[2].plot(stiff_res[label]["grad_norm"], color=col, ls=ls, lw=2.0, label=label)

axes[1].set_xlabel("Optimization Step")
axes[1].set_ylabel(r"Curvature Margin Loss ($\tau = 0.65$)")
axes[1].set_title("Unclipped 3rd-Order Curvature Loss Trajectory")
axes[1].legend(fontsize=8.2)
axes[1].grid(alpha=0.3)

axes[2].set_yscale("log")
axes[2].set_xlabel("Optimization Step")
axes[2].set_ylabel(r"Euclidean Gradient Norm $\|\nabla_\theta \mathcal{L}_{\mathrm{curv}}\|_2$")
axes[2].set_title("Euclidean Gradient Spikes vs. Coupled Stability")
axes[2].legend(fontsize=8.2)
axes[2].grid(alpha=0.3, which="both")

plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 5. Ablation: Universal Unsupervised DeWitt Metric ($G^{\Theta,\text{DeWitt}}$) vs. Task-Coupled Metric ($G^{\Theta,\text{coupled}}$)
""",
        },
        {
            "type": "code",
            "source": r"""
def train_pure_dewitt_ngd(task: str, n_steps: int = 45, seed: int = 7):
    cfg = TASK_CONFIGS[task]
    margin = cfg["margin"]
    m_mode = "euclidean" if task == "volume" else "graph"
    mdl = MoonsMLP(
        layer_dims=LAYER_DIMS,
        activation=HIDDEN_ACTIVATION,
        output_activation=OUTPUT_ACTIVATION,
        seed=seed,
    )
    hist = {"loss": [], "test_acc": []}

    for _ in range(n_steps):
        mdl.zero_grad()
        loss, _, te_acc = evaluate_geometric_task(mdl, X_train, y_train, X_test, y_test, task, margin)
        hist["loss"].append(loss.item())
        hist["test_acc"].append(te_acc * 100.0)
        loss.backward()

        g_flat = torch.cat([
            p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel(), dtype=X_train.dtype)
            for p in mdl.parameters()
        ])
        G_dw = compute_coupled_dewitt_metric(mdl, X_train, metric_mode=m_mode, eps=1e-6, inv_eps=1e-3, damping=cfg["damping"])
        nat_dir = torch.linalg.solve(G_dw, g_flat)
        r_norm = torch.sqrt(torch.clamp(nat_dir @ (G_dw @ nat_dir), min=1e-12)).item()
        scale = min(1.0, 0.85 / r_norm)
        lr_dw = 0.35 if task != "curvature" else 0.20
        with torch.no_grad():
            offset = 0
            for p in mdl.parameters():
                n = p.numel()
                p -= (lr_dw * scale) * nat_dir[offset : offset + n].reshape(p.shape)
                offset += n

    loss_f, _, te_f = evaluate_geometric_task(mdl, X_train, y_train, X_test, y_test, task, margin)
    hist["loss"].append(loss_f.item())
    hist["test_acc"].append(te_f * 100.0)
    return hist

dewitt_runs = {t: train_pure_dewitt_ngd(t, n_steps=N_STEPS, seed=7) for t in ["volume", "anisotropy", "curvature"]}

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.2), sharey=False)
for idx, task_name in enumerate(["volume", "anisotropy", "curvature"]):
    axes[idx].plot(all_runs[task_name]["gd"][1]["loss"], ":", lw=1.9, color="#7f7f7f", label=r"Uncoupled GD ($I_P$)")
    axes[idx].plot(dewitt_runs[task_name]["loss"], "--", lw=2.2, color="#1f77b4", label=r"Coupled DeWitt-NGD ($G^{\Theta,\mathrm{DeWitt}}$)")
    axes[idx].plot(all_runs[task_name]["coupled_ngd"][1]["loss"], "-", lw=2.5, color="#2ca02c", label=r"Task-Coupled Geom-NGD ($G^{\Theta,\mathrm{coupled}}$)")
    axes[idx].set_yscale("log")
    axes[idx].set_xlabel("Optimization Step")
    axes[idx].set_title(TASK_CONFIGS[task_name]["title"])
    axes[idx].grid(alpha=0.3, which="both")

axes[0].set_ylabel("Geometric Loss (log scale)")
axes[0].legend(fontsize=8.3)
plt.suptitle("Ablation: Unsupervised Data-Metric Pullback (DeWitt) vs. Full Sobolev/Task-Coupled Pullback", y=1.03, fontsize=13)
plt.tight_layout()
plt.show()

print(f"{'Task':<12} | {'Uncoupled GD Final Loss (Acc)':<32} | {'Pure DeWitt-NGD Final Loss (Acc)':<34} | {'Task-Coupled NGD Final Loss (Acc)':<34}")
print("-" * 120)
for t in ["volume", "anisotropy", "curvature"]:
    h_gd = all_runs[t]["gd"][1]
    h_dw = dewitt_runs[t]
    h_cp = all_runs[t]["coupled_ngd"][1]
    print(
        f"{t:<12} | {h_gd['loss'][-1]:8.4f} ({h_gd['test_acc'][-1]:6.2f}%)               | "
        f"{h_dw['loss'][-1]:8.4f} ({h_dw['test_acc'][-1]:6.2f}%)                 | "
        f"{h_cp['loss'][-1]:8.4f} ({h_cp['test_acc'][-1]:6.2f}%)"
    )
""",
        },
    ]
    build_and_execute_notebook(ROOT / "notebooks" / "06-coupled-optimization.ipynb", cells)


def build_notebook_07() -> None:
    cells = [
        {
            "type": "markdown",
            "source": r"""
# Notebook 07: Evaluating Efficient First-Order Approximations for Geometric Training

## Research Objective
Training neural networks with **exact differential-geometric losses** requires nested symbolic automatic differentiation:
- **Volume-Element** ($\log\sqrt{\det g^{(L)}}$) and **Metric Anisotropy** ($\log(\tilde{g}^{(L)}_{11}/\tilde{g}^{(L)}_{22})$) require **2nd-order** mixed derivatives $\nabla_\theta \partial_x z$.
- **Gaussian Curvature** ($K^{(L)} = R_{1212}/\det\tilde{g}^{(L)}$) requires **3rd-order** mixed derivatives $\nabla_\theta \partial_x^2 z$.

In this notebook, we implement and benchmark **four principled approaches** that reduce compute and memory toward standard **1st-order `SGD` with Cross-Entropy loss** by replacing symbolic spatial derivatives with forward-only discrete or algebraic geometric operators:

1. **Approach 1 — Batched Finite-Difference Stencils (Pure 1st-Order Autodiff)**
2. **Approach 2 — Precomputed Point-Cloud $k$-NN Taylor Jet Operators (`1N` Single Forward Pass)**
3. **Approach 3 — Randomized Tangent-Plane Sketching / Hutchinson Directional Probes ($O(1)$ in Input Dimension $d_0$)**
4. **Approach 4 — Algebraic Order Reduction Before Autodiff (Gauss's Extrinsic-to-Intrinsic Identity)**
""",
        },
        {
            "type": "code",
            "source": r"""
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path.cwd()
for p in [Path.cwd(), Path.cwd().parent, Path.cwd() / Path("projects") / Path("diffgeom")]:
    if (p / "src" / "geometry.py").exists():
        PROJECT_ROOT = p
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

DATASET_NAME = CFG["dataset"]["name"]
N_CLASSES = CFG["dataset"].get("n_classes", 2)
DATASET_PARAMS = dict(CFG["dataset"].get("params", {}))
LAYER_DIMS = list(CFG["model"]["layer_dims"])
HIDDEN_ACTIVATION = CFG["model"].get("hidden_activation", "tanh")
OUTPUT_ACTIVATION = CFG["model"].get("output_activation", "identity")

import time
import numpy as np
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from src.geometry import (
    MoonsMLP,
    get_synthetic_dataset,
    make_evaluation_grid,
    compute_layer_geometry,
    compute_layer_pullback_metric,
    compute_fast_graph_geometry,
    compute_fd_pullback_metric,
    compute_fd_graph_curvature,
    build_knn_jet_operator,
    compute_knn_jet_geometry,
    compute_sketched_pullback_metric,
    compute_sketched_plane_curvature,
    compute_coupled_geometric_metric,
    woodbury_coupled_ngd_direction,
    curvature_margin_loss,
    volume_contrastive_loss,
    anisotropy_margin_loss,
    scalar_geometric_predict,
)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

CE_LAYER_DIMS = [*LAYER_DIMS[:-1], max(LAYER_DIMS[-1], N_CLASSES)]
CLASS_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

X_train, y_train, X_test, y_test = get_synthetic_dataset(
    dataset_name=DATASET_NAME,
    n_classes=N_CLASSES,
    input_dim=LAYER_DIMS[0],
    **DATASET_PARAMS,
)
xx, yy, grid_pts = make_evaluation_grid(X_train, n_grid=28, padding=0.45)

# Precompute offline t=0 point-cloud k-NN operators (Approach 2)
t0_knn = time.perf_counter()
knn_idx_lin, knn_op_lin = build_knn_jet_operator(X_train, k_neighbors=10, order=1, ridge=1e-4)
knn_idx_quad, knn_op_quad = build_knn_jet_operator(X_train, k_neighbors=16, order=2, ridge=1e-4)
knn_precomp_ms = (time.perf_counter() - t0_knn) * 1000.0
print(
    f"Dataset: {DATASET_NAME} (classes={N_CLASSES}) | MLP: {LAYER_DIMS} (hidden={HIDDEN_ACTIVATION}, out={OUTPUT_ACTIVATION})\n"
    f"Offline t=0 k-NN Jet Precomputation: {knn_precomp_ms:.2f} ms | Linear op: {tuple(knn_op_lin.shape)}, Quadratic op: {tuple(knn_op_quad.shape)}"
)

# Section 1: Numerical approximation fidelity vs. Exact Autodiff on a trained reference MLP
ref_mlp = MoonsMLP(
    layer_dims=CE_LAYER_DIMS,
    activation=HIDDEN_ACTIVATION,
    output_activation=OUTPUT_ACTIVATION,
    seed=7,
)
L_ref = ref_mlp.num_layers
ref_opt = torch.optim.SGD(ref_mlp.parameters(), lr=0.12, momentum=0.9)
ce_loss_fn = nn.CrossEntropyLoss()
for _ in range(100):
    ref_opt.zero_grad()
    ce_loss_fn(ref_mlp(X_train), y_train).backward()
    ref_opt.step()

with torch.no_grad():
    geom_exact_levi = compute_layer_geometry(ref_mlp, X_train, layer_idx=L_ref, metric_mode="graph")
    g_exact_gauss, K_exact_gauss = compute_fast_graph_geometry(ref_mlp, X_train)
    g_eucl_exact = compute_layer_pullback_metric(ref_mlp, X_train, layer_idx=L_ref, metric_mode="euclidean", eps=1e-6)
    lv_exact = 0.5 * torch.log(torch.clamp(torch.linalg.det(g_eucl_exact), min=1e-8))

    h_grid = np.logspace(-6, -1, 16)
    fd_vol_errs, fd_curv_errs = [], []
    for h_val in h_grid:
        g_fd_h = compute_fd_pullback_metric(ref_mlp, X_train, h=float(h_val), metric_mode="euclidean", eps=1e-6)
        lv_fd_h = 0.5 * torch.log(torch.clamp(torch.linalg.det(g_fd_h), min=1e-8))
        fd_vol_errs.append((lv_fd_h - lv_exact).abs().mean().item())

        _, K_fd_h = compute_fd_graph_curvature(ref_mlp, X_train, h=float(h_val))
        fd_curv_errs.append((K_fd_h - K_exact_gauss).abs().mean().item())

    _, K_fd_opt = compute_fd_graph_curvature(ref_mlp, X_train, h=2e-3)
    _, K_knn_est = compute_knn_jet_geometry(ref_mlp, X_train, knn_idx_quad, knn_op_quad, metric_mode="graph")

print(f"Approach 4 (Gauss Eq.) max |K_gauss - K_levi_civita| : {(K_exact_gauss - geom_exact_levi['K']).abs().max().item():.2e}")
print(f"Approach 1 (7-pt FD, h=2e-3) mean |K_fd - K_exact|   : {(K_fd_opt - K_exact_gauss).abs().mean().item():.2e}")
print(f"Approach 2 (1N k-NN Jet) Pearson r(K_knn, K_exact)   : {np.corrcoef(K_knn_est.numpy(), K_exact_gauss.numpy())[0, 1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.3))
axes[0].plot(h_grid, fd_vol_errs, "o-", lw=2.0, color="#1f77b4", label=r"Approach 1: 3-pt Forward FD $|\Delta \log\sqrt{\det g}|$ ($O(h)$)")
axes[0].plot(h_grid, fd_curv_errs, "s-", lw=2.0, color="#d62728", label=rf"Approach 1: 7-pt Central FD $|\Delta K^{{({L_ref})}}|$ ($O(h^2)$)")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel(r"Finite-Difference Step Size $h$")
axes[0].set_ylabel("Mean Absolute Approximation Error")
axes[0].set_title("Stencil Truncation vs. Floating-Point Roundoff Scaling")
axes[0].legend(fontsize=8.3)
axes[0].grid(alpha=0.3, which="both")

axes[1].scatter(np.arcsinh(K_exact_gauss.numpy()), np.arcsinh(K_fd_opt.numpy()), s=20, alpha=0.75, color="#2ca02c", label=r"Approach 1: 7-pt FD Stencil ($h=2\times 10^{-3}$)")
axes[1].scatter(np.arcsinh(K_exact_gauss.numpy()), np.arcsinh(K_knn_est.numpy()), s=20, alpha=0.65, color="#ff7f0e", marker="^", label="Approach 2: 1N Point-Cloud $k$-NN Jet")
lim_v = max(np.abs(np.arcsinh(K_exact_gauss.numpy())).max() * 1.1, 1.0)
axes[1].plot([-lim_v, lim_v], [-lim_v, lim_v], "k--", lw=1.4, label="Exact Identity $y = x$")
axes[1].set_xlabel(rf"Exact Analytical $\mathrm{{arcsinh}}(K^{{({L_ref})}}(x))$")
axes[1].set_ylabel(rf"1st-Order Approximated $\mathrm{{arcsinh}}(\hat{{K}}^{{({L_ref})}}(x))$")
axes[1].set_title("Pointwise Curvature Fidelity on Trained Network")
axes[1].legend(fontsize=8.3)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 2. Step-Level Compute, Memory & Autograd Graph Benchmark (vs. `SGD` Cross-Entropy Baseline)

We benchmark a complete **forward + backward + `SGD.step()`** iteration across **14 configurations**, anchored by **`SGD + Cross-Entropy`** (`1st-Order` reference).
""",
        },
        {
            "type": "code",
            "source": r"""
def count_autograd_nodes(grad_fn) -> int:
    visited = set()
    stack = [grad_fn]
    while stack:
        node = stack.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        for nxt, _ in getattr(node, "next_functions", []):
            stack.append(nxt)
    return len(visited)


def benchmark_sgd_step(loss_fn, model: MoonsMLP, n_repeats: int = 25):
    sgd = torch.optim.SGD(model.parameters(), lr=1e-4)

    sgd.zero_grad()
    l_warm = loss_fn()
    n_nodes = count_autograd_nodes(l_warm.grad_fn)
    l_warm.backward()
    sgd.step()

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        sgd.zero_grad()
        l_mem = loss_fn()
        l_mem.backward()
        sgd.step()
    mem_mb = sum(e.cpu_memory_usage for e in prof.key_averages() if e.cpu_memory_usage > 0) / (1024.0 * 1024.0)

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        sgd.zero_grad()
        l_run = loss_fn()
        l_run.backward()
        sgd.step()
    step_ms = (time.perf_counter() - t0) * 1000.0 / n_repeats
    return step_ms, mem_mb, n_nodes


bench_net_ce = MoonsMLP(layer_dims=CE_LAYER_DIMS, activation=HIDDEN_ACTIVATION, output_activation=OUTPUT_ACTIVATION, seed=7)
bench_net = MoonsMLP(layer_dims=LAYER_DIMS, activation=HIDDEN_ACTIVATION, output_activation=OUTPUT_ACTIVATION, seed=7)
L = bench_net.num_layers
rng_probe = torch.Generator().manual_seed(42)

BENCHMARK_SUITE = [
    ("Baseline", "SGD + Cross-Entropy", "1st", "1N", bench_net_ce, lambda: ce_loss_fn(bench_net_ce(X_train), y_train), "#111111"),
    # Volume-Element Loss
    ("Volume", "Vol: Exact 2nd-Order Autodiff", "2nd", "jacrev", bench_net, lambda: volume_contrastive_loss(compute_layer_pullback_metric(bench_net, X_train, L, "euclidean", 1e-6, params=dict(bench_net.named_parameters())), y_train, 0.60, n_classes=N_CLASSES)[0], "#8c564b"),
    ("Volume", "Vol: App1 (3-pt FD Stencil)", "1st", "3N", bench_net, lambda: volume_contrastive_loss(compute_fd_pullback_metric(bench_net, X_train, metric_mode="euclidean"), y_train, 0.60, n_classes=N_CLASSES)[0], "#1f77b4"),
    ("Volume", "Vol: App2 (1N k-NN Linear)", "1st", "1N", bench_net, lambda: volume_contrastive_loss(compute_knn_jet_geometry(bench_net, X_train, knn_idx_lin, knn_op_lin, metric_mode="euclidean")[0], y_train, 0.60, n_classes=N_CLASSES)[0], "#2ca02c"),
    ("Volume", "Vol: App3 (2-Probe Sketch)", "1st", "3N", bench_net, lambda: volume_contrastive_loss(compute_sketched_pullback_metric(bench_net, X_train, n_probes=2, metric_mode="euclidean", generator=rng_probe)[0], y_train, 0.60, n_classes=N_CLASSES)[0], "#9467bd"),
    # Metric Anisotropy Loss
    ("Anisotropy", "Aniso: Exact 2nd-Order Autodiff", "2nd", "jacrev", bench_net, lambda: anisotropy_margin_loss(compute_layer_pullback_metric(bench_net, X_train, L, "graph", params=dict(bench_net.named_parameters())), y_train, 0.45, n_classes=N_CLASSES)[0], "#8c564b"),
    ("Anisotropy", "Aniso: App1 (3-pt FD Stencil)", "1st", "3N", bench_net, lambda: anisotropy_margin_loss(compute_fd_pullback_metric(bench_net, X_train, metric_mode="graph"), y_train, 0.45, n_classes=N_CLASSES)[0], "#1f77b4"),
    ("Anisotropy", "Aniso: App2 (1N k-NN Linear)", "1st", "1N", bench_net, lambda: anisotropy_margin_loss(compute_knn_jet_geometry(bench_net, X_train, knn_idx_lin, knn_op_lin, metric_mode="graph")[0], y_train, 0.45, n_classes=N_CLASSES)[0], "#2ca02c"),
    ("Anisotropy", "Aniso: App3 (1-Probe Hutch.)", "1st", "2N", bench_net, lambda: anisotropy_margin_loss(compute_sketched_pullback_metric(bench_net, X_train, n_probes=1, metric_mode="graph", generator=rng_probe)[0], y_train, 0.45, n_classes=N_CLASSES)[0], "#9467bd"),
    # Gaussian Curvature Loss
    ("Curvature", "Curv: Exact 3rd-Order (Levi-Civ.)", "3rd", "jacrev^3", bench_net, lambda: (lambda g: curvature_margin_loss(g["K"], y_train, g["g"], margin=0.40, n_classes=N_CLASSES)[0])(compute_layer_geometry(bench_net, X_train, L, "graph", params=dict(bench_net.named_parameters()))), "#d62728"),
    ("Curvature", "Curv: App4 (Exact Gauss Eq.)", "3rd", "jacfwd^2", bench_net, lambda: (lambda gK: curvature_margin_loss(gK[1], y_train, gK[0], margin=0.40, n_classes=N_CLASSES)[0])(compute_fast_graph_geometry(bench_net, X_train, params=dict(bench_net.named_parameters()))), "#ff7f0e"),
    ("Curvature", "Curv: App1 (7-pt FD Stencil)", "1st", "7N", bench_net, lambda: (lambda gK: curvature_margin_loss(gK[1], y_train, gK[0], margin=0.40, n_classes=N_CLASSES)[0])(compute_fd_graph_curvature(bench_net, X_train)), "#1f77b4"),
    ("Curvature", "Curv: App2 (1N k-NN Quad Jet)", "1st", "1N", bench_net, lambda: (lambda gK: curvature_margin_loss(gK[1], y_train, gK[0], margin=0.40, n_classes=N_CLASSES)[0])(compute_knn_jet_geometry(bench_net, X_train, knn_idx_quad, knn_op_quad, metric_mode="graph")), "#2ca02c"),
    ("Curvature", "Curv: App3 (Random 2-Plane)", "1st", "7N", bench_net, lambda: (lambda gK: curvature_margin_loss(gK[1], y_train, gK[0], margin=0.40, n_classes=N_CLASSES)[0])(compute_sketched_plane_curvature(bench_net, X_train, generator=rng_probe)), "#9467bd"),
]

bench_records = []
for grp, lbl, order_str, bmult, net_m, fn, col in BENCHMARK_SUITE:
    ms, mb, nodes = benchmark_sgd_step(fn, net_m)
    bench_records.append({
        "group": grp,
        "label": lbl,
        "order": order_str,
        "batch": bmult,
        "ms": ms,
        "mb": mb,
        "nodes": nodes,
        "color": col,
    })

exact_ref_by_group = {
    "Baseline": bench_records[0],
    "Volume": bench_records[1],
    "Anisotropy": bench_records[5],
    "Curvature": bench_records[9],
}

print(f"{'Method':<34} | {'Order':^5} | {'Evals':^8} | {'Latency (ms)':>12} | {'Speedup':>8} | {'Memory (MB)':>11} | {'Mem Save':>9} | {'Nodes':>6}")
print("-" * 114)
for rec in bench_records:
    ref = exact_ref_by_group[rec["group"]]
    sp = ref["ms"] / rec["ms"]
    msave = ref["mb"] / rec["mb"]
    print(
        f"{rec['label']:<34} | {rec['order']:^5} | {rec['batch']:^8} | {rec['ms']:12.2f} | {sp:7.1f}x | "
        f"{rec['mb']:11.2f} | {msave:8.1f}x | {rec['nodes']:6d}"
    )

plot_subset = [bench_records[i] for i in [0, 1, 2, 3, 9, 10, 11, 12, 13]]
short_labels = [
    "SGD + CE\n(1st, 1N)",
    "Vol Exact\n(2nd)",
    "Vol App1\nFD (1st)",
    "Vol App2\nkNN (1st)",
    "Curv Exact\nLevi (3rd)",
    "Curv App4\nGauss (3rd)",
    "Curv App1\n7pt FD (1st)",
    "Curv App2\n1N kNN (1st)",
    "Curv App3\nSketch (1st)",
]

fig, axes = plt.subplots(1, 3, figsize=(17.2, 4.8))
y_idx = np.arange(len(plot_subset))
cols_sub = [r["color"] for r in plot_subset]
ms_sub = [r["ms"] for r in plot_subset]
mb_sub = [r["mb"] for r in plot_subset]

bars0 = axes[0].barh(y_idx, ms_sub, color=cols_sub, edgecolor="k", alpha=0.88, height=0.65)
axes[0].set_xscale("log")
axes[0].set_yticks(y_idx)
axes[0].set_yticklabels(short_labels, fontsize=8.5)
axes[0].invert_yaxis()
axes[0].set_xlabel("Step Latency (ms / SGD step, log scale)")
axes[0].set_title("Compute Cost per Training Step")
axes[0].set_xlim(0.4, max(ms_sub) * 3.2)
axes[0].grid(axis="x", alpha=0.3, which="both")
for b, v in zip(bars0, ms_sub):
    axes[0].text(v * 1.12, b.get_y() + b.get_height() / 2, f"{v:.2f} ms", va="center", fontsize=8, fontweight="bold")

bars1 = axes[1].barh(y_idx, mb_sub, color=cols_sub, edgecolor="k", alpha=0.88, height=0.65)
axes[1].set_xscale("log")
axes[1].set_yticks(y_idx)
axes[1].set_yticklabels([""] * len(plot_subset))
axes[1].invert_yaxis()
axes[1].set_xlabel("CPU Tensor Memory (MB / SGD step, log scale)")
axes[1].set_title("Memory Footprint per Training Step")
axes[1].set_xlim(0.2, max(mb_sub) * 4.0)
axes[1].grid(axis="x", alpha=0.3, which="both")
for b, v in zip(bars1, mb_sub):
    axes[1].text(v * 1.12, b.get_y() + b.get_height() / 2, f"{v:.2f} MB", va="center", fontsize=8, fontweight="bold")

for rec in bench_records:
    marker = "*" if rec["group"] == "Baseline" else ("D" if rec["order"] != "1st" else "o")
    sz = 220 if rec["group"] == "Baseline" else max(rec["nodes"] * 0.22, 65)
    axes[2].scatter(rec["ms"], rec["mb"], s=sz, c=rec["color"], marker=marker, edgecolors="k", lw=1.2, alpha=0.88)

for i in [0, 1, 3, 9, 10, 11, 12]:
    r = bench_records[i]
    axes[2].annotate(r["label"].replace("Curv: ", "").replace("Vol: ", ""), (r["ms"], r["mb"]), textcoords="offset points", xytext=(6, 4), fontsize=7.8)

axes[2].set_xscale("log")
axes[2].set_yscale("log")
axes[2].set_xlabel("Step Latency (ms / step, log scale)")
axes[2].set_ylabel("Step Memory (MB / step, log scale)")
axes[2].set_title("Compute–Memory Pareto Frontier vs. SGD + CE")
axes[2].set_xlim(0.5, max(ms_sub) * 4.5)
axes[2].set_ylim(0.25, max(mb_sub) * 3.0)
axes[2].grid(alpha=0.3, which="both")

plt.suptitle("Approaching First-Order SGD Efficiency: Compute, Memory, and Graph Scaling Across Approaches 1–4", y=1.03, fontsize=13)
plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 3. End-to-End `SGD` Training Convergence & Exact Analytical Test Accuracy

Below we train **every method using `torch.optim.SGD` (`momentum = 0.9`)** from the same initialization (`seed = 7`) alongside the **`SGD + Cross-Entropy` baseline**, and evaluate held-out test accuracy at every checkpoint using the **exact analytical differential-geometric operators**.
""",
        },
        {
            "type": "code",
            "source": r"""
def exact_test_accuracy(mdl: MoonsMLP, task: str) -> float:
    l_out = mdl.num_layers
    with torch.no_grad():
        if task == "ce":
            return (mdl(X_test).argmax(dim=-1) == y_test).float().mean().item() * 100.0
        if task == "volume":
            g = compute_layer_pullback_metric(mdl, X_test, layer_idx=l_out, metric_mode="euclidean", eps=1e-6)
            lv = 0.5 * torch.log(torch.clamp(torch.linalg.det(g), min=1e-8))
            return (scalar_geometric_predict(lv, n_classes=N_CLASSES, margin=0.60) == y_test).float().mean().item() * 100.0
        if task == "anisotropy":
            g = compute_layer_pullback_metric(mdl, X_test, layer_idx=l_out, metric_mode="graph")
            lr = torch.log(torch.clamp(g[:, 0, 0], min=1e-8) / torch.clamp(g[:, 1, 1], min=1e-8))
            return (scalar_geometric_predict(lr, n_classes=N_CLASSES, margin=0.45) == y_test).float().mean().item() * 100.0
        if task == "curvature":
            _, K = compute_fast_graph_geometry(mdl, X_test)
            return (scalar_geometric_predict(K, n_classes=N_CLASSES, margin=0.40) == y_test).float().mean().item() * 100.0
    raise ValueError(task)


def run_sgd_training(task: str, approx_mode: str, epochs: int = 120, lr: float = 0.06, seed: int = 7):
    dims_use = CE_LAYER_DIMS if task == "ce" else LAYER_DIMS
    mdl = MoonsMLP(
        layer_dims=dims_use,
        activation=HIDDEN_ACTIVATION,
        output_activation=OUTPUT_ACTIVATION,
        seed=seed,
    )
    l_out = mdl.num_layers
    opt = torch.optim.SGD(mdl.parameters(), lr=lr, momentum=0.9)
    rng = torch.Generator().manual_seed(seed + 100)

    eval_epochs, acc_curve, time_ms_curve, loss_curve = [0], [exact_test_accuracy(mdl, task)], [0.0], []
    elapsed_train_ms = 0.0

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        opt.zero_grad()

        if task == "ce":
            loss = ce_loss_fn(mdl(X_train), y_train)
        elif task == "volume":
            if approx_mode == "exact":
                g = compute_layer_pullback_metric(mdl, X_train, l_out, "euclidean", 1e-6, params=dict(mdl.named_parameters()))
            elif approx_mode == "app1_fd":
                g = compute_fd_pullback_metric(mdl, X_train, metric_mode="euclidean")
            elif approx_mode == "app2_knn":
                g, _ = compute_knn_jet_geometry(mdl, X_train, knn_idx_lin, knn_op_lin, metric_mode="euclidean")
            elif approx_mode == "app3_sketch":
                g, _ = compute_sketched_pullback_metric(mdl, X_train, n_probes=2, metric_mode="euclidean", generator=rng)
            loss, _ = volume_contrastive_loss(g, y_train, margin=0.60, n_classes=N_CLASSES)
        elif task == "anisotropy":
            if approx_mode == "exact":
                g = compute_layer_pullback_metric(mdl, X_train, l_out, "graph", params=dict(mdl.named_parameters()))
            elif approx_mode == "app1_fd":
                g = compute_fd_pullback_metric(mdl, X_train, metric_mode="graph")
            elif approx_mode == "app2_knn":
                g, _ = compute_knn_jet_geometry(mdl, X_train, knn_idx_lin, knn_op_lin, metric_mode="graph")
            elif approx_mode == "app3_sketch":
                g, _ = compute_sketched_pullback_metric(mdl, X_train, n_probes=1, metric_mode="graph", generator=rng)
            loss, _ = anisotropy_margin_loss(g, y_train, margin=0.45, n_classes=N_CLASSES)
        elif task == "curvature":
            if approx_mode == "exact_levi":
                gm = compute_layer_geometry(mdl, X_train, l_out, "graph", params=dict(mdl.named_parameters()))
                g, K = gm["g"], gm["K"]
            elif approx_mode == "app4_gauss":
                g, K = compute_fast_graph_geometry(mdl, X_train, params=dict(mdl.named_parameters()))
            elif approx_mode == "app1_fd":
                g, K = compute_fd_graph_curvature(mdl, X_train)
            elif approx_mode == "app2_knn":
                g, K = compute_knn_jet_geometry(mdl, X_train, knn_idx_quad, knn_op_quad, metric_mode="graph")
            elif approx_mode == "app3_sketch":
                g, K = compute_sketched_plane_curvature(mdl, X_train, generator=rng)
            loss, _ = curvature_margin_loss(K, y_train, g, margin=0.40, lambda_vol=0.02, lambda_iso=0.005, k_clip=10.0, n_classes=N_CLASSES)
        else:
            raise ValueError(task)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=5.0)
        opt.step()
        elapsed_train_ms += (time.perf_counter() - t0) * 1000.0
        loss_curve.append(loss.item())

        if ep % 10 == 0 or ep == epochs:
            eval_epochs.append(ep)
            acc_curve.append(exact_test_accuracy(mdl, task))
            time_ms_curve.append(elapsed_train_ms)

    return mdl, {
        "epochs": eval_epochs,
        "acc": acc_curve,
        "time_ms": time_ms_curve,
        "final_loss": loss_curve[-1],
        "total_ms": elapsed_train_ms,
    }


N_EPOCHS = 120
ce_mdl, ce_hist = run_sgd_training("ce", "exact", epochs=N_EPOCHS, lr=0.12)

TRAINING_MATRIX = {
    "volume": [
        ("exact", "Exact 2nd-Order Autodiff", 0.06, "#8c564b", "--"),
        ("app1_fd", "App 1: 3-pt FD Stencil (1st)", 0.06, "#1f77b4", "-"),
        ("app2_knn", "App 2: 1N k-NN Linear (1st)", 0.025, "#2ca02c", "-"),
        ("app3_sketch", "App 3: Random 2-Probe (1st)", 0.06, "#9467bd", "-."),
    ],
    "anisotropy": [
        ("exact", "Exact 2nd-Order Autodiff", 0.08, "#8c564b", "--"),
        ("app1_fd", "App 1: 3-pt FD Stencil (1st)", 0.08, "#1f77b4", "-"),
        ("app2_knn", "App 2: 1N k-NN Linear (1st)", 0.08, "#2ca02c", "-"),
        ("app3_sketch", "App 3: 1-Probe Hutch. (1st)", 0.08, "#9467bd", "-."),
    ],
    "curvature": [
        ("exact_levi", "Exact 3rd-Order (Levi-Civita)", 0.045, "#d62728", ":"),
        ("app4_gauss", "App 4: Exact Gauss Eq. (3rd)", 0.045, "#ff7f0e", "--"),
        ("app1_fd", "App 1: 7-pt FD Stencil (1st)", 0.045, "#1f77b4", "-"),
        ("app2_knn", "App 2: 1N k-NN Quad Jet (1st)", 0.035, "#2ca02c", "-"),
        ("app3_sketch", "App 3: Random 2-Plane (1st)", 0.045, "#9467bd", "-."),
    ],
}

trained_models = {t: {} for t in TRAINING_MATRIX}
print(f"{'Task & Training Method (All via SGD)':<44} | {'Train Time (ms)':>15} | {'vs SGD+CE Time':>14} | {'Exact Analytical Test Acc':>25}")
print("-" * 108)
print(f"{'Baseline: SGD + Cross-Entropy (1st-Order)':<44} | {ce_hist['total_ms']:15.1f} | {'1.00x':>14} | {ce_hist['acc'][-1]:24.2f}%")
print("-" * 108)

for t_name, specs in TRAINING_MATRIX.items():
    for mode_key, label, lr_val, col, ls in specs:
        m_tr, h_tr = run_sgd_training(t_name, mode_key, epochs=N_EPOCHS, lr=lr_val)
        trained_models[t_name][mode_key] = (m_tr, h_tr, label, col, ls)
        rel_time = h_tr["total_ms"] / ce_hist["total_ms"]
        print(f"{f'{t_name.capitalize()} — {label}':<44} | {h_tr['total_ms']:15.1f} | {f'{rel_time:.2f}x':>14} | {h_tr['acc'][-1]:24.2f}%")
    print("-" * 108)

fig, axes = plt.subplots(2, 3, figsize=(17.0, 7.8))
task_titles = {
    "volume": rf"Volume-Element ($\log\sqrt{{\det g^{{({L})}}}}$)",
    "anisotropy": r"Metric Anisotropy ($\log(\tilde{g}_{11}/\tilde{g}_{22})$)",
    "curvature": rf"Gaussian Curvature ($K^{{({L})}}$)",
}

for col_idx, t_name in enumerate(["volume", "anisotropy", "curvature"]):
    axes[0, col_idx].plot(ce_hist["epochs"], ce_hist["acc"], "k:", lw=2.0, label="Baseline: SGD + Cross-Entropy")
    axes[1, col_idx].plot(np.maximum(ce_hist["time_ms"], 1.0), ce_hist["acc"], "k:", lw=2.0, label="Baseline: SGD + Cross-Entropy")

    for mode_key, (_, h_tr, label, col, ls) in trained_models[t_name].items():
        axes[0, col_idx].plot(h_tr["epochs"], h_tr["acc"], color=col, ls=ls, lw=2.1, label=label)
        axes[1, col_idx].plot(np.maximum(h_tr["time_ms"], 1.0), h_tr["acc"], color=col, ls=ls, lw=2.1, label=label)

    axes[0, col_idx].set_title(task_titles[t_name])
    axes[0, col_idx].set_xlabel("SGD Training Epoch")
    axes[0, col_idx].set_ylim(min(38.0, 100.0 / N_CLASSES - 10.0), 102.5)
    axes[0, col_idx].grid(alpha=0.3)
    axes[0, col_idx].legend(fontsize=7.6, loc="lower right")

    axes[1, col_idx].set_xscale("log")
    axes[1, col_idx].set_xlabel("Cumulative Wall-Clock Training Time (ms, log scale)")
    axes[1, col_idx].set_ylim(min(38.0, 100.0 / N_CLASSES - 10.0), 102.5)
    axes[1, col_idx].grid(alpha=0.3, which="both")

axes[0, 0].set_ylabel("Exact Analytical Test Accuracy (%)")
axes[1, 0].set_ylabel("Exact Analytical Test Accuracy (%)")
plt.suptitle("End-to-End SGD Convergence by Epoch (Top) and Wall-Clock Compute Time (Bottom)", y=1.02, fontsize=13)
plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 4. Exact Analytical Decision Landscapes Learned via 1st-Order Approximations

We evaluate the **exact analytical** log-volume field $\log\sqrt{\det g^{(L)}(x)}$ (Row 0) and **exact analytical** Gaussian curvature $\mathrm{arcsinh}(K^{(L)}(x))$ (Row 1) across the ambient 2D plane.
""",
        },
        {
            "type": "code",
            "source": r"""
fig, axes = plt.subplots(2, 4, figsize=(17.0, 7.6), sharex=True, sharey=True)

vol_cols = [
    ("exact", "Exact 2nd-Order Autodiff"),
    ("app1_fd", "App 1: 3-pt FD Stencil (1st)"),
    ("app2_knn", "App 2: 1N k-NN Linear (1st)"),
    ("app3_sketch", "App 3: Random 2-Probe (1st)"),
]
curv_cols = [
    ("app4_gauss", "App 4: Exact Gauss Eq. (3rd)"),
    ("app1_fd", "App 1: 7-pt FD Stencil (1st)"),
    ("app2_knn", "App 2: 1N k-NN Quad Jet (1st)"),
    ("app3_sketch", "App 3: Random 2-Plane (1st)"),
]

for c_idx, (m_key, c_title) in enumerate(vol_cols):
    mdl, h_tr, _, _, _ = trained_models["volume"][m_key]
    with torch.no_grad():
        g_grid = compute_layer_pullback_metric(mdl, grid_pts, layer_idx=mdl.num_layers, metric_mode="euclidean", eps=1e-6)
        fld = (0.5 * torch.log(torch.clamp(torch.linalg.det(g_grid), min=1e-8))).numpy().reshape(xx.shape)
    vm = float(max(np.percentile(np.abs(fld), 96), 0.25))
    im = axes[0, c_idx].contourf(xx, yy, fld, levels=np.linspace(-vm, vm, 25), cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vm, vcenter=0.0, vmax=vm), extend="both")
    if fld.min() < 0.0 < fld.max():
        axes[0, c_idx].contour(xx, yy, fld, levels=[0.0], colors="k", linewidths=1.8)
    for c in range(N_CLASSES):
        mc = y_test == c
        axes[0, c_idx].scatter(X_test[mc, 0], X_test[mc, 1], c=CLASS_COLORS[c % len(CLASS_COLORS)], edgecolors="w", s=18)
    axes[0, c_idx].set_title(f"{c_title}\nExact Acc: {h_tr['acc'][-1]:.1f}% ({h_tr['total_ms']:.0f} ms)", fontsize=9.5)
    plt.colorbar(im, ax=axes[0, c_idx], fraction=0.046)

for c_idx, (m_key, c_title) in enumerate(curv_cols):
    mdl, h_tr, _, _, _ = trained_models["curvature"][m_key]
    with torch.no_grad():
        _, K_grid = compute_fast_graph_geometry(mdl, grid_pts)
        fld = np.arcsinh(K_grid.numpy()).reshape(xx.shape)
    vm = float(max(np.percentile(np.abs(fld), 96), 0.25))
    im = axes[1, c_idx].contourf(xx, yy, fld, levels=np.linspace(-vm, vm, 25), cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vm, vcenter=0.0, vmax=vm), extend="both")
    if fld.min() < 0.0 < fld.max():
        axes[1, c_idx].contour(xx, yy, fld, levels=[0.0], colors="k", linewidths=1.8)
    for c in range(N_CLASSES):
        mc = y_test == c
        axes[1, c_idx].scatter(X_test[mc, 0], X_test[mc, 1], c=CLASS_COLORS[c % len(CLASS_COLORS)], edgecolors="w", s=18)
    axes[1, c_idx].set_title(f"{c_title}\nExact Acc: {h_tr['acc'][-1]:.1f}% ({h_tr['total_ms']:.0f} ms)", fontsize=9.5)
    axes[1, c_idx].set_xlabel("$x^1$")
    plt.colorbar(im, ax=axes[1, c_idx], fraction=0.046)

axes[0, 0].set_ylabel("Exact Log Volume\n" + rf"$\log\sqrt{{\det g^{{({L})}}(x)}}$" + "\n$x^2$")
axes[1, 0].set_ylabel("Exact Curvature\n" + rf"$\mathrm{{arcsinh}}(K^{{({L})}}(x))$" + "\n$x^2$")
plt.suptitle("Exact Analytical Geometric Landscapes Learned via Exact vs. First-Order Approximated SGD Training", y=1.02, fontsize=13)
plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 5. High-Dimensional Input Scaling ($d_0 = 2 \to 128$): Why Randomized Tangent Sketching Wins
""",
        },
        {
            "type": "code",
            "source": r"""
dims_to_test = [2, 4, 8, 16, 32, 64, 128]
N_hd = 128
scaling_res = {
    "SGD + Cross-Entropy": {"ms": [], "mb": [], "col": "#111111", "ls": ":"},
    "Exact Metric Autodiff (2nd)": {"ms": [], "mb": [], "col": "#d62728", "ls": "--"},
    "App 1: Full Coord FD ((d0+1)N)": {"ms": [], "mb": [], "col": "#1f77b4", "ls": "-."},
    "App 3: 1-Probe Metric Sketch (2N)": {"ms": [], "mb": [], "col": "#2ca02c", "ls": "-"},
    "App 3: Random 2-Plane Curv. (7N)": {"ms": [], "mb": [], "col": "#9467bd", "ls": "-"},
}

rng_hd = torch.Generator().manual_seed(77)
for d0 in dims_to_test:
    X_hd = torch.randn(N_hd, d0, dtype=torch.float64)
    y_hd = torch.randint(0, N_CLASSES, (N_hd,))
    mlp_hd = MoonsMLP(
        layer_dims=(d0, 16, 16, max(LAYER_DIMS[-1], N_CLASSES)),
        activation=HIDDEN_ACTIVATION,
        output_activation=OUTPUT_ACTIVATION,
        seed=7,
    )
    l_hd = mlp_hd.num_layers

    ms, mb, _ = benchmark_sgd_step(lambda: ce_loss_fn(mlp_hd(X_hd), y_hd), mlp_hd, n_repeats=12)
    scaling_res["SGD + Cross-Entropy"]["ms"].append(ms)
    scaling_res["SGD + Cross-Entropy"]["mb"].append(mb)

    def exact_hd_loss():
        g = compute_layer_pullback_metric(mlp_hd, X_hd, layer_idx=l_hd, metric_mode="euclidean", eps=1e-6, params=dict(mlp_hd.named_parameters()))
        tr_log = torch.log(torch.clamp(torch.diagonal(g, dim1=-2, dim2=-1).sum(-1) / d0, min=1e-8))
        return torch.nn.functional.softplus(0.5 - (2.0 * (y_hd % 2).to(torch.float64) - 1.0) * tr_log).mean()

    ms, mb, _ = benchmark_sgd_step(exact_hd_loss, mlp_hd, n_repeats=8)
    scaling_res["Exact Metric Autodiff (2nd)"]["ms"].append(ms)
    scaling_res["Exact Metric Autodiff (2nd)"]["mb"].append(mb)

    def coord_fd_hd_loss():
        g = compute_fd_pullback_metric(mlp_hd, X_hd, metric_mode="euclidean")
        tr_log = torch.log(torch.clamp(torch.diagonal(g, dim1=-2, dim2=-1).sum(-1) / d0, min=1e-8))
        return torch.nn.functional.softplus(0.5 - (2.0 * (y_hd % 2).to(torch.float64) - 1.0) * tr_log).mean()

    ms, mb, _ = benchmark_sgd_step(coord_fd_hd_loss, mlp_hd, n_repeats=8)
    scaling_res["App 1: Full Coord FD ((d0+1)N)"]["ms"].append(ms)
    scaling_res["App 1: Full Coord FD ((d0+1)N)"]["mb"].append(mb)

    def sketch1_hd_loss():
        _, g_sub = compute_sketched_pullback_metric(mlp_hd, X_hd, n_probes=1, metric_mode="euclidean", generator=rng_hd)
        tr_log = torch.log(torch.clamp(g_sub[:, 0, 0], min=1e-8))
        return torch.nn.functional.softplus(0.5 - (2.0 * (y_hd % 2).to(torch.float64) - 1.0) * tr_log).mean()

    ms, mb, _ = benchmark_sgd_step(sketch1_hd_loss, mlp_hd, n_repeats=12)
    scaling_res["App 3: 1-Probe Metric Sketch (2N)"]["ms"].append(ms)
    scaling_res["App 3: 1-Probe Metric Sketch (2N)"]["mb"].append(mb)

    def sketch_curv_hd_loss():
        g_p, K_p = compute_sketched_plane_curvature(mlp_hd, X_hd, generator=rng_hd)
        return curvature_margin_loss(K_p, y_hd, g_p, margin=0.40, n_classes=N_CLASSES)[0]

    ms, mb, _ = benchmark_sgd_step(sketch_curv_hd_loss, mlp_hd, n_repeats=12)
    scaling_res["App 3: Random 2-Plane Curv. (7N)"]["ms"].append(ms)
    scaling_res["App 3: Random 2-Plane Curv. (7N)"]["mb"].append(mb)

print(f"{'d_0':>5} | {'SGD+CE (ms/MB)':>16} | {'Exact 2nd (ms/MB)':>19} | {'Coord FD (ms/MB)':>19} | {'App3 1-Probe (ms/MB)':>21} | {'App3 2-Plane K (ms/MB)':>23}")
print("-" * 116)
for i, d0 in enumerate(dims_to_test):
    print(
        f"{d0:5d} | "
        f"{scaling_res['SGD + Cross-Entropy']['ms'][i]:6.2f} / {scaling_res['SGD + Cross-Entropy']['mb'][i]:5.2f} | "
        f"{scaling_res['Exact Metric Autodiff (2nd)']['ms'][i]:8.2f} / {scaling_res['Exact Metric Autodiff (2nd)']['mb'][i]:6.2f} | "
        f"{scaling_res['App 1: Full Coord FD ((d0+1)N)']['ms'][i]:8.2f} / {scaling_res['App 1: Full Coord FD ((d0+1)N)']['mb'][i]:6.2f} | "
        f"{scaling_res['App 3: 1-Probe Metric Sketch (2N)']['ms'][i]:9.2f} / {scaling_res['App 3: 1-Probe Metric Sketch (2N)']['mb'][i]:6.2f} | "
        f"{scaling_res['App 3: Random 2-Plane Curv. (7N)']['ms'][i]:10.2f} / {scaling_res['App 3: Random 2-Plane Curv. (7N)']['mb'][i]:6.2f}"
    )

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
for name, d in scaling_res.items():
    axes[0].plot(dims_to_test, d["ms"], marker="o", color=d["col"], ls=d["ls"], lw=2.1, label=name)
    axes[1].plot(dims_to_test, d["mb"], marker="s", color=d["col"], ls=d["ls"], lw=2.1, label=name)

for ax, ylabel, title in [
    (axes[0], "Step Latency (ms / SGD step)", "Compute Scaling vs. Ambient Input Dimension $d_0$"),
    (axes[1], "Step Memory (MB / SGD step)", "Memory Scaling vs. Ambient Input Dimension $d_0$"),
]:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(dims_to_test)
    ax.set_xticklabels([str(x) for x in dims_to_test])
    ax.set_xlabel(r"Input Dimension $d_0$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8.0)
    ax.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.show()
""",
        },
        {
            "type": "markdown",
            "source": r"""
## 6. Matrix-Free Woodbury Coupled Natural Gradient (`O(S^2 P)` + 1st-Order Stencils)
""",
        },
        {
            "type": "code",
            "source": r"""
def compare_dense_vs_woodbury_coupled_ngd(n_steps: int = 35):
    results_ngd = {}
    rng_sub = torch.Generator().manual_seed(19)

    for mode, label, col, ls in [
        ("sgd_fd", "Uncoupled SGD + 7-pt FD (1st)", "#7f7f7f", ":"),
        ("woodbury_fd", "Woodbury Coupled NGD (S=16, 7-pt FD)", "#1f77b4", "--"),
        ("dense_exact", "Dense Coupled NGD (Exact Gauss)", "#2ca02c", "-"),
    ]:
        mdl = MoonsMLP(
            layer_dims=LAYER_DIMS,
            activation=HIDDEN_ACTIVATION,
            output_activation=OUTPUT_ACTIVATION,
            seed=7,
        )
        losses, accs = [], []

        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            mdl.zero_grad()
            if mode == "dense_exact":
                g_m, K_m = compute_fast_graph_geometry(mdl, X_train, params=dict(mdl.named_parameters()))
            else:
                g_m, K_m = compute_fd_graph_curvature(mdl, X_train)
            l_m, _ = curvature_margin_loss(K_m, y_train, g_m, margin=0.40, lambda_vol=0.02, lambda_iso=0.005, k_clip=10.0, n_classes=N_CLASSES)
            l_m.backward()
            if mode == "dense_exact":
                _ = compute_coupled_geometric_metric(mdl, X_train, y_train, task="curvature", margin=0.40, damping=0.025, n_classes=N_CLASSES)
            elif mode == "woodbury_fd":
                gf = torch.cat([p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel(), dtype=X_train.dtype) for p in mdl.parameters()])
                _ = woodbury_coupled_ngd_direction(mdl, X_train[:16], y_train[:16], gf, task="curvature", margin=0.40, n_classes=N_CLASSES)
        mem_mb = sum(e.cpu_memory_usage for e in prof.key_averages() if e.cpu_memory_usage > 0) / (1024.0 * 1024.0)

        mdl = MoonsMLP(
            layer_dims=LAYER_DIMS,
            activation=HIDDEN_ACTIVATION,
            output_activation=OUTPUT_ACTIVATION,
            seed=7,
        )
        sgd_opt = torch.optim.SGD(mdl.parameters(), lr=0.045, momentum=0.9) if mode == "sgd_fd" else None
        t0 = time.perf_counter()

        for _ in range(n_steps):
            mdl.zero_grad()
            if mode == "dense_exact":
                g_tr, K_tr = compute_fast_graph_geometry(mdl, X_train, params=dict(mdl.named_parameters()))
            else:
                g_tr, K_tr = compute_fd_graph_curvature(mdl, X_train)
            loss, _ = curvature_margin_loss(K_tr, y_train, g_tr, margin=0.40, lambda_vol=0.02, lambda_iso=0.005, k_clip=10.0, n_classes=N_CLASSES)
            losses.append(loss.item())
            accs.append(exact_test_accuracy(mdl, "curvature"))
            loss.backward()

            if mode == "sgd_fd":
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 5.0)
                sgd_opt.step()
            elif mode == "woodbury_fd":
                gf = torch.cat([p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel(), dtype=X_train.dtype) for p in mdl.parameters()])
                idx_s = torch.randperm(X_train.shape[0], generator=rng_sub)[:16]
                nat_d, r_norm = woodbury_coupled_ngd_direction(mdl, X_train[idx_s], y_train[idx_s], gf, task="curvature", margin=0.40, damping=0.03, n_classes=N_CLASSES)
                sc = min(1.0, 0.85 / r_norm)
                with torch.no_grad():
                    off = 0
                    for p in mdl.parameters():
                        n = p.numel()
                        p -= (0.26 * sc) * nat_d[off : off + n].reshape(p.shape)
                        off += n
            else:
                gf = torch.cat([p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel(), dtype=X_train.dtype) for p in mdl.parameters()])
                G_d = compute_coupled_geometric_metric(mdl, X_train, y_train, task="curvature", margin=0.40, damping=0.025, n_classes=N_CLASSES)
                nat_d = torch.linalg.solve(G_d, gf)
                r_norm = torch.sqrt(torch.clamp(nat_d @ (G_d @ nat_d), min=1e-12)).item()
                sc = min(1.0, 0.85 / r_norm)
                with torch.no_grad():
                    off = 0
                    for p in mdl.parameters():
                        n = p.numel()
                        p -= (0.28 * sc) * nat_d[off : off + n].reshape(p.shape)
                        off += n

        ms_per_step = (time.perf_counter() - t0) * 1000.0 / n_steps
        accs.append(exact_test_accuracy(mdl, "curvature"))
        with torch.no_grad():
            g_f, K_f = compute_fast_graph_geometry(mdl, X_train)
            losses.append(curvature_margin_loss(K_f, y_train, g_f, margin=0.40, lambda_vol=0.02, lambda_iso=0.005, k_clip=10.0, n_classes=N_CLASSES)[0].item())

        results_ngd[label] = {"loss": losses, "acc": accs, "ms": ms_per_step, "mb": mem_mb, "col": col, "ls": ls}
        print(f"{label:<40} | {ms_per_step:7.2f} ms/step | {mem_mb:7.2f} MB/step | Final Loss: {losses[-1]:.4f} | Exact Test Acc: {accs[-1]:6.2f}%")

    return results_ngd

ngd_comp = compare_dense_vs_woodbury_coupled_ngd(n_steps=35)

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.2))
for label, d in ngd_comp.items():
    axes[0].plot(d["loss"], color=d["col"], ls=d["ls"], lw=2.2, label=f"{label} ({d['ms']:.1f} ms, {d['mb']:.1f} MB)")
    axes[1].plot(d["acc"], color=d["col"], ls=d["ls"], lw=2.2, label=label)

axes[0].set_yscale("log")
axes[0].set_xlabel("Optimization Step")
axes[0].set_ylabel("Gaussian Curvature Margin Loss")
axes[0].set_title("Curvature Loss: Matrix-Free Woodbury vs. Dense Coupled NGD")
axes[0].legend(fontsize=8.0)
axes[0].grid(alpha=0.3, which="both")

axes[1].set_xlabel("Optimization Step")
axes[1].set_ylabel("Exact Analytical Test Accuracy (%)")
axes[1].set_title("Held-Out Curvature Test Accuracy Trajectory")
axes[1].set_ylim(min(45.0, 100.0 / N_CLASSES - 10.0), 102.5)
axes[1].legend(fontsize=8.3)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
""",
        },
    ]
    build_and_execute_notebook(ROOT / "notebooks" / "07-evaluating-efficient-training.ipynb", cells)


if __name__ == "__main__":
    build_notebook_01()
    build_notebook_02()
    build_notebook_03()
    build_notebook_04()
    build_notebook_05()
    build_notebook_06()
    build_notebook_07()
