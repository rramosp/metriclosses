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

### Setup & Dataset Initialization
The setup cell below loads `config.yaml`, promotes the output width via `CE_LAYER_DIMS = [*LAYER_DIMS[:-1], max(LAYER_DIMS[-1], N_CLASSES)]` when `N_CLASSES` exceeds `LAYER_DIMS[-1]`, generates the standardized synthetic train/test split, and constructs a uniform $32 \times 32$ evaluation grid (`grid_pts`) spanning the ambient 2D input domain.
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
    if (p / "config.yaml").exists():
        PROJECT_ROOT = p
        if str(p / "src") not in sys.path:
            sys.path.insert(0, str(p / "src"))
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

from diffgeom.geometry import (
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

### What This Experiment Does
We instantiate `MoonsMLP` with `CE_LAYER_DIMS`, train it for `250` epochs using `Adam` (`lr = 0.015`) and standard `CrossEntropyLoss`, and save deep-copied parameter snapshots at epochs $t \in \{0, 25, 80, 250\}$ so downstream sections can analyze both the converged representation manifold and its evolution from random initialization.

### How to Interpret the Printed Diagnostics & Plots
- **Printed Summary**: Reports exact layer widths, total trainable parameter count `P`, final Cross-Entropy loss, and train/held-out test accuracy at Epoch 250.
- **Left Plot (`Baseline Training Convergence`)**: Shows Cross-Entropy loss decaying toward zero (blue) alongside held-out test accuracy climbing toward $1.0$ (green), verifying clean coordinate-based separation.
- **Right Plot (`Learned Decision Regions`)**: Evaluates softmax probabilities across `grid_pts` (`P(y=1 | x)` for binary classification, or predicted class regions for $C > 2$). The **solid black contour** marks the learned decision boundary (`P = 0.5`), overlaid with held-out `X_test` samples.
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

### What This Experiment Does
At each layer $\ell \in \{1, \dots, L\}$ of the trained MLP, we compute the exact $2 \times 2$ Euclidean pullback metric $g^{(\ell)}(x) = (J^{(\ell)}(x))^T J^{(\ell)}(x) + \varepsilon I_2$ across the 2D grid via `torch.func.jacrev`, and extract:
1. **Log Riemannian Volume Element**: $\log_{10}\sqrt{\det g^{(\ell)}(x)}$, measuring local 2D area expansion ($> 0$) vs. contraction ($< 0$).
2. **Log Spectral Condition Number**: $\log_{10}(\lambda_1^{(\ell)}(x) / \lambda_2^{(\ell)}(x))$ where $\lambda_1 \ge \lambda_2 > 0$ are the eigenvalues of $g^{(\ell)}(x)$, quantifying directional stretching relative to orthogonal compression.

### How to Interpret the $2 \times L$ Contour Grid & Summary Table
- **Top Row (`magma` heatmaps, Volume Element)**: Bright yellow/orange ridges along the **dashed cyan decision boundary** indicate intense local area expansion where the network separates adjacent classes, whereas dark purple/black regions inside class basins indicate strong area contraction ($\det g^{(\ell)} \ll 1$), confirming **Zavatone-Veth et al. (2023)**.
- **Bottom Row (`viridis` heatmaps, Metric Anisotropy)**: Bright yellow bands along the **dashed white decision boundary** show where $\lambda_1 \gg \lambda_2$—especially in later dimension-contracting layers ($d_{\ell-1} \to d_\ell$), where the representation collapses directions parallel to class manifolds while stretching the normal direction across the boundary.
- **Summary Table (`Layer`, `Width`, `Median Vol`, `Max Vol`, `Median Cond (l1/l2)`)**:
  - Compare `Median Vol` (typical interior point) against `Max Vol` (peak decision-boundary magnification) across depth.
  - Watch `Median Cond (l1/l2)` grow from modest values in early expanding layers (`l = 1`) to orders of magnitude in the output projection (`l = L`), reflecting progressive rank-1 foliation collapse.
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

### What This Experiment Does
Scalar invariants ($\det g$ and $\lambda_1/\lambda_2$) hide the **principal eigenvector directions** of the metric tensor. Here we evaluate $g^{(\ell)}(x)$ on an $11 \times 11$ coarse grid at three representative depths (first layer `1`, middle layer, and final layer `L`), diagonalize $g^{(\ell)}(x) = \sum_{i=1}^2 \lambda_i v_i v_i^T$, and render **Tissot indicatrix ellipses** with semi-axes proportional to $\sqrt{\lambda_1}, \sqrt{\lambda_2}$ oriented along $v_1, v_2$.

### How to Interpret the Ellipse Maps
- **Shape**: Nearly circular disks indicate locally isotropic/conformal representations ($\lambda_1 \approx \lambda_2$); elongated needles indicate strong rank-1 directional stretching ($\lambda_1 \gg \lambda_2$).
- **Size**: Larger ellipses indicate stronger overall Jacobian sensitivity at $x$; tiny dots inside confident class interiors show local contraction where perturbations in $x$ barely move the layer representation.
- **Orientation relative to the Dashed Black Boundary**: By Layer `L`, ellipses along the decision boundary collapse into thin needles oriented **orthogonal to the boundary**—visually confirming **Benfenati & Marta (2023)**: the network crushes tangent vectors parallel to constant-prediction equivalence leaves while amplifying perturbations that cross from one class into another.
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

### What This Experiment Does
We reload the saved parameter checkpoints from Section 1 at epochs $t \in \{0, 25, 80, 250\}$ and evaluate the penultimate-layer log-volume field $\log_{10}\sqrt{\det g^{(L-1)}(x)}$ across `grid_pts` to observe *when* and *how* geometric structure emerges during optimization.

### How to Interpret the Four-Epoch Progression
- **`Epoch 0` (Random Initialization)**: Exhibits a smooth, broad, nearly uniform volume landscape determined solely by initial weight variance and activation slope, with zero alignment to the data geometry.
- **`Epoch 25 -> Epoch 80` (Feature Formation)**: Translational/rotational homogeneity breaks rapidly as bright yellow high-volume ridges (`plasma` colormap) nucleate along the separating interface between classes.
- **`Epoch 250` (Convergence)**: High-volume expansion sharpens into a localized tube tracking the nonlinear decision boundary, flanked by dark contractive basins over class interiors.
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

### Setup & Closed-Form Analytical Curvature Validation
Before evaluating neural manifolds, the setup cell below loads `config.yaml` and validates our exact `torch.func` Christoffel and Riemann curvature operator (`_christoffel_and_curvature_from_g_single`) against two textbook surfaces with known analytical Gaussian curvature:
- **2-Sphere of radius $R = 2$ in stereographic coordinates**: Exact constant curvature $K = 1 / R^2 = 0.250000$.
- **Hyperbolic paraboloid Monge patch $z(x^1, x^2) = (x^1)^2 - \frac{3}{4}(x^2)^2$ at the origin**: Exact saddle curvature $K(0) = \det \mathrm{Hess}(z)(0) = (2)(-1.5) = -3.000000$.
Matching both values to 6+ decimal places confirms exact higher-order autodiff behavior.
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
    if (p / "config.yaml").exists():
        PROJECT_ROOT = p
        if str(p / "src") not in sys.path:
            sys.path.insert(0, str(p / "src"))
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

from diffgeom.geometry import (
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

### What This Experiment Does
We train `MoonsMLP` for `200` `Adam` epochs with Cross-Entropy (saving weight checkpoints at $t \in \{0, 20, 70, 200\}$) and evaluate `compute_layer_geometry` on 50 held-out test points under **four Riemannian metric definitions**:
1. **Layer `L` Euclidean Pullback (`metric_mode="euclidean"`)**: $g^{(L)} = (J^{(L)})^T J^{(L)} + \varepsilon I_2$.
2. **Layer `L` Graph Immersion (`metric_mode="graph"`)**: $\tilde{g}^{(L)} = I_2 + (J^{(L)})^T J^{(L)}$, viewing $(x, z(x))$ as an immersed 2D surface in $\mathbb{R}^{2 + d_L}$.
3. **Layer `L` Conformal Gaussian Metric (`metric_mode="conformal_gaussian"`)**: Multiplies $(J^{(L)})^T J^{(L)}$ by $e^{-\|z(x)\|^2/\sigma^2}$, endowing output space itself with non-flat ambient curvature.
4. **Penultimate Layer `L-1` Euclidean Immersion**: Pullback of $I_{d_{L-1}}$ from the higher-dimensional hidden space $\mathbb{R}^{d_{L-1}}$.

### How to Interpret the Printed Comparison
- **Line 1 (`Layer L Euclidean Pullback`)**: Whenever $d_L = 2$, `max |K|` is zero up to floating-point/regularization noise ($\sim 10^{-11}$–$10^{-9}$) away from singular folds—confirming Gauss's *Theorema Egregium* (reparameterizing a flat 2D plane cannot create intrinsic curvature).
- **Lines 2–4 (`Graph`, `Conformal`, and `Penultimate` metrics)**: Exhibit O(1) to O(100) `mean |K|` and wide negative-to-positive `[min, max]` ranges. This explains why **Graph Immersion (`I_2 + J^T J`)** is used throughout Notebooks 04–07 whenever intrinsic output curvature $K^{(L)}(x)$ is evaluated or optimized on $d_L \le 2$ layers.
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

### What This Experiment Does
We evaluate the exact intrinsic Gaussian curvature $K^{(\ell)}(x) = R^{(\ell)}_{1212}(x) / \det g^{(\ell)}(x)$ across the 2D ambient grid for every layer $\ell \in \{1, \dots, L\}$ (using `euclidean` immersion when $d_\ell > 2$ and `graph` immersion when $d_\ell \le 2$). Because curvature spikes span several orders of magnitude near sharp folds, we plot $\mathrm{arcsinh}(K^{(\ell)}(x))$, which behaves linearly ($\approx K$) near $0$ and logarithmically ($\approx \pm \ln(2|K|)$) in the tails while preserving exact sign.

### How to Interpret the $1 \times L$ Curvature Maps
- **Diverging `RdBu_r` Colors**:
  - **Blue ($K^{(\ell)} < 0$)**: Hyperbolic saddle geometry (principal curvatures of opposite sign).
  - **White ($K^{(\ell)} \approx 0$)**: Locally flat / developable sheet (at least one vanishing principal curvature).
  - **Red ($K^{(\ell)} > 0$)**: Elliptic dome or bowl geometry (principal curvatures of the same sign).
- **Spatial Progression Across Depth**: Early wide layers show gentle, distributed positive and negative curvature bands, whereas deeper layers concentrate sharp alternating positive/negative curvature ridges directly along the **dashed black decision boundary** while flattening confident class interiors toward white ($K \approx 0$).
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

### What This Experiment Does
We partition `X_train` into **uncertain decision-boundary samples** (`max_c P(y=c|x) < 1/C + 0.20`) and **confident class-interior samples** (the complement), then reload checkpoints $t \in \{0, 20, 70, 200\}$ to track two layerwise profiles across training:
1. **Mean Curvature Magnitude**: $\mathbb{E}[\mathrm{arcsinh}(|K^{(\ell)}(x)|)]$ separately on interior vs. boundary points.
2. **Mean Levi-Civita Connection Strength**: $\mathbb{E}[\|\Gamma^{(\ell)k}_{ij}(x)\|_F]$, measuring coordinate-chart distortion and geodesic acceleration on $\mathcal{M}_\ell$.

### How to Interpret the Two-Panel Evolution Plot
- **Left Plot (`Curvature Evolution: Class Interior vs. Decision Boundary`)**: Compare solid lines (`Class Interior`) against dashed lines (`Boundary`) of the same color (epoch). At `Epoch 0` (gray), interior and boundary curvature are small and indistinguishable. By `Epoch 200` (green), boundary curvature (dashed) rises substantially above interior curvature (solid), showing that Cross-Entropy selectively bends the manifold along separating interfaces while smoothing class interiors (consistent with discrete Ricci-like interior smoothing).
- **Right Plot (`Levi-Civita Connection Strength Across Layers`)**: Tracks $\|\Gamma^{(\ell)}\|_F$ from `Epoch 0` to `Epoch 200`. Rising connection norms in deeper layers quantify increasing nonlinear coordinate stretching as the network warps curved input classes into linearly separable output regions.
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

### What This Experiment Does
Alongside curvature $K^{(\ell)}(x)$, the pullback metric $g^{(\ell)}_{ij}(x; \theta)$ defines the **local Riemannian log-volume density**:
$$\log \rho_{\text{vol}}^{(\ell)}(x; \theta) = \log\sqrt{\det g^{(\ell)}(x; \theta)} = \frac{1}{2}\log \det g^{(\ell)}(x; \theta)$$
We map $\log\sqrt{\det g^{(\ell)}(x)}$ across `grid_pts` for every layer $\ell \in \{1, \dots, L\}$ and track its mean trajectory across training epochs $t \in \{0, 20, 70, 200\}$, comparing `Interior vs. Boundary` points as well as `Class C-1 vs. Class 0`.

### How to Interpret the Table & Two Figures
- **Summary Table (`Min log Vol`, `Mean log Vol`, `Max log Vol`, `Boundary Mean`)**: Negative entries indicate net local area contraction ($\det g < 1$); positive entries indicate area expansion ($\det g > 1$). Notice `Boundary Mean` consistently exceeds overall `Mean log Vol`, especially in deeper layers.
- **Top $1 \times L$ Contour Row**: Blue ($< 0$) marks locally contractive basins, red ($> 0$) marks locally expansive ridges, the **dotted gold contour** marks the equiareal locus ($\det g^{(\ell)}(x) = 1$), and the **dashed black contour** marks the classifier boundary.
- **Bottom Two-Panel Plot**:
  - **Left (`Interior vs. Boundary`)**: Shows progressive divergence between boundary magnification (dashed squares) and interior compression (solid circles) over training epochs.
  - **Right (`Class C-1 vs. Class 0`)**: Under standard Cross-Entropy training, solid (`Class C-1`) and dashed (`Class 0`) curves remain nearly superimposed across all layers—showing that Cross-Entropy treats classes symmetrically in volume and motivating Notebook 05's contrastive Volume-Element loss when class-specific volume separation is desired.
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

### What This Experiment Does
To capture how each layer stretches space directionally beyond isotropic area change, we evaluate two complementary anisotropy diagnostics across layers $\ell \in \{1, \dots, L\}$ and epochs $t \in \{0, 20, 70, 200\}$:
1. **Coordinate Directional Polarization**: $\log(g^{(\ell)}_{11}(x) / g^{(\ell)}_{22}(x))$ (under the graph metric), measuring horizontal ($x^1$) vs. vertical ($x^2$) stretching.
2. **Rotation-Invariant Spectral Anisotropy**: $\log_{10}(\lambda_1^{(\ell)}(x) / \lambda_2^{(\ell)}(x))$ (under the Euclidean pullback metric), measuring the ratio of maximum-to-minimum directional stretching regardless of orientation.

### How to Interpret the Table & Two Figures
- **Summary Table (`Mean log(g11/g22)`, `Mean log10(l1/l2)`, `Boundary log10(l1/l2)`)**: Quantifies overall axis bias vs. principal eigenvalue conditioning across depth, with `Boundary log10(l1/l2)` peaking sharply at the output projection.
- **Top $2 \times L$ Contour Grid**:
  - **Row 0 (`PuOr_r` diverging)**: Positive orange/brown regions stretch $x^1$ more than $x^2$ (vertical boundary segments), whereas negative purple regions stretch $x^2$ more than $x^1$ (horizontal boundary segments).
  - **Row 1 (`viridis`)**: Rotation-invariant spectral condition number $\log_{10}(\lambda_1/\lambda_2)$, forming a continuous high-anisotropy ridge along the entire **dashed white decision boundary**.
- **Bottom Two-Panel Evolution Plot**: Left panel shows spectral anisotropy escalating with depth and epoch (highest along the boundary); Right panel shows that Cross-Entropy induces very small net class gap $\Delta \log(g_{11}/g_{22})$ between classes, contrasting with the dedicated Metric Anisotropy loss studied in Notebook 05.
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

### Setup & Reference Trajectory Capture
The setup cell below loads `config.yaml`, initializes the `MoonsMLP` (`P_total` trainable parameters), and trains it for `180` `Adam` epochs with Cross-Entropy while recording parameter snapshots at epochs $t \in \{0, 15, 60, 180\}$ so we can track the coupled information geometry of $\Theta$ and $\mathcal{M}_0$ across training.
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
    if (p / "config.yaml").exists():
        PROJECT_ROOT = p
        if str(p / "src") not in sys.path:
            sys.path.insert(0, str(p / "src"))
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

from diffgeom.geometry import (
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

### What This Experiment Does
At each saved training checkpoint $t \in \{0, 15, 60, 180\}$, we compute:
1. **The $P \times P$ Parameter Fisher Information Matrix (`compute_parameter_fim`)**: Diagonalized into sorted eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_P$, along with its trace $\mathrm{Tr}(F(\theta)) = \sum_{k=1}^P \lambda_k$ and Shannon-entropy **Effective Rank** $r_{\text{eff}} = \exp\!\big(-\sum_k \hat{\lambda}_k \log \hat{\lambda}_k\big)$ where $\hat{\lambda}_k = \lambda_k / \mathrm{Tr}(F)$.
2. **The $2 \times 2$ Data Information Matrix (`compute_data_information_matrix`)**: Evaluated across `X_train` to track average input Fisher sensitivity $\mathbb{E}_x[\mathrm{Tr}(\mathrm{DIM}(x; \theta))]$.

### How to Interpret the Plots & Printed Diagnostics
- **Left Plot (`Parameter Manifold Fisher Spectrum`, log–log)**: Reveals a classic **sloppy-model hierarchy**—a small number of top eigenvalues carry nearly all Fisher curvature, while the remaining spectrum decays by orders of magnitude into flat/redundant weight directions.
- **Right Plot & Printed Table (`Tr(FIM)`, `E[Tr(DIM)]`, `FIM Effective Rank / P`)**:
  - Compare `Parameter Tr(FIM)` (blue circles) and `Data E[Tr(DIM)]` (red squares) across epochs to observe dual sensitivity growth/consolidation between $\Theta$ and $\mathcal{M}_0$.
  - `FIM Effective Rank` (green triangles, right axis) stays far smaller than total parameter dimension $P$, showing that classification geometry is controlled by a compact low-dimensional stiff subspace on $\Theta$ (**Tron et al., 2024**).
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

### What This Experiment Does
We evaluate the $2 \times 2$ Data Information Matrix $\mathrm{DIM}_{ij}(x; \theta)$ across the 2D ambient grid (`grid_pts`), diagonalize $\mathrm{DIM}(x)$ at every grid point, plot $\log_{10}\mathrm{Tr}(\mathrm{DIM}(x))$ as a background heatmap, and overlay a `quiver` field of its **smallest-eigenvalue (kernel) eigenvector** $v_{\ker}(x)$.

### How to Interpret the Heatmap & Quiver Field
- **Background Heatmap (`inferno`, $\log_{10}\mathrm{Tr}(\mathrm{DIM}(x))$ )**: Bright yellow/orange bands pinpoint regions of high input Fisher information—strictly localized along the decision boundary where class probabilities transition sharply. Dark purple/black basins inside confident class interiors have $\mathrm{DIM}(x) \approx 0$ (local perturbations of $x$ barely alter $p_\theta(y \mid x)$).
- **Cyan Quiver Vectors (`DIM Kernel / Foliation Tangent`)**: Because $\mathrm{DIM}(x)$ has rank at most $\min(d_0, C - 1)$ (rank `1` in binary 2D classification!), it possesses an exact null eigenvector $v_{\ker}(x)$ satisfying $D_x p_\theta(y \mid x)\, v_{\ker}(x) = 0$. Along the bright boundary ridge, the cyan arrows align **parallel to the decision boundary**, tracing out the 1D equivalence leaves of constant class probability (**Benfenati & Marta, 2023**; **Tron et al., 2024**).
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

### What This Experiment Does
To directly test **Halverson & Ruehle (2023)**'s *Metric Flow* framework—how movements on the parameter manifold $\Theta$ deform the Riemannian geometry of the data representation manifold $\mathcal{M}_L$—we diagonalize $F(\theta)$ on the trained network, select logarithmically spaced eigenvector modes $k \in \{1, \dots, P\}$ from stiffest ($k = 1$) to flattest ($k = P$), perturb weights by equal Euclidean step length $\theta \mapsto \theta + \delta v^{(k)}$ ($\delta = 0.08$, $\|v^{(k)}\|_2 = 1$), and measure two induced deformations on `X_test`:
1. **Data Metric Shift**: $\mathbb{E}_{x \in X_{\text{test}}}\!\left[\|g^{(L)}(x; \theta + \delta v^{(k)}) - g^{(L)}(x; \theta)\|_F\right]$.
2. **Data Curvature Shift**: $\mathbb{E}_{x \in X_{\text{test}}}\!\left[|K^{(L)}(x; \theta + \delta v^{(k)}) - K^{(L)}(x; \theta)|\right]$.

### How to Interpret the Log–Log Sensitivity Plot
- Moving left-to-right from `k = 1` (stiffest Fisher mode) to `k = P` (flattest/null Fisher mode), both **Data Metric Shift** (blue circles) and **Data Curvature Shift** (red squares) fall by orders of magnitude—even though every perturbation vector has the **exact same Euclidean norm** $\delta = 0.08$ in $\mathbb{R}^P$!
- This directly demonstrates parameter–data geometric duality: stiff directions of $F(\theta)$ drive macroscopic metric and curvature flows on $\mathcal{M}_L$, whereas null directions of $F(\theta)$ correspond to approximate symmetries/reparameterizations that leave data-manifold geometry nearly invariant.
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

### What This Experiment Does
We train two identically initialized networks (`seed = 11`) for `45` full-batch Cross-Entropy steps:
1. **Euclidean Gradient Descent**: $\Delta \theta = -\eta_{\text{GD}} \nabla_\theta \mathcal{L}$ (`lr = 0.15`), treating weight space as flat $\mathbb{R}^P$.
2. **Riemannian Natural Gradient Descent**: $\Delta \theta = -\eta_{\text{NGD}} (F(\theta) + \lambda I)^{-1} \nabla_\theta \mathcal{L}$ (`lr = 0.08`, `damping = 0.02`), following steepest descent with respect to the Fisher metric on $\Theta$.
Every 3 steps, we also evaluate the mean output Gaussian curvature magnitude $\mathbb{E}[|K^{(L)}(x)|]$ on held-out test samples.

### How to Interpret the Two-Panel Trajectory Plot
- **Left Plot (`Optimization Trajectory on Parameter Manifold`)**: Natural Gradient (green) achieves much steeper per-step Cross-Entropy loss reduction than Euclidean GD (blue) by amplifying progress along informative low-curvature directions while damping stiff directions.
- **Right Plot (`Induced Curvature Flow on Data Manifold`)**: Shows the corresponding geometric evolution on $\mathcal{M}_L$—Natural Gradient rapidly and smoothly sculpts the output manifold's curvature profile as class separation locks in, whereas Euclidean GD lags behind.
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
- **Purely Geometric Inference Rule**: At test time, any point $x^*$ is classified strictly by local manifold curvature ($\hat{y}(x^*) = \mathbb{I}[K(x^*; \theta) > 0]$ for $C=2$, or ordered curvature bands for $C > 2$).

### Setup & Dataset Loading
The setup cell below loads `config.yaml` using the exact `LAYER_DIMS` (no extra output logit channels are needed even when `N_CLASSES > 2`, because classification uses the scalar intrinsic curvature field $K^{(L)}(x)$) and builds the 2D evaluation grid.
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
    if (p / "config.yaml").exists():
        PROJECT_ROOT = p
        if str(p / "src") not in sys.path:
            sys.path.insert(0, str(p / "src"))
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

from diffgeom.geometry import (
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

### What This Experiment Does
We train `geom_model` from scratch for `90` `Adam` epochs (`lr = 0.025`, gradient clipping `5.0`) using **exclusively** `curvature_margin_loss` (`margin = 0.40`, `lambda_vol = 0.02`, `lambda_iso = 0.005`, `k_clip = 10.0`) computed from the exact final-layer Graph immersion metric $\tilde{g}^{(L)} = I_2 + J^T J$. Raw output coordinates $z(x)$ never appear in any target or classifier; both train and test accuracy are evaluated strictly from the sign/band of $K^{(L)}(x)$ via `scalar_geometric_predict`.

### How to Interpret the Printed Checkpoints & Plots
- **Printed Checkpoints (`Epoch 1, 20, 45, 90`)**: Track elapsed seconds, total geometric loss, purely curvature-based `Train Acc` and `Test Acc`, and class-conditional mean curvatures `E[K|y=0]` and `E[K|y=1]`.
- **Left Plot (`Purely Geometric Curvature Training Convergence`)**: Purple curve shows `curvature_margin_loss` decreasing smoothly while purely geometric `Train Acc` (blue dashed) and held-out `Test Acc` (green solid) climb from random chance toward high classification accuracy.
- **Right Plot (`Bifurcation of Manifold Curvature by Class`)**: Starting near $K \approx 0$ at initialization, `Class 0` mean curvature (blue) is driven below the negative target margin $-\tau$ (hyperbolic saddle regime) while `Class 1` mean curvature (red) is driven above $+\tau$ (elliptic dome regime).
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

### What This Experiment Does
We evaluate the trained network's exact intrinsic Gaussian curvature $K^{(L)}(x)$ across the 2D ambient grid (`grid_pts`) and across all held-out test points `X_test`, classifying test samples purely via `scalar_geometric_predict(K_test)`.

### How to Interpret the Spatial Contour Map & Test Histogram
- **Left Plot (`Learned Curvature Field & Decision Boundary`)**: Diverging `RdBu_r` contour map of $\mathrm{arcsinh}(K^{(L)}(x))$. **Blue regions ($K < 0$)** are hyperbolic saddle basins assigned to Class `0`; **red regions ($K > 0$)** are elliptic dome regions assigned to Class `1`; the **solid black contour** marks the zero-curvature parabolic inflection curve $K^{(L)}(x) = 0$ (or multi-class band thresholds $b_c$) acting as the intrinsic decision boundary.
- **Right Plot (`Held-Out Test Set Curvature Distribution`)**: Class-colored histograms of raw $K^{(L)}(x)$ values on unseen `X_test` points relative to the **dashed black decision threshold**. Clean separation on opposite sides of $K = 0$ confirms generalization of the learned curvature geometry to held-out data.
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

### What This Experiment Does
To see physically *how* the neural network sculpts positive vs. negative Gaussian curvature without coordinate supervision, we plot the 3D Monge graph $(x^1, x^2, z^1(x)) \subset \mathbb{R}^3$ over the 2D input plane and color every surface element by its exact intrinsic Gaussian curvature $\mathrm{arcsinh}(K^{(L)}(x))$, overlaying held-out test points slightly above the surface.

### How to Interpret the 3D Surface Geometry
- **Blue Saddle Regions ($K = \kappa_1 \kappa_2 < 0$, Class `0`)**: The surface curves upward along one principal direction and downward along the orthogonal direction (hyperbolic saddle geometry).
- **Red Dome/Bowl Regions ($K = \kappa_1 \kappa_2 > 0$, Class `1`)**: Both principal curvatures bend in the same direction (elliptic bowl or dome geometry).
- **White Inflection Crease ($K = 0$)**: Marks the parabolic transition locus where one principal curvature changes sign—forming the purely differential-geometric decision boundary between classes.
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

### Setup & Shared Dataset Initialization
The setup cell below loads `config.yaml` and builds the shared train/test split and evaluation grid (`grid_pts`) so all four training paradigms start from identical data and random seeds (`seed = 7`).
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
    if (p / "config.yaml").exists():
        PROJECT_ROOT = p
        if str(p / "src") not in sys.path:
            sys.path.insert(0, str(p / "src"))
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

from diffgeom.geometry import (
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

### What This Experiment Does
We train four identically initialized MLPs (`seed = 7`) using `Adam`:
1. **`m_ce` (`Cross-Entropy Baseline`, 120 epochs)**: Standard coordinate logit supervision (`argmax_c z_c(x)`).
2. **`m_curv` (`Gaussian Curvature Margin Loss`, 85 epochs)**: 3rd-order autodiff separating classes by intrinsic graph curvature $K^{(L)}(x)$ (`margin = 0.40`).
3. **`m_vol` (`Volume-Element Contrastive Loss`, 100 epochs)**: 2nd-order autodiff separating classes by log area density $\log\sqrt{\det g^{(L)}(x)}$ (`margin = 0.60`).
4. **`m_aniso` (`Metric Anisotropy Loss`, 100 epochs)**: 2nd-order autodiff separating classes by directional stretching $\log(g^{(L)}_{11}(x)/g^{(L)}_{22}(x))$ (`margin = 0.45`).

### How to Interpret the Summary Table
- **`Decision Criterion`**: Highlights the fundamental conceptual divide—Row 1 inspects raw output coordinates $z_c(x)$, whereas Rows 2–4 classify unseen test points strictly via **coordinate-free differential invariants** (`scalar_geometric_predict`).
- **`Test Accuracy`**: Shows that first-fundamental-form metric invariants (`Volume-Element` and `Metric Anisotropy`, which need only 2nd-order derivatives) match or surpass 3rd-order `Gaussian Curvature` and rival standard `Cross-Entropy`.
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

### What This Experiment Does
We evaluate each of the four trained models across the 2D ambient grid (`grid_pts`) using **its own native classification scalar**: logit gap $z_1(x) - z_0(x)$ for Cross-Entropy, $\mathrm{arcsinh}(K^{(L)}(x))$ for Curvature, $\log\sqrt{\det g^{(L)}(x)}$ for Volume-Element, and $\log(g^{(L)}_{11}(x)/g^{(L)}_{22}(x))$ for Metric Anisotropy.

### How to Interpret the $1 \times 4$ Landscape Comparison
- **Diverging `RdBu_r` Scale (`Blue < 0`, `White = 0`, `Red > 0`)**: In all four panels, blue regions predict Class `0`, red regions predict Class `1` (for binary classification), and the **solid black `0.0` level curve** marks the exact decision boundary:
  - Panel 1: Equal-logit hyperplane pullback ($z_1 = z_0$).
  - Panel 2: Zero-curvature parabolic inflection curve ($K = 0$).
  - Panel 3: Equiareal volume-preserving curve ($\det g = 1$).
  - Panel 4: Conformal/isotropic directional balance curve ($g_{11} = g_{22}$).
- Despite optimizing four distinct mathematical quantities, all four objectives recover consistent nonlinear decision boundaries separating the test data.
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

### What This Experiment Does
We compare how the four training objectives distribute curvature across network depth $\ell \in \{1, \dots, L\}$ and how they reshape the eigenspectrum of the $P \times P$ Generalized Gauss-Newton / Output Pullback Metric on the parameter manifold $\Theta$:
$$G^\Theta_{\mu\nu}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_{a=1}^{d_L} \frac{\partial z^a(x; \theta)}{\partial \theta^\mu} \frac{\partial z^a(x; \theta)}{\partial \theta^\nu} \right]$$

### How to Interpret the Two-Panel Plot
- **Left Plot (`Layerwise Class Curvature Separation`)**: Plots $\Delta \mathrm{arcsinh}(K^{(\ell)}) = \mathbb{E}[\mathrm{arcsinh}(K^{(\ell)}) \mid y = C-1] - \mathbb{E}[\mathrm{arcsinh}(K^{(\ell)}) \mid y = 0]$ across layers. Notice that **only** `Curvature Loss` (red) produces a large positive jump at the final layer `L`, whereas `Cross-Entropy`, `Volume`, and `Anisotropy` stay near zero—demonstrating that curvature separation does not arise by accident under lower-order objectives.
- **Right Plot (`Induced Parameter Manifold Metric Spectrum`, log–log)**: Reveals the **back-reaction of data-space losses on $\Theta$**. Compare the eigenvalue decay rate and top stiffness of the three geometric models against `Cross-Entropy` (blue): geometric metric regularizers prevent extreme rank collapse and redistribute sensitivity across more active parameter directions.
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

### What This Experiment Does
We evaluate the **local Riemannian log-volume density** across all layers $\ell \in \{1, \dots, L\}$ on `X_test` for all four trained models:
$$\log \rho_{\text{vol}}^{(\ell)}(x; \theta) = \log\sqrt{\det g^{(\ell)}(x; \theta)} = \frac{1}{2}\log \det g^{(\ell)}(x; \theta)$$
tracking both per-class means (`Class C-1` vs. `Class 0`) and the signed class separation gap $\Delta \log\sqrt{\det g^{(\ell)}}$.

### How to Interpret the Table & Two-Panel Figure
- **Printed Gap Table (`L1 Gap ... LL Gap`)**: Reports the signed difference in mean log-volume between `Class C-1` and `Class 0` at every layer. Look at the final column (`LL Gap`): `Volume-Element Loss` achieves a large positive gap ($> +1$), whereas all other methods sit near `0`.
- **Left Plot (`Per-Class Layerwise Log-Volume Trajectories`)**: Solid lines (`Class C-1`) vs. dashed lines (`Class 0`). For `Volume-Element Loss` (green), the two classes split cleanly onto opposite sides of the dotted zero line ($\det g = 1$) at Layer `L`; for `Curvature` (red) and `Anisotropy` (orange), both classes stay anchored near `0` thanks to $(\log\det g)^2$ stabilization.
- **Right Plot (`Layerwise Class Log-Volume Separation`)**: Directly visualizes how volume contrast builds progressively through hidden layers and peaks sharply at the output manifold exclusively under `Volume-Element Loss`.
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

### What This Experiment Does
We evaluate how all four training objectives shape directional stretching across depth $\ell \in \{1, \dots, L\}$ on `X_test`, measuring both the **signed class polarization gap** $\Delta \log(g^{(\ell)}_{11}/g^{(\ell)}_{22})$ and the **mean spectral log-condition number** $\mathbb{E}[\log_{10}(\lambda_1^{(\ell)}/\lambda_2^{(\ell)})]$.

### How to Interpret the Table & Two-Panel Figure
- **Printed Table & Left Plot (`Layerwise Class Directional Anisotropy Separation`)**: `Metric Anisotropy Loss` (orange) uniquely drives a large positive polarization gap at Layer `L` (stretching $x^1$ on Class `C-1` and $x^2$ on Class `0`), while the other three methods remain near zero directional gap.
- **Right Plot (`Layerwise Spectral Condition Number Across Depth`)**: Highlights a striking geometric difference between coordinate and geometric training: `Cross-Entropy` (blue) exhibits severe spectral ill-conditioning ($\lambda_1/\lambda_2 \gg 10^2$) at the output layer because unconstrained softmax training collapses within-class directions, whereas all three geometric losses preserve well-conditioned, non-degenerate 2D immersions via volume regularization.
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

### What This Experiment Does & How to Interpret the Results
We profile a single `forward + backward` step for each of the four exact objectives (`benchmark_method`), recording wall-clock step latency (`ms/step`), CPU tensor memory allocation via `torch.profiler` (`MB/step`), and total backward graph nodes (`count_autograd_graph_nodes`):
- **Printed Table & Bar Charts (Left & Middle)**: Show a clean staircase determined by differential order—`1st-order` Cross-Entropy is cheapest, `2nd-order` metric losses (`Volume-Element` and `Metric Anisotropy`) add only a small constant factor in latency and memory while preserving high test accuracy, and `3rd-order` `Gaussian Curvature` jumps by an order of magnitude due to nested reverse-mode graph expansion.
- **Right Scatter Plot (`Compute vs. Memory Scaling by Differential Order`)**: Plots `Latency` vs. `Memory` on log–log axes with bubble area scaled by memory footprint and labels showing autograd node counts and held-out test accuracy—motivating the 1st-order approximations developed in Notebook 07.
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

### What This Experiment Does
To test **geometric selectivity (orthogonality of objectives)**, we evaluate **all three differential-geometric invariants (rows)**—Gaussian Curvature $\mathrm{arcsinh}(K^{(L)})$, Log Volume $\log\sqrt{\det g^{(L)}}$, and Metric Anisotropy $\log(g^{(L)}_{11}/g^{(L)}_{22})$—across **all four trained networks (columns)** (`Cross-Entropy`, `Curvature`, `Volume-Element`, `Metric Anisotropy`), computing the test accuracy each geometric invariant achieves on every model.

### How to Interpret the $3 \times 4$ Matrix
- **Bold Diagonal Target Panels (`★ Target` at `(Row 0, Col 1)`, `(Row 1, Col 2)`, `(Row 2, Col 3)`)**: Each geometric model cleanly separates Class `0` (blue, `< 0`) from Class `1` (red, `> 0`) along the solid black `0.0` contour **in the exact geometric invariant it was trained to optimize**, achieving high held-out accuracy.
- **Off-Diagonal Panels & Column 0 (`Cross-Entropy Baseline`)**: Hover near random-chance geometric accuracy (~`40%–65%`) because non-target invariants are either regularized toward `0.0` (white) or concentrated symmetrically along the boundary rather than taking opposite signs on the two classes. This confirms that curvature, volume density, and metric anisotropy act as **independent, controllable geometric channels** on $\mathcal{M}_L$.
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

### Setup & Dataset Initialization
The setup cell below loads `config.yaml` and generates the train/test split and ambient 2D grid (`grid_pts`) used across all coupled vs. uncoupled optimization experiments.
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
    if (p / "config.yaml").exists():
        PROJECT_ROOT = p
        if str(p / "src") not in sys.path:
            sys.path.insert(0, str(p / "src"))
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

from diffgeom.geometry import (
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

### What This Experiment Does
1. **Gauss-Equation Consistency Check**: Verifies that `compute_fast_graph_geometry` (algebraic Gauss identity via `jacfwd^2`) matches `compute_layer_geometry` (`jacrev^3`) to `float64` machine precision ($\sim 10^{-16}$).
2. **Gauge-Direction Norm Test**: Constructs four unit parameter vectors $v \in \mathbb{R}^P$ ($\|v\|_2 = 1$) at initialization:
   - Two pure output translations (`Trans. z^1 (bL_0)` and `Trans. z^2 (bL_1)`), shifting $b^{(L)}$.
   - One infinitesimal output rotation (`Rot. so(2) (WL,bL)`), rotating $(W^{(L)}, b^{(L)}) \mapsto (W^{(L)} + \epsilon \Omega W^{(L)},\; b^{(L)} + \epsilon \Omega b^{(L)})$.
   - One generic random hidden-layer direction (`Generic Hidden Dir.`).
   We evaluate the undamped quadratic form $|v^T G^\Theta v|$ across `Uncoupled I_P`, `Uncoupled Coord-GGN`, `Coupled DeWitt`, and `Coupled Curvature-GGN`, and plot the full $P \times P$ eigenspectra of the four parameter metrics.

### How to Interpret the Table & Two-Panel Figure
- **Printed Table & Left Bar Chart (`Exact Quotienting of E(2) Output Symmetries`, log scale)**:
  - When `OUTPUT_ACTIVATION = "identity"`, translating or rotating output coordinates $z \mapsto Qz + b$ leaves $J_x^T J_x$ and $K(x)$ **identically unchanged**—so the first three directions are pure zero-effect gauge symmetries.
  - `Uncoupled I_P` (`1.0`) and `Uncoupled Coord-GGN` ($\sim 10^{-1}$–$10^0$) falsely treat these output motions as stiff directions, whereas `Coupled DeWitt` and `Coupled Curvature-GGN` send $|v^T G^\Theta v|$ to numerical zero ($\sim 10^{-32}$–$10^{-18}$) while remaining strictly positive on `Generic Hidden Dir.`! *(Note: if `OUTPUT_ACTIVATION` is set to a nonlinear function such as `"tanh"`, output shifts/rotations alter activation saturation and cease to be exact symmetries, which the coupled metrics automatically reflect.)*
- **Right Plot (`Parameter Metric Spectra at Initialization`, log–log)**: Shows how coupled metrics concentrate spectral weight onto genuine metric- and curvature-deforming parameter modes while quotienting redundant gauge directions into the nullspace.
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

### What This Experiment Does
We train identically initialized MLPs (`seed = 7`) for `N_STEPS = 45` optimization steps across all three purely geometric objectives (`volume`, `anisotropy`, `curvature`), comparing four parameter-space optimizers:
1. **`Uncoupled GD (I_P)`**: Standard Euclidean gradient descent in $\mathbb{R}^P$.
2. **`Uncoupled Adam`**: Coordinate-wise diagonal second-moment preconditioning.
3. **`Uncoupled Coord-NGD`**: Natural Gradient using the uncoupled output-coordinate metric $G^{\Theta,\text{coord}}$ with geodesic trust-region radius $0.85$.
4. **`Coupled Geom-NGD`**: Natural Gradient using the data-geometry-coupled metric $G^{\Theta,\text{coupled}}$ with the exact same trust-region radius $0.85$.

### How to Interpret the Table & $2 \times 3$ Convergence Grid
- **Summary Table Columns**:
  - `Step 10 Acc` & `Step 25 Acc`: Early-step held-out geometric test accuracy—highlighting how fast each optimizer sculpts class-separating geometry in minimal iterations.
  - `Final Loss` & `Final Test Acc`: Solution quality after 45 steps.
  - `bL Gauge Drift`: Euclidean drift $\|b^{(L)}(45) - b^{(L)}(0)\|_2$ of the final-layer bias vector (diagnostic for spurious motion along gauge directions when `OUTPUT_ACTIVATION = "identity"`).
- **Top Row (`Geometric Loss`, log scale) & Bottom Row (`Held-Out Test Accuracy`)**: Across all three columns (`Volume-Element`, `Metric Anisotropy`, `Gaussian Curvature`), `Coupled Geom-NGD` (bold green solid) achieves the steepest loss descent and earliest plateau at high test accuracy, with its largest advantage appearing on stiff 3rd-order `Gaussian Curvature`.
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

### What This Experiment Does
We take all `12` trained models from Section 2 (`3 geometric tasks x 4 optimizers` at step 45) and evaluate each model's target geometric scalar field $\psi(x^1, x^2)$ across the ambient 2D grid (`grid_pts`).

### How to Interpret the $3 \times 4$ Spatial Grid
- **Rows**: Target geometric invariant (`Row 0`: Log Volume $\log\sqrt{\det g^{(L)}}$; `Row 1`: Metric Anisotropy $\log(\tilde{g}^{(L)}_{11}/\tilde{g}^{(L)}_{22})$; `Row 2`: Gaussian Curvature $\mathrm{arcsinh}(K^{(L)})$).
- **Columns**: Optimizer (`Col 0`: `Uncoupled GD`; `Col 1`: `Uncoupled Adam`; `Col 2`: `Uncoupled Coord-NGD`; `Col 3`: `Coupled Geom-NGD` in bold).
- **Visual Takeaway**: Compare Column 3 against Columns 0–2 along each row. `Coupled Geom-NGD` produces sharper blue-vs.-red margin saturation and cleaner black `0.0` decision boundaries in 45 steps—especially in Row 2 (`Curvature`), where `Uncoupled GD` remains stuck in a high-loss, poorly separated geometry.
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

### What This Experiment Does
We isolate the **two concrete mathematical mechanisms** responsible for `Coupled Geom-NGD`'s advantage:
1. **Gauge-Drift Tracking**: Compares the trajectory of final-bias drift $\|b^{(L)}(t) - b^{(L)}(0)\|_2$ across steps `0..45` between `Uncoupled Coord-NGD` and `Coupled Geom-NGD` for all three geometric tasks.
2. **Unclipped High-Margin Curvature Stress Test**: Runs `35` curvature-loss optimization steps at a wider margin (`margin = 0.65`, `k_clip = 25.0`) **with all heuristic gradient clipping removed**, comparing `Uncoupled GD (Unclipped, lr=0.07)` against `Coupled Geom-NGD (Unclipped, lr=0.28)`.

### How to Interpret the Three-Panel Diagnostic Plot
- **Left Plot (`Spurious Translation Gauge Drift on Theta`)**: When `OUTPUT_ACTIVATION = "identity"`, the true gradient with respect to $b^{(L)}$ is identically zero ($\nabla_{b^{(L)}}\mathcal{L}_{\mathrm{geom}} = 0$). Yet `Uncoupled Coord-NGD` (dash-dot curves) continuously drifts $b^{(L)}(t)$ because $G^{\Theta,\text{coord}}$ mixes hidden weights $W$ with output shifts $b^{(L)}$ via off-diagonal cross-covariances! `Coupled Geom-NGD` (solid flat lines at `0`) respects the $E(d_L)$ quotient exactly, eliminating wasted gauge motion.
- **Middle Plot (`Unclipped 3rd-Order Curvature Loss`) & Right Plot (`Euclidean Gradient Norm`, log scale)**:
  - Notice the violent multi-order-of-magnitude spikes in $\|\nabla_\theta \mathcal{L}_{\mathrm{curv}}\|_2$ (Right, red dashed) whenever surface folds sharpen (`K = R_{1212}/\det\tilde{g}`).
  - Without gradient clipping, `Uncoupled GD` (red dashed) overshoots across curvature cliffs and destabilizes in loss (Middle). `Coupled Geom-NGD` (green solid) preconditions directly by $(\nabla_\theta K)^T \nabla_\theta K$, automatically shrinking Euclidean step lengths whenever curvature sensitivity spikes and descending smoothly **without needing manual gradient clipping**.
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

### What This Experiment Does
$G^{\Theta,\text{coupled}}(\theta)$ combines two geometric pullbacks:
1. **The unsupervised, task-agnostic DeWitt metric $G^{\Theta,\text{DeWitt}}(\theta)$**: Pulls back the canonical affine-invariant metric $\mathrm{Tr}(g^{-1}\partial_\mu g\, g^{-1}\partial_\nu g)$ on $2 \times 2$ SPD matrices $\mathrm{Sym}^+_2$, depending **only** on $X$ and $g(x; \theta)$ (zero dependence on class labels $y$ or the chosen loss invariant $\psi$).
2. **The task-specific GGN / Sobolev term $\mathbb{E}[w_\psi \nabla_\theta \psi \otimes \nabla_\theta \psi]$**: Pulls back the sensitivity of the target geometric scalar $\psi \in \{\log\sqrt{\det g}, \log(\tilde{g}_{11}/\tilde{g}_{22}), K\}$.

Here we train `Pure DeWitt-NGD` (using $G^{\Theta,\text{DeWitt}}$ alone) for 45 steps across all three tasks and compare its loss curves and final test accuracies against `Uncoupled GD` and `Task-Coupled Geom-NGD`.

### How to Interpret the Three-Panel Ablation & Summary Table
- **Columns 1 & 2 (`Volume-Element` and `Metric Anisotropy`, 1st-fundamental-form tasks)**: `Coupled DeWitt-NGD` (blue dashed) tracks `Task-Coupled Geom-NGD` (green solid) almost identically, capturing virtually the entire speedup over `Uncoupled GD` (gray dotted)—because $\det g$ and $g_{11}/g_{22}$ are smooth algebraic functions of $g(x; \theta)$ whose stiffness is already controlled by the DeWitt metric on $\mathrm{Sym}^+_2$.
- **Column 3 (`Gaussian Curvature`, 2nd-fundamental-form task)**: `Coupled DeWitt-NGD` still strongly outperforms `Uncoupled GD`, while `Task-Coupled Geom-NGD` achieves an additional loss and accuracy improvement—because Gaussian curvature $K(x)$ depends on **spatial derivatives of the metric** (2nd-order jets $\partial_x^2 z$), requiring the extra 2nd-order Sobolev stiffness carried by $\nabla_\theta K \otimes \nabla_\theta K$.
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

1. **Approach 1 — Batched Finite-Difference Stencils (Pure 1st-Order Autodiff)**: Evaluates spatial Jacobians and Hessians via batched perturbed forward passes (`3N` points for metric losses, `7N` central stencil points for Gaussian curvature) so `loss.backward()` differentiates through standard forward passes only.
2. **Approach 2 — Precomputed Point-Cloud $k$-NN Taylor Jet Operators (`1N` Single Forward Pass)**: Fits Gaussian-weighted local linear (`order=1`) and quadratic (`order=2`) least-squares Taylor operators on the training point cloud once offline at $t = 0$, estimating spatial derivatives at every training step from a **single `1N` forward pass** $Z = f_\theta(X)$.
3. **Approach 3 — Randomized Tangent-Plane Sketching / Hutchinson Directional Probes ($O(1)$ in Input Dimension $d_0$)**: Replaces full $d_0$-coordinate stencils with $m = 1$ or $2$ random orthonormal tangent probes (`2N`–`3N` evaluations for metric invariants, `7N` along a random 2-plane for sectional/Gaussian curvature), eliminating dependence on ambient input dimension $d_0$.
4. **Approach 4 — Algebraic Order Reduction Before Autodiff (Gauss's Extrinsic-to-Intrinsic Identity)**: Rewrites intrinsic graph curvature $K = R_{1212}/\det\tilde{g}$ algebraically via Gauss's equation directly in terms of $(J, H)$ using 2nd-order forward-mode spatial AD (`jacfwd(jacfwd(phi))`) instead of differentiating the metric tensor with 3rd-order reverse-mode AD (`jacrev^3`).

---

## 1. Setup, Offline Jet Precomputation & Numerical Approximation Fidelity

### What This Experiment Does
1. **Configuration & Offline $k$-NN Precomputation**: Loads the dataset and MLP architecture from `config.yaml` and precomputes the fixed point-cloud $k$-NN Taylor operators (`knn_op_lin` with $k = 10$ for 1st-order Jacobians; `knn_op_quad` with $k = 16$ for 2nd-order jets) once before training begins.
2. **Reference Manifold Preparation**: Trains a reference MLP (`ref_mlp`) for 100 `SGD` epochs with Cross-Entropy so its output representation forms a genuinely curved, non-trivial manifold on which approximation errors can be measured meaningfully.
3. **Step-Size Sweep & Pointwise Curvature Validation**: Evaluates exact log-volume $\log\sqrt{\det g}$ and exact Gaussian curvature $K^{(L)}$ (`Levi-Civita` vs. `Gauss equation`), sweeps finite-difference step sizes $h \in [10^{-6}, 10^{-1}]$, and compares pointwise curvature estimates from the 7-point FD stencil (`Approach 1`) and the single-pass $k$-NN quadratic jet (`Approach 2`) against exact analytical curvature.

### How to Interpret the Printed Metrics & Plots
- **Printed Diagnostics**:
  - `Offline t=0 k-NN Jet Precomputation`: One-time setup latency (typically a few milliseconds) amortized over the entire training run, plus operator tensor shapes `(N, m_dim, k)`.
  - `Approach 4 (Gauss Eq.) max |K_gauss - K_levi_civita|`: Should sit near `float64` machine precision ($\sim 10^{-15}$–$10^{-14}$), confirming that Approach 4 is **mathematically exact** rather than an approximation.
  - `Approach 1 (7-pt FD, h=2e-3) mean |K_fd - K_exact|`: Typical pointwise curvature error ($\sim 10^{-5}$–$10^{-4}$), showing near-exact fidelity from pure forward evaluations.
  - `Approach 2 (1N k-NN Jet) Pearson r(K_knn, K_exact)`: Measures spatial correlation between the zero-perturbation `1N` graph-regression curvature and exact analytical curvature; high positive $r$ confirms it captures the true sign and coarse magnitude of $K(x)$ despite finite sample spacing.
- **Left Plot (`Stencil Truncation vs. Floating-Point Roundoff Scaling`)**: Classic numerical-analysis "V-curve" on log–log axes. Moving right-to-left (shrinking $h$), truncation error drops at rate $O(h)$ for the 3-point forward metric stencil (blue) and $O(h^2)$ for the symmetric 7-point central curvature stencil (red), until `float64` subtractive cancellation causes roundoff error to rise for very small $h \lesssim 10^{-5}$ (especially for 2nd-order differences dividing by $h^2$). The minimum around $h \in [10^{-4}, 3 \times 10^{-3}]$ justifies our default stencil widths.
- **Right Plot (`Pointwise Curvature Fidelity on Trained Network`)**: Scatter plot of approximated vs. exact $\mathrm{arcsinh}(K(x))$ relative to the dashed identity line $y = x$. Green circles (`7-pt FD`) lie directly on $y = x$; orange triangles (`1N k-NN Jet`) cluster tightly along the diagonal with mild smoothing from local neighborhood averaging over noisy training points.
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
    if (p / "config.yaml").exists():
        PROJECT_ROOT = p
        if str(p / "src") not in sys.path:
            sys.path.insert(0, str(p / "src"))
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

from diffgeom.geometry import (
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

### What This Experiment Does
We profile a single complete training iteration—`sgd.zero_grad() -> loss_fn() -> loss.backward() -> sgd.step()`—across **14 configurations** spanning four groups:
1. **`Baseline`**: Standard 1st-order `SGD + Cross-Entropy` (`1N` forward batch).
2. **`Volume` (`volume_contrastive_loss`)**: Exact 2nd-order autodiff vs. `App 1` (`3-pt FD`), `App 2` (`1N k-NN Linear`), and `App 3` (`2-Probe Sketch`).
3. **`Anisotropy` (`anisotropy_margin_loss`)**: Exact 2nd-order autodiff vs. `App 1` (`3-pt FD`), `App 2` (`1N k-NN Linear`), and `App 3` (`1-Probe Hutchinson`).
4. **`Curvature` (`curvature_margin_loss`)**: Exact 3rd-order Levi-Civita autodiff vs. `App 4` (`Exact Gauss Eq.`), `App 1` (`7-pt FD`), `App 2` (`1N k-NN Quad Jet`), and `App 3` (`Random 2-Plane`).

All 14 rows use identical `torch.optim.SGD` parameter updates so every difference in latency, CPU tensor memory (`torch.profiler`), and autograd DAG size (`count_autograd_nodes`) comes purely from evaluating and backpropagating the loss.

### How to Interpret the Table & Three-Panel Figure
- **Benchmark Table Columns**:
  - `Order`: Highest symbolic differentiation order traced by PyTorch autograd (`1st` for Cross-Entropy and Approaches 1–3; `2nd` for exact metric losses; `3rd` for exact curvature losses).
  - `Evals`: Effective forward batch multiplier (`1N` for Cross-Entropy and `k-NN` jets; `2N`–`7N` for stencils/sketches; nested `jacrev` / `jacfwd` transforms for exact autodiff).
  - `Latency (ms)` & `Speedup`: Wall-clock milliseconds per step and relative speedup **within that geometric loss family** against its own exact autodiff baseline (`1.0x`).
  - `Memory (MB)` & `Mem Save`: Total CPU tensor memory allocated during forward + backward and the memory reduction factor relative to that group's exact autodiff baseline.
  - `Nodes`: Number of backward `grad_fn` nodes in PyTorch's autograd graph—explaining *why* 1st-order methods shrink memory and overhead dramatically (hundreds/thousands of higher-order graph nodes collapse down toward the compact DAG of standard backprop).
- **Left & Middle Bar Charts (`Compute Cost` & `Memory Footprint`)**: Horizontal log-scale bars comparing representative methods from `SGD + CE` (black, top) down through exact and approximated Volume and Curvature losses. Shorter bars (further left) are faster / lighter.
- **Right Scatter Plot (`Compute–Memory Pareto Frontier`)**: Plots `Latency` (x-axis) vs. `Memory` (y-axis) on log–log scales. The **bottom-left corner** is optimal (matching `SGD + Cross-Entropy`, black star `*`). Diamonds (`D`) denote higher-order symbolic autodiff (`2nd`/`3rd` order), circles (`o`) denote 1st-order approximations, and marker area scales with autograd `Nodes`—visualizing how Approaches 1–3 jump orders of magnitude down and left toward the `SGD + CE` frontier.
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

### What This Experiment Does
Step speedups only matter if the approximated gradients still **train the network to high classification accuracy**. Here we train fresh, identically initialized MLPs (`seed = 7`) for `N_EPOCHS = 120` epochs using `torch.optim.SGD(..., momentum=0.9)` and gradient clipping across all three geometric tasks (`Volume-Element`, `Metric Anisotropy`, `Gaussian Curvature`) and all approximation modes, alongside the `SGD + Cross-Entropy` baseline.

Crucially, we use a strict **train-approximate / evaluate-exact protocol**:
- **During training**, forward and backward passes use only the chosen approximation (`App 1`, `App 2`, `App 3`, `App 4`, or `Exact`).
- **At evaluation checkpoints** (`exact_test_accuracy`), held-out `X_test` points are classified using the **exact analytical `torch.func` differential-geometric operators** ($\log\sqrt{\det g_{\text{exact}}}$, $\log(\tilde{g}_{11,\text{exact}}/\tilde{g}_{22,\text{exact}})$, and $K_{\text{exact}}$)—verifying that the network genuinely sculpted the true continuous manifold geometry rather than overfitting discrete approximation artifacts.

### How to Interpret the Table & $2 \times 3$ Convergence Grid
- **Summary Table Columns**:
  - `Train Time (ms)`: Total cumulative wall-clock time spent inside the 120 training steps (excluding test evaluation checkpoints).
  - `vs SGD+CE Time`: Ratio of total training time to the `SGD + Cross-Entropy` baseline (`1.00x`). Values around `1.2x`–`3x` indicate near-first-order training cost, whereas `Exact 3rd-Order (Levi-Civita)` can exceed `30x`–`80x`.
  - `Exact Analytical Test Acc`: Final held-out test accuracy (%) computed from the exact analytical geometric invariant on `X_test`.
- **Top Row (`Accuracy vs. SGD Training Epoch`)**: Measures **per-step optimization fidelity**. When an approximation's curve tracks the dashed `Exact` curve closely step-for-step, its 1st-order gradient estimator is nearly unbiased/noiseless; mild lag in stochastic probes (`App 3`) or neighborhood regression (`App 2`) reflects probe variance or local smoothing bias.
- **Bottom Row (`Accuracy vs. Cumulative Wall-Clock Training Time`, log scale)**: Measures **real-world compute efficiency**. Curves reaching the top-left earliest achieve high exact geometric test accuracy in minimal wall-clock time—showing that Approaches 1–3 converge nearly an order of magnitude faster in wall-clock time than exact 3rd-order curvature autodiff, closely approaching the `SGD + Cross-Entropy` dotted black curve.
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

### What This Experiment Does
To inspect *how* each training approximation shapes the representation manifold across the entire 2D input domain $(x^1, x^2)$, we take the trained models from Section 3 and evaluate their **exact analytical** geometric scalar fields on a dense ambient grid (`grid_pts`):
- **Row 0 (`Volume-Element` models)**: Plots exact analytical log-volume density $\log\sqrt{\det g^{(L)}(x)}$ comparing `Exact 2nd-Order Autodiff` (Col 0) against `App 1: 3-pt FD`, `App 2: 1N k-NN Linear`, and `App 3: Random 2-Probe` (Cols 1–3).
- **Row 1 (`Gaussian Curvature` models)**: Plots exact analytical compressed Gaussian curvature $\mathrm{arcsinh}(K^{(L)}(x))$ comparing `App 4: Exact Gauss Eq.` (Col 0) against `App 1: 7-pt FD`, `App 2: 1N k-NN Quad Jet`, and `App 3: Random 2-Plane` (Cols 1–3).

### How to Interpret the $2 \times 4$ Spatial Contour Maps
- **Diverging Color Scale (`RdBu_r` centered at `0.0 = white`)**:
  - **Blue regions ($< 0$)**: Locally contractive basins ($\det g < 1$ in Row 0) or negatively curved hyperbolic saddles ($K < 0$ in Row 1), targeted to Class `0` in binary classification.
  - **Red regions ($> 0$)**: Locally expansive regions ($\det g > 1$ in Row 0) or positively curved elliptic domes ($K > 0$ in Row 1), targeted to Class `1`.
- **Solid Black Contour (`level = 0.0`)**: The exact intrinsic geometric decision boundary ($\det g(x) = 1$ in Row 0; parabolic zero-curvature curve $K(x) = 0$ in Row 1) separating classes without consulting raw output coordinates.
- **Panel Titles & Visual Comparison**: Compare Columns 1–3 against Column 0 (exact reference). Matching contour topology, smooth class-separating zero contours, and comparable `Exact Acc` achieved in a fraction of the training milliseconds (`ms`) demonstrate that 1st-order stencils, $k$-NN jets, and tangent sketches learn the **same qualitative Riemannian geometry** as higher-order symbolic autodiff.
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

### What This Experiment Does
On 2D toy inputs ($d_0 = 2$), full coordinate finite differences (`Approach 1`) only require $(d_0 + 1)N = 3N$ forward evaluations. However, on high-dimensional inputs, any method that differentiates or perturbs along **every** input axis suffers from the curse of dimensionality. Here we sweep input dimension $d_0 \in \{2, 4, 8, 16, 32, 64, 128\}$ on synthetic batches of size $N = 128$ with architecture `[d_0, 16, 16, C]`, benchmarking five methods via `benchmark_sgd_step`:
1. **`SGD + Cross-Entropy`** (`1N` forward pass, 1st-order reference).
2. **`Exact Metric Autodiff (2nd)`** (`jacrev` pullback of the full $d_0 \times d_0$ metric).
3. **`App 1: Full Coord FD ((d0+1)N)`** (perturbs every coordinate axis $e_1, \dots, e_{d_0}$).
4. **`App 3: 1-Probe Metric Sketch (2N)`** (estimates directional metric stretch along a single random unit tangent vector $u \sim \mathrm{Unif}(\mathbb{S}^{d_0-1})$).
5. **`App 3: Random 2-Plane Curv. (7N)`** (evaluates sectional curvature along a random orthonormal 2-plane $Q \in \mathbb{R}^{d_0 \times 2}$ via the 7-point stencil).

> **Why the y-axes report `ms / SGD step` and `MB / SGD step`**: All five curves—both `Cross-Entropy` and the four geometric loss variants—use **`torch.optim.SGD` as the parameter optimizer** inside `benchmark_sgd_step` (`sgd.zero_grad() -> loss_fn() -> backward() -> sgd.step()`). Holding the optimizer fixed to vanilla `SGD` eliminates confounding memory or compute overhead from Adam state buffers or $P \times P$ Natural Gradient solves (evaluated separately in Section 6), isolating purely how each loss operator scales with $d_0$.

### How to Interpret the Table & Log–Log Scaling Plots
- **Printed Table (`ms / MB` per cell)**: Each column reports `Latency (ms) / Memory (MB)` at input dimension `d_0`. Watch the middle two columns (`Exact 2nd` and `Coord FD`) balloon to tens/hundreds of megabytes by `d_0 = 128`, while `App3 2-Plane K (7N)` stays nearly constant across the entire column.
- **Left Plot (`Compute Scaling vs. Ambient Input Dimension`) & Right Plot (`Memory Scaling vs. Ambient Input Dimension`)**:
  - **Steeply rising curves (`Exact Metric Autodiff` red dashed, `Full Coord FD` blue dash-dot)**: Exhibit $O(d_0)$ to $O(d_0^2)$ batch/Jacobian growth as $d_0 \to 128$.
  - **Nearly flat curves (`App 3: 1-Probe Sketch` green, `App 3: Random 2-Plane Curv.` purple)**: Maintain **$O(1)$ forward batch multiplicity (`2N` and `7N`)** with respect to $d_0$, tracking parallel to `SGD + Cross-Entropy` (black dotted) up to minor input-layer matrix-multiply growth ($d_0 \times 16$).
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

### What This Experiment Does
Sections 1–5 eliminated higher-order **data-space** autodiff inside `loss.backward()` using first-order `SGD`. However, Notebook 06 showed that **Coupled Geometric Natural Gradient (`Coupled Geom-NGD`)** dramatically accelerates and stabilizes Gaussian curvature optimization—at the cost of forming and solving a dense $P \times P$ parameter metric $G^{\Theta,\text{coupled}}(\theta)$ ($O(P^2)$ memory and $O(P^3)$ Cholesky solve).

In this final experiment, we eliminate **both** bottlenecks simultaneously via `woodbury_coupled_ngd_direction`, comparing three optimizers over `35` curvature-loss steps from identical initialization (`seed = 7`):
1. **`Uncoupled SGD + 7-pt FD (1st)`**: Momentum `SGD` in flat Euclidean weight space using the 1st-order 7-point curvature stencil.
2. **`Woodbury Coupled NGD (S=16, 7-pt FD)`**: Evaluates the 7-point FD stencil on a random subsample of $S = 16$ points to build a skinny low-rank Jacobian $U \in \mathbb{R}^{R \times P}$ ($R = 4S = 64 \ll P$, stacking 1 curvature residual + 3 independent whitened DeWitt metric components per sample) and computes the exact damped Natural Gradient step **matrix-free** via the Sherman–Morrison–Woodbury identity:
   $$\left(\lambda I_P + \frac{1}{S} U^T U\right)^{-1} g = \frac{1}{\lambda}\left[g - U^T \left(\lambda S I_R + U U^T\right)^{-1} (U g)\right]$$
   inverting only a tiny $64 \times 64$ matrix ($O(R^2 P + R^3)$ compute, zero $P \times P$ allocation).
3. **`Dense Coupled NGD (Exact Gauss)`**: Full-batch exact Gauss-equation curvature + dense $P \times P$ `compute_coupled_geometric_metric` linear solve from Notebook 06.

### How to Interpret the Printed Benchmark & Two-Panel Trajectory Plot
- **Printed Summary Rows**:
  - `ms/step` & `MB/step`: Total wall-clock step time and CPU tensor memory for the entire parameter update (loss forward/backward **plus** parameter metric preconditioning). Look for `Woodbury Coupled NGD` to cut step latency and memory substantially relative to `Dense Coupled NGD`, especially as parameter count $P$ grows.
  - `Final Loss` & `Exact Test Acc`: Final curvature margin loss and held-out test accuracy evaluated via exact analytical curvature (`exact_test_accuracy`).
- **Left Plot (`Curvature Loss: Matrix-Free Woodbury vs. Dense Coupled NGD`, log scale)** & **Right Plot (`Held-Out Curvature Test Accuracy Trajectory`)**:
  - Compare `Woodbury Coupled NGD` (blue dashed) against `Uncoupled SGD + 7-pt FD` (gray dotted) and `Dense Coupled NGD` (green solid).
  - `Woodbury Coupled NGD` captures the steep per-step loss descent and rapid test-accuracy climb of `Dense Coupled NGD` (far outpacing uncoupled `SGD` in 35 steps) while requiring only 1st-order spatial stencils and a $64 \times 64$ inner solve.
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
