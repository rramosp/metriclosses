# Differential Geometry of Deep Learning: Parameter and Representation Manifolds

## 1. Research Overview & Problem Statement

This project investigates the differential geometric interplay between two fundamental manifolds in deep learning:

1. **The Parameter Manifold ($\Theta$)**: The space of neural network weights and biases $\theta \in \Theta \subset \mathbb{R}^P$, endowed with an information-geometric Riemannian metric (such as the Fisher Information Metric or Generalized Gauss-Newton metric) that governs optimization dynamics, parameter sensitivity, and generalization.
2. **The Data / Representation Manifolds ($\{\mathcal{M}_\ell\}_{\ell=0}^L$)**: The sequence of geometric spaces formed as input data $x \in \mathcal{M}_0 \subset \mathbb{R}^{d_0}$ is progressively transformed by the layers of a deep neural network $\Phi^{(\ell)}_\theta : \mathbb{R}^{d_0} \to \mathbb{R}^{d_\ell}$, inducing layerwise pullback metric tensors $g^{(\ell)}(x; \theta)$ and intrinsic/extrinsic curvature fields $K^{(\ell)}(x; \theta)$.

Our goal is twofold:
- **Layerwise Geometric Evolution & Manifold Duality**: Quantify how the induced Riemannian metric tensor $g^{(\ell)}$ and Gaussian/scalar curvature $K^{(\ell)}$ evolve across network depth $\ell \in \{1, \dots, L\}$ and training time $t$, and characterize how movements on the parameter manifold $\Theta$ drive metric and curvature flows on the representation manifolds $\mathcal{M}_\ell$.
- **Purely Geometric Loss Functions**: Design and evaluate training objectives defined exclusively in terms of differential-geometric quantities (curvature $K(x)$, volume form $\sqrt{\det g(x)}$, and metric tensor eigenvalues), such as driving Class $0$ points into regions of negative curvature ($K(x) < 0$) and Class $1$ points into regions of positive curvature ($K(x) > 0$).

---

## 2. Experimental Testbed Specification

To compute exact, non-stochastic higher-order derivatives (Jacobians, Christoffel symbols, and Riemann curvature tensors) via automatic differentiation without approximation artifacts, we use a low-dimensional synthetic classification benchmark and a bottleneck-expansion-contraction Multi-Layer Perceptron (MLP).

### 2.1 Dataset: `sklearn.datasets.make_moons`
- **Input Domain**: $x = (x^1, x^2) \in \mathbb{R}^2$ ($d_0 = 2$), binary labels $y \in \{0, 1\}$.
- **Configuration**: $N_{\text{train}} = 1000$, $N_{\text{test}} = 500$, noise $\sigma \in \{0.05, 0.10, 0.20\}$, standardized to zero mean and unit variance.
- **Evaluation Grid**: A dense 2D meshgrid $\mathcal{G} \subset [-2.5, 2.5]^2$ (e.g., $80 \times 80$) for visualizing continuous metric fields, volume elements, and curvature landscapes across the ambient input plane.

### 2.2 Model Architecture: `[2 -> 5 -> 10 -> 15 -> 10 -> 2]` MLP
The network maps 2D inputs through four hidden layers before projecting back to a 2D output space:

| Layer $\ell$ | Transformation | Width $d_\ell$ | Geometric Regime (relative to $d_0 = 2$) | Parameters (weights + bias) |
| :--- | :--- | :---: | :--- | :---: |
| $\ell = 0$ | Input $x$ | $2$ | Base chart $\mathcal{M}_0 \subset \mathbb{R}^2$ | — |
| $\ell = 1$ | $h^{(1)} = \phi(W^{(1)} x + b^{(1)})$ | $5$ | Dimensional expansion (immersion into $\mathbb{R}^5$) | $10 + 5 = 15$ |
| $\ell = 2$ | $h^{(2)} = \phi(W^{(2)} h^{(1)} + b^{(2)})$ | $10$ | Dimensional expansion (immersion into $\mathbb{R}^{10}$) | $50 + 10 = 60$ |
| $\ell = 3$ | $h^{(3)} = \phi(W^{(3)} h^{(2)} + b^{(3)})$ | $15$ | Maximum expansion (immersion into $\mathbb{R}^{15}$) | $150 + 15 = 165$ |
| $\ell = 4$ | $h^{(4)} = \phi(W^{(4)} h^{(3)} + b^{(4)})$ | $10$ | Dimensional contraction (submersion to $\mathbb{R}^{10}$) | $150 + 10 = 160$ |
| $\ell = 5$ | $z = W^{(5)} h^{(4)} + b^{(5)}$ | $2$ | Output representation manifold $\mathcal{M}_5 \subset \mathbb{R}^2$ | $20 + 2 = 22$ |
| **Total** | $f_\theta : \mathbb{R}^2 \to \mathbb{R}^2$ | — | — | **$P = 422$** |

> **Critical Note on Activation Smoothness ($\phi$)**: Piecewise-linear activations (`ReLU`) have vanishing second derivatives almost everywhere, concentrating curvature onto a measure-zero singular set (polyhedral fold boundaries). Computing point-wise Christoffel symbols $\Gamma^k_{ij}$ and Riemann curvature requires $C^2$ (ideally $C^\infty$) activations such as `tanh`, `GELU`, `SiLU`, or tunable `Softplus` ($\phi_\beta(u) = \beta^{-1}\log(1 + e^{\beta u})$). Our experiments will default to `tanh` and `GELU`, using $\beta$-annealed `Softplus` to study the smooth-to-ReLU singular limit.

---

## 3. Research Objectives & Literature Context

### Objective 1: Layerwise Evolution of the Metric Tensor and Curvature
**Goal**: Formulate, compute, and visualize the induced Riemannian metric tensor $g^{(\ell)}_{ij}(x; \theta)$, local volume magnification $\sqrt{\det g^{(\ell)}(x; \theta)}$, and Gaussian/scalar curvature $K^{(\ell)}(x; \theta)$ at every layer $\ell \in \{1, \dots, 5\}$ across training epochs.

#### Mathematical Formulation
Because the input space is 2-dimensional ($x \in \mathbb{R}^2$) and $d_\ell \ge 2$ for all layers, the map $\Phi^{(\ell)}_\theta : \mathbb{R}^2 \to \mathbb{R}^{d_\ell}$ generically immerses a 2D surface $\mathcal{M}_\ell = \Phi^{(\ell)}_\theta(\mathbb{R}^2)$ into $\mathbb{R}^{d_\ell}$. Pulling back the ambient Euclidean metric $\delta_{ab}$ of $\mathbb{R}^{d_\ell}$ to the input chart $(x^1, x^2)$ yields a $2 \times 2$ Riemannian metric tensor at every layer $\ell$:
$$g^{(\ell)}_{ij}(x; \theta) = \left( (J^{(\ell)}(x))^T J^{(\ell)}(x) \right)_{ij} = \sum_{a=1}^{d_\ell} \frac{\partial \Phi^{(\ell)}_a(x)}{\partial x^i} \frac{\partial \Phi^{(\ell)}_a(x)}{\partial x^j}, \quad i, j \in \{1, 2\}$$
where $J^{(\ell)}(x) \in \mathbb{R}^{d_\ell \times 2}$ is the layerwise Jacobian. From $g^{(\ell)}_{ij}(x)$, the Levi-Civita connection coefficients (Christoffel symbols of the second kind) are:
$$\Gamma^{(\ell)k}_{ij}(x) = \frac{1}{2} \sum_{m=1}^2 g^{(\ell)km}(x) \left( \frac{\partial g^{(\ell)}_{mi}}{\partial x^j} + \frac{\partial g^{(\ell)}_{mj}}{\partial x^i} - \frac{\partial g^{(\ell)}_{ij}}{\partial x^m} \right)$$
For a 2D Riemannian manifold, the full Riemann curvature tensor has a single independent component $R_{1212}$, and the intrinsic **Gaussian curvature** $K^{(\ell)}(x)$ (half the scalar curvature $R^{(\ell)} = 2K^{(\ell)}$) is given by:
$$K^{(\ell)}(x) = \frac{R^{(\ell)}_{1212}(x)}{\det g^{(\ell)}(x)}$$

> **Resolving the 2D-to-2D Flatness Degeneracy at Layer $\ell = 5$**:
> For hidden layers $\ell \in \{1, 2, 3, 4\}$ ($d_\ell > 2$), $\mathcal{M}_\ell$ is a curved 2D surface embedded in a higher-dimensional Euclidean space $\mathbb{R}^{d_\ell}$, so $K^{(\ell)}(x) \neq 0$ generically. However, at the output layer $\ell = 5$ ($d_5 = 2$), $\Phi^{(5)}_\theta : \mathbb{R}^2 \to \mathbb{R}^2$ maps between two 2D spaces. If $J^{(5)}(x)$ is invertible, pulling back the *flat Euclidean metric* $I_2$ from $\mathbb{R}^2$ to $\mathbb{R}^2$ is merely a coordinate reparameterization of flat space, yielding $K^{(5)}_{\text{eucl}}(x) \equiv 0$ identically! To obtain a non-trivial intrinsic curvature on the final representation manifold $\mathcal{M}_5$, we must use one of three non-degenerate constructions:
> 1. **Graph Embedding Metric (Monge Patch in $\mathbb{R}^4$)**: Define the graph immersion $\tilde{\Phi}^{(5)}(x) = (x^1, x^2, z^1(x), z^2(x)) \in \mathbb{R}^4$, yielding the strictly positive-definite metric $\tilde{g}^{(5)}_{ij}(x) = \delta_{ij} + \left((J^{(5)}(x))^T J^{(5)}(x)\right)_{ij}$, which has rich, non-zero intrinsic Gaussian curvature $K^{(5)}_{\text{graph}}(x)$.
> 2. **Non-Flat Output Ambient Metric**: Pull back a curved metric $h_{ab}(z)$ on the output space $\mathbb{R}^2$ (e.g., a Gaussian/radial metric $h_{ab}(z) = e^{-\|z\|^2/\sigma^2}\delta_{ab}$ or spherical/hyperbolic output chart) via $g^{(5)}(x) = (J^{(5)}(x))^T h(z(x)) J^{(5)}(x)$.
> 3. **Penultimate Surface Curvature**: Evaluate the intrinsic surface curvature $K^{(4)}(x)$ on the 10-dimensional penultimate manifold $\mathcal{M}_4 \subset \mathbb{R}^{10}$ before the final linear projection $W^{(5)}$.

#### Connections to Recent Literature
- **Hauser & Ray (2017)** established the foundational framework for pulling back metric tensors across feedforward and residual neural network layers via coordinate charts and fiber bundles.
- **Kaul & Lall (2020)** formulated explicit Riemannian and sectional curvature computations for deep neural networks to quantify how hidden layers warp input geometry.
- **Benfenati & Marta (2023)** developed *singular Riemannian geometry* for DNNs, analyzing how dimension-contracting layers ($15 \to 10 \to 2$ in our architecture) induce degenerate pullback pseudometrics and equivalence classes (foliation leaves) in input space.
- **Zavatone-Veth et al. (2023)** proved that while randomly initialized wide networks induce highly symmetric pullback metrics, feature learning breaks this symmetry and concentrates the metric volume element $\sqrt{\det g(x)}$ along class decision boundaries.
- **Baptista et al. (2024)** and **Hehl, von Renesse, & Weber (2025)** demonstrated that layer-by-layer geometric transformations in trained classifiers act analogously to discrete **Ricci flow**, progressively smoothing manifold curvature within classes while expanding inter-class separation.
- **Brandon, Chadwick, & Pellegrino (2025)** tracked the layerwise pullback metric and Gaussian curvature of MLPs learning discrete binary tasks on 2D continuous manifolds, revealing a two-stage geometric mechanism: early layers discretize continuous coordinates via high-curvature folds, while deeper layers perform logical separation.

---

### Objective 2: Interplay and Duality Between Parameter Manifold ($\Theta$) and Data Manifolds ($\mathcal{M}_\ell$)
**Goal**: Characterize how the geometry of the 422-dimensional parameter manifold $\Theta$ couples to the geometry of the 2-dimensional data representation manifolds $\mathcal{M}_\ell$.

#### Mathematical Formulation
Let $h^{\text{out}}_{ab}(z)$ be a Riemannian metric on the 2D output space (either the flat identity $\delta_{ab}$, the graph metric, or the categorical Fisher metric on $\text{softmax}(z)$). Pulling $h^{\text{out}}$ back with respect to inputs $x$ versus parameters $\theta$ exposes an exact structural duality:
$$\underbrace{g^{(L)}_{ij}(x; \theta) = \sum_{a,b=1}^2 \frac{\partial z^a}{\partial x^i} h^{\text{out}}_{ab}(z) \frac{\partial z^b}{\partial x^j}}_{\text{Data Information Matrix (DIM) / Input Pullback ($2 \times 2$)}} \qquad \longleftrightarrow \qquad \underbrace{F_{\mu\nu}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_{a,b=1}^2 \frac{\partial z^a}{\partial \theta^\mu} h^{\text{out}}_{ab}(z) \frac{\partial z^b}{\partial \theta^\nu} \right]}_{\text{Parameter Fisher / Gauss-Newton Metric ($P \times P$)}}$$
Both metrics share the same output inner product and are coupled by the mixed spatio-parametric Jacobian $M_{i\mu}(x; \theta) = \frac{\partial^2 z}{\partial x^i \partial \theta^\mu}$. Furthermore, an infinitesimal parameter update $\mathrm{d}\theta = -\eta \, \xi(\theta)$ along the parameter manifold (via Euclidean gradient descent $\xi = \nabla_\theta \mathcal{L}$ or Natural Gradient Descent $\xi = (F + \lambda I)^{-1} \nabla_\theta \mathcal{L}$) induces a **metric flow** on each representation manifold:
$$\frac{\mathrm{d} g^{(\ell)}_{ij}(x; \theta(t))}{\mathrm{d} t} = \sum_{\mu=1}^P \frac{\partial g^{(\ell)}_{ij}(x; \theta)}{\partial \theta^\mu} \frac{\mathrm{d}\theta^\mu}{\mathrm{d}t}$$

#### Connections to Recent Literature
- **Amari (2016)** formalized the Fisher-Rao information metric on the parameter manifold $\Theta$ and proved the invariance of Natural Gradient Descent under coordinate reparameterizations.
- **Fasina et al. (2023)** introduced *Neural FIM* to learn continuous Fisher information metrics and geodesics directly on data point clouds.
- **Halverson & Ruehle (2023)** formulated *Metric Flows with Neural Networks*, showing that gradient descent in parameter space $\Theta$ induces a coupled differential equation (governed by a metric-Neural Tangent Kernel) on the space of pullback Riemannian metrics and scalar curvatures on the data manifold.
- **Tron, Fioresi, Novelli, & Latini (2024, 2026)** formally defined the **Data Information Matrix (DIM)** as the data-space dual of the parameter Fisher Information Matrix, proving that its eigenspectrum and kernel induce singular foliations whose leaves capture learned class manifolds.

---

### Objective 3: Purely Geometric Loss Functions for Classification
**Goal**: Train the `[2 -> 5 -> 10 -> 15 -> 10 -> 2]` MLP using loss functions constructed *exclusively* from differential-geometric invariants of the representation manifold (metric tensor $g_{ij}$, volume form $\sqrt{\det g}$, and Gaussian/scalar curvature $K$), without standard cross-entropy or MSE on logit coordinates.

#### Mathematical Formulation & Hypotheses
To place Class $0$ data points ($y_n = 0$, sign $s_n = -1$) in regions of negative Gaussian curvature ($K(x_n) < 0$, hyperbolic/saddle geometry) and Class $1$ data points ($y_n = 1$, sign $s_n = +1$) in regions of positive Gaussian curvature ($K(x_n) > 0$, spherical/elliptic geometry), we define the **Curvature Margin Loss**:
$$\mathcal{L}_{\text{curv}}(\theta) = \frac{1}{N} \sum_{n=1}^N \text{softplus}\Big( \tau - s_n \, K(x_n; \theta) \Big), \quad s_n = 2y_n - 1 \in \{-1, +1\}$$
where $\tau > 0$ is a target curvature margin (e.g., $\tau = 1.0$) and $K(x_n; \theta)$ is computed at the final representation manifold (via the Graph Embedding Metric $\tilde{g}^{(5)} = I_2 + (J^{(5)})^T J^{(5)}$ or penultimate surface metric $g^{(4)}$).

**Geometric Regularization to Prevent Degenerate Scaling**:
Because curvature scales inversely with metric area ($K \propto (\det g)^{-1}$ under uniform rescaling of $g$), optimizing $\mathcal{L}_{\text{curv}}$ alone can trigger trivial metric collapse ($\det g \to 0$) or explosion. To keep the optimization purely geometric and well-posed, we pair $\mathcal{L}_{\text{curv}}$ with an **Isometric / Volume Stabilization Regularizer**:
$$\mathcal{L}_{\text{geom}}(\theta) = \mathcal{L}_{\text{curv}}(\theta) + \lambda_{\text{vol}} \frac{1}{N} \sum_{n=1}^N \left( \log \det g(x_n; \theta) \right)^2 + \lambda_{\text{iso}} \frac{1}{N} \sum_{n=1}^N \| g(x_n; \theta) - I_2 \|_F^2$$
- **Purely Geometric Inference Rule**: At test time, a point $x^*$ is classified strictly by the sign of the local manifold curvature:
  $$\hat{y}(x^*) = \mathbb{I}\big[ K(x^*; \theta) > 0 \big]$$

#### Alternative Purely Geometric Contrastive Losses: Local Volume Density ($\sqrt{\det g(x)}$)
As an alternative to second-order curvature $K(x)$, we also investigate contrastive losses based on **local volume density** (as well as directional metric anisotropy $\lambda_{\max}(g)/\lambda_{\min}(g)$).

- **What is Local Volume Density?**
  On a 2D Riemannian manifold equipped with pullback metric $g_{ij}(x; \theta)$, the canonical Riemannian volume (area) form is:
  $$\mathrm{d}V_g(x) = \sqrt{\det g(x; \theta)} \, \mathrm{d}x^1 \wedge \mathrm{d}x^2$$
  The scalar field $\rho_{\text{vol}}(x; \theta) = \sqrt{\det g(x; \theta)} = \sqrt{\lambda_1(x)\lambda_2(x)}$ is the **local volume density** (or Jacobian area magnification factor). It measures the ratio between an infinitesimal area element on the transformed representation manifold $\mathcal{M}_\ell$ and the corresponding area element $\mathrm{d}x^1\mathrm{d}x^2$ in the input space:
  - **Volume Contraction ($\sqrt{\det g(x)} < 1$, or $\log\sqrt{\det g(x)} < 0$)**: The network locally compresses space around $x$, pulling neighboring input trajectories closer together on the representation manifold.
  - **Volume Expansion ($\sqrt{\det g(x)} > 1$, or $\log\sqrt{\det g(x)} > 0$)**: The network locally stretches space around $x$, pushing neighboring points apart and magnifying local geometric resolution.
  - **Equiareal / Area-Preserving Regime ($\sqrt{\det g(x)} = 1$, or $\log\sqrt{\det g(x)} = 0$)**: The transformation locally preserves Euclidean area.

- **Intuition & Why Local Volume Density is a Compelling Geometric Loss Foundation**:
  We can define a **Volume-Element Contrastive Loss** that drives Class $0$ points into contractive regions ($\log\sqrt{\det g(x)} \le -\tau$) and Class $1$ points into expansive regions ($\log\sqrt{\det g(x)} \ge +\tau$):
  $$\mathcal{L}_{\text{vol-contrast}}(\theta) = \frac{1}{N} \sum_{n=1}^N \text{softplus}\!\left( \tau - s_n \cdot \frac{1}{2}\log \det g(x_n; \theta) \right), \quad s_n = 2y_n - 1 \in \{-1, +1\}$$
  with the purely geometric test-time decision rule $\hat{y}(x^*) = \mathbb{I}[\log\sqrt{\det g(x^*; \theta)} > 0]$ (where the decision boundary is the **equiareal contour** $\det g(x) = 1$). This is an attractive foundation for four reasons:
  1. **Invariance to Rigid Output Coordinate Motions**: Standard losses (Cross-Entropy, MSE) tie classes to specific coordinate locations or half-spaces in the output vector space $\mathbb{R}^2$. By contrast, $g(x) = J(x)^T J(x)$—and therefore $\sqrt{\det g(x)}$—is strictly invariant under arbitrary output translations $z \mapsto z + b$ and orthogonal rotations or reflections $z \mapsto Qz$ ($Q \in O(d_\ell)$). The network learns to classify purely through *how it warps local metric volume* (compression vs. expansion) rather than *where* it places points in ambient coordinates.
  2. **Duality with Probability Measure Transport**: By the change-of-variables theorem for pushforward measures ($p_X(x) = p_Z(f_\theta(x))\sqrt{\det g(x)}$), contrasting $\log\sqrt{\det g(x)}$ between classes is equivalent to sculpting the induced probability density on the representation manifold: Class $0$ is mapped into high-density contractive attractors (geometric sinks), whereas Class $1$ is mapped into low-density expansive regions (geometric sources).
  3. **No 2D-to-2D Flatness Degeneracy**: Unlike Gaussian curvature $K^{(5)}_{\text{eucl}}(x)$—which vanishes identically when pulling back the flat 2D Euclidean metric to a 2D input space—the volume density $\sqrt{\det((J^{(5)})^T J^{(5)})} = |\det J^{(5)}(x)|$ is non-trivial directly on the raw $2 \to 2$ Euclidean output pullback metric without requiring a 4D graph embedding.
  4. **Lower Differential Order & Optimization Stability**: Computing $K(x; \theta)$ requires 2nd-order spatial derivatives of the network (and **3rd-order** mixed derivatives $\nabla_\theta K(x; \theta)$ during backpropagation), whereas $\sqrt{\det g(x; \theta)}$ depends only on 1st-order spatial Jacobians $J(x; \theta)$ (requiring only **2nd-order** mixed derivatives $\nabla_\theta J(x; \theta)$), leading to faster, better-conditioned gradient dynamics.

---

## 4. Experimentation Plan via IPython Notebooks

All experiments are structured into a progressive 5-notebook suite under `notebooks/`, backed by a shared, differentiable PyTorch geometry utility module (`diffgeom_mlx/` or `src/geometry.py` using `torch.func` / `vmap`, `jacrev`, and `hessian` for exact, vectorized higher-order derivatives).

### Shared Technical Stack
- **Framework**: PyTorch (`torch.func.jacrev`, `torch.func.vmap`) for exact composable 1st, 2nd, and 3rd derivatives required when backpropagating through curvature $K(x; \theta)$ into parameters $\theta$.
- **Architecture**: `MLP([2, 5, 10, 15, 10, 2])` with smooth activations (`tanh`, `GELU`, and $\beta$-tunable `Softplus`).
- **Dataset**: `make_moons(n_samples=1000, noise=0.1, random_state=42)`.

---

### Notebook 01: `01_baseline_and_pullback_metrics.ipynb`
**Title**: *Baseline MLP Training and Layerwise Pullback Metric Tensors*
- **Objectives**:
  1. Train the `[2 -> 5 -> 10 -> 15 -> 10 -> 2]` MLP with standard Cross-Entropy loss on `make_moons` and record parameter checkpoints across training epochs.
  2. Implement vectorized layerwise Jacobian computation $J^{(\ell)}(x) \in \mathbb{R}^{d_\ell \times 2}$ and induced pullback metrics $g^{(\ell)}_{ij}(x) = (J^{(\ell)}(x))^T J^{(\ell)}(x)$ for $\ell \in \{1, 2, 3, 4, 5\}$ using `torch.func.vmap(torch.func.jacrev(...))`.
  3. Compute and plot across all 5 layers:
     - Volume magnification factor $\sqrt{\det g^{(\ell)}(x)}$ over the 2D input grid.
     - Metric condition number / anisotropy $\kappa(g^{(\ell)}(x)) = \lambda_1 / \lambda_2$.
     - Visual Tissot indicatrices (metric ellipses) showing how each layer stretches and compresses the `make_moons` plane along the decision boundary (testing Zavatone-Veth et al., 2023).
  4. Compare smooth (`tanh`, `GELU`) vs. piecewise-linear (`ReLU`, $\beta$-`Softplus`) pullback metrics.

### Notebook 02: `02_layerwise_curvature_evolution.ipynb`
**Title**: *Christoffel Symbols, Riemann Tensor, and Gaussian Curvature Flow Across Layers*
- **Objectives**:
  1. Implement exact automatic differentiation for $\partial_k g^{(\ell)}_{ij}(x)$, Christoffel symbols $\Gamma^{(\ell)k}_{ij}(x)$, Riemann tensor $R^{(\ell)}_{1212}(x)$, and Gaussian curvature $K^{(\ell)}(x)$ across layers $\ell \in \{1, \dots, 5\}$.
  2. Empirically verify the 2D-to-2D Euclidean pullback flatness identity ($K^{(5)}_{\text{eucl}}(x) \equiv 0$) and compare the three non-degenerate output curvature constructions:
     - Graph Embedding Metric $\tilde{g}^{(5)} = I_2 + (J^{(5)})^T J^{(5)}$.
     - Conformal Gaussian Output Metric $h(z) = e^{-\|z\|^2}\delta_{ab}$.
     - Penultimate Immersion Metric $g^{(4)}$ on $\mathcal{M}_4 \subset \mathbb{R}^{10}$.
  3. Track the distribution of $K^{(\ell)}(x)$ for Class $0$, Class $1$, and the decision boundary across layers $\ell = 1 \to 5$ and across training epochs $t = 0 \to T$ to test the discrete Ricci flow and two-stage discretization hypotheses (Baptista et al., 2024; Brandon et al., 2025).

### Notebook 03: `03_parameter_data_manifold_duality.ipynb`
**Title**: *Duality and Metric Flow Between the Parameter Manifold ($\Theta$) and Data Manifolds ($\mathcal{M}_\ell$)*
- **Objectives**:
  1. Compute the full $422 \times 422$ Parameter Fisher Information Matrix $F_{\mu\nu}(\theta)$ and the $2 \times 2$ Data Information Matrix $\text{DIM}_{ij}(x; \theta)$ across training checkpoints.
  2. Analyze their spectral coupling: track the effective rank, eigenvalue decay, and top eigenvectors of $F(\theta)$ alongside the trace and determinant of $g^{(\ell)}(x; \theta)$.
  3. Perturb parameters $\theta$ along the top vs. bottom eigenvectors of $F(\theta)$ and measure the induced metric velocity $\|\partial_\theta g^{(\ell)} \cdot \delta\theta\|_F$ and curvature response $\|\partial_\theta K^{(\ell)} \cdot \delta\theta\|$ on the `make_moons` manifold (testing Halverson & Ruehle, 2023).
  4. Compare training trajectories under SGD, Adam, and damped Natural Gradient Descent ($\Delta \theta \propto -(F + \lambda I)^{-1} \nabla_\theta \mathcal{L}$), visualizing how natural gradient updates regularize the layerwise curvature flow.

### Notebook 04: `04_purely_geometric_curvature_losses.ipynb`
**Title**: *Classification via Purely Geometric Curvature Losses ($K < 0$ vs. $K > 0$)*
- **Objectives**:
  1. Train the `[2 -> 5 -> 10 -> 15 -> 10 -> 2]` MLP from scratch using **only** the geometric objective $\mathcal{L}_{\text{geom}}(\theta) = \mathcal{L}_{\text{curv}}(\theta) + \lambda_{\text{vol}}\mathcal{L}_{\text{vol}}(\theta)$ (requiring 3rd-order autodiff: $\nabla_\theta K(x; \theta)$).
  2. Enforce $K(x_n; \theta) \le -\tau$ for Class $0$ ($y_n = 0$) and $K(x_n; \theta) \ge +\tau$ for Class $1$ ($y_n = 1$) on the final representation manifold (comparing Graph Metric $\tilde{g}^{(5)}$ and Penultimate Surface Metric $g^{(4)}$).
  3. Evaluate test classification accuracy using the purely geometric decision rule $\hat{y}(x^*) = \mathbb{I}[K(x^*; \theta) > 0]$ across noise levels and margin hyperparameters $(\tau, \lambda_{\text{vol}})$.
  4. Visualize the learned 2D curvature landscape $K(x^1, x^2)$ over the ambient plane (verifying the zero-curvature decision boundary $K(x) = 0$ separating negative-curvature Class $0$ basins from positive-curvature Class $1$ domes) and embed the learned 2D output manifold into $\mathbb{R}^3$ via PCA / MDS to inspect the hyperbolic saddle vs. spherical dome geometry.

### Notebook 05: `05_geometric_loss_ablations_and_synthesis.ipynb`
**Title**: *Comparative Analysis of Geometric Losses, Layerwise Curvature Profiles, and Parameter Geometry*
- **Objectives**:
  1. Benchmark three purely geometric training objectives against standard Cross-Entropy:
     - **Sign-Curvature Margin Loss** ($K < -\tau$ vs. $K > +\tau$).
     - **Volume-Element Contrastive Loss** ($\log\det g < -\tau$ vs. $\log\det g > +\tau$).
     - **Anisotropy / Metric-Eigenvalue Loss** (aligning principal metric directions with class tangents).
  2. Compare how Cross-Entropy training vs. Purely Geometric Curvature training shapes:
     - The intermediate hidden layer curvatures $K^{(1)}, K^{(2)}, K^{(3)}, K^{(4)}$.
     - The spectrum and condition number of the $422 \times 422$ Parameter Fisher Information Matrix $F(\theta)$.
  3. Synthesize findings on how explicit geometric constraints on the data manifold reshape the curvature and conditioning of the parameter manifold.

---

## 5. References

1. **Amari, S.** (2016). *Information Geometry and Its Applications*. Applied Mathematical Sciences, Vol. 194, Springer.
2. **Hauser, M., & Ray, A.** (2017). Principles of Riemannian Geometry in Neural Networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 2807–2816. [Paper](https://papers.nips.cc/paper/6873-principles-of-riemannian-geometry-in-neural-networks)
3. **Kaul, P., & Lall, B.** (2020). Riemannian Curvature of Deep Neural Networks. *IEEE Transactions on Neural Networks and Learning Systems*, 31(4), 1410–1416. [arXiv:1902.03892](https://arxiv.org/abs/1902.03892)
4. **Benfenati, A., & Marta, A.** (2023). A singular Riemannian geometry approach to Deep Neural Networks I. Theoretical foundations. *Neural Networks*, 158, 331–343. [arXiv:2201.09656](https://arxiv.org/abs/2201.09656)
5. **Zavatone-Veth, J. A., Yang, S., Rubinfien, J. A., & Pehlevan, C.** (2023). How does training shape the Riemannian geometry of neural network representations? *arXiv preprint*. [arXiv:2301.11375](https://arxiv.org/abs/2301.11375)
6. **Fasina, O., Huguet, G., Tong, A., Zhang, Y., Wolf, G., Nickel, M., Adelstein, I., & Krishnaswamy, S.** (2023). Neural FIM for learning Fisher information metrics from point cloud data. *Proceedings of the 40th International Conference on Machine Learning (ICML)*, PMLR 202, 9813–9826. [arXiv:2306.06062](https://arxiv.org/abs/2306.06062)
7. **Halverson, J., & Ruehle, F.** (2023). Metric Flows with Neural Networks. *arXiv preprint*. [arXiv:2310.19870](https://arxiv.org/abs/2310.19870)
8. **Baptista, A., Barp, A., Chakraborti, T., Harbron, C., MacArthur, B. D., & Banerji, C. R. S.** (2024). Deep Learning as Ricci Flow. *arXiv preprint*. [arXiv:2404.14265](https://arxiv.org/abs/2404.14265)
9. **Tron, E., Fioresi, R., Novelli, G., & Latini, E.** (2024/2026). Manifold Learning via Foliations and Knowledge Transfer / The Geometry of Manifold Learning: Singular foliations via Deep Learning models. *arXiv preprints*. [arXiv:2409.07412](https://arxiv.org/abs/2409.07412) / [arXiv:2610.37370](https://arxiv.org/abs/2610.37370)
10. **Hehl, M., von Renesse, M.-K., & Weber, M.** (2025). Neural Feature Geometry Evolves as Discrete Ricci Flow. *arXiv preprint*. [arXiv:2509.22362](https://arxiv.org/abs/2509.22362)
11. **Brandon, J., Chadwick, A., & Pellegrino, A.** (2025). Emergent Riemannian geometry over learning discrete computations on continuous manifolds. *arXiv preprint*. [arXiv:2512.00196](https://arxiv.org/abs/2512.00196)
