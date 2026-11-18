# Conversation Log: Fisher Metric & Coupled Parameter–Data Optimization

---

## Turn 1

### User
how is the fisher metric used in this project

### Assistant
In this project, the **Fisher Information Metric** endows the neural network's parameter space and input data space with dual information-geometric Riemannian structures ([spec.md](spec.md), [README.md](README.md)). Rather than serving as a classification loss itself, it is used to analyze **parameter–data manifold duality**, **metric/curvature flows**, and **Natural Gradient optimization**.

#### 1. Mathematical Formulation & Implementation
Pulling back the categorical Fisher metric of the 2-class output distribution $p_\theta(y \mid x) = \mathrm{softmax}(z(x; \theta))$ with respect to parameters versus inputs yields two dual tensors implemented in [src/geometry.py](src/geometry.py) via exact `torch.func` (`vmap` + `jacrev`) automatic differentiation:

* **Parameter Fisher Information Matrix (`FIM`, $422 \times 422$)** — `compute_parameter_fim` in [src/geometry.py](src/geometry.py):
  $$F_{\mu\nu}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_{c=0}^1 p_\theta(y=c \mid x) \frac{\partial \log p_\theta(y=c \mid x)}{\partial \theta^\mu} \frac{\partial \log p_\theta(y=c \mid x)}{\partial \theta^\nu} \right] + \lambda I_{422}$$
  Computed by taking per-sample parameter Jacobians of `log_softmax` logits of shape `(N, 2, 422)`, scaling each class row by $\sqrt{p_\theta(y=c \mid x)}$, and forming the damped outer product over all `P = 422` weights and biases of `MoonsMLP`.
* **Data Information Matrix (`DIM`, $2 \times 2$ per input)** — `compute_data_information_matrix` in [src/geometry.py](src/geometry.py):
  $$\mathrm{DIM}_{ij}(x; \theta) = \sum_{c=0}^1 p_\theta(y=c \mid x) \frac{\partial \log p_\theta(y=c \mid x)}{\partial x^i} \frac{\partial \log p_\theta(y=c \mid x)}{\partial x^j} + \epsilon I_2$$
  The spatial counterpart of the FIM on the 2D input manifold, differentiating `log_softmax` with respect to input coordinates `x` instead of parameters `theta`.

#### 2. How It Is Used in Experiments
All primary Fisher metric experiments live in [notebooks/03_parameter_data_manifold_duality.ipynb](notebooks/03_parameter_data_manifold_duality.ipynb) (built in [scripts/build_all_notebooks.py](scripts/build_all_notebooks.py)):

1. **Spectral Compression & Parameter–Data Co-Evolution**
   Across training checkpoints (epochs 0, 15, 60, 180), the notebook tracks the 422 eigenvalues of `fim`, its trace, and its entropy-based effective rank alongside the spatial average trace of `DIM`. This shows that Fisher information concentrates into a low-dimensional stiff subspace (effective rank ~2.4–3.9 out of 422 parameters) while data-space information concentrates along the decision boundary.

2. **Singular Foliations of Input Space via `DIM`**
   Because the unregularized binary `DIM` has rank at most 1, its null/smallest eigenvector field spans the 1D tangent leaves of constant class probability (`quiver` plot over `make_moons`), while its trace highlights the separating boundary.

3. **FIM Eigenvector Perturbations → Representation Metric & Curvature Flows**
   To test how parameter-space geometry drives representation-space geometry, weights are perturbed by fixed norm (`delta_scale = 0.08`) along unit eigenvectors of `fim` ranging from stiffest (`k = 1`) to flattest (`k = 351`). Measuring the resulting shifts in the output graph pullback metric $\|\Delta g^{(5)}(x)\|_F$ and Gaussian curvature $|\Delta K^{(5)}(x)|$ shows that stiff Fisher directions induce large geometric deformations on the data manifold, whereas flat/null Fisher modes leave the representation geometry nearly unchanged.

4. **Riemannian Natural Gradient Descent**
   The damped parameter FIM (`damping = 0.02`) is used directly as a Riemannian preconditioner in `run_optimizer_comparison`:
   $$\Delta \theta = -\eta \big(F(\theta) + \lambda I\big)^{-1} \nabla_\theta \mathcal{L}$$
   solved exactly at each step via `torch.linalg.solve(fim_curr, g_flat)`, comparing optimization speed and the smoothness of the induced output curvature trajectory $|K^{(5)}(x)|$ against standard Euclidean gradient descent.

#### 3. Relation to Parameter Geometry Under Purely Geometric Losses (Notebook 05)
In [notebooks/05_geometric_loss_ablations_and_synthesis.ipynb](notebooks/05_geometric_loss_ablations_and_synthesis.ipynb), models trained with purely geometric objectives (curvature, volume element, and anisotropy) do not interpret raw outputs `z` as `softmax` logits. To compare their induced $422 \times 422$ parameter-space geometry against the Cross-Entropy baseline on equal footing, `compute_param_pullback_spectrum` replaces the categorical Fisher weight with the flat output identity metric—computing the unweighted **Generalized Gauss-Newton / parameter pullback metric** $G^\Theta_{\mu\nu}(\theta) = \mathbb{E}_x[(J_\theta z)^T (J_\theta z)]$.

---

## Turn 2

### User
what is the justification for using fisher metric, and not other metric?

### Assistant
The justification for choosing the **Fisher Information Metric** over alternatives (such as the Euclidean parameter metric, the exact loss Hessian, or an unweighted Jacobian pullback) comes from both **information geometry** and the **specific architecture of probabilistic classifiers** studied in [spec.md](spec.md):

#### 1. Why the Fisher Metric Over Other Metrics on Parameter & Data Space

1. **Uniqueness & Reparameterization Invariance (Čencov's Theorem & Amari)**
   * **Why not the Euclidean metric $\|\delta\theta\|_2^2$?** Euclidean distance in weight space is an artifact of parameterization: rescaling weights in one layer while inversely compensating in another changes $\|\delta\theta\|_2$ arbitrarily without changing the learned function, and differently scaled layers distort SGD trajectories.
   * By **Čencov's theorem**, the Fisher-Rao metric is the **unique** Riemannian metric (up to a positive constant) on a manifold of probability distributions that is invariant under sufficient statistics / Markov morphisms and smooth coordinate reparameterizations of $\theta$ or $x$.

2. **Exact Local Measure of Predictive Change (KL Divergence)**
   * For a probabilistic classifier $p_\theta(y \mid x) = \mathrm{softmax}(z(x; \theta))$, the Fisher metric is the local second-order Taylor expansion of the expected Kullback–Leibler divergence:
     $$\mathbb{E}_x\big[D_{\mathrm{KL}}\big(p_\theta(\cdot \mid x) \,\|\, p_{\theta + \delta\theta}(\cdot \mid x)\big)\big] = \frac{1}{2}\, \delta\theta^T F(\theta)\, \delta\theta + \mathcal{O}(\|\delta\theta\|^3)$$
   * It directly measures how much a parameter perturbation $\delta\theta$ (for `FIM`) or an input perturbation $\delta x$ (for `DIM`) changes the **predicted class distribution**, aligning naturally with the Cross-Entropy objective used in Notebooks 01–03.

3. **Guaranteed Positive Semi-Definiteness (Why Not the Loss Hessian?)**
   * A natural candidate metric for optimization is the exact loss Hessian $\nabla_\theta^2 \mathcal{L}(\theta)$. However, in a non-convex multi-layer network (`MoonsMLP`), the Hessian is **indefinite**—it has negative eigenvalues around saddle points and non-convex regions, so it fails to define a valid Riemannian inner product and breaks Newton-like steps.
   * By contrast, $F(\theta)$ is an outer product of score vectors ([src/geometry.py](src/geometry.py)), guaranteeing it is symmetric positive semi-definite everywhere on $\Theta$ (and for Cross-Entropy with `softmax`, it coincides exactly with the Generalized Gauss-Newton approximation to the Hessian).

4. **Automatic Quotienting of `softmax` Symmetries & Saturation (`FIM` ↔ `DIM` Duality)**
   * As formulated in [spec.md](spec.md), both the parameter metric and input data metric are pullbacks of an output-space metric $h^{\mathrm{out}}_{ab}(z)$:
     $$g^{(L)}_{ij}(x; \theta) = J_x^T\, h^{\mathrm{out}}(z)\, J_x \quad \longleftrightarrow \quad F_{\mu\nu}(\theta) = \mathbb{E}_x\!\left[J_\theta^T\, h^{\mathrm{out}}(z)\, J_\theta\right]$$
   * **Why not use the flat Euclidean metric $h^{\mathrm{out}} = I_2$ on `z` for a classifier?** Because `softmax(z)` is invariant to common logit shifts $z \mapsto z + c\mathbf{1}$: moving both logits equally changes Euclidean output coordinates (`J^T J > 0`) while changing the classifier's probabilities by **zero**. Moreover, deep inside a class region where logits are large, raw logit gradients can remain nonzero even when class probabilities are already saturated ($p \approx 0$ or $1$).
   * Choosing $h^{\mathrm{out}}$ to be the **categorical Fisher metric** weights logit differences by $p(1-p)$ and projects out the $(1, 1)$ shift gauge direction. That is why the Data Information Matrix in [notebooks/03_parameter_data_manifold_duality.ipynb](notebooks/03_parameter_data_manifold_duality.ipynb) has exact rank ≤ 1, vanishes in saturated class interiors, and isolates the 1D singular foliation along the decision boundary.

#### 2. Where the Fisher Metric Is *Not* Used (and Why This Project Switches Metrics)

Crucially, the Fisher metric is **only** justified when outputs parameterize a probability distribution $p_\theta(y \mid x)$. Where no likelihood model exists, this codebase deliberately uses other Riemannian metrics instead:

* **Hidden representations ($\ell = 1, \dots, 4$)**: Hidden activations have no categorical likelihood, so [src/geometry.py](src/geometry.py) pulls back the ambient **Euclidean metric** $g^{(\ell)} = (J^{(\ell)})^T J^{(\ell)}$.
* **Output curvature & purely geometric losses (Notebooks 02, 04, 05)**: When training via curvature, volume, or anisotropy losses, the 2D output `z` is treated as a geometric surface rather than `softmax` logits (and a rank-1 Fisher metric has $\det = 0$, making 2D Gaussian curvature $K = R_{1212}/\det g$ singular). There, [src/geometry.py](src/geometry.py) uses the **Graph immersion metric** $I_2 + J^T J$ or **Conformal Gaussian metric** $e^{-\|z\|^2/\sigma^2} J^T J$.
* **Cross-paradigm parameter comparison (Notebook 05)**: Because models trained with geometric losses never use `softmax`, applying `compute_parameter_fim` to them would artificially impose probabilistic semantics they were not trained under. Accordingly, [scripts/build_all_notebooks.py](scripts/build_all_notebooks.py) switches from $F(\theta)$ to the unweighted **parameter output pullback metric** $G^\Theta(\theta) = \mathbb{E}_x[J_\theta^T J_\theta]$ (`h_out = I_2`) to compare all four training paradigms fairly.

---

## Turn 3

### User
so when the geometric losses are used (like curvature or density), what is the metric tensor on the parameter space, and what is it on the data space

### Assistant
When training with the purely geometric losses in Notebooks 04 and 05, there is no `softmax` likelihood on the output space, so both manifolds replace the Fisher metric with **Euclidean / immersion pullback metrics**:

#### 1. Metric Tensor on the Data Space ($2 \times 2$ per input $x$)

For an input $x \in \mathbb{R}^2$ mapped to the 2D output $z = \Phi^{(5)}_\theta(x) \in \mathbb{R}^2$ with spatial Jacobian $J_x(x; \theta) = \partial z / \partial x \in \mathbb{R}^{2 \times 2}$, the project uses **two different $2 \times 2$ pullback metrics** in [src/geometry.py](src/geometry.py) depending on the geometric invariant:

1. **For Gaussian Curvature Loss & Metric Anisotropy Loss → Graph Immersion Metric (`metric_mode="graph"`)**
   $$\tilde{g}^{(5)}_{ij}(x; \theta) = \delta_{ij} + \sum_{a=1}^2 \frac{\partial z^a}{\partial x^i} \frac{\partial z^a}{\partial x^j} = \big(I_2 + J_x^T J_x\big)_{ij}$$
   * **Where used**: Notebooks 04 and 05 ([scripts/build_all_notebooks.py](scripts/build_all_notebooks.py)).
   * **Why**: Pulling back the flat metric $I_2$ from $\mathbb{R}^2 \to \mathbb{R}^2$ is just a coordinate reparameterization of flat space, which has $K^{(5)}_{\mathrm{eucl}}(x) \equiv 0$ everywhere ([spec.md](spec.md)). Embedding the network as a 2D Monge surface $(x^1, x^2, z^1(x), z^2(x)) \in \mathbb{R}^4$ makes $\tilde{g}^{(5)}$ strictly positive-definite (eigenvalues $\ge 1$) and gives non-trivial Gaussian curvature $K^{(5)} = R_{1212} / \det\tilde{g}^{(5)}$ as well as non-singular diagonal ratios $\tilde{g}_{11} / \tilde{g}_{22}$.

2. **For Local Volume Density Loss → Raw Euclidean Pullback Metric (`metric_mode="euclidean"`)**
   $$g^{(5)}_{ij}(x; \theta) = \sum_{a=1}^2 \frac{\partial z^a}{\partial x^i} \frac{\partial z^a}{\partial x^j} + \epsilon \delta_{ij} = \big(J_x^T J_x + \epsilon I_2\big)_{ij}$$
   * **Where used**: Notebook 05 (`eps = 1e-6`).
   * **Why**: `volume_contrastive_loss` separates classes into **area-contracting** ($\det g^{(5)} < 1$, i.e. $\log\sqrt{\det g^{(5)}} < 0$) versus **area-expanding** ($\det g^{(5)} > 1$) regions. Under the Graph metric $I_2 + J_x^T J_x$, $\det\tilde{g}^{(5)} \ge 1$ everywhere so contraction below 1 is impossible; by contrast, $\det(J_x^T J_x) = (\det J_x)^2$ allows both compression and expansion across the equiareal boundary $\det g^{(5)} = 1$ without suffering from 2D flatness degeneracy.

*(For all intermediate hidden layers $\ell \in \{1, 2, 3, 4\}$ where $d_\ell > 2$, the data manifold is already a curved 2D surface in $\mathbb{R}^{d_\ell}$, so it uses the Euclidean immersion pullback $g^{(\ell)} = (J_x^{(\ell)})^T J_x^{(\ell)} + \epsilon I_2$.)*

#### 2. Metric Tensor on the Parameter Space ($422 \times 422$ on $\Theta$)

In Notebooks 04 and 05, parameter space $\Theta \subset \mathbb{R}^{422}$ is treated in two ways:

1. **During Optimization (Training Dynamics)**
   * Both [notebooks/04_purely_geometric_curvature_losses.ipynb](notebooks/04_purely_geometric_curvature_losses.ipynb) and [notebooks/05_geometric_loss_ablations_and_synthesis.ipynb](notebooks/05_geometric_loss_ablations_and_synthesis.ipynb) optimize $\theta$ with standard `Adam`—effectively using Euclidean gradients $\nabla_\theta \mathcal{L}_{\mathrm{geom}}$ with a diagonal empirical second-moment preconditioner rather than a full $422 \times 422$ Riemannian natural gradient.

2. **For Parameter Manifold Geometry Analysis (`G_theta`)**
   * In Section 3 of Notebook 05 (`compute_param_pullback_spectrum` in [scripts/build_all_notebooks.py](scripts/build_all_notebooks.py)), $\Theta$ is endowed with the **$422 \times 422$ Euclidean Output Pullback Metric** (the Generalized Gauss-Newton matrix for identity output metric $h^{\mathrm{out}} = I_2$):
     $$G^\Theta_{\mu\nu}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_{a=1}^2 \frac{\partial z^a(x; \theta)}{\partial \theta^\mu} \frac{\partial z^a(x; \theta)}{\partial \theta^\nu} \right]$$
   * Geometrically, $\delta\theta^T G^\Theta(\theta)\,\delta\theta = \mathbb{E}_x[\|\delta z(x)\|_2^2]$ measures the expected $L^2$ displacement of the 2D output map $z(x; \theta)$ induced by a parameter perturbation $\delta\theta$.

#### 3. A Geometric Subtlety Worth Noting on Parameter Space

While $G^\Theta_{\mu\nu}(\theta) = \mathbb{E}_x[J_\theta^T J_\theta]$ pulls back changes in **output coordinates** $z(x; \theta)$, the geometric losses depend only on $g(x; \theta)$, which is invariant under rigid Euclidean motions of the output $z \mapsto Qz + b$ ([spec.md](spec.md)). For example, shifting the final-layer bias $b^{(5)}$ has nonzero length under $G^\Theta$ even though it changes $g(x)$, $\sqrt{\det g(x)}$, and $K(x)$ by **zero**.

If you wanted a parameter-space metric that quotients out those output symmetries—analogous to how the Fisher metric quotients out `softmax` logit shifts—the natural theoretical counterpart would pull back the **affine-invariant (DeWitt) metric on the space of $2 \times 2$ SPD matrices**:
$$G^{\Theta,\mathrm{DeWitt}}_{\mu\nu}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathrm{Tr}\!\left( g(x)^{-1} \frac{\partial g(x)}{\partial \theta^\mu} \, g(x)^{-1} \frac{\partial g(x)}{\partial \theta^\nu} \right) \right]$$

---

## Turn 4

### User
when using geometric loses from the data space to do an optimization in the parameter space, the geometry of the parameter space should depend on the geometry of the data space. is that correct?

### Assistant
**Yes, that is completely correct.**

When your training objective depends purely on the differential geometry of the data manifold rather than output coordinates or class probabilities, a parameter vector `theta` is no longer selecting a probability distribution $p_\theta(y \mid x)$—it is selecting a **Riemannian metric field on the data manifold**:
$$\theta \in \Theta \;\longmapsto\; g(\cdot\,; \theta) \in \mathrm{Met}(\mathcal{M})$$
Consequently, the natural geometry of the parameter space **must be pulled back from the geometry of the data space**, in two distinct ways:

#### 1. Defining the Parameter Metric Directly From Data-Space Geometry

Just as the Fisher metric pulls back the geometry of the probability simplex, optimizing data-space geometry induces two natural Riemannian metrics on parameter space $\Theta$:

1. **The Metric-Space Pullback (Ebin / DeWitt Metric)**
   On the infinite-dimensional manifold of $2 \times 2$ Riemannian metrics $\mathrm{Met}(\mathcal{M})$, the canonical coordinate- and scale-covariant inner product is the **Ebin (or DeWitt) $L^2$ metric** (the geometric foundation behind the metric-flow framework cited in [spec.md](spec.md)). Pulling it back to $\Theta$ gives:
   $$G^{\Theta,\mathrm{Ebin}}_{\mu\nu}(\theta) = \int_{\mathcal{M}} g^{ik}(x; \theta)\, g^{jl}(x; \theta)\, \frac{\partial g_{ij}(x; \theta)}{\partial \theta^\mu} \frac{\partial g_{kl}(x; \theta)}{\partial \theta^\nu} \sqrt{\det g(x; \theta)}\, \mathrm{d}^2 x$$
   *(or its empirical average over training points $x \sim \mathcal{D}$, $\mathbb{E}_x[\mathrm{Tr}(g^{-1}\partial_\mu g\, g^{-1}\partial_\nu g)]$, optionally weighted by local volume density $\sqrt{\det g(x)}$).*
   Notice how every ingredient of $G^{\Theta,\mathrm{Ebin}}$—the inverse tensor $g^{ij}(x; \theta)$ contracting indices and the volume form $\sqrt{\det g(x; \theta)}$ weighting space—comes **directly from the induced data-space metric**.

2. **The Task-Specific Gauss-Newton Pullback (For a Specific Geometric Loss)**
   When optimizing a specific scalar invariant $\psi(x; \theta)$—such as Gaussian curvature $K(x; \theta)$ or log-volume density $\log\sqrt{\det g(x; \theta)}$—under a convex margin penalty $\ell(\psi, y)$, the exact Generalized Gauss-Newton metric on $\Theta$ is:
   $$G^{\Theta,\psi}_{\mu\nu}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \frac{\partial^2 \ell(\psi(x; \theta), y)}{\partial \psi^2} \, \frac{\partial \psi(x; \theta)}{\partial \theta^\mu} \frac{\partial \psi(x; \theta)}{\partial \theta^\nu} \right]$$
   which measures parameter distance strictly by how much $\delta\theta$ changes the target geometric field $\psi$ on the data manifold.

#### 2. Why This Coupling Matters for Optimization

If parameter space ignores data-space geometry (e.g., using Euclidean SGD or even the coordinate pullback $G^\Theta = \mathbb{E}_x[J_\theta^T J_\theta]$), two mismatches occur:

* **Unquotiented Symmetries (Gauge Directions)**: Rigid motions of the output space $z \mapsto Qz + b$ change network weights and raw outputs $z$, yet leave $g(x)$, $\sqrt{\det g(x)}$, and $K(x)$ strictly invariant ([spec.md](spec.md)). Both $G^{\Theta,\mathrm{Ebin}}$ and $G^{\Theta,\psi}$ automatically assign **zero** metric length to those symmetry directions.
* **Ill-Conditioning Near Metric Singularities**: Because $K = R_{1212} / \det g$ and $\log\sqrt{\det g} = \frac{1}{2}\log\det g$ depend on $g^{-1}$, regions where $\det g(x; \theta)$ becomes small make the map $\theta \mapsto (g, K)$ extraordinarily stiff: tiny Euclidean parameter steps trigger huge curvature spikes. Preconditioning updates by $(G^{\Theta,\mathrm{Ebin}} + \lambda I)^{-1}$ or $(G^{\Theta,\psi} + \lambda I)^{-1}$ (a **Geometric Natural Gradient**) automatically shrinks parameter steps where the data manifold is nearly degenerate or sharply curved.

#### 3. How This Compares to Notebooks 04 & 05

In Notebooks 04 and 05, the codebase captured only the **empirical back-reaction**, not a **metric-coupled optimizer**:

* **During optimization**, Notebooks 04 and 05 used `Adam` in Euclidean weight coordinates, compensating for stiffness manually via `tanh` curvature clipping (`k_clip`), isometric/volume regularizers in [src/geometry.py](src/geometry.py), and gradient norm clipping.
* **During post-hoc analysis**, Section 3 of [notebooks/05_geometric_loss_ablations_and_synthesis.ipynb](notebooks/05_geometric_loss_ablations_and_synthesis.ipynb) evaluated the simpler coordinate pullback $G^\Theta(\theta) = \mathbb{E}_x[J_\theta^T J_\theta]$ rather than $G^{\Theta,\mathrm{Ebin}}(\theta)$ or $G^{\Theta,\psi}(\theta)$.

---

## Turn 5

### User
ok. implement a new notebook named notebooks/06-coupled-optimization.ipynb that uses the parameter space metric coupled with the data space metric to do the optimization with the three geometric losses, and compares it with the uncoupled version

### Assistant
Implemented and executed [notebooks/06-coupled-optimization.ipynb](notebooks/06-coupled-optimization.ipynb), along with reusable coupled parameter-metric operators in [src/geometry.py](src/geometry.py) (exported in [src/__init__.py](src/__init__.py) and registered in [scripts/build_all_notebooks.py](scripts/build_all_notebooks.py)).

#### What Was Implemented

1. **Coupled Parameter–Data Metric Operators ([src/geometry.py](src/geometry.py))**
   * `compute_coupled_dewitt_metric`: Pulls back the affine-invariant DeWitt / Ebin metric on $2 \times 2$ SPD matrices from the induced data-space metric $g(x; \theta)$:
     $$G^{\Theta,\mathrm{DeWitt}}_{\mu\nu}(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\!\left[ g^{ik}(x; \theta)\, g^{jl}(x; \theta)\, \frac{\partial g_{ij}(x; \theta)}{\partial \theta^\mu} \frac{\partial g_{kl}(x; \theta)}{\partial \theta^\nu} \right] + \lambda I_{422}$$
   * `compute_coupled_geometric_metric`: Combines $G^{\Theta,\mathrm{DeWitt}}$ with the Generalized Gauss-Newton pullback of the target geometric field $\psi(x; \theta) \in \{K^{(5)}, \log\sqrt{\det g^{(5)}}, \log(\tilde{g}^{(5)}_{11}/\tilde{g}^{(5)}_{22})\}$ (forming a 2nd-order Sobolev-DeWitt metric on $\mathrm{Met}(\mathcal{M}_0)$ for Gaussian curvature).
   * `compute_uncoupled_coord_metric`: Computes the uncoupled output-coordinate pullback $G^{\Theta,\mathrm{coord}}(\theta) = \mathbb{E}_x[(J_\theta z)^T (J_\theta z)] + \lambda I_{422}$ as a controlled natural-gradient baseline.
   * `compute_fast_graph_geometry`: Computes $\tilde{g} = I_2 + J_x^T J_x$ and Gaussian curvature $K^{(5)}$ via 2nd-order `jacfwd` and Gauss's equation ($R_{1212} = \langle \mathrm{II}_{11}, \mathrm{II}_{22}\rangle - \|\mathrm{II}_{12}\|^2$), matching `compute_layer_geometry` to `< 9e-16` while speeding up parameter curvature Jacobians ~45×.

#### Key Empirical Results in [notebooks/06-coupled-optimization.ipynb](notebooks/06-coupled-optimization.ipynb)

* **Exact $E(2)$ Output Symmetry Quotienting (Section 1)**:
  Along all three rigid Euclidean output generators—translations $\delta b^{(5)}_1, \delta b^{(5)}_2$ and $\mathfrak{so}(2)$ rotation $(\Omega W^{(5)}, \Omega b^{(5)})$—uncoupled parameter metrics assign positive length (`1.00` under $I_{422}$; `1.00` and `7.98e-2` under $G^{\Theta,\mathrm{coord}}$), whereas both coupled metrics ($G^{\Theta,\mathrm{DeWitt}}$ and $G^{\Theta,\mathrm{coupled}}$) evaluate to machine zero (`0.0` to `1.5e-16`).

* **45-Step Optimization Comparison Across All Three Geometric Losses (Sections 2 & 3)**:

  | Geometric Objective | Optimizer | Step 10 Test Acc | Step 25 Test Acc | Final Loss | Final Test Acc | `b5` Gauge Drift |
  | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
  | **Volume-Element** | Uncoupled GD (`I_P`) | `85.83%` | `97.50%` | `0.1236` | `100.00%` | `0.00` |
  | **Volume-Element** | Uncoupled Adam | `85.00%` | `95.00%` | `0.0891` | `99.17%` | `0.00` |
  | **Volume-Element** | Uncoupled Coord-NGD | `90.83%` | `90.00%` | `0.1632` | `95.83%` | `0.732` |
  | **Volume-Element** | **Coupled Geom-NGD** | `67.50%` | **`98.33%`** | **`0.0766`** | `98.33%` | **`0.00`** |
  | **Metric Anisotropy** | Uncoupled GD (`I_P`) | `50.83%` | `54.17%` | `0.3819` | `79.17%` | `0.00` |
  | **Metric Anisotropy** | Uncoupled Adam | `68.33%` | `93.33%` | `0.1166` | `98.33%` | `0.00` |
  | **Metric Anisotropy** | Uncoupled Coord-NGD | `99.17%` | `100.00%` | `0.0962` | `100.00%` | `0.524` |
  | **Metric Anisotropy** | **Coupled Geom-NGD** | **`100.00%`** | **`100.00%`** | **`0.0837`** | **`100.00%`** | **`0.00`** |
  | **Gaussian Curvature** | Uncoupled GD (`I_P`) | `61.67%` | `71.67%` | `0.3970` | `72.50%` | `0.00` |
  | **Gaussian Curvature** | Uncoupled Adam | `60.00%` | `94.17%` | `0.0899` | `100.00%` | `0.00` |
  | **Gaussian Curvature** | Uncoupled Coord-NGD | `70.83%` | `93.33%` | `0.1356` | `95.83%` | `0.808` |
  | **Gaussian Curvature** | **Coupled Geom-NGD** | **`95.83%`** | **`100.00%`** | **`0.0273`** | `97.50%` | **`0.00`** |

* **Elimination of Gauge Drift & Unclipped Stiffness Stability (Sections 4 & 5)**:
  * **Zero Gauge Drift**: `Uncoupled Coord-NGD` drags the unused output bias $b^{(5)}$ (`0.52`–`0.81` norm drift) via off-diagonal cross-covariance $\mathbb{E}_x[\partial_{\theta_{\mathrm{hid}}} z]$, wasting trust-region budget on pure translations; `Coupled Geom-NGD` keeps `b5` drift identically `0.00`.
  * **No Gradient Clipping Needed**: With `clip_grad_norm_` disabled on high-margin 3rd-order curvature training, uncoupled Euclidean GD suffers sharp gradient and loss spikes, while `Coupled Geom-NGD` preconditions stiff curvature directions smoothly via $(G^{\Theta,\mathrm{coupled}})^{-1}$.
  * **DeWitt vs. Sobolev-Coupled Ablation**: Even the purely unsupervised data-metric pullback (`Coupled DeWitt-NGD`, which sees neither class labels nor curvature derivatives) drops final loss from `0.3819` → `0.0835` (`100%` acc) on anisotropy and `0.3970` → `0.0330` on curvature, with full task coupling pushing curvature loss further down to `0.0273`.

---

## Turn 6

### User
all these geometric losses require at least second order derivatives, right?

### Assistant
**Yes—for training (optimization), every one of these geometric losses requires at least 2nd-order derivatives** of the network map $z = \Phi_\theta(x)$ ([README.md](README.md), [spec.md](spec.md)).

Specifically, because the induced pullback metric $g_{ij}(x; \theta)$ already depends on first-order spatial derivatives $J_x = \partial z / \partial x$, taking parameter gradients $\nabla_\theta \mathcal{L}_{\mathrm{geom}}$ adds one more derivative with respect to $\theta$:

| Loss / Operation | Forward Pass / Inference (wrt `x` only) | Training Gradient $\nabla_\theta \mathcal{L}$ (mixed `x`, `theta`) | Coupled Metric $G^{\Theta,\mathrm{coupled}}$ | Highest Derivative Tensor |
| :--- | :---: | :---: | :---: | :--- |
| **Cross-Entropy (Baseline)** | **0th-order** (`z`) | **1st-order** | **1st-order** (`FIM`) | $\frac{\partial z^a}{\partial \theta^\mu}$ |
| **Volume-Element Loss** | **1st-order** | **2nd-order** | **2nd-order** | $\frac{\partial^2 z^a}{\partial \theta^\mu \partial x^i}$ |
| **Metric Anisotropy Loss** | **1st-order** | **2nd-order** | **2nd-order** | $\frac{\partial^2 z^a}{\partial \theta^\mu \partial x^i}$ |
| **Gaussian Curvature Loss** | **2nd-order** | **3rd-order** | **3rd-order** | $\frac{\partial^3 z^a}{\partial \theta^\mu \partial x^i \partial x^j}$ |

#### Three Important Consequences in This Codebase

1. **Inference is one order cheaper than training**: At test time, classifying via volume density ($\log\sqrt{\det g(x)} > 0$) or anisotropy ($\log(g_{11}/g_{22}) > 0$) only needs **1st-order** spatial Jacobians `J_x`, while classifying via Gaussian curvature ($K(x) > 0$) needs **2nd-order** spatial derivatives.
2. **Coupled NGD does not increase the derivative order beyond backprop**: Because both $G^{\Theta,\mathrm{DeWitt}}$ and $G^{\Theta,\mathrm{coupled}}$ in [src/geometry.py](src/geometry.py) are outer-product (Gauss-Newton) pullbacks $\partial_\mu \psi \, \partial_\nu \psi$ rather than exact loss Hessians $\partial_\mu \partial_\nu \mathcal{L}_{\mathrm{geom}}$, they stay at **2nd-order** for volume/anisotropy and **3rd-order** for curvature (using Gauss's equation in `compute_fast_graph_geometry` at [src/geometry.py](src/geometry.py) to avoid redundant 4th-order autodiff nodes).
3. **Smooth activations are mandatory**: As noted in [spec.md](spec.md), piecewise-linear `ReLU` has vanishing second derivatives almost everywhere (making spatial Hessians $\partial_x^2 z = 0$ and mixed derivatives $\partial_\theta \partial_x z$ piecewise constant with zero gradient inside linear regions), which is why all experiments use smooth $C^\infty$ activations like `tanh`, `GELU`, or `Softplus`.

---

## Turn 7

### User
save this conversation to coupled-loss.md and continue doing so as I ask you more things in this conversationm

### Assistant
Saved the full conversation log to [coupled-loss.md](coupled-loss.md). I will continue appending each new question and response to [coupled-loss.md](coupled-loss.md) as we go.

---

## Turn 8

### User
could you explore methods to approximate the second order geometrical objects during training, so that  it approaches first order methods (like SGD) in terms of computing and memory

### Assistant
The reason exact geometric training is expensive is **nested symbolic differentiation**: computing $\partial_x z$ or $\partial_x^2 z$ via `jacrev`/`jacfwd` builds a higher-order computation graph that must be retained in memory so `loss.backward()` can differentiate through it a second or third time with respect to $\theta$.

To bring both **compute** and **memory** down toward standard 1st-order SGD, the key idea is to **replace symbolic spatial derivatives ($\partial_x, \partial_x^2$) with forward-only discrete differential operators**, so backpropagation with respect to $\theta$ becomes a **pure 1st-order backward pass** ($\nabla_\theta z$).

---

#### 1. Four Families of 1st-Order Approximations for Spatial Geometric Losses

1. **Batched Finite-Difference Stencils (Pure 1st-Order Autodiff, $O(h^2)$ Accurate)**
   Instead of differentiating `model(x)` symbolically with respect to `x`, evaluate `model` on a small displaced stencil stacked into a **single batched forward pass**:
   * **For $g_{ij}(x; \theta)$ (Volume & Anisotropy — 3-point stencil, `3N` batch)**:
     Evaluate $Z = \Phi_\theta([X,\, X + h e_1,\, X + h e_2])$ in one forward pass, form $\hat{\partial}_i z = (z(x + h e_i) - z(x))/h$, and assemble $\hat{g} = \hat{J}_x^T \hat{J}_x$.
   * **For $K(x; \theta)$ (Gaussian Curvature — 7-point central stencil + Gauss's equation, `7N` batch)**:
     Evaluate $\Phi_\theta$ at $\{x,\, x \pm h e_1,\, x \pm h e_2,\, x \pm h(e_1 + e_2)\}$ in one forward pass to obtain all five 1st and 2nd spatial derivatives $(\hat{\partial}_1 z, \hat{\partial}_2 z, \hat{\partial}_{11} z, \hat{\partial}_{12} z, \hat{\partial}_{22} z)$ at $O(h^2)$ accuracy, then plug directly into Gauss's equation ($R_{1212} = \langle \mathrm{II}_{11}, \mathrm{II}_{22}\rangle - \|\mathrm{II}_{12}\|^2$).
   * **Why it approaches SGD**: There is **zero** nested `jacrev`/`jacfwd` graph—`loss.backward()` differentiates standard `Linear` + `tanh` activations once.

2. **Precomputed Point-Cloud $k$-NN Taylor Jet Operators (`1N` Forward Pass — Zero Extra Evaluations!)**
   You can eliminate even the `3N`/`7N` stencil multiplier by using **Discrete Differential Geometry (DDG) on the training batch itself**:
   * **Offline ($t = 0$, once before training)**: For each training point $x_i$ and its $k$ nearest neighbors $\{x_j\}_{j \in \mathcal{N}(i)}$, fit a Gaussian-weighted quadratic Taylor design matrix on $\Delta x_{ij} = x_j - x_i$ (`[dx1, dx2, 0.5*dx1^2, dx1*dx2, 0.5*dx2^2]`) and store the pseudoinverse jet operator $Q_i \in \mathbb{R}^{5 \times k}$.
   * **Online (every training step)**: Run **only one standard `1N` forward pass** `Z = model(X)` (identical to Cross-Entropy SGD!), gather neighbor output differences $\Delta Z_{ij} = z_j - z_i$, and contract `jets = einsum("nmk,nka->nma", Q, dZ)` to get $(\hat{J}_i, \hat{H}_i) \implies (\hat{g}_i, \hat{K}_i)$.
   * **Bonus**: Works directly on curved point clouds where no global flat input chart exists.

3. **Randomized Tangent-Plane Sketching / Hutchinson Probes (Scaling to High Input Dimension $d_0 \gg 2$)**
   In high-dimensional input spaces ($d_0 = 100$–$10{,}000$), coordinate stencils would scale as $O(d_0)$ or $O(d_0^2)$. Instead, sample **two random orthonormal probe vectors** $v_1, v_2 \in \mathbb{R}^{d_0}$ per step and evaluate the 3-point or 7-point finite-difference stencil on the random 2D slice $x + u_1 v_1 + u_2 v_2$:
   * By Hutchinson's trace identity, $\mathbb{E}_v[\|D_v z\|_2^2] = \mathrm{Tr}(g(x))$, and the expected 2-plane curvature $\mathbb{E}_{v_1, v_2}[K(v_1, v_2)]$ is proportional to the intrinsic **scalar curvature** $R(x)$—keeping compute and memory **$O(1)$ independent of $d_0$**.

4. **Algebraic Order Reduction Before Autodiff (Gauss's Equation)**
   Even when exact autodiff is retained, reformulating intrinsic curvature via Gauss's equation (`compute_fast_graph_geometry` in [src/geometry.py](src/geometry.py)) cancels the cancelling 3rd-order spatial derivatives $\partial_x^3 z$ analytically before autodiff runs, cutting memory **15×** with zero approximation error.

---

#### 2. Verified Benchmark on `MoonsMLP` (`[2 -> 5 -> 10 -> 15 -> 10 -> 2]`, $N = 220$)

Below are measured forward+backward step costs on this project's exact architecture—and held-out test accuracies when **trained purely with the 1st-order approximation** and **evaluated at test time using the exact analytical differential-geometric operators**:

| Training Method | Autodiff Order (wrt `theta`) | Latency (`ms/step`) | Memory (`MB/step`) | Autograd Nodes | Held-Out Test Acc (Exact Eval) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cross-Entropy Reference (SGD/Adam)** | **1st-Order** | `1.02 ms` | `0.51 MB` | `26` | `100.00%` |
| **Volume — Exact 2nd-Order Autodiff** | 2nd-Order | `4.13 ms` | `3.47 MB` | `107` | `100.00%` |
| **Volume — 1st-Order 3-pt FD Stencil** | **1st-Order** | **`1.85 ms`** (2.2× faster) | **`1.87 MB`** | **`57`** | **`100.00%`** |
| **Volume — 1st-Order `1N` $k$-NN Jet** | **1st-Order** | **`2.38 ms`** | **`1.40 MB`** (2.5× less) | **`63`** | **`98.33%`** |
| **Curvature — Exact 3rd-Order (Levi-Civita)** | 3rd-Order | `92.34 ms` | `502.95 MB` | `1,122` | `96.67%` |
| **Curvature — Exact 3rd-Order (Gauss Eq.)** | 3rd-Order | `29.39 ms` | `32.92 MB` | `615` | `97.50%` |
| **Curvature — 1st-Order 7-pt FD Stencil** | **1st-Order** | **`4.49 ms`** (**20.6× faster**) | **`5.26 MB`** (**95.6× less**) | **`194`** | **`98.33%`** |
| **Curvature — 1st-Order `1N` $k$-NN Jet** | **1st-Order** | **`4.64 ms`** (**19.9× faster**) | **`2.26 MB`** (**222.5× less**) | **`182`** | **`93.33%`** |

*(Anisotropy trained with the 1st-order 3-pt FD stencil likewise achieves **`100.00%`** exact analytical test accuracy in `274 ms` total across 100 epochs.)*

---

#### 3. Approximating the Coupled Parameter Metric $G^{\Theta,\mathrm{coupled}}$ at Near-SGD Cost

If you also want the **coupled Natural Gradient** benefits from [notebooks/06-coupled-optimization.ipynb](notebooks/06-coupled-optimization.ipynb) without forming or inverting a dense $P \times P$ matrix every step:

* **Low-Rank Subsampled Woodbury Sketch ($O(S^2 P)$ vs. $O(P^3)$)**: Compute the 1st-order FD/jet Jacobian $U \in \mathbb{R}^{S \times P}$ on only $S \ll P$ active margin-violating points (e.g. $S = 16$) and invert $(\frac{1}{S} U^T U + \lambda I_P)^{-1} g$ via the **Woodbury identity** without ever allocating a $P \times P$ matrix:
  $$( \lambda I_P + \tfrac{1}{S} U^T U )^{-1} g = \frac{1}{\lambda}\!\left( g - U^T (\lambda S I_S + U U^T)^{-1} U g \right)$$
* **Amortized / Lazy Metric Refresh**: Recompute $G^{\Theta,\mathrm{coupled}}$ (and its Cholesky factor) only every $K = 10$–$20$ steps while running cheap 1st-order stencil gradients on intermediate steps.
* **Empirical Diagonal / K-FAC Coupled Preconditioner**: Accumulate an exponential moving average of per-parameter (or layerwise Kronecker-factored) squared gradients of the whitened metric residual $g_0^{-1/2} \hat{g}(\theta) g_0^{-1/2}$, preserving $E(2)$ output translation invariance at $O(P)$ Adam-equivalent memory.

---

## Turn 9

### User
ok. create a notebook 07-evaluating-efficient-training.ipynb with the implemnetation of the four approaches above and the proper comparison of compute and memory metrics, including the baseline of training with SGD with cross-entropy loss

### Assistant
Implemented and executed [notebooks/07-evaluating-efficient-training.ipynb](notebooks/07-evaluating-efficient-training.ipynb), backed by modular first-order approximation operators in [src/geometry.py](src/geometry.py) (exported in [src/__init__.py](src/__init__.py) and registered in [scripts/build_all_notebooks.py](scripts/build_all_notebooks.py)).

#### 1. Operators Implemented in [src/geometry.py](src/geometry.py)
* **Approach 1 — Batched Finite-Difference Stencils**: `compute_fd_pullback_metric` (`3N` forward evaluations for $g_{ij}$) and `compute_fd_graph_curvature` (`7N` central stencil + Gauss's equation for $K$), reducing backpropagation to pure 1st-order autodiff.
* **Approach 2 — Precomputed Point-Cloud $k$-NN Taylor Jet Operators**: `build_knn_jet_operator` (`order=1` linear Jacobian stencil and `order=2` quadratic curvature jet, precomputed once at $t=0$ in `14.8 ms`) and `compute_knn_jet_geometry` (online single `1N` forward pass, zero extra evaluations beyond standard `SGD`).
* **Approach 3 — Randomized Tangent-Plane / Hutchinson Sketching**: `compute_sketched_pullback_metric` (`(m+1)N` directional probes) and `compute_sketched_plane_curvature` (`7N` random orthonormal 2-plane evaluations, $O(1)$ in ambient input dimension $d_0$).
* **Approach 4 — Algebraic Order Reduction (Gauss's Equation)**: `compute_fast_graph_geometry` (exact analytical cancellation of $\partial_x^3 z$ terms before autodiff, `< 1.6e-13` deviation from Levi-Civita curvature).
* **Matrix-Free Coupled NGD Extension**: `woodbury_coupled_ngd_direction` ($O(R^2 P)$ Woodbury solve on $S = 16$ subsampled 1st-order FD stencil features without allocating a $P \times P$ matrix).

#### 2. Step-Level Compute, Memory & Autograd Graph Comparison (All Under `torch.optim.SGD`)

| Method | Autodiff Order | Forward Evals | Step Latency (`ms/step`) | Speedup vs. Exact | Step Memory (`MB/step`) | Memory Saving | Graph Nodes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`SGD + Cross-Entropy` (Baseline)** | **1st** | **`1N`** | **`1.12 ms`** | — | **`0.51 MB`** | — | **`26`** |
| **Vol: Exact 2nd-Order Autodiff** | 2nd | `jacrev` | `3.89 ms` | `1.0x` | `3.47 MB` | `1.0x` | `107` |
| **Vol: App 1 (3-pt FD Stencil)** | **1st** | `3N` | **`1.96 ms`** | **`2.0x`** | **`1.86 MB`** | **`1.9x`** | **`56`** |
| **Vol: App 2 (`1N` $k$-NN Linear)** | **1st** | **`1N`** | **`1.85 ms`** | **`2.1x`** | **`0.99 MB`** | **`3.5x`** | **`61`** |
| **Vol: App 3 (2-Probe Sketch)** | **1st** | `3N` | **`2.18 ms`** | **`1.8x`** | **`2.05 MB`** | **`1.7x`** | **`72`** |
| **Aniso: Exact 2nd-Order Autodiff** | 2nd | `jacrev` | `4.19 ms` | `1.0x` | `3.58 MB` | `1.0x` | `114` |
| **Aniso: App 1 (3-pt FD Stencil)** | **1st** | `3N` | **`2.08 ms`** | **`2.0x`** | **`1.96 MB`** | **`1.8x`** | **`63`** |
| **Aniso: App 2 (`1N` $k$-NN Linear)** | **1st** | **`1N`** | **`2.37 ms`** | **`1.8x`** | **`1.10 MB`** | **`3.3x`** | **`68`** |
| **Aniso: App 3 (1-Probe Hutch.)** | **1st** | `2N` | **`3.51 ms`** | **`1.2x`** | **`1.43 MB`** | **`2.5x`** | **`69`** |
| **Curv: Exact 3rd-Order (Levi-Civita)** | 3rd | `jacrev^3` | `65.89 ms` | `1.0x` | `324.81 MB` | `1.0x` | `1,036` |
| **Curv: App 4 (Exact Gauss Eq.)** | 3rd | `jacfwd^2` | `21.96 ms` | **`3.0x`** | `32.92 MB` | **`9.9x`** | `615` |
| **Curv: App 1 (7-pt FD Stencil)** | **1st** | `7N` | **`3.50 ms`** | **`18.8x`** | **`5.26 MB`** | **`61.8x`** | **`194`** |
| **Curv: App 2 (`1N` $k$-NN Quad Jet)** | **1st** | **`1N`** | **`3.42 ms`** | **`19.2x`** | **`2.26 MB`** | **`143.7x`** | **`182`** |
| **Curv: App 3 (Random 2-Plane)** | **1st** | `7N` | **`3.54 ms`** | **`18.6x`** | **`5.26 MB`** | **`61.8x`** | **`194`** |

#### 3. End-to-End 120-Epoch `SGD` Training & Exact Analytical Test Accuracy
Training every model with `torch.optim.SGD` (`momentum = 0.9`) and evaluating held-out test accuracy with the **exact analytical** geometric operators (`compute_layer_pullback_metric` / `compute_fast_graph_geometry`):

* **`SGD + Cross-Entropy` Baseline**: `167.8 ms` (`1.00x`) → **`100.00%`** test accuracy.
* **Volume-Element**: App 1 (`322.4 ms`, `1.92x`) and App 3 (`351.6 ms`, `2.10x`) both achieve **`100.00%`** exact analytical test accuracy (matching Exact 2nd-Order's `100.00%` in ~half the time); App 2 (`1N` $k$-NN, `338.3 ms`) achieves **`97.50%`**.
* **Metric Anisotropy**: Both App 1 (`332.5 ms`, `1.98x`) and App 2 (`1N` $k$-NN, `367.0 ms`, `2.19x`) achieve **`98.33%`** exact analytical test accuracy (matching Exact 2nd-Order's `98.33%`).
* **Gaussian Curvature**: App 1 (`560.7 ms`, `3.34x` SGD+CE) and App 3 (`609.1 ms`, `3.63x` SGD+CE) both achieve **`97.50%`** exact analytical test accuracy—matching Exact 3rd-Order Levi-Civita (`6,837.1 ms`, `40.75x` SGD+CE) with a **12.2× end-to-end training speedup**.
* **High-Dimensional Scaling ($d_0 = 2 \to 128$, Section 5)**: At $d_0 = 128$, Exact 2nd-Order (`165.3 MB/step`) and Full Coordinate FD (`195.9 MB/step`) blow up quadratically, whereas **App 3 Random 2-Plane Curvature Sketch** stays flat at **`3.17 ms/step`** and **`4.33 MB/step`** (**38×–45× less memory**).
* **Woodbury Coupled NGD (Section 6)**: Combining App 1 (`7-pt FD`) with a matrix-free $S=16$ Woodbury solve cuts dense coupled NGD latency **4.7×** (`285.4 ms` → `61.3 ms/step`) and profiler tensor allocation **61×** while reaching **`95.00%`** exact curvature test accuracy in 35 steps (vs. `83.33%` for uncoupled SGD at step 35).

---

## Turn 10

### User
so we using the coupled metric with a geometry loss, I understand we have following situation:
- the metric tensor of the parameter manifold depends on the metric tensor of the data manifold, because the data manifold metric tensor is part of the loss equation and shapes the parameter manifold
- the metric tensor of the data manifold depends on the metric tensor of the parameter manifold, because the parameters of the model defines the shape of the data manifold.

is this correct? if so, isn't this like some circular definition?

### Assistant
You put your finger on a crucial subtlety—and resolving it comes down to distinguishing between **a parameter *point* $\theta \in \Theta$** (the current weights) and **the parameter *metric tensor* $G^\Theta(\theta)$** (how we measure distances and gradients on $\Theta$).

Because of that distinction, **the static definition at any instant is strictly one-way (not circular)**, while **across training time $t$ they form a well-posed dynamical feedback loop**:

---

#### 1. At Any Fixed Instant $t$: Strictly One-Way (Not Circular)

Look closely at your second bullet point:
* The data-space metric $g_{ij}(x; \theta)$ depends on the **parameter values (point)** $\theta \in \mathbb{R}^{422}$—**not** on the **parameter metric tensor** $G^\Theta_{\mu\nu}(\theta)$.
* Specifically, $g_{ij}(x; \theta)$ is anchored to a **fixed ambient Euclidean metric** ($I_2$ on $\mathbb{R}^2$ or $I_4$ on the graph space $\mathbb{R}^4$) pulled back through the network's spatial Jacobian $J_x \Phi_\theta(x)$ ([src/geometry.py](src/geometry.py)). You could throw away $G^\Theta$ completely and $g_{ij}(x; \theta)$ at a given $\theta$ would not change at all.

So at any fixed weight vector $\theta$, the mathematical definition is a **directed acyclic chain**:
$$\text{Fixed Output Metric } (I_2 \text{ or } I_4) \;+\; \text{Point } \theta \;\xrightarrow{\;\partial_x \Phi_\theta\;}\; \text{Data Metric } g_{ij}(x; \theta) \;\xrightarrow{\;\partial_\theta g,\, g^{-1}\;}\; \text{Parameter Metric } G^\Theta_{\mu\nu}(\theta)$$
Nothing on the left ever needs $G^\Theta_{\mu\nu}(\theta)$ to be defined, so there is **zero circularity**.

---

#### 2. Across Optimization Steps $t \to t + 1$: A Coupled Geometric Flow (Feedback in Time)

Your intuition about mutual dependence becomes completely true **dynamically** as soon as we step forward in training time $t$:
1. At step $t$, the current **point** $\theta_t$ determines the current **data metric** $g(x; \theta_t)$ (and curvature $K(x; \theta_t)$).
2. That data metric determines both the geometric loss gradient $\nabla_\theta \mathcal{L}_{\mathrm{geom}}(\theta_t)$ and the **parameter metric** $G^\Theta(\theta_t)$ ([notebooks/06-coupled-optimization.ipynb](notebooks/06-coupled-optimization.ipynb)).
3. The parameter metric $G^\Theta(\theta_t)$ then determines the **velocity** $\Delta\theta_t = -G^\Theta(\theta_t)^{-1} \nabla_\theta \mathcal{L}_{\mathrm{geom}}(\theta_t)$, moving the weights to a **new point** $\theta_{t+1} = \theta_t + \Delta\theta_t$.
4. At step $t + 1$, the new point $\theta_{t+1}$ reshapes the **new data metric** $g(x; \theta_{t+1})$.

Unrolled in time, the dependency is causal rather than circular:
$$\theta_t \;\longrightarrow\; g(\cdot\,; \theta_t) \;\longrightarrow\; G^\Theta(\theta_t) \;\longrightarrow\; \theta_{t+1} \;\longrightarrow\; g(\cdot\,; \theta_{t+1}) \;\longrightarrow\; G^\Theta(\theta_{t+1}) \;\longrightarrow\; \cdots$$

This is an **ordinary differential equation (a coupled geometric flow)**, directly analogous to:
* **Ricci Flow ($\partial_t g = -2\,\mathrm{Ric}(g)$)**: The current metric $g(t)$ defines its curvature $\mathrm{Ric}(g(t))$, which in turn dictates how the metric evolves at $t + \mathrm{d}t$.
* **General Relativity**: *"Matter tells spacetime how to curve; spacetime tells matter how to move"*—a coupled dynamical system, not a circular definition.
* **Standard Fisher Natural Gradient**: Even with Cross-Entropy, $\theta_t$ defines $p_{\theta_t}(y \mid x)$, which defines $F(\theta_t)$, which dictates $\theta_{t+1}$, which changes $p_{\theta_{t+1}}(y \mid x)$.

---

#### 3. What *Would* Be Truly Circular?
It would only be circular if computing $g_{ij}(x; \theta_t)$ **at the same step $t$** required already knowing $G^\Theta_{\mu\nu}(\theta_t)$ (a simultaneous algebraic fixed-point system $g_t = \mathcal{F}(G^\Theta_t)$ and $G^\Theta_t = \mathcal{H}(g_t)$). Because $g_{ij}(x; \theta_t)$ is pulled back from the fixed ambient output metric rather than from $G^\Theta_t$, the base of the hierarchy is grounded at every step.

---

## Turn 11

### User
if it were truly circular, what would it imply for the optimization process (training)?

### Assistant
If the two metrics were **truly circular at the same instant $t$**—meaning $g(x; \theta)$ depended on $G^\Theta(\theta)$ and $G^\Theta(\theta)$ simultaneously depended on $g(x; \theta)$ without an external Euclidean output ruler—for example via the mixed spatio-parametric sensitivity tensor $M_{i\mu}(x; \theta) = \frac{\partial^2 z}{\partial x^i \partial \theta^\mu}$ ([spec.md](spec.md)):
$$g(x) = \mathcal{F}_\theta(G^\Theta)(x) = M(x)\,(G^\Theta)^{-1} M(x)^T, \qquad G^\Theta = \mathcal{H}_\theta(g) = \mathbb{E}_{x \sim \mathcal{D}}\!\left[M(x)^T\, g(x)^{-1} M(x)\right]$$
then training would fundamentally change from **standard feedforward optimization** into **self-consistent equilibrium (bilevel) optimization**, with five concrete consequences:

---

#### 1. Every Forward Pass Would Require an Inner Fixed-Point Solver
* You could no longer evaluate the geometric loss $\mathcal{L}_{\mathrm{geom}}(\theta_t)$ in one pass, because $g^*(x; \theta_t)$ is only defined implicitly as a fixed point $g^* = (\mathcal{F}_{\theta_t} \circ \mathcal{H}_{\theta_t})(g^*)$.
* Just **computing the loss** at step $t$ would require running an inner alternating Sinkhorn / matrix-scaling loop $g^{(k+1)} = \mathcal{F}_{\theta_t}(\mathcal{H}_{\theta_t}(g^{(k)}))$ until convergence $\|g^{(k+1)} - g^{(k)}\| < \varepsilon$—analogous to **Deep Equilibrium Models (DEQs)** or **Hartree–Fock self-consistent field** iterations in quantum physics.

#### 2. Backpropagation Would Require Implicit Differentiation (IFT), Not Standard Autograd
* Unrolling $K_{\mathrm{inner}}$ matrix-inversion iterations inside PyTorch autograd would multiply memory and graph size by $K_{\mathrm{inner}}\times$ and suffer from vanishing/exploding gradients through repeated inversions.
* Instead, `loss.backward()` would have to use the **Implicit Function Theorem (IFT)** at the converged equilibrium $g^* = \mathcal{T}_\theta(g^*)$, solving a linear **adjoint system**:
  $$\left(I - \frac{\partial \mathcal{T}_\theta}{\partial g^*}\right)^T \lambda = \frac{\partial \mathcal{L}_{\mathrm{geom}}}{\partial g^*}, \qquad \nabla_\theta \mathcal{L} = \frac{\partial \mathcal{L}_{\mathrm{geom}}}{\partial \theta} + \left(\frac{\partial \mathcal{T}_\theta}{\partial \theta}\right)^T \lambda$$
  This restores $O(1)$ memory with respect to inner iterations, but requires a linear solver (e.g., Conjugate Gradient or Neumann series) on every backward pass.

#### 3. Global Scale Indeterminacy (Runaway Dilation Without Gauge Fixing)
* Without an external ambient ruler ($I_2$ in [src/geometry.py](src/geometry.py)) to set the unit of length, any mutually inverse system has a **projective scale gauge symmetry**: if $(g^*, G^{\Theta*})$ is a solution, then $(\alpha^{-1} g^*, \alpha\, G^{\Theta*})$ is *also* an exact solution for **every** $\alpha > 0$.
* **Training consequence**: Unless you explicitly enforce a normalization constraint at every step (such as $\mathrm{Tr}(G^\Theta) = P$ or $\mathbb{E}_x[\det g(x)] = 1$, as in Tyler's M-estimator / operator Sinkhorn scaling), the inner loop drifts toward $0$ or $\infty$, and absolute volume thresholds ($\det g = 1$) become undefined—only *relative* volume ratios $\det g(x_i) / \det g(x_j)$ remain meaningful.

#### 4. Phase Transitions & Bifurcation Cliffs in the Loss Landscape
* When $\mathcal{T}_\theta = \mathcal{F}_\theta \circ \mathcal{H}_\theta$ is a contraction ($\rho(\partial_g \mathcal{T}_\theta) < 1$), the fixed point $g^*(\theta)$ is unique and smooth in $\theta$.
* However, if the network enters a weight region where Jacobian ranks drop (singular foliations) and $\rho(\partial_g \mathcal{T}_\theta) \to 1$, the Jacobian $(I - \partial_g \mathcal{T}_\theta)$ becomes singular: the fixed point can **fail to exist** or **bifurcate** into multiple competing geometries for the same weights $\theta$, creating discontinuous jumps ("metric cliffs") and exploding implicit gradients during training.

#### 5. Practical Workaround: Two-Timescale Adiabatic Co-Evolution
* In practice, you would avoid solving the inner fixed point to completion at every step by turning the circularity into a **two-timescale dynamical system**—maintaining persistent running estimates $(g_t, G^\Theta_t)$ that update faster than the weights ($\eta \ll \beta_g, \beta_G$):
  $$\begin{aligned}
  g_{t+1} &= \mathrm{Normalize}\!\Big((1 - \beta_g)\,g_t + \beta_g\,\mathcal{F}_{\theta_t}(G^\Theta_t)\Big) \\
  G^\Theta_{t+1} &= (1 - \beta_G)\,G^\Theta_t + \beta_G\,\mathcal{H}_{\theta_t}(g_{t+1}) \\
  \theta_{t+1} &= \theta_t - \eta\,(G^\Theta_{t+1} + \lambda I)^{-1} \nabla_\theta \mathcal{L}_{\mathrm{geom}}(\theta_t; g_{t+1})
  \end{aligned}$$
* By Borkar's two-timescale stochastic approximation principle, $(g_t, G^\Theta_t)$ adiabatically tracks the self-consistent equilibrium $(g^*(\theta_t), G^{\Theta*}(\theta_t))$ at **$O(1)$ per-step cost**—giving **complete background independence** (no arbitrary choice of ambient Euclidean vs. Graph metric on the output space) without needing an inner loop.


---

## Turn 12

### User
update the notebooks so that, at the beginning of each notebook the user can set:

1- what is the toy dataset to use (from scikit learn synthetic datasets)  and with what parameters (noise, number of classes, etc.) ... number of classes should default to 2
2- a sequence of integers specifying the number of layers and number of neurons per layer for the neural network
3- the activation function for the hidden layers, and for the output layer

### Assistant
All seven notebooks and the supporting differential geometry library in [src/geometry.py](src/geometry.py) have been updated, regenerated, and verified.

#### 1. Top-of-Notebook `# USER CONFIGURATION` Block Across All 7 Notebooks
At the very top of Code Cell 1 in every notebook:
- [notebooks/01_baseline_and_pullback_metrics.ipynb](notebooks/01_baseline_and_pullback_metrics.ipynb)
- [notebooks/02_layerwise_curvature_evolution.ipynb](notebooks/02_layerwise_curvature_evolution.ipynb)
- [notebooks/03_parameter_data_manifold_duality.ipynb](notebooks/03_parameter_data_manifold_duality.ipynb)
- [notebooks/04_purely_geometric_curvature_losses.ipynb](notebooks/04_purely_geometric_curvature_losses.ipynb)
- [notebooks/05_geometric_loss_ablations_and_synthesis.ipynb](notebooks/05_geometric_loss_ablations_and_synthesis.ipynb)
- [notebooks/06-coupled-optimization.ipynb](notebooks/06-coupled-optimization.ipynb)
- [notebooks/07-evaluating-efficient-training.ipynb](notebooks/07-evaluating-efficient-training.ipynb)

users can now configure:
```python
# 1. Toy Dataset Configuration (from sklearn.datasets synthetic generators)
#    Options: "moons", "circles", "blobs", "classification", "gaussian_quantiles"
DATASET_NAME = "moons"
N_CLASSES = 2                  # Number of classes (defaults to 2)
DATASET_PARAMS = {
    "n_train": 400,
    "n_test": 200,
    "noise": 0.10,             # Noise std (or flip_y for "classification")
    "seed": 42,
    # Optional dataset-specific parameters: e.g. factor=0.5 ("circles"),
    # cluster_std=0.65 ("blobs"), class_sep=1.25 ("classification")
}

# 2. Neural Network Layer Architecture: [input_dim, *hidden_dims, output_dim]
LAYER_DIMS = [2, 5, 10, 15, 10, 2]

# 3. Activation Functions for Hidden Layers and Output Layer
#    Options: "tanh", "gelu", "silu", "softplus", "elu", "sigmoid", "sin", "relu", "identity"
HIDDEN_ACTIVATION = "tanh"
OUTPUT_ACTIVATION = "identity"
```

#### 2. Supporting Library & Downstream Dynamic Adaptations ([src/geometry.py](src/geometry.py), [scripts/build_all_notebooks.py](scripts/build_all_notebooks.py))
- **Configurable `scikit-learn` Synthetic Datasets (`get_synthetic_dataset`)**: Supports `"moons"`, `"circles"`, `"blobs"`, `"classification"`, and `"gaussian_quantiles"`, defaulting `n_classes = 2` (with multi-class support for $C \ge 2$, including shifted/rotated crescents and concentric rings when `n_classes > 2` on `"moons"` or `"circles"`) plus generator-specific keyword parameters (`factor`, `cluster_std`, `class_sep`, `cov`).
- **Dynamic Layer Depth & Widths (`LAYER_DIMS`)**: Replaced hardcoded layer counts (`5`), parameter counts (`422`), and subplot grids across all 7 notebooks with dynamic queries (`model.num_layers`, `P_total`, and `CE_LAYER_DIMS = [*LAYER_DIMS[:-1], max(LAYER_DIMS[-1], N_CLASSES)]` for Cross-Entropy baselines when `N_CLASSES > LAYER_DIMS[-1]`).
- **Separate Hidden & Output Activations (`HIDDEN_ACTIVATION`, `OUTPUT_ACTIVATION`)**: Extended `MoonsMLP` and `_make_single_point_layer_fn` with `output_activation` (`"identity"`, `"tanh"`, `"gelu"`, `"silu"`, `"softplus"`, `"elu"`, `"sigmoid"`, `"sin"`, `"softmax"`, `"log_softmax"`, `"relu"`), ensuring exact consistency between `model.forward` and `torch.func` pullback metric/curvature operators.
- **Multi-Class Generalization**: Generalized `compute_parameter_fim`, `compute_data_information_matrix`, `curvature_margin_loss`, `volume_contrastive_loss`, `anisotropy_margin_loss`, `scalar_geometric_predict`, `compute_coupled_geometric_metric`, and `woodbury_coupled_ngd_direction` to arbitrary `N_CLASSES >= 2` while preserving exact numerical equivalence for default `N_CLASSES = 2`.


---

## Turn 13

### User
explain how the losses are constructed in the case of multiclass classification and the different geometric losses, and include the explanation into README.md

### Assistant
#### 1. Coordinate Cross-Entropy vs. Multiclass Scalar Geometric Band Partitioning
- **Cross-Entropy Baseline (`CE`)**: Uses $C$ coordinate channels ($z(x) \in \mathbb{R}^{\max(d_L, C)}$, promoted automatically via `CE_LAYER_DIMS` when `N_CLASSES > LAYER_DIMS[-1]`) and classifies via $\hat{y}(x) = \arg\max_c z_c(x)$.
- **Purely Differential-Geometric Losses**: Never inspect output coordinates $z_a(x)$ and do **not** require $C$ output neurons (even a 2D output map $\Phi^{(L)}_\theta: \mathbb{R}^2 \to \mathbb{R}^2$ works for any $C \ge 2$). Instead, each loss computes a **single geometric scalar invariant** $\psi(x; \theta) \in \mathbb{R}$ from the pullback metric $g^{(L)}(x; \theta)$ and partitions $\mathbb{R}$ into **$C$ ordered, margin-separated geometric bands** (`_scalar_geometric_margin_loss_and_weights` and `scalar_geometric_predict` in [src/geometry.py](src/geometry.py)).

With margin $\tau > 0$ (`margin`), sharpness $\beta > 0$ (`beta_hinge = 4.0`), and smooth hinge $\sigma_\beta(u) = \frac{1}{\beta}\log(1 + e^{\beta u})$:
- **Binary ($C = 2$)**: Enforces opposite half-lines $\psi \le -\tau$ ($y = 0$) and $\psi \ge +\tau$ ($y = 1$) via $\ell_y(\psi) = \sigma_\beta(\tau - (2y - 1)\psi)$, predicting $\hat{y}(x) = \mathbb{I}[\psi(x) > 0]$.
- **Multiclass ($C \ge 3$)**: Assigns symmetric target centers $t_c$ spaced by $2\tau$ and midpoint decision thresholds $b_c$:
  $$t_c = 2\tau \left(c - \frac{C - 1}{2}\right), \qquad b_c = \frac{t_c + t_{c+1}}{2} = 2\tau \left(c + \frac{1}{2} - \frac{C - 1}{2}\right)$$
  Outer classes ($y = 0$ and $y = C - 1$) use one-sided rays so stronger separation on the outer ends is free, whereas interior classes ($0 < y < C - 1$) use two-sided deadzone tubes $[t_y - \frac{\tau}{2}, t_y + \frac{\tau}{2}]$ so they cannot encroach on adjacent classes:
  $$\ell_y(\psi) = \begin{cases}
  \sigma_\beta\!\left(\psi - \left(t_0 + \tfrac{\tau}{2}\right)\right), & y = 0 \\[4pt]
  \sigma_\beta\!\left(\left(t_y - \tfrac{\tau}{2}\right) - \psi\right) + \sigma_\beta\!\left(\psi - \left(t_y + \tfrac{\tau}{2}\right)\right), & 0 < y < C - 1 \\[4pt]
  \sigma_\beta\!\left(\left(t_{C-1} - \tfrac{\tau}{2}\right) - \psi\right), & y = C - 1
  \end{cases}$$
  Consecutive acceptance bands are separated by a gap of width $\tau$ centered on $b_c$, and test prediction buckets $\psi(x)$ via $\hat{y}(x) = \sum_{c=0}^{C-2} \mathbb{I}[\psi(x) > b_c]$. For $C = 3$, the three acceptance regimes are $(-\infty, -\frac{3}{2}\tau]$, $[-\frac{\tau}{2}, +\frac{\tau}{2}]$, and $[+\frac{3}{2}\tau, +\infty)$ separated at $-\tau$ and $+\tau$.

#### 2. Construction of the Three Geometric Losses
Let $J(x; \theta) = \partial_x \Phi^{(L)}_\theta(x) \in \mathbb{R}^{d_L \times d_0}$ and $\mathcal{L}_{\mathrm{band}}(\psi, y) = \frac{1}{N}\sum_{n=1}^N \ell_{y_n}(\psi(x_n))$:
1. **Gaussian Curvature Margin Loss (`curvature_margin_loss`, 3rd-order wrt $(\theta, x)$)**:
   - Uses the **graph immersion metric** $\tilde{g} = I_2 + J^T J$ on $(x, \Phi^{(L)}_\theta(x)) \subset \mathbb{R}^{2 + d_L}$ (necessary when $d_L = 2$ because a direct map $\mathbb{R}^2 \to \mathbb{R}^2$ has zero intrinsic Euclidean curvature away from singularities) and saturates curvature via $\psi_{\mathrm{curv}}(x) = K_{\max}\tanh(K(x)/K_{\max})$ where $K = R_{1212}/\det\tilde{g}$.
   - For $C = 3$, Class `0` maps to negatively curved saddles ($K \lesssim -\frac{3}{2}\tau$), Class `1` to a nearly flat parabolic sheet ($|K| \lesssim \frac{\tau}{2}$), and Class `2` to positively curved domes ($K \gtrsim +\frac{3}{2}\tau$).
   - Objective: $\mathcal{L}_{\mathrm{curv}} = \mathcal{L}_{\mathrm{band}}(\psi_{\mathrm{curv}}, y) + \lambda_{\mathrm{vol}}\mathbb{E}[(\log\det\tilde{g})^2] + \lambda_{\mathrm{iso}}\mathbb{E}[\|\tilde{g} - I_2\|_F^2]$, preventing curvature blowup through trivial area collapse ($\det\tilde{g} \to 0$).
2. **Riemannian Volume-Element Contrastive Loss (`volume_contrastive_loss`, 2nd-order wrt $(\theta, x)$)**:
   - Uses the **Euclidean pullback metric** $g = J^T J + \varepsilon I_2$ and log area element $\psi_{\mathrm{vol}}(x) = \log\sqrt{\det g(x)} = \frac{1}{2}\log\det g(x)$.
   - For $C = 3$, separates classes into locally contractive ($\det g < 1$), locally equiareal/volume-preserving ($\det g \approx 1$), and locally expansive ($\det g > 1$) regimes.
   - Objective: $\mathcal{L}_{\mathrm{vol}} = \mathcal{L}_{\mathrm{band}}(\psi_{\mathrm{vol}}, y) + \lambda_{\mathrm{iso}}\mathbb{E}[\psi_{\mathrm{vol}}^2]$, keeping outer rays bounded away from $0$ and $\infty$.
3. **Directional Metric Anisotropy Loss (`anisotropy_margin_loss`, 2nd-order wrt $(\theta, x)$)**:
   - Uses the directional stretching ratio $\psi_{\mathrm{aniso}}(x) = \log(g_{11}(x)/g_{22}(x))$ under the pullback metric $g$.
   - For $C = 3$, separates classes into vertical-dominant stretching ($g_{11} \ll g_{22}$), locally conformal/isotropic representations ($g_{11} \approx g_{22}$), and horizontal-dominant stretching ($g_{11} \gg g_{22}$).
   - Objective: $\mathcal{L}_{\mathrm{aniso}} = \mathcal{L}_{\mathrm{band}}(\psi_{\mathrm{aniso}}, y) + \lambda_{\mathrm{vol}}\mathbb{E}[(\log\det g)^2]$, anchoring $\det g \approx 1$ so polarization arises from area-preserving shear rather than uniform scaling.
4. **Coupled Parameter Metric (`compute_coupled_geometric_metric`)**: Weights $\nabla_\theta \psi(x)$ by the square root of the local second derivative of $\ell_y(\psi)$ across both lower and upper band walls ($t_y \pm \frac{\tau}{2}$) alongside the DeWitt metric on $\mathrm{Sym}^+_2$.

*(Also added this explanation as a dedicated section in [README.md](README.md).)*


---

## Turn 14

### User
ok, take the user configuration out of the beginning of the notebooks into a file named config.yaml, so that each notebook reads it at the beginning.

also, update the sys.path updating in all notebooks in the same way as it is in notebook 01:
`for p in [Path.cwd(), Path.cwd().parent, Path.cwd() / Path("projects") / Path("diffgeom")]:`

### Assistant
1. **Extracted Shared Configuration into [config.yaml](config.yaml)**:
   - `dataset`: `name` (`"moons"`, `"circles"`, `"blobs"`, `"classification"`, `"gaussian_quantiles"`), `n_classes: 2`, and `params` (`n_train`, `n_test`, `noise`, `seed`, plus optional generator-specific parameters).
   - `model`: `layer_dims` (`[2, 5, 10, 15, 10, 2]`), `hidden_activation` (`"tanh"`), and `output_activation` (`"identity"`).

2. **Unified `sys.path` Resolution & `config.yaml` Loading Across All 7 Notebooks ([scripts/build_all_notebooks.py](scripts/build_all_notebooks.py))**:
   Every notebook (`01` through `07`) now starts its first code cell with:
   ```python
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
   ```
   All seven notebooks were rebuilt and executed end-to-end, and [README.md](README.md) was updated to document [config.yaml](config.yaml).
