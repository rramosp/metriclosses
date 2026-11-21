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


---

## Turn 15

### User
in notebook 7, in figure in section "5. High Dimensional Input Scaling..." why are the y-axis referring to SGD (like ms/SGD step or MB?SGD step) ? ... since the graphs include SGD, but also the other methods with geometric losses

### Assistant
In Sections 2–5 of [notebooks/07-evaluating-efficient-training.ipynb](notebooks/07-evaluating-efficient-training.ipynb), **all five curves use `torch.optim.SGD` as the parameter optimizer**—the difference between the curves is only the **loss function** (`Cross-Entropy` vs. the exact/approximated geometric losses), not the optimizer.

Specifically, every curve in Section 5 calls the same `benchmark_sgd_step(loss_fn, mlp_hd)` helper:
```python
sgd = torch.optim.SGD(model.parameters(), lr=1e-4)
...
sgd.zero_grad()
l_run = loss_fn()   # Cross-Entropy OR Exact Autodiff OR FD Stencil OR Tangent Sketch
l_run.backward()
sgd.step()          # Same first-order SGD weight update for every curve
```

### Why it was set up that way (and why the label looks ambiguous)
1. **Controlled benchmark**: Holding the optimizer fixed to vanilla `SGD` across all five methods ensures zero confounding overhead from Adam moment buffers or $P \times P$ Natural Gradient solves (which are benchmarked separately in Section 6), isolating the compute and memory cost of `loss_fn()` + `loss.backward()` as $d_0$ grows from `2` to `128`.
2. **Naming ambiguity**: Because the baseline legend entry is labeled `"SGD + Cross-Entropy"` while the geometric curves omit `"SGD +"` (e.g., `"Exact Metric Autodiff (2nd)"`, `"App 1: Full Coord FD"`), writing `ms / SGD step` and `MB / SGD step` on the y-axes makes `"SGD"` look like it refers to the baseline curve rather than to one `zero_grad() + forward + backward + sgd.step()` iteration shared by all methods.

If you'd like, I can update those axis labels (in Sections 2 and 5 of [notebooks/07-evaluating-efficient-training.ipynb](notebooks/07-evaluating-efficient-training.ipynb)) to `"Step Latency (ms / training step)"` and `"Step Memory (MB / training step)"` so the distinction between the optimizer (`SGD`) and the loss (`Cross-Entropy` vs. geometric) is unambiguous.


---

## Turn 16

### User
ok, great. please add this explanation at the top of section 5 in the notebook

### Assistant
Added the explanation directly to the Section 5 introductory Markdown cell in [notebooks/07-evaluating-efficient-training.ipynb](notebooks/07-evaluating-efficient-training.ipynb) (and kept [scripts/build_all_notebooks.py](scripts/build_all_notebooks.py) synchronized), clarifying why `benchmark_sgd_step` uses `torch.optim.SGD` across all five loss curves so the `ms / SGD step` and `MB / SGD step` axes isolate pure forward + backward operator cost without optimizer state/matrix-solve overhead.


---

## Turn 17

### User
in notebook 7 add an explanation at the beginning of each section explaining in detail what experiment is doing and how to interpret the tables and results

### Assistant
Added detailed `What This Experiment Does` and `How to Interpret the Table & Plots` subsections at the start of **every section (`Sections 1–6`)** in [notebooks/07-evaluating-efficient-training.ipynb](notebooks/07-evaluating-efficient-training.ipynb) (and synchronized [scripts/build_all_notebooks.py](scripts/build_all_notebooks.py)):

1. **Section 1 (`Setup, Offline Jet Precomputation & Numerical Approximation Fidelity`)**: Explains the offline $k$-NN Taylor jet precomputation, the 100-epoch reference MLP manifold setup, how to read the machine-precision Gauss identity check vs. FD/jet error diagnostics, the truncation-vs.-roundoff V-curve over $h \in [10^{-6}, 10^{-1}]$, and the pointwise $\mathrm{arcsinh}(K)$ identity scatter plot.
2. **Section 2 (`Step-Level Compute, Memory & Autograd Graph Benchmark`)**: Explains the 14-configuration `benchmark_sgd_step` suite across `Baseline`, `Volume`, `Anisotropy`, and `Curvature`, every table column (`Order`, `Evals`, `Latency`, `Speedup`, `Memory`, `Mem Save`, `Nodes`), and the three-panel figure (horizontal compute/memory log-bar charts + Compute–Memory Pareto scatter plot with marker shapes and node-scaled bubble sizes).
3. **Section 3 (`End-to-End SGD Training Convergence & Exact Analytical Test Accuracy`)**: Explains the 120-epoch `SGD` training matrix, the strict **train-approximate / evaluate-exact** protocol (`exact_test_accuracy`), the runtime-ratio summary table (`vs SGD+CE Time`), and the difference between per-epoch statistical fidelity (top row) and log wall-clock compute efficiency (bottom row).
4. **Section 4 (`Exact Analytical Decision Landscapes Learned via 1st-Order Approximations`)**: Explains the $2 \times 4$ dense ambient grid evaluation of exact $\log\sqrt{\det g^{(L)}(x)}$ and $\mathrm{arcsinh}(K^{(L)}(x))$, the diverging `RdBu_r` color scale (`< 0` blue vs. `> 0` red), and the solid black `0.0` geometric decision boundary contour.
5. **Section 5 (`High-Dimensional Input Scaling, d_0 = 2 -> 128`)**: Explains the ambient input dimension sweep (`N = 128`), why `ms / SGD step` and `MB / SGD step` hold `torch.optim.SGD` fixed across all five objectives, how to read the `ms / MB` scaling table, and why `Approach 3` (`2N` metric probe and `7N` random 2-plane curvature) stays $O(1)$ in batch multiplicity while coordinate-wise methods grow steeply with $d_0$.
6. **Section 6 (`Matrix-Free Woodbury Coupled Natural Gradient`)**: Explains how `woodbury_coupled_ngd_direction` combines 1st-order 7-point FD stencils on an $S = 16$ subsample ($R = 64$ coupled DeWitt + curvature features) with the Sherman–Morrison–Woodbury identity to avoid $P \times P$ matrix allocation/inversion, and how to interpret the step-cost summary rows and loss/accuracy trajectories against `Uncoupled SGD` and `Dense Coupled NGD`.


---

## Turn 18

### User
do the same for the rest of the notebooks

### Assistant
Added detailed **`What This Experiment Does`** and **`How to Interpret ...`** subsections at the beginning of every section across all six remaining notebooks (preserving all executed outputs and inline figures while keeping [scripts/build_all_notebooks.py](scripts/build_all_notebooks.py) synchronized):

1. **[notebooks/01_baseline_and_pullback_metrics.ipynb](notebooks/01_baseline_and_pullback_metrics.ipynb) (`Sections 1–4`)**:
   - Baseline Cross-Entropy training & checkpoint capture (`t = 0, 25, 80, 250`), $2 \times L$ layerwise volume magnification ($\log_{10}\sqrt{\det g^{(\ell)}}$) and spectral anisotropy ($\log_{10}(\lambda_1/\lambda_2)$) heatmaps + summary table, Tissot indicatrix ellipses illustrating singular foliation compression, and 4-epoch symmetry-breaking progression.
2. **[notebooks/02_layerwise_curvature_evolution.ipynb](notebooks/02_layerwise_curvature_evolution.ipynb) (`Setup + Sections 1–5`)**:
   - Closed-form sphere ($K = 0.25$) and hyperbolic saddle ($K(0) = -3.0$) analytical validation, empirical verification of the 2D-to-2D Euclidean flatness identity ($K \equiv 0$) vs. Graph/Conformal/Penultimate metrics, $1 \times L$ $\mathrm{arcsinh}(K^{(\ell)})$ depth maps, interior-vs.-boundary curvature & Levi-Civita connection $\|\Gamma^{(\ell)}\|_F$ evolution, layerwise log-volume density profiles, and coordinate vs. spectral anisotropy evolution.
3. **[notebooks/03_parameter_data_manifold_duality.ipynb](notebooks/03_parameter_data_manifold_duality.ipynb) (`Setup + Sections 1–4`)**:
   - Co-evolution of the $P \times P$ Parameter FIM spectrum, Effective Rank, and $2 \times 2$ Data Information Matrix trace, input singular foliation tangent `quiver` map (`DIM` kernel eigenvector $v_{\ker}$), Halverson–Ruehle metric/curvature flow sensitivity across stiff vs. null FIM modes, and Euclidean GD vs. Fisher Natural Gradient trajectories.
4. **[notebooks/04_purely_geometric_curvature_losses.ipynb](notebooks/04_purely_geometric_curvature_losses.ipynb) (`Setup + Sections 1–3`)**:
   - End-to-end 3rd-order autodiff curvature-margin training & class curvature bifurcation, 2D $\mathrm{arcsinh}(K^{(L)})$ zero-curvature decision boundary + held-out test curvature histogram, and 3D Monge surface $(x^1, x^2, z^1(x))$ colored by hyperbolic saddle ($K < 0$) vs. elliptic dome ($K > 0$) curvature.
5. **[notebooks/05_geometric_loss_ablations_and_synthesis.ipynb](notebooks/05_geometric_loss_ablations_and_synthesis.ipynb) (`Setup + Sections 1–7`)**:
   - Head-to-head training across Cross-Entropy, Curvature, Volume-Element, and Metric Anisotropy losses, $1 \times 4$ native decision landscapes, layerwise curvature gaps & $P \times P$ parameter metric back-reaction, layerwise log-volume and anisotropy gap tables/profiles, differential-order compute/memory/node benchmarks, and the $3 \times 4$ cross-evaluation selectivity matrix (`★ Target` diagonal vs. off-diagonal controls).
6. **[notebooks/06-coupled-optimization.ipynb](notebooks/06-coupled-optimization.ipynb) (`Setup + Sections 1–5`)**:
   - Machine-precision Gauss identity check & exact $E(2)$ output translation/rotation quotienting (`|v^T G v|` table and initial spectra), 45-step Coupled Geom-NGD vs. uncoupled optimizers (`GD`, `Adam`, `Coord-NGD`), $3 \times 4$ learned geometric landscapes, mechanistic diagnostics (spurious bias gauge drift & unclipped 3rd-order curvature gradient spikes), and the unsupervised `DeWitt-NGD` vs. `Task-Coupled Geom-NGD` ablation.


---

## Turn 19

### User
make this project into a proper uv python project

### Assistant
Converted the repository into a standard `src`-layout `uv` Python package (`diffgeom`):

1. **Packaging, Python Pin & Lockfile**:
   - Added [pyproject.toml](pyproject.toml) (`hatchling` build backend, PEP 621 metadata, runtime dependencies `torch`, `numpy`, `scikit-learn`, `matplotlib`, `pyyaml`, `ipykernel`, dev group `pytest` & `ruff`, and CLI entrypoint `diffgeom-build-notebooks`).
   - Added [.python-version](.python-version) (`3.13`) and generated [uv.lock](uv.lock) + `.venv/` via `uv sync` with `diffgeom` installed editable.
2. **Standard `src/diffgeom/` Package Layout**:
   - Moved library modules into [src/diffgeom/\_\_init\_\_.py](src/diffgeom/__init__.py) and [src/diffgeom/geometry.py](src/diffgeom/geometry.py), and added [src/diffgeom/cli.py](src/diffgeom/cli.py) (`uv run diffgeom-build-notebooks [1..7]`).
   - Updated all seven notebooks (`01`–`07`), [scripts/build_all_notebooks.py](scripts/build_all_notebooks.py), and [scripts/nb_executor.py](scripts/nb_executor.py) to import from `diffgeom.geometry` and target the `diffgeom` kernelspec.
3. **Unit Test Suite**:
   - Added [tests/test_geometry.py](tests/test_geometry.py) covering synthetic dataset generators, sphere/saddle curvature identities, Gauss's equation & $E(2)$ DeWitt nullspace quotienting, custom activations, multiclass geometric losses, and first-order efficient operators (`uv run pytest` passes `5/5` tests).
4. **Documentation**:
   - Updated [README.md](README.md) with the new file layout and `uv sync`, `uv run pytest`, and `uv run diffgeom-build-notebooks` commands.


---

## Turn 20

### User
how do I run a jupyter notebook to serve kernels for the ntoebooks in this project

### Assistant
Added `jupyterlab` and `notebook` to [pyproject.toml](pyproject.toml) (synced into `.venv` and [uv.lock](uv.lock)) and pre-registered the **`Python (diffgeom)`** kernelspec inside `.venv/share/jupyter/kernels/diffgeom` to match the kernelspec of all 7 notebooks.

### 1. Launch Jupyter Notebook (or JupyterLab) from the Project Root
```bash
cd /usr/local/google/home/raulramos/projects/diffgeom

# Classic Notebook UI
uv run jupyter notebook

# Or JupyterLab UI
uv run jupyter lab
```

### 2. Running Over SSH / Headless (or Connecting VS Code / Antigravity / Colab to a Kernel Server)
```bash
uv run jupyter notebook --no-browser --port 8888
```
Then either open via SSH port-forwarding (`ssh -L 8888:127.0.0.1:8888 <host>`) or paste the printed `http://127.0.0.1:8888/?token=...` URL into your IDE's **"Select Kernel -> Existing Jupyter Server"**.

### 3. Selecting `.venv` Directly in VS Code / Antigravity Without Starting a Server
In the top-right **Select Kernel** picker of any notebook, choose **Python Environments -> `.venv (Python 3.13.13)`** (`/usr/local/google/home/raulramos/projects/diffgeom/.venv/bin/python`), or register it globally in `~/.local/share/jupyter/kernels/` once via:
```bash
uv run python -m ipykernel install --user --name diffgeom --display-name "Python (diffgeom)"
```


---

## Turn 21

### User
hos is multiclass classification formulated when using geometric losses ... like in what regions of metric volume, anisotropy or curvature is each class targetted to fall into

### Assistant
Because each purely geometric loss computes a **single scalar differential invariant** $\psi(x; \theta) \in \mathbb{R}$ from the final-layer pullback metric $g^{(L)}(x; \theta)$ (rather than $C$ separate coordinate logits), **$C$-class classification (`n_classes = C`) is formulated by partitioning the 1D geometric spectrum $\mathbb{R}$ into $C$ ordered, margin-separated bands** (`_scalar_geometric_margin_loss_and_weights` and `scalar_geometric_predict` in [src/diffgeom/geometry.py](src/diffgeom/geometry.py)).

---

### 1. General Band Layout on $\psi(x)$ for $C$ Classes

For a chosen margin $\tau > 0$, each class $c \in \{0, 1, \dots, C - 1\}$ is assigned a symmetric target center $t_c$ spaced by $2\tau$, with midpoint decision boundaries $b_c$:
$$t_c = 2\tau\left(c - \frac{C - 1}{2}\right), \qquad b_c = \frac{t_c + t_{c+1}}{2} = 2\tau\left(c + \frac{1}{2} - \frac{C - 1}{2}\right)$$

- **Class `0` (lowest class)**: Targets the **open left ray** $(-\infty,\; t_0 + \tfrac{\tau}{2}]$ (one-sided penalty—more negative is free).
- **Interior classes (`0 < c < C - 1`)**: Target **bounded intervals** $[t_c - \tfrac{\tau}{2},\; t_c + \tfrac{\tau}{2}]$ of width $\tau$ (two-sided soft-tube penalty so they stay inside their assigned band without spilling into Class $c-1$ or Class $c+1$).
- **Class `C - 1` (highest class)**: Targets the **open right ray** $[t_{C-1} - \tfrac{\tau}{2},\; +\infty)$ (one-sided penalty—more positive is free).
- *(For binary `C = 2`, this simplifies to $(-\infty, -\tau]$ vs. $[+\tau, +\infty)$ separated at `0`.)*

Visually along the $\psi(x)$ axis for **`C = 3`** and **`C = 4`**:
```text
C = 3 (centers -2τ, 0, +2τ | decision boundaries at -τ and +τ):
  Class 0 target          gap      Class 1 target     gap          Class 2 target
<=================] . . . | . . . [==============] . . . | . . . [=================>
                -1.5τ    -τ    -0.5τ     0     +0.5τ    +τ     +1.5τ

C = 4 (centers -3τ, -τ, +τ, +3τ | decision boundaries at -2τ, 0, +2τ):
  Class 0      gap     Class 1     gap     Class 2     gap       Class 3
<=========] . . | . . [=======] . . | . . [=======] . . | . . [=========>
        -2.5τ  -2τ  -1.5τ   -0.5τ   0   +0.5τ   +1.5τ  +2τ  +2.5τ
```

---

### 2. Concrete Geometric Regimes Targeted by Each Loss

Below is what those $\psi$-bands mean geometrically for **`C = 3` classes** (and **`C = 4`**) using the notebook margins:

| Geometric Loss & Scalar $\psi(x)$ | Class `0` Target Region (`y = 0`) | Class `1` Target Region (`y = 1`, interior) | Class `2` Target Region (`y = 2`) | What Happens for `C = 4` Classes |
| :--- | :--- | :--- | :--- | :--- |
| **1. Volume-Element** (`margin τ = 0.60`)<br>$\psi_{\mathrm{vol}} = \log\sqrt{\det g(x)}$<br>*(Area factor $\sqrt{\det g} = e^{\psi_{\mathrm{vol}}}$)* | **Strongly contractive basin**<br>$\psi_{\mathrm{vol}} \le -0.90$<br>Local area shrunk to $\sqrt{\det g} \le 0.41\times$ *(predicted if $\le 0.55\times$)* | **Equiareal / area-preserving sheet**<br>$\psi_{\mathrm{vol}} \in [-0.30, +0.30]$<br>Local area near unity: $\sqrt{\det g} \in [0.74\times, 1.35\times]$ | **Strongly expansive region**<br>$\psi_{\mathrm{vol}} \ge +0.90$<br>Local area stretched to $\sqrt{\det g} \ge 2.46\times$ *(predicted if $> 1.82\times$)* | Splits around $\det g = 1$ into: **strong contraction** ($\le -1.5$), **mild contraction** ($[-0.9, -0.3]$), **mild expansion** ($[+0.3, +0.9]$), **strong expansion** ($\ge +1.5$). |
| **2. Metric Anisotropy** (`margin τ = 0.45`)<br>$\psi_{\mathrm{aniso}} = \log\frac{g_{11}(x)}{g_{22}(x)}$<br>*(With $\det g \approx 1$ area anchor)* | **Vertical-dominant shear**<br>$\psi_{\mathrm{aniso}} \le -0.675$<br>$g_{11}/g_{22} \le 0.51$ ($x^2$ stretched, $x^1$ compressed) | **Conformal / isotropic band**<br>$\psi_{\mathrm{aniso}} \in [-0.225, +0.225]$<br>$g_{11} \approx g_{22}$ ($g_{11}/g_{22} \in [0.80, 1.25]$, circles stay near-circles) | **Horizontal-dominant shear**<br>$\psi_{\mathrm{aniso}} \ge +0.675$<br>$g_{11}/g_{22} \ge 1.96$ ($x^1$ stretched, $x^2$ compressed) | Splits around $g_{11}=g_{22}$ into: **strong vertical stretch**, **mild vertical stretch**, **mild horizontal stretch**, **strong horizontal stretch**. |
| **3. Gaussian Curvature** (`margin τ = 0.40`)<br>$\psi_{\mathrm{curv}} = K_{\max}\tanh\!\left(\frac{K(x)}{K_{\max}}\right)$<br>*($\psi_{\mathrm{curv}} \approx K$ when $|K| \ll K_{\max}$)* | **Hyperbolic saddle**<br>$K(x) \lesssim -0.60$<br>Principal curvatures $\kappa_1, \kappa_2$ have opposite signs | **Flat / parabolic sheet**<br>$K(x) \in [-0.20, +0.20]$<br>Nearly zero intrinsic curvature ($K \approx 0$) | **Elliptic dome / bowl**<br>$K(x) \gtrsim +0.60$<br>Principal curvatures $\kappa_1, \kappa_2$ share the same sign | Splits around $K = 0$ into: **sharp saddles** ($\lesssim -1.0$), **gentle saddles** ($[-0.6, -0.2]$), **gentle domes** ($[+0.2, +0.6]$), **sharp domes** ($\gtrsim +1.0$). |

---

### 3. Geometric / Topological Consequence in Input Space
Because $\psi(x; \theta)$ is a **smooth, continuous scalar field** over $x \in \mathbb{R}^2$ (like altitude on a topographic map), any continuous path in input space from a Class `0` region ($\psi \le -1.5\tau$) to a Class `2` region ($\psi \ge +1.5\tau$) must pass through the intermediate Class `1` interval $(-\tau, +\tau)$ by the Intermediate Value Theorem. When Class `0` and Class `2` clusters are adjacent in input space without Class `1` points between them, the network learns a narrow, steep transition crease between them while placing broad, flat plateaus at the target levels $t_c$ over the actual data clusters.


---

## Turn 22

### User
hos is multiclass classification formulated when using geometric losses ... like in what regions of metric volume, anisotropy or curvature is each class targetted to fall into

*(Follow-up)*:
this seems to induce an artificial ordering in the data space, cant we better profit from the parameter and data space coupling to define a more adequate loss without these artificial artifacts?

### Assistant
Yes—your intuition is spot on. Mapping $C$ unordered categorical classes onto a 1D scalar invariant $\psi(x) \in \mathbb{R}$ (`-2τ, 0, +2τ`) breaks **class permutation symmetry ($S_C$)** and creates two concrete artifacts for $C \ge 3$:
1. **Artificial Metric Spacing**: Class `0` and Class `2` are forced twice as far apart ($4\tau$) as Class `0` and Class `1` ($2\tau$).
2. **Intermediate-Value Theorem (IVT) Phantom Fringes**: Because $\psi(x; \theta)$ is continuous on $\mathbb{R}^2$, any path crossing directly from a Class `0` cluster ($\psi \le -1.5\tau$) to a Class `2` cluster ($\psi \ge +1.5\tau$) *must* pass through $\psi \in (-\tau, +\tau)$—creating a narrow phantom strip misclassified as Class `1` along the `0–2` border.

By exploiting the **coupled geometry between the parameter manifold $(\Theta, G^\Theta)$ and the data manifold $(\mathcal{M}, g)$**, we can eliminate 1D ordering completely. Here are the **three most principled $S_C$-symmetric geometric loss constructions**, ordered by how directly they leverage the $\Theta \leftrightarrow \mathcal{M}$ coupling:

---

### 1. Coupled Parameter–Data Subspace Orthogonality (`DeWitt Tangent Kernel` on $\Theta \leftrightarrow \mathcal{M}$)
This profits **directly** from the coupled metric structure in [src/diffgeom/geometry.py](src/diffgeom/geometry.py). Recall that our global DeWitt parameter metric is the expectation of a **pointwise pullback operator** $G^\Theta(x; \theta)$ at each data point $x$:
$$G^{\Theta,\mathrm{DeWitt}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\big[G^\Theta(x; \theta)\big], \qquad G^\Theta(x; \theta) = U(x; \theta)^T U(x; \theta)$$
where $U(x; \theta) \in \mathbb{R}^{3 \times P}$ stacks the 3 independent components of the whitened metric derivative $g(x)^{-1/2}\, \partial_\theta g(x)\, g(x)^{-1/2}$ (or $4 \times P$ when including $\nabla_\theta K(x)$).

* **Key Geometric Idea**: Instead of assigning classes to numbers on $\mathbb{R}$, we assign classes to **mutually $G^\Theta$-orthogonal tangent subspaces** $\mathcal{V}_0 \perp_{G^\Theta} \mathcal{V}_1 \perp_{G^\Theta} \dots \perp_{G^\Theta} \mathcal{V}_{C-1}$ inside the high-dimensional parameter tangent space $T_\theta \Theta$ ($\dim \Theta = P \gg C$)!
* **Coupled Loss Construction**: Define the normalized Riemannian subspace overlap (Grassmannian cosine similarity on $(\Theta, G^\Theta)$) between how $\theta$ deforms the data metric at $x_i$ vs. $x_j$:
  $$\mathcal{K}_{\Theta \leftrightarrow \mathcal{M}}(x_i, x_j; \theta) = \frac{\left\| U(x_i; \theta)\,\big(G^{\Theta,\mathrm{DeWitt}}(\theta) + \lambda I_P\big)^{-1} U(x_j; \theta)^T \right\|_F^2}{\left\| U(x_i)\, (G^\Theta + \lambda I)^{-1} U(x_i)^T \right\|_F \cdot \left\| U(x_j)\, (G^\Theta + \lambda I)^{-1} U(x_j)^T \right\|_F} \in [0, 1]$$
  $$\mathcal{L}_{\mathrm{coupled\text{-}ortho}}(\theta) = \mathbb{E}_{y_i = y_j}\!\big[1 - \mathcal{K}_{\Theta \leftrightarrow \mathcal{M}}(x_i, x_j)\big] + \mathbb{E}_{y_i \neq y_j}\!\big[\mathcal{K}_{\Theta \leftrightarrow \mathcal{M}}(x_i, x_j)^2\big] + \lambda_{\mathrm{vol}}\,\mathbb{E}_x\!\big[(\log\det g(x))^2\big]$$
* **Why This Eliminates Every Artifact**:
  1. **Exact $S_C$ Symmetry (scales to any $C \ll P$)**: In $\mathbb{R}^P$, $C$ subspaces can all be mutually $90^\circ$ orthogonal—every pair of classes $(c, c')$ is equidistant, even when the network output is just 2D ($d_L = 2$)!
  2. **Zero IVT Phantom Fringes**: Moving in data space from Class `0` ($\mathcal{V}_0$) to Class `2` ($\mathcal{V}_2$) rotates $U(x)$ inside $\mathrm{span}(\mathcal{V}_0, \mathcal{V}_2)$, remaining **strictly orthogonal to $\mathcal{V}_1$** across the entire boundary.
  3. **Exact $E(d_L)$ Gauge Invariance + Non-Interfering Optimization**: Built purely from $g^{-1/2}\partial_\theta g\, g^{-1/2}$, it quotients out rigid output motions $z \mapsto Qz + b$ AND block-diagonalizes $G^\Theta(\theta) \approx \bigoplus_{c=0}^{C-1} G^\Theta_c$, so Natural Gradient updates to Class $c$'s geometry do not interfere with Class $c'$!
  4. **Purely Geometric Inference**: Classify $x^*$ by maximum subspace alignment $\hat{y}(x^*) = \arg\max_c \mathbb{E}_{x \in \mathcal{D}_c}[\mathcal{K}_{\Theta \leftrightarrow \mathcal{M}}(x^*, x)]$.

---

### 2. Equidistant Geodesic Simplex on the Metric Cone $(\mathrm{Sym}^+_d, d_{\mathrm{AI}})$ (Exact Dual of $G^{\Theta,\mathrm{DeWitt}}$)
Even without pairwise comparisons, collapsing $g(x; \theta) \in \mathrm{Sym}^+_2$ to a 1D scalar $\psi(x)$ threw away 2 of its 3 degrees of freedom!
* The space of $2 \times 2$ SPD metrics $\mathrm{Sym}^+_2$ equipped with the affine-invariant metric is isometric to $\mathbb{R}^+ \times \mathbb{H}^2$, where the unit-determinant slice ($\det g = 1$) is the **2D hyperbolic Poincaré disk $\mathbb{H}^2$** (parameterized by Tissot ellipse eccentricity and orientation angle).
* For $C = 3$, place three target metrics $\{G^*_0, G^*_1, G^*_2\} \subset \mathrm{Sym}^+_2$ at the vertices of an **equilateral hyperbolic triangle**—identical area ($\det G^*_c = 1$) and identical anisotropy ($e^{2\tau}$), rotated by $\theta_c = 0^\circ, 60^\circ, 120^\circ$ (since ellipse axes are $\pi$-periodic, $0^\circ, 60^\circ, 120^\circ$ are mutually equidistant on $\mathbb{RP}^1$!):
  $$G^*_c = R(\theta_c)\begin{pmatrix} e^\tau & 0 \\ 0 & e^{-\tau} \end{pmatrix} R(\theta_c)^T, \qquad \mathcal{L}_{\mathrm{DeWitt\text{-}simplex}}(\theta) = \frac{1}{N}\sum_{n=1}^N \left\|\log\!\Big((G^*_{y_n})^{-1/2}\, g(x_n; \theta)\, (G^*_{y_n})^{-1/2}\Big)\right\|_F^2$$
* **Direct $\Theta \leftrightarrow \mathcal{M}$ Coupling**: The Generalized Gauss–Newton pullback of $\mathcal{L}_{\mathrm{DeWitt\text{-}simplex}}$ onto parameter space $\Theta$ **is literally the coupled DeWitt metric $G^{\Theta,\mathrm{DeWitt}}(\theta)$**! *(For $C > 3$, pulling back onto a width-$d$ representation gives $\mathrm{Sym}^+_d$ of dimension $d(d+1)/2$, hosting an equidistant $C$-simplex whenever $d(d+1)/2 \ge C$.)*

---

### 3. Relational Riemannian Metric-Barrier (Geodesic Contrastive) Loss on $\mathcal{M}$
Instead of giving different classes *different* metric signatures, give **every class the exact same signature**—a locally flat, contracted geodesic basin ($\det g \ll 1$, $K \approx 0$)—separated from all other classes by **high-metric / high-curvature Riemannian walls**:
$$\mathcal{L}_{\mathrm{barrier}}(\theta) = \mathbb{E}_{y_i = y_j}\!\Big[\Delta x_{ij}^T\, g(\bar{x}_{ij}; \theta)\, \Delta x_{ij}\Big] + \mathbb{E}_{y_i \neq y_j}\!\Big[\sigma_\beta\!\Big(\tau - \sqrt{\Delta x_{ij}^T\, g(\bar{x}_{ij}; \theta)\, \Delta x_{ij}}\Big)\Big]$$
where $\Delta x_{ij} = x_j - x_i$ and $\bar{x}_{ij} = \frac{1}{2}(x_i + x_j)$ (or midpoint quadrature along segment $\overline{x_i x_j}$).
* **Result**: Completely invariant to label permutations $S_C$ for **any** $C$, directly implements the singular-foliation geometry observed in Cross-Entropy networks (**Zavatone-Veth et al., 2023; Benfenati & Marta, 2023**) without using logits, and classifies test points via **Riemannian geodesic k-NN / basin distance** $d_g(x^*, x_i)$!


---

## Turn 23

### User
before implementing that, are these methods going to induce a penalty on computation time or memory?

### Assistant
They differ **dramatically** in compute and memory cost—one of them (**Method 1**) carries a heavy higher-order penalty, whereas **Method 2** (and **Method 3** with endpoint quadrature) has **essentially zero penalty (`~1.0x`)** relative to the current Volume and Anisotropy losses.

Here is the exact breakdown of why:

---

### Side-by-Side Compute & Memory Comparison

| Method | Autodiff Order (Exact / w/ FD Stencil) | Step Time vs. Current Volume/Aniso Loss | Memory vs. Current Volume/Aniso Loss | Standalone $O(1)$ Test Inference? | Why |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Current 1D Band Loss** *(Baseline)* | `2nd` (`3rd` for $K$) / `1st` | `1.0x` (~`4 ms` exact, ~`2 ms` FD) | `1.0x` (~`3.5 MB` exact, ~`1.8 MB` FD) | Yes | Evaluates $g(x) \in \mathbb{R}^{N \times 2 \times 2}$ once per step. |
| **1. Coupled Subspace Orthogonality** ($\Theta \leftrightarrow \mathcal{M}$) | **`3rd` (`4th` for $K$) / `2nd`** | **`~10x–30x` slower** *(`~4x–8x` if sketched)* | **`~10x–25x` higher** *($O(NP)$ active graph)* | Needs class projectors $\Pi_c$ | **Heavy**: puts $\partial_\theta g(x; \theta)$ *inside* the forward loss, forcing `backward()` to take **2nd derivatives in $\theta$** ($\partial_\theta^2 g$) and store $(3N \times P)$ parameter Jacobians with autograd history. |
| **2. DeWitt Geodesic Simplex on $\mathrm{Sym}^+_2$** | **`2nd` / `1st`** | **`~1.00x` (Zero penalty!)** | **`~1.00x` (Zero penalty!)** | **Yes** ($\arg\min_c d_{\mathrm{AI}}(g(x^*), G^*_c)$) | **Zero overhead**: uses the *exact same* $g(x) \in \mathbb{R}^{N \times 2 \times 2}$ tensor plus closed-form $2 \times 2$ eigenvalues against tiny constant $2 \times 2$ prototypes $(G^*_c)^{-1/2}$. |
| **3. Geodesic Metric-Barrier Loss** | **`2nd` / `1st`** | **`~1.05x–1.20x`** *(endpoint $k$-NN)*<br>**`~100x`** *(naive all-midpoints)* | **`~1.05x–1.15x`** *(endpoint $k$-NN)* | Needs training anchors ($k$-NN in $d_g$) | Naive midpoint evaluation runs the MLP on $\binom{N}{2}$ pairs (`~100x` cost), **but** trapezoidal endpoint quadrature $\frac{1}{2}\Delta x^T(g(x_i)+g(x_j))\Delta x$ on $k$-NN pairs reuses the $1N$ metrics $g(x_i)$ with `< 0.2 ms` overhead. |

---

### Detailed Takeaways on Where Penalties Arise (and How to Avoid Them)

1. **Why Method 1 (`Coupled Subspace Orthogonality`) Is Expensive**:
   * Because the loss itself depends on the parameter Jacobian $U(x_i; \theta) \propto \partial_\theta g(x_i; \theta)$, calling `loss.backward()` differentiates with respect to $\theta$ **twice** ($\nabla_\theta U \sim \partial_\theta^2 g(x; \theta)$), analogous to differentiating through a Neural Tangent Kernel or MAML inner step.
   * Even if we replace spatial autodiff with 1st-order FD stencils from [notebooks/07-evaluating-efficient-training.ipynb](notebooks/07-evaluating-efficient-training.ipynb) and use Woodbury for $(G^\Theta + \lambda I)^{-1}$, PyTorch still has to keep a $(3N \times P)$ parameter-Jacobian graph in memory to backpropagate through $U(x_i; \theta)$.

2. **Why Method 2 (`DeWitt Geodesic Simplex on Sym^+_2`) Gets the Coupling for Free (`1.00x` Time & Memory)**:
   * Instead of putting $\partial_\theta g$ *inside* the forward loss, Method 2 keeps the forward loss purely on the 3-dimensional Riemannian manifold of $2 \times 2$ data metrics $(\mathrm{Sym}^+_2, d_{\mathrm{AI}})$:
     $$M_n = (G^*_{y_n})^{-1/2}\, g(x_n; \theta)\, (G^*_{y_n})^{-1/2}, \qquad d_{\mathrm{AI}}(g(x_n), G^*_{y_n})^2 = (\log \lambda_1(M_n))^2 + (\log \lambda_2(M_n))^2$$
   * For $2 \times 2$ matrices, $\lambda_{1,2}(M_n) = \frac{1}{2}\big(\mathrm{Tr} M_n \pm \sqrt{(\mathrm{Tr} M_n)^2 - 4\det M_n}\big)$ is **five arithmetic operations** on the exact same $g(x_n)$ tensor we already compute—so it works seamlessly with `SGD` + Notebook 07's `3-pt FD` or `1N k-NN` approximations at **~`1.5–2 ms/step` and ~`1.5 MB/step`**!
   * Yet when paired with Coupled Natural Gradient (or Notebook 07's fast `O(S^2 P)` Woodbury Coupled NGD), the Gauss–Newton pullback of $d_{\mathrm{AI}}^2$ onto parameter space $\Theta$ **is still the exact coupled DeWitt metric $G^{\Theta,\mathrm{DeWitt}}(\theta)$**—giving full parameter–data metric coupling without increasing derivative order!
   * **Class capacity**: `Sym^+_2` has dimension `3`, so up to **`C = 4` classes** fit as an exact equidistant regular simplex in a single 2D metric (`C = 3` on the equiareal circle $\det G^*_c = 1$ at $0^\circ, 60^\circ, 120^\circ$; `C = 4` as a regular 3D hyperbolic tetrahedron combining volume + anisotropy). For `C >= 5`, using $k = \lceil (C-1)/3 \rceil$ 2D output heads gives $(\mathrm{Sym}^+_2)^k$ with cost growing only in the tiny final linear layer.

3. **Why Method 3 (`Geodesic Metric-Barrier`) Is Cheap Only with Endpoint Quadrature**:
   * Evaluating $g(\cdot)$ at pair midpoints $\frac{1}{2}(x_i + x_j)$ blows up batch size from $N$ to $O(N^2)$ (or $O(kN)$).
   * Using **endpoint quadrature** $E_g(\gamma_{ij}) \approx \frac{1}{2}\Delta x_{ij}^T (g(x_i) + g(x_j))\Delta x_{ij}$ on precomputed $k$-NN pairs reuses the single $N$-point metric batch $g(x_1), \dots, g(x_N)$, keeping training cost within **`~5%–20%`** of Method 2.


---

## Turn 24

### User
ok, then implement this with the new notebook that you suggested, but add a section where you compare compute times and memory consumption with the previous methods (SGD + previous geometric losses exact + their approximations)

### Assistant
Implemented all three $S_C$-symmetric parameter–data coupled multiclass formulations in [src/diffgeom/geometry.py](src/diffgeom/geometry.py), added unit tests in [tests/test_geometry.py](tests/test_geometry.py) (`uv run pytest`: `6/6` passed), and built/executed [notebooks/08-symmetric-coupled-multiclass.ipynb](notebooks/08-symmetric-coupled-multiclass.ipynb) (registered in [scripts/build_all_notebooks.py](scripts/build_all_notebooks.py) and [src/diffgeom/cli.py](src/diffgeom/cli.py)).

### Key Results from [notebooks/08-symmetric-coupled-multiclass.ipynb](notebooks/08-symmetric-coupled-multiclass.ipynb)

1. **Section 1 — Elimination of 1D Ordering & IVT Phantom Boundaries**:
   - **Exact Geodesic Simplex Equidistance**: Pairwise DeWitt geodesic distances $d_{\mathrm{AI}}(G^*_a, G^*_b)$ have `1.33e-15` spread across both $C = 3$ ($0^\circ, 60^\circ, 120^\circ$ Tissot ellipses) and $C = 4$ (regular 3D Riemannian tetrahedron in $\mathrm{Sym}^+_2 \cong \mathbb{R}_{\mathrm{vol}} \times \mathbb{H}^2$).
   - **$S_3$ Permutation Spread (all $3! = 6$ relabelings)**: `1D Volume Band` (`1.77e-01`) and `1D Anisotropy Band` (`1.83e-01`) vary under relabeling, whereas `DeWitt Geodesic Simplex` (`4.44e-16`), `Geodesic Metric-Barrier` (`0.00e+00`), and `Coupled Subspace Ortho` (`0.00e+00`) are strictly permutation-invariant.
   - **IVT Phantom Strip**: Along the direct `Class 0 <-> Class 2` segment on 3-class triple-junction blobs, `1D Volume Band` misclassifies **`16.0%`** of the border as phantom `Class 1`, whereas `DeWitt Geodesic Simplex` reduces phantom intrusion to **`0.0%`**.

2. **Section 2 — Multiclass Accuracy Across 3-Class, 4-Class & `config.yaml` (`circles`, `noise=0.2`, `[2, 10, 2]`)**:
   - **3-Class Triple-Junction Blobs**: `DeWitt Simplex (SGD+FD)` (`98.33%`), `DeWitt Simplex (Coupled NGD)` (`98.33%`), and `Geodesic Barrier` (`98.33%`) match `SGD + Cross-Entropy` (`98.33%`); `Coupled Subspace Ortho` reaches `97.50%`.
   - **4-Class Regular Tetrahedron (`d_L = 2` output layer only!)**: `DeWitt Simplex (SGD+FD)` (`95.83%`) and `Coupled Subspace Ortho` (`97.50%`) surpass both `1D Volume Band` (`88.33%`) and 4-logit `SGD + Cross-Entropy` (`95.00%`).
   - **3-Class Noisy Concentric Circles (`config.yaml`)**: `DeWitt Simplex (SGD+FD)` reaches **`85.00%`** vs. `75.00%` for `1D Volume Band` (matching `85.83%` for `SGD + Cross-Entropy`).

3. **Section 3 — 16-Method Compute & Memory Benchmark (vs. `SGD + CE`, Previous Exact & Previous Approximations)**:
   - `SGD + Cross-Entropy (1st, 1N)`: `2.04 ms/step`, `0.18 MB`
   - `Prev Vol Band (App1: 3-pt FD, 1st)`: `2.06 ms/step` (`1.00x`), `0.98 MB`
   - `New DeWitt Simplex (3-pt FD, 1st)`: **`2.06 ms/step` (`1.00x`)**, `2.44 MB`
   - `New DeWitt Simplex (1N k-NN, 1st)`: **`1.99 ms/step` (`0.97x`)**, `2.25 MB`
   - `New Geodesic Barrier (3-pt FD, 1st)`: **`2.18 ms/step` (`1.06x`)**, `2.87 MB`
   - `New Coupled Subspace Ortho (FD 2nd-θ / Exact 3rd)`: `15.93 ms` / `14.69 ms` (`~7.1x–7.7x`) and `56.7–64.5 MB` (comparable to exact 3rd-order Levi-Civita curvature at `18.40 ms` / `69.64 MB`).

4. **Section 4 — Internal Geometric Visualizations**:
   - Visualizes the learned $0^\circ / 60^\circ / 120^\circ$ Tissot ellipses (`DeWitt Simplex`), the symmetric Y-junction high-volume barrier wall (`Geodesic Barrier`), and the $3 \times 3$ block-diagonal Grassmannian overlap matrix $\mathcal{K}_{\Theta \leftrightarrow \mathcal{M}}(x_i, x_j)$ (`Coupled Subspace Ortho`).


---

## Turn 25

### User
add a section with two experiments in this last notebook measuring computation time and memory consumption 1) as the number of dimensions, 2) as the number of classes increases

### Assistant
Added **Section 5** (`## 5. Scaling Compute Time & Memory Consumption: (1) Input Dimension d_0 and (2) Number of Classes C`) with detailed explanatory headers, two empirical scaling tables, and a $2 \times 2$ log–log scaling plot at the end of [notebooks/08-symmetric-coupled-multiclass.ipynb](notebooks/08-symmetric-coupled-multiclass.ipynb) (generated via [scripts/build_all_notebooks.py](scripts/build_all_notebooks.py)), along with two supporting helpers in [src/diffgeom/geometry.py](src/diffgeom/geometry.py):
- `fibered_dewitt_simplex_loss`: Exact $S_C$-equidistant product-manifold $(\mathrm{Sym}^+_2)^C$ DeWitt simplex loss ($d_L = 2C$, $E(2)^C$-invariant) supporting arbitrary $C \ge 2$ without geometric crowding in $O(NC)$ operations.
- `return_ambient: bool = True` in `compute_sketched_pullback_metric` (and `plane_sketch: bool = False` in `compute_coupled_subspace_factors` / `coupled_subspace_orthogonality_loss`) so 2-probe tangent-plane sketches avoid allocating $O(N d_0^2)$ ambient metric tensors when $d_0 \gg 2$.

---

### Summary of Measured Scaling Results (Section 5 of `08-symmetric-coupled-multiclass.ipynb`)

#### 1. Experiment 5.1 — Scaling vs. Ambient Input Dimension (`d_0 = 2, 4, 8, 16, 32, 64, 128`, `C = 3`, `N = 96`)

| `d_0` | `SGD + Cross-Entropy` (`ms` / `MB`) | `Prev Volume Band` (`Exact jacrev`) | `DeWitt Simplex` (`Coord FD, (d_0+1)N`) | `DeWitt Simplex` (`2-Probe Sketch, 3N`) | `Geodesic Barrier` (`Coord FD, (d_0+1)N`) | `Coupled Subspace` (`2-Probe, 2nd-θ`) |
| ---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `2` | `7.53 / 0.19` | `9.51 / 1.18` | `1.86 / 1.32` | **`1.80 / 1.32`** | `1.61 / 1.25` | `45.96 / 526.6` |
| `8` | `10.96 / 0.19` | `10.08 / 1.75` | `2.32 / 3.04` | **`2.07 / 1.34`** | `2.72 / 9.20` | `50.80 / 675.9` |
| `32` | `7.90 / 0.20` | `10.12 / 9.32` | `3.35 / 15.14` | **`2.47 / 1.44`** | `8.47 / 118.06` | `124.85 / 1305.4` |
| `64` | `7.47 / 0.21` | `10.31 / 32.59` | `4.91 / 44.49` | **`2.00 / 1.56`** | `23.51 / 454.94` | `278.93 / 2225.3` |
| `128` | `8.78 / 0.24` | `12.00 / 124.32` | `8.04 / 148.52` | **`3.04 / 1.83`** | `210.59 / 1786.06` | `1022.07 / 4341.0` |

* **Takeaway**: Coordinate-wise pullback metrics (`Exact jacrev`, `Coord FD`) grow as $O(d_0)$–$O(d_0^2)$ in memory (`124–149 MB` at `d_0 = 128`, and `1786 MB` when contracted across $k$-NN pairs in `Geodesic Barrier`), while `Coupled Subspace` grows steepest (`1022 ms / 4341 MB` at `d_0 = 128`) because parameter count $P(d_0)$ grows with $d_0$ inside its 2nd-order-in-$\theta$ graph. By contrast, **`DeWitt Simplex (2-Probe Sketch, 3N)` stays nearly $O(1)$ in both latency (`1.80 -> 3.04 ms`) and memory (`1.32 -> 1.83 MB`)** across the entire $64\times$ jump from `d_0 = 2` to `d_0 = 128`.

---

#### 2. Experiment 5.2 — Scaling vs. Number of Classes (`C = 2, 3, 4, 8, 16, 32, 64`, `d_0 = 2`, `N = 192`)

| `C` | `SGD + CE` (`d_L = C`) | `Prev 1D Vol Band` (`d_L = 2`) | `DeWitt Single Sym^+_2` (`d_L = 2`) | `DeWitt Fibered (Sym^+_2)^C` (`d_L = 2C`) | `Geodesic Barrier` (`d_L = 2`) | `Coupled Subspace` (`d_L = 2`) |
| ---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `2` | `4.13 / 0.13` | `1.63 / 0.71` | `2.18 / 1.49` | `2.59 / 2.76` | **`2.24 / 1.65`** | `12.41 / 53.8` |
| `4` | `3.97 / 0.16` | `1.53 / 0.79` | `1.79 / 2.41` | `2.45 / 5.18` | **`2.54 / 1.65`** | `12.60 / 54.4` |
| `16` | `3.91 / 0.34` | `1.55 / 0.79` | `4.29 / 7.90` | `3.96 / 19.71` | **`2.05 / 1.65`** | `12.31 / 57.9` |
| `32` | `3.87 / 0.58` | `1.68 / 0.79` | `2.91 / 15.26` | `5.32 / 39.08` | **`1.84 / 1.65`** | `12.74 / 62.5` |
| `64` | `4.00 / 1.05` | `1.58 / 0.79` | `3.61 / 29.91` | **`6.30 / 77.82`** | **`1.95 / 1.65`** | `19.62 / 71.7` |

* **Takeaway**:
  - **`Geodesic Barrier` (`d_L = 2`)** is strictly **$O(1)$ in both latency (`~1.9–2.5 ms`) and memory (`1.65 MB` constant)** from `C = 2` to `C = 64` while remaining 100% $S_C$-symmetric with zero crowding.
  - **`DeWitt Single Sym^+_2` (`d_L = 2`)** keeps the network $O(1)$ in $C$ (`2.18 -> 3.61 ms`, `1.49 -> 29.91 MB` from $(N, C)$ margin distances), though $\dim(\mathrm{Sym}^+_2) = 3$ only hosts up to $C = 4$ *mutually equidistant* vertices before prototypes crowd on $\mathbb{H}^2$.
  - **`DeWitt Fibered (Sym^+_2)^C` (`d_L = 2C`)** resolves crowding for arbitrary $C > 4$ with exact $S_C$ equidistance and standalone prototype inference, scaling gently and linearly $O(C)$ in its final output layer (`2.59 -> 6.30 ms/step`, `2.76 -> 77.82 MB/step` at `C = 64`).


---

## Turn 26

### User
ok. i need the new methods still to scale much better when the number of dimensions and classes grows, specially in memory usage, please investigate alternatives

### Assistant
### 1. Root-Cause Memory Diagnosis (Why Each Method Grew in Section 5)

Profiling the autograd allocations in [notebooks/08-symmetric-coupled-multiclass.ipynb](notebooks/08-symmetric-coupled-multiclass.ipynb) pinpoints four exact tensor bottlenecks:

1. **Materializing & Pair-Indexing $(d_0 \times d_0)$ Metrics (`Geodesic Barrier`: `1,786 MB` at `d_0 = 128`)**:
   - Evaluating full coordinate FD runs $(d_0 + 1)N$ forward passes (`12,384` samples at $d_0 = 128$), and `0.5 * (g[src] + g[dst])` copies a $(d_0 \times d_0)$ matrix across $P_{\mathrm{pairs}} \approx 14N$ edges in the autograd graph ($O(P_{\mathrm{pairs}} d_0^2)$ memory).
2. **One-Hot Fiber Coding & `(N, C, 2, 2)` 3-Way `einsum` (`DeWitt Simplex`: `77.8 MB` for `(Sym^+_2)^C` and `29.9 MB` for `Sym^+_2` at `C = 64`)**:
   - Assigning a dedicated 2D output fiber to every class (`d_L = 2C`) scales final-layer activations and fiber metrics linearly as $O(NC)$.
   - Even for a single fiber (`d_L = 2`), `einsum("cij,njk,ckl->ncil", ...)` materializes `(N, C, 2, 2)` intermediate buffers to compute eigenvalues against all $C$ prototypes.
3. **$P$-Tangent `jacfwd` Inside a 2nd-Order Autograd Graph (`Coupled Subspace`: `4,341 MB` at `d_0 = 128`, `71.7 MB` at `C = 64`)**:
   - Calling `backward()` through `jacfwd(sym3_features)(params)` holds $P(d_0)$ simultaneous forward-over-reverse autograd tapes in RAM ($P = 2,402$ at $d_0 = 128$), builds a dense $P \times P$ DeWitt matrix, and forms an $O(N^2)$ (or $O(C^2)$) pairwise overlap tensor.
4. **High-$d_0$ Random-Plane Orientation Pitfall**:
   - Redrawing a *random* 2-plane $Q \in V_2(\mathbb{R}^{d_0})$ every step washes out anisotropic prototype orientations ($0^\circ$ vs. $60^\circ$ vs. $120^\circ$ have no consistent meaning on a rotating plane, dropping 8-class $\mathbb{R}^8$ accuracy to `65%`). Fixing $Q_0 \in V_2(\mathbb{R}^{d_0})$ once at $t = 0$ as the **canonical top-2 data PCA Stiefel 2-frame** preserves full $E(2)$ output invariance while keeping `3N` forward passes ($O(1)$ in $d_0$) and restoring **`99%–100%`** accuracy.

---

### 2. Three Memory-Optimal Alternatives (`30x` to `1,387x` Less Memory)

#### Alternative A — **Canonical-Frame Logarithmic-Fiber Hyperboloid DeWitt (`O(1)` in $d_0$, `O(log C)` in $C$)**
Instead of $K = C$ fibers (`d_L = 2C`) and $2 \times 2$ matrix contractions, combine three exact geometric identities:
1. **Logarithmic Spherical Metric Codebook ($K = \lceil \log_2 C \rceil$ fibers, $d_L = 2\lceil \log_2 C \rceil$)**:
   - $(\mathrm{Sym}^+_2)^K$ has Riemannian dimension $3K$. Placing $C$ prototypes via repulsive Riesz/Tammes dispersion on $\mathbb{S}^{3K-1}$ mapped by $\mathrm{Exp}_{(e^{s_0}I_2)^K}$ eliminates crowding using only **`K = 6` fibers (`d_L = 12`) for `C = 64`** and **`K = 8` (`d_L = 16`) for `C = 256`**—smaller than Cross-Entropy's `C`-wide logit layer!
2. **Exact Minkowski Hyperboloid Isometry ($\mathrm{Sym}^+_2 \cong \mathbb{R}_{\mathrm{vol}} \times \mathbb{H}^2$)**:
   - Map each fiber metric $g_k \in \mathrm{Sym}^+_2$ once ($O(NK)$) to $s_k = \frac{1}{2}\log\det g_k$ and unimodular hyperboloid vector $u_k = \big(\frac{\tilde{g}_{00}+\tilde{g}_{11}}{2}, \frac{\tilde{g}_{00}-\tilde{g}_{11}}{2}, \tilde{g}_{01}\big)$ ($\tilde{g} = (\det g)^{-1/2} g$). Exact squared DeWitt distance reduces to scalar dot products (verified to `1.1e-10` against `compute_dewitt_dist_sq` without `(N, C, 2, 2)` tensors):
     $$d_{\mathrm{AI}}^2(g_k, G^*_{c,k}) = 2(s_k - s^*_{c,k})^2 + 2\,\mathrm{arccosh}^2\!\big(u_{k,0} u^*_{c,k,0} - u_{k,1} u^*_{c,k,1} - u_{k,2} u^*_{c,k,2}\big).$$
3. **Sampled $m_{\mathrm{neg}}$-Negative Voronoi Margin ($m_{\mathrm{neg}} = 8 \ll C$)**:
   - Evaluates exact distance to the ground-truth prototype $\mathbf{G}^*_{y_n}$ plus $m_{\mathrm{neg}}$ sampled negative prototypes per step, eliminating the $O(NC)$ logit matrix during training (`O(N log C)` total memory).

#### Alternative B — **2-Segment Midpoint Polygonal Geodesic Barrier (`O(1)` in *Both* $d_0$ and $C$)**
Instead of forming $g(x) \in \mathbb{R}^{d_0 \times d_0}$, discretize Riemannian path energy and arc length **directly in output space** along 1 sampled same-class edge $(i, j^+)$ and 1 cross-class boundary edge $(i, j^-)$ per sample via their midpoints $x_{\mathrm{mid}}^\pm = \frac{1}{2}(x_i + x_{j^\pm})$ (`3N` forward passes total, `d_L = 2` or `4`):
$$E_{\mathrm{same}}(i) = 2\big\|f(x_{\mathrm{mid},i}^+) - f(x_i)\big\|_2^2 + 2\big\|f(x_{j^+(i)}) - f(x_{\mathrm{mid},i}^+)\big\|_2^2, \qquad \ell_{\mathrm{diff}}(i) = \big\|f(x_{\mathrm{mid},i}^-) - f(x_i)\big\|_2 + \big\|f(x_{j^-(i)}) - f(x_{\mathrm{mid},i}^-)\big\|_2.$$
- Evaluates the exact midpoint quadrature of $\int_0^1 \|J_f(\gamma(t))\Delta x\|\,dt$ along the true chord $\Delta x \in \mathbb{R}^{d_0}$ with **zero $(d_0 \times d_0)$ matrices**, **zero step-size $h$**, and **strictly constant memory in $C$**.

#### Alternative C — **1st-Order Coupled Subspace Orthogonality + Gram Trace Identity (`~1,000x` Less Memory)**
Eliminates all three memory bottlenecks of `Coupled Subspace Orthogonality`:
1. **Eliminate 2nd-Order Autograd (`jacfwd` -> Analytic Head Jacobian *or* $r$-Probe $\Theta$-FD)**:
   - **Variant C1 (`ParamFD-Subsp`, full-network coupling)**: Project parameter tangents onto $r = 6$ random orthonormal directions $V \in \mathbb{R}^{P \times r}$ via 1st-order forward differences in $\theta$ and whiten with an **`r x r` (`6 x 6`)** sketched DeWitt matrix instead of $P \times P$.
   - **Variant C2 (`Analytic Head-Subsp`, `3N` forward passes total)**: Compute the **exact closed-form** derivative $U_{\mathrm{head}}(x) = \partial_{W_L}(g_0^{-1/2} g_{Q_0}(x) g_0^{-1/2}) \in \mathbb{R}^{3 \times 2d_{\mathrm{hid}}}$ analytically from the penultimate Jacobian $H(x) = D_{Q_0} h_{L-1}(x)$, keeping parameter subspace width $P_{\mathrm{head}} = 2 d_{\mathrm{hid}}$ (`32`–`64`) **100% independent of $d_0$** while `backward()` updates all trunk layers via pure 1st-order autograd!
2. **Eliminate $O(N^2)$ and $O(C^2)$ Pairwise Overlap Matrices via the Gram Identity**:
   - Aggregate normalized class-mean subspaces $\bar{W}_c \in \mathbb{R}^{3 \times r}$ ($B_c \le \min(N, C)$ active classes in batch) and use the exact trace identity:
     $$\sum_{a \neq b} \big\|\bar{W}_a \bar{W}_b^T\big\|_F^2 = \left\|\sum_{c=1}^{B_c} \bar{W}_c^T \bar{W}_c\right\|_F^2 - \sum_{c=1}^{B_c} \big\|\bar{W}_c \bar{W}_c^T\big\|_F^2$$
     which forms a tiny **`r x r` Gram matrix** (`O(B_c r^2)`) instead of an $O(N^2)$ or $O(C^2)$ pairwise tensor!

---

### 3. Prototype Benchmark & Accuracy Verification

| Regime | Baseline in Notebook `08` (`ms` / `MB`) | Scalable Alternative (`ms` / `MB`) | Memory Reduction |
| :--- | :---: | :---: | :---: |
| **`d_0 = 128`, `C = 3`** (`DeWitt`) | `8.04 ms` / `148.52 MB` *(Coord FD)* | **`3.18 ms` / `1.37 MB`** *(LogFib-DeWitt, `3.39 MB` at `d_0 = 512`)* | **`108x` less** *(1.0x vs 2-probe)* |
| **`d_0 = 128`, `C = 3`** (`Barrier`) | `210.59 ms` / `1,786.06 MB` *(Coord FD)* | **`2.18 ms` / `1.42 MB`** *(Midpt-Barrier, `4.56 MB` at `d_0 = 512`)* | **`1,258x` less (`96x` faster)** |
| **`d_0 = 128`, `C = 3`** (`Coupled Subsp.`) | `1,022.07 ms` / `4,341.0 MB` *(`jacfwd`)* | **`4.72 ms` / `3.13 MB`** *(Analytic Head)*<br>**`12.64 ms` / `4.41 MB`** *(6-Probe $\Theta$-FD)* | **`1,387x` less (`216x` faster)**<br>**`984x` less (`81x` faster)** |
| **`d_0 = 2`, `C = 64`** (`DeWitt`) | `6.30 ms` / `77.82 MB` *(`(Sym^+_2)^C`, `d_L=128`)* | **`4.17 ms` / `5.05 MB`** *(LogFib `K=6`, `d_L=12`, `m_neg=8`)* | **`15.4x` less** *(zero crowding)* |
| **`d_0 = 2`, `C = 64`** (`Coupled Subsp.`) | `19.62 ms` / `71.70 MB` *(`jacfwd`)* | **`11.72 ms` / `8.48 MB`** *(ParamFD + Gram)*<br>**`5.59 ms` / `10.92 MB`** *(Head + Gram)* | **`8.5x` less** |
| **Joint `d_0 = 128`, `C = 256`** (`N = 256`) | *OOM / multi-GB for all 3 baselines* | **LogFib (`K=8, d_L=16`)**: `5.00 ms` / **`10.40 MB`**<br>**Midpt-Barrier (`d_L=2`)**: `2.44 ms` / **`4.13 MB`**<br>**ParamFD-Subsp**: `14.03 ms` / **`14.09 MB`**<br>**Head-Subsp**: `9.63 ms` / **`20.71 MB`** | **All `< 21 MB` at `d_0=128, C=256`** |

* **Multiclass Test Accuracy in $\mathbb{R}^{16}$ (`d_0 = 16`, `blobs`, `N_train = 320`)**:
  - **`C = 8` classes**: `SGD+CE`: `100.00%` | `LogFib-DeWitt (K=3, d_L=6)`: **`99.38%`** | `Midpt-Barrier`: **`100.00%`** | `Analytic Head-Subsp`: **`99.38%`**
  - **`C = 16` classes**: `SGD+CE`: `100.00%` | `LogFib-DeWitt (K=4, d_L=8)`: **`98.75%`** | `Midpt-Barrier`: **`100.00%`** | `Analytic Head-Subsp`: **`100.00%`**


---

## Turn 27

### User
yes, implement them in a new notebook

### Assistant
Implemented all four memory-scalable $S_C$-symmetric coupled geometric formulations in [src/diffgeom/geometry.py](src/diffgeom/geometry.py) (exported in [src/diffgeom/__init__.py](src/diffgeom/src/diffgeom/__init__.py) and tested in [tests/test_geometry.py](tests/test_geometry.py): `7/7` pytest suites passing, `0` ruff lint warnings), registered Notebook `09` in [src/diffgeom/cli.py](src/diffgeom/cli.py) and [scripts/build_all_notebooks.py](scripts/build_all_notebooks.py), and built/executed [notebooks/09-scalable-coupled-multiclass.ipynb](notebooks/09-scalable-coupled-multiclass.ipynb).

### Key Verified Results from [notebooks/09-scalable-coupled-multiclass.ipynb](notebooks/09-scalable-coupled-multiclass.ipynb)

1. **Section 1 — Exact Algebraic Identities, Logarithmic Prototype Separation & Canonical 2-Frames**:
   - **Minkowski-Hyperboloid Isometry vs. $2 \times 2$ Eigenvalue DeWitt Distance**: `7.19e-11` max difference.
   - **$r \times r$ Gram Trace Identity vs. Explicit $O(C^2)$ Off-Diagonal Subspace Sum**: `2.78e-17` max difference.
   - **Zero Crowding at $C = 256$**: Minimum pairwise DeWitt prototype separation $\min_{a \neq b} d_{\mathrm{AI}}$ collapses to `0.0616` on a single $\mathrm{Sym}^+_2$, whereas `Log-Fibered (K = 8, d_L = 16)` maintains **`1.7873`** (`16` output neurons vs. `256` Cross-Entropy logits and `512` one-hot fibers).
   - **Canonical PCA 2-Frame ($Q_0$) vs. Rotating Random 2-Plane ($Q_t$) on `8-Class R^16`**: **`100.00%`** test accuracy vs. **`34.38%`**.

2. **Section 2 — Multiclass Accuracy & `0.0%` IVT Phantom Intrusion**:
   - **`3-Class Triple-Junction (R^2)`**: `Midpoint Polygonal Barrier`: **`100.00%`** (`0.0%` phantom), `LogFib Hyperboloid DeWitt`: **`98.33%`** (`0.0%` phantom), matching `SGD + Cross-Entropy` (`98.33%`).
   - **Active `config.yaml` (`circles`, `noise=0.2`, `[2, 10, 2]`)**: `LogFib Hyperboloid DeWitt` (**`84.17%`**) and `Midpoint Polygonal Barrier` (**`83.33%`**) match/exceed `SGD + Cross-Entropy` (`83.33%`).
   - **`8-Class Blobs (R^16)`**: `SGD+CE`: `100.00%` | `LogFib-DeWitt`: **`100.00%`** | `Midpt-Barrier`: **`100.00%`** | `ParamFD-Subsp`: **`99.38%`** | `Analytic Head-Subsp`: **`98.75%`**.
   - **`16-Class Blobs (R^16)`**: `SGD+CE`: `100.00%` | `Midpt-Barrier`: **`100.00%`** | `LogFib-DeWitt`: **`99.48%`** | `Analytic Head-Subsp`: **`99.48%`** | `ParamFD-Subsp`: **`97.92%`**.

3. **Section 3 — High-Dimensional Scaling (`d_0 = 2 -> 512`, `C = 3`, `N = 96`)**:
   - At `d_0 = 128`: `Nb08 Barrier CoordFD` (`113.75 ms / 1,125.41 MB`) and `Nb08 Coupled Subspace` (`998.17 ms / 1,767.38 MB` at `N=64`) drop to:
     - **`Nb09 LogFib-DeWitt`**: `2.86 ms / 1.38 MB` (**`3.68 ms / 3.40 MB`** at `d_0 = 512`)
     - **`Nb09 Midpt-Barrier`**: `2.16 ms / 1.42 MB` (**`2.55 ms / 4.57 MB`** at `d_0 = 512`)
     - **`Nb09 Analytic Head-Subspace`**: `4.21 ms / 3.12 MB` (**`4.50 ms / 5.14 MB`** at `d_0 = 512`)
     - **`Nb09 ParamFD-Subspace (r=6)`**: `12.28 ms / 4.40 MB` (**`24.58 ms / 8.34 MB`** at `d_0 = 512`)

4. **Section 4 — Many-Class Scaling (`C = 2 -> 256`, `d_0 = 2`, `N = 256`)**:
   - **`Nb09 Midpt-Barrier (d_L = 2)`**: **`1.85 ms / 1.42 MB`** at `C = 256` (strictly constant `1.42 MB` from `C = 2` to `C = 256`—smaller than `SGD + Cross-Entropy`'s `2.29 MB` at `C = 256`!).
   - **`Nb09 LogFib-DeWitt (d_L = 16, m_neg = 8)`**: **`5.47 ms / 9.04 MB`** at `C = 256` (vs. `78.38 MB` already at `C = 128` for `Nb08 OneHot`).
   - **`Nb09 ParamFD` & `Head-Subspace + Gram`**: **`11.70 MB`** and **`18.95 MB`** at `C = 256` (vs. `2,373.86 MB` for `Nb08 Subspace` at `C = 128`).

5. **Section 5 — Simultaneous High-Dimension + Many-Class Diagonal Stress Test (`d_0 = C = 4 -> 256`, `N = 256`)**:
   - At **`(d_0 = 256, C = 256)`**: `Midpt-Barrier`: **`2.56 ms / 7.32 MB`** | `LogFib-DeWitt (d_L = 16)`: **`5.21 ms / 13.34 MB`** | `ParamFD-Subsp`: **`18.84 ms / 20.78 MB`** | `Analytic Head-Subsp`: **`8.81 ms / 22.86 MB`**.


---

## Turn 28

### User
update both charts so that dims and classes include 512 and 1024

### Assistant
Updated all dimension and class-count sweeps and charts across Sections 1, 3, 4, and 5 of [notebooks/09-scalable-coupled-multiclass.ipynb](notebooks/09-scalable-coupled-multiclass.ipynb) (via [scripts/build_all_notebooks.py](scripts/build_all_notebooks.py), plus fast vectorized no-grad Riesz prototype dispersion in `build_log_fiber_prototypes` in [src/diffgeom/geometry.py](src/diffgeom/geometry.py)) to span **`512` and `1024`** (`d_0 = 2 -> 1024`, `C = 2 -> 1024`, and joint diagonal `d_0 = C = 4 -> 1024`).

### Verified Extended Scaling Results (`512` and `1024`)

1. **Section 3 — Ambient Input Dimension (`d_0 = 2 -> 1024`, `C = 3`, `N = 96`)**:

| `d_0` | `Nb08 Barrier` (`ms` / `MB`) | `Nb08 Subspace` (`ms` / `MB`) | `Nb09 LogFib-DW` | `Nb09 Midpt-Barrier` | `Nb09 Analytic Head` | `Nb09 ParamFD (r=6)` |
| ---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `128` | `69.88 / 1125.41` | `830.97 / 1767.38` | **`2.50 / 1.38`** | **`2.00 / 1.42`** | **`4.61 / 3.12`** | **`10.22 / 4.40`** |
| `256` | `> 4 GB / skip` | `> 4 GB / skip` | **`2.52 / 2.06`** | **`1.71 / 2.47`** | **`3.51 / 3.80`** | **`10.10 / 5.71`** |
| `512` | `> 4 GB / skip` | `> 4 GB / skip` | **`3.20 / 3.40`** | **`2.08 / 4.57`** | **`3.89 / 5.14`** | **`14.42 / 8.34`** |
| `1024` | `> 4 GB / skip` | `> 4 GB / skip` | **`3.48 / 6.10`** | **`2.84 / 8.75`** | **`5.02 / 7.84`** | **`18.57 / 13.60`** |

2. **Section 4 — Number of Classes (`C = 2 -> 1024`, `d_0 = 2`, `N = 256`)**:

| `C` | `SGD+CE (d_L=C)` | `Nb08 OneHot (d_L=2C)` | `Nb08 Subspace` | `Nb09 LogFib-DW` | `Nb09 Midpt-Barrier` | `Nb09 Analytic Head` | `Nb09 ParamFD` |
| ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `128` | `0.90 / 1.27` | `16.47 / 78.38` | `344.55 / 2373.86` | **`4.83 / 8.03`** | **`1.79 / 1.42`** | **`5.88 / 15.52`** | **`11.54 / 11.33`** |
| `256` | `0.94 / 2.29` | `> 4 GB / skip` | `> 4 GB / skip` | **`4.00 / 9.04`** | **`1.72 / 1.42`** | **`6.47 / 18.95`** | **`12.26 / 11.70`** |
| `512` | `1.08 / 4.32` | `> 4 GB / skip` | `> 4 GB / skip` | **`3.71 / 10.05`** | **`1.79 / 1.42`** | **`5.94 / 18.95`** | **`13.89 / 11.70`** |
| `1024` | `1.32 / 8.39` | `> 4 GB / skip` | `> 4 GB / skip` | **`4.06 / 11.07`** | **`2.12 / 1.42`** | **`7.38 / 18.95`** | **`21.88 / 11.70`** |

3. **Section 5 — Joint Diagonal (`d_0 = C = 4 -> 1024`, `N = 256`)**:
   - **`512 x 512`** (`d_L(LogFib) = 18`): `SGD+CE`: `1.34 ms / 4.76 MB` | `Midpt-Barrier`: **`2.86 ms / 12.88 MB`** | `LogFib-DeWitt`: **`4.93 ms / 17.92 MB`** | `Head-Subsp`: **`7.30 ms / 26.43 MB`** | `ParamFD-Subsp`: **`23.94 ms / 26.91 MB`**.
   - **`1024 x 1024`** (`d_L(LogFib) = 20`): `SGD+CE`: `2.03 ms / 9.02 MB` | `Midpt-Barrier`: **`4.27 ms / 24.01 MB`** | `LogFib-DeWitt`: **`9.00 ms / 26.06 MB`** | `Head-Subsp`: **`7.17 ms / 33.56 MB`** | `ParamFD-Subsp`: **`25.89 ms / 39.16 MB`**.


---

## Turn 29

### User
ok, now add 2048 and 4096

### Assistant
Extended all dimension and class-count sweeps and charts across Sections 1, 3, 4, and 5 of [notebooks/09-scalable-coupled-multiclass.ipynb](notebooks/09-scalable-coupled-multiclass.ipynb) (via [scripts/build_all_notebooks.py](scripts/build_all_notebooks.py) and [src/diffgeom/geometry.py](src/diffgeom/geometry.py)) to include **`2048` and `4096`** (`d_0 = 2 -> 4096`, `C = 2 -> 4096`, and joint diagonal `d_0 = C = 4 -> 4096`).

### Verified Extended Scaling Results (`2048` & `4096`)

1. **Section 1 — Prototype Separation & Output Width at `C = 2048` and `4096`**:
   - **`C = 2048` (`K = 11`, `d_L = 22` vs. `2048` CE logits)**: `Min d_AI Single = 0.0077` vs. **`Min d_AI LogFib = 1.3724`**.
   - **`C = 4096` (`K = 12`, `d_L = 24` vs. `4096` CE logits)**: `Min d_AI Single = 0.0039` vs. **`Min d_AI LogFib = 1.3362`**.

2. **Section 3 — Ambient Input Dimension (`d_0 = 2 -> 4096`, `C = 3`, `N = 96`)**:

| `d_0` | `Nb08 Barrier` (`ms` / `MB`) | `Nb08 Subspace` (`ms` / `MB`) | `Nb09 LogFib-DW` | `Nb09 Midpt-Barrier` | `Nb09 Analytic Head` | `Nb09 ParamFD (r=6)` |
| ---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `128` | `98.75 / 1125.41` | `915.59 / 1767.38` | **`2.23 / 1.38`** | **`1.67 / 1.42`** | **`3.98 / 3.12`** | **`12.07 / 4.40`** |
| `512` | `> 4 GB / skip` | `> 4 GB / skip` | **`3.04 / 3.40`** | **`2.18 / 4.57`** | **`3.77 / 5.14`** | **`15.62 / 8.34`** |
| `1024` | `> 4 GB / skip` | `> 4 GB / skip` | **`3.16 / 6.10`** | **`2.43 / 8.75`** | **`9.48 / 7.84`** | **`18.31 / 13.60`** |
| `2048` | `> 4 GB / skip` | `> 4 GB / skip` | **`3.97 / 11.49`** | **`3.35 / 17.13`** | **`5.93 / 13.23`** | **`22.02 / 24.12`** |
| `4096` | `> 4 GB / skip` | `> 4 GB / skip` | **`5.22 / 22.27`** | **`4.64 / 33.88`** | **`10.18 / 24.01`** | **`31.14 / 45.15`** |

3. **Section 4 — Number of Classes (`C = 2 -> 4096`, `d_0 = 2`, `N = 256`) — Memory Crossover Below `SGD + Cross-Entropy`**:

| `C` | `SGD+CE (d_L=C)` | `Nb08 OneHot (d_L=2C)` | `Nb08 Subspace` | `Nb09 LogFib-DW` | `Nb09 Midpt-Barrier` | `Nb09 Analytic Head` | `Nb09 ParamFD` |
| ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `512` | `1.17 / 4.32` | `> 4 GB / skip` | `> 4 GB / skip` | **`4.10 / 10.05`** | **`1.87 / 1.42`** | **`6.65 / 18.95`** | **`16.28 / 11.70`** |
| `1024` | `1.95 / 8.39` | `> 4 GB / skip` | `> 4 GB / skip` | **`5.51 / 11.07`** | **`1.87 / 1.42`** | **`6.81 / 18.95`** | **`12.13 / 11.70`** |
| `2048` | `2.14 / 16.52` | `> 4 GB / skip` | `> 4 GB / skip` | **`4.18 / 12.08`** | **`1.74 / 1.42`** | **`9.55 / 18.95`** | **`12.02 / 11.70`** |
| `4096` | `3.43 / 32.78` | `> 4 GB / skip` | `> 4 GB / skip` | **`4.98 / 13.09`** | **`1.86 / 1.42`** | **`6.43 / 18.95`** | **`13.83 / 11.70`** |

* At **`C = 4096`**, **`SGD + Cross-Entropy`** grows to **`32.78 MB`** (`3.43 ms`) due to its $O(NC)$ logit layer (`d_L = 4096`), crossing **above all four scalable geometric losses** (`Midpt-Barrier`: **`1.42 MB` (`23.1x` less memory & faster at `1.86 ms`)**, `ParamFD`: **`11.70 MB`**, `LogFib-DeWitt`: **`13.09 MB`**, `Analytic Head`: **`18.95 MB`**).

4. **Section 5 — Simultaneous Joint Diagonal (`d_0 = C = 4 -> 4096`, `N = 256`)**:
   - **`2048 x 2048`** (`d_L(LogFib) = 22`): `SGD+CE`: `2.80 ms / 17.52 MB` | `LogFib-DeWitt`: **`6.93 ms / 41.34 MB`** | `Midpt-Barrier`: **`4.05 ms / 46.26 MB`** | `Head-Subsp`: **`8.74 ms / 47.83 MB`** | `ParamFD-Subsp`: **`35.77 ms / 63.68 MB`**.
   - **`4096 x 4096`** (`d_L(LogFib) = 24`): `SGD+CE`: `5.20 ms / 34.54 MB` | `LogFib-DeWitt`: **`10.71 ms / 70.88 MB`** | `Head-Subsp`: **`11.63 ms / 76.36 MB`** | `Midpt-Barrier`: **`6.10 ms / 90.76 MB`** | `ParamFD-Subsp`: **`56.09 ms / 112.71 MB`**.


---

## Turn 30

### User
explain DeWitt / Ebin metric  and why and when is it used in this project 

### Assistant
### 1. What Is the DeWitt / Ebin Metric?

In standard deep learning, a parameter vector $\theta \in \Theta$ maps an input $x \in \mathcal{M}$ to an output vector $z(x; \theta) \in \mathbb{R}^{d_L}$. In **geometric deep learning on pullback manifolds**, however, the object the network is sculpting is not a vector of logits—it is an entire **Riemannian metric field** $x \mapsto g(x; \theta) \in \mathrm{Sym}^+_d$ on the data space $\mathcal{M}$.

To measure distances or take steepest-descent (Natural Gradient) steps on a space whose "points" are **metrics themselves**, we need a **metric on the space of metrics** $\mathrm{Met}(\mathcal{M})$:

1. **Pointwise Level — Affine-Invariant Metric on the SPD Cone $\mathrm{Sym}^+_d$**:
   At a single data point $x$, $g(x) \in \mathrm{Sym}^+_d$ is a $d \times d$ symmetric positive-definite matrix. For two tangent perturbations $h, k \in T_{g}\mathrm{Sym}^+_d$ (symmetric matrices representing infinitesimal metric changes $\delta g$), the **affine-invariant Riemannian metric** is:
   $$\langle h, k \rangle_{g} = \mathrm{Tr}\!\left(g^{-1}\, h\, g^{-1}\, k\right) = \left\langle g^{-1/2}\, h\, g^{-1/2},\; g^{-1/2}\, k\, g^{-1/2} \right\rangle_F$$
   - Under this metric, $(\mathrm{Sym}^+_d, \langle \cdot, \cdot \rangle)$ is a **Hadamard manifold** (complete, simply connected, non-positively curved: $\mathrm{Sym}^+_2 \cong \mathbb{R}_{\mathrm{vol}} \times \mathbb{H}^2$).
   - The boundary of singular/collapsed matrices ($\det g = 0$) lies at **infinite geodesic distance**, and the exact geodesic distance between two SPD matrices $g_1, g_2 \in \mathrm{Sym}^+_d$ is:
     $$d_{\mathrm{AI}}(g_1, g_2) = \left\| \log\!\Big(g_1^{-1/2}\, g_2\, g_1^{-1/2}\Big) \right\|_F = \sqrt{\sum_{i=1}^d \big(\log \lambda_i(g_1^{-1} g_2)\big)^2}$$

2. **Global Field Level — The Ebin / DeWitt $L^2$ Metric on $\mathrm{Met}(\mathcal{M})$**:
   Integrating this pointwise inner product over all $x \in \mathcal{M}$ (with either the empirical data measure $d\mathbb{P}(x)$ or the intrinsic Riemannian volume element $\sqrt{\det g(x)}\,dx$) yields the **Ebin metric** (**David Ebin, 1970**, global analysis on the manifold of Riemannian metrics; **Bryce DeWitt, 1967**, *supermetric* on 3-geometries in canonical general relativity, where the trace-term coefficient $\lambda$ is fixed to $\lambda = 0$ so the metric is strictly positive-definite rather than Lorentzian):
   $$\mathcal{G}_{\mathrm{Ebin}}(h, k)_{g} = \int_{\mathcal{M}} \mathrm{Tr}\!\Big(g(x)^{-1}\, h(x)\, g(x)^{-1}\, k(x)\Big)\, w(x)\, d\mu(x)$$

3. **Parameter-Space Pullback — Coupling $\Theta \leftrightarrow \mathcal{M}$**:
   Pulling $\mathcal{G}_{\mathrm{Ebin}}$ back through the neural network map $\theta \mapsto g(\cdot; \theta)$ defines a $P \times P$ Riemannian metric directly on the **parameter manifold $\Theta$**:
   $$G^{\Theta,\mathrm{DeWitt}}_{\mu\nu}(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\!\left[ w(x)\, \mathrm{Tr}\!\Big(g(x; \theta)^{-1}\, \partial_{\theta_\mu} g(x; \theta)\, g(x; \theta)^{-1}\, \partial_{\theta_\nu} g(x; \theta)\Big) \right]$$

---

### 2. Why Is It Used in This Project?

Training neural networks through **intrinsic geometric losses** (volume, anisotropy, curvature, geodesic barriers) breaks standard coordinate-based optimizers and metrics for four fundamental reasons—each directly solved by the DeWitt / Ebin geometry:

1. **Quotients Out Rigid Output Gauge Symmetries ($E(d_L) = O(d_L) \ltimes \mathbb{R}^{d_L}$)**:
   - Every intrinsic geometric quantity ($g = J^T J$ or $I + J^T J$, $\det g$, $K(x)$) is **100% invariant** under global orthogonal rotations/reflections and translations of the network output $z(x) \mapsto Q z(x) + b$ ($Q \in O(d_L), b \in \mathbb{R}^{d_L}$).
   - Standard coordinate Gauss–Newton or Fisher matrices ($J_\theta z^T J_\theta z$, `compute_uncoupled_coord_metric`) assign **large positive stiffness** to these pure gauge directions—wasting preconditioner capacity resisting harmless output rotations/translations.
   - Because $\partial_\theta g = 0$ along any rigid motion of $z$, **$G^{\Theta,\mathrm{DeWitt}}(\theta)$ assigns exact zero norm ($\ker G^{\Theta,\mathrm{DeWitt}} \supseteq \mathfrak{e}(d_L)$) to output gauge symmetries**, conditioning *only* genuine deformations of data-space geometry.

2. **Relative Scale Invariance & Protection Against Metric Collapse ($\mathrm{GL}(d, \mathbb{R})$ Invariance)**:
   - Euclidean Frobenius norm $\|\delta g\|_F^2$ treats a change of $0.01$ identically whether $g \approx 100\,I$ (a tiny $0.01\%$ tweak on a steep decision wall) or $g \approx 0.01\,I$ (a catastrophic $100\%$ singular collapse inside a contracted class basin!).
   - Whitening via $g^{-1/2} \delta g\, g^{-1/2}$ measures **multiplicative / logarithmic strain** ($\delta \log g$): singular rank collapse ($\det g \to 0$) is pushed to $d_{\mathrm{AI}} = +\infty$, and conditioning automatically adapts across both compressed class interiors ($\det g \ll 1$) and magnified decision boundaries ($\det g \gg 1$).

3. **Eliminates Artificial 1D Class Ordering in Multiclass Classification**:
   - Scalar geometric losses ($\log\sqrt{\det g}, \log(g_{11}/g_{22}), K$) force $C \ge 3$ unordered classes onto a 1D number line (`-2τ, 0, +2τ`), creating unequal class spacing and Intermediate-Value Theorem (IVT) phantom fringes.
   - Under the DeWitt metric, $\mathrm{Sym}^+_2 \cong \mathbb{R}_{\mathrm{vol}} \times \mathbb{H}^2$ is a **3D curved manifold** (and $(\mathrm{Sym}^+_2)^K$ has dimension $3K$), allowing us to place $C$ target metrics at **equidistant simplex / spherical-code vertices** measured by $d_{\mathrm{AI}}$—with exact $S_C$ permutation symmetry and `0.0%` phantom fringes.

4. **Exact Gauss–Newton Duality**:
   - When the data-space loss is squared DeWitt geodesic distance $\mathcal{L}(\theta) = \frac{1}{2N}\sum_n d_{\mathrm{AI}}^2(g(x_n; \theta), G^*_{y_n})$, its Generalized Gauss–Newton Hessian on parameter space $\Theta$ **is literally $G^{\Theta,\mathrm{DeWitt}}(\theta)$**—unifying the loss geometry on $\mathcal{M}$ and the optimization geometry on $\Theta$.

---

### 3. When & Where Is It Used Across This Project?

The DeWitt / Ebin metric is used in **three distinct roles** across [src/diffgeom/geometry.py](src/diffgeom/geometry.py) and Notebooks `06`–`09`:

| Role | Functions in `src/diffgeom/geometry.py` | Notebooks | When to Use It |
| :--- | :--- | :---: | :--- |
| **A. Parameter-Space Natural Gradient Preconditioner** ($G^{\Theta,\mathrm{DeWitt}}$) | `compute_coupled_dewitt_metric`, `compute_coupled_geometric_metric`, `woodbury_coupled_ngd_direction` | `06`, `07`, `08` | **During optimization** when training any geometric loss (Curvature, Volume, Anisotropy, or DeWitt Simplex) and ill-conditioning / $E(d_L)$ gauge drift slows first-order SGD. Use `woodbury_coupled_ngd_direction` ($O(S^2 P)$) rather than dense $P \times P$ inversion when $P$ is large. |
| **B. Direct Data-Space Multiclass Loss & Prototype Classifier** ($d_{\mathrm{AI}}^2$ on $\mathrm{Sym}^+_2$ / $(\mathrm{Sym}^+_2)^K$) | `compute_dewitt_dist_sq`, `dewitt_simplex_loss`, `fibered_dewitt_simplex_loss`, `sym2_to_hyperboloid`, `hyperboloid_dewitt_dist_sq`, `log_fiber_dewitt_loss`, `log_fiber_dewitt_predict` | `08`, `09` | **As the classification loss and standalone test-time rule** whenever $C \ge 3$ (eliminating 1D ordering & IVT fringes). Use single-fiber `dewitt_simplex_loss` for low-dim $C \le 4$, and **`log_fiber_dewitt_loss`** ($K = \lceil\log_2 C\rceil$, Hyperboloid isometry + canonical PCA 2-frame $Q_0$) when $d_0 \gg 2$ or $C \gg 4$ (up to $4096 \times 4096$). |
| **C. Coupled $\Theta \leftrightarrow \mathcal{M}$ Subspace Orthogonality** ($G^\Theta$-whitened Grassmannian kernel) | `compute_coupled_subspace_factors`, `coupled_subspace_orthogonality_loss`, `analytic_head_subspace_loss`, `sketched_param_subspace_loss` | `08`, `09` | **When you want non-interfering parameter subspaces across classes** ($\mathcal{V}_a \perp_{G^\Theta} \mathcal{V}_b$). Whitens parameter Jacobians $g_0^{-1/2}\partial_\theta g\,g_0^{-1/2}$ by $(G^{\Theta,\mathrm{DeWitt}} + \lambda I)^{-1/2}$; use `analytic_head_subspace_loss` or `sketched_param_subspace_loss` (`09`) for $O(1)$-memory scaling. |

#### When Is It *Not* Used?
- **Standard Coordinate Cross-Entropy (`Notebooks 01, 03`)**: Logits *depend* on output coordinates and rotations, so the classical **Parameter Fisher Information Matrix** (`compute_parameter_fim` in Notebook `03`) is the canonical parameter metric instead.
- **Pure Geodesic Barrier Training under First-Order SGD/Adam (`geodesic_barrier_loss` / `midpoint_polygonal_barrier_loss` in `08`, `09`)**: Separates classes via relational path length $\int \sqrt{\dot\gamma^T g\,\dot\gamma}\,dt$ on $\mathcal{M}$ directly without needing SPD prototype targets or $G^\Theta$ whitening.


---

## Turn 31

### User
why do we need to measure distances between metrics?

### Assistant
### The Core Shift: What Is the Network's "Output" at $x$?

In **standard classification**, a neural network maps each input $x$ to a **probability distribution** $p(y \mid x; \theta)$ (via logits $z(x; \theta) \in \mathbb{R}^C$). Everything in training and optimization comes down to comparing two distributions:
- **Loss / Inference**: Compare the predicted distribution $p(\cdot \mid x; \theta)$ to the target label distribution $e_y$ (via Cross-Entropy / KL divergence).
- **Natural Gradient**: Compare the current model $p(\cdot \mid x; \theta)$ to a perturbed model $p(\cdot \mid x; \theta + \Delta\theta)$ via KL divergence, whose 2nd-order Taylor expansion gives the **Fisher Information Matrix**.

In **purely geometric classification**, we intentionally discard coordinate logits so the model depends only on **intrinsic representation geometry** (invariant to rotations/translations $z \mapsto Qz + b$). Once logits are gone, the coordinate-free object the network produces at each point $x$ is **no longer a vector—it is a local Riemannian metric tensor $g(x; \theta) \in \mathrm{Sym}^+_d$** (visualized as a Tissot ellipse encoding local volume stretching and directional shear).

As soon as the network's output at $x$ is a **metric**, three fundamental operations require measuring distances between metrics:

---

### 1. Supervision & Anchor-Free Test Inference: Comparing $g(x_n; \theta)$ to Class Prototypes $G^*_c$
- **Training**: Suppose Class `0`, Class `1`, and Class `2` are assigned three symmetric target geometries $G^*_0, G^*_1, G^*_2 \in \mathrm{Sym}^+_2$ (e.g., Tissot ellipses rotated at $0^\circ, 60^\circ, 120^\circ$ in Notebook `08`, or logarithmic spherical-code fibers in Notebook `09`). Pulling $g(x_n; \theta)$ toward its ground-truth prototype $G^*_{y_n}$ and away from $G^*_{c \neq y_n}$ requires a distance $d_{\mathrm{AI}}\!\big(g(x_n; \theta), G^*_c\big)$.
- **Test-time prediction**: For a single unseen point $x^*$, we evaluate its local metric $g(x^*; \theta)$ and pick the nearest prototype:
  $$\hat{y}(x^*) = \arg\min_{c} d_{\mathrm{AI}}\!\big(g(x^*; \theta),\, G^*_c\big)$$
  without needing to store or search $k$-NN training points!
- *Why not collapse $g(x)$ to a single number (like $\det g$) and use 1D scalar differences?* Because $\mathrm{Sym}^+_2$ is 3-dimensional; projecting it onto a 1D number line throws away 2 degrees of freedom and forces artificial class ordering (`Class 0 < Class 1 < Class 2`) and Intermediate-Value Theorem phantom boundaries (Notebook `08`, Section 1).

---

### 2. Natural Gradient & Non-Interfering Optimization: Comparing $g(\cdot; \theta)$ to $g(\cdot; \theta + \Delta\theta)$
Every steepest-descent optimizer solves:
$$\Delta\theta^* = \arg\min_{\Delta\theta} \langle \nabla_\theta \mathcal{L}(\theta),\, \Delta\theta \rangle \quad \text{subject to} \quad \mathrm{Dist}\big(\text{model}(\theta),\, \text{model}(\theta + \Delta\theta)\big)^2 \le \varepsilon^2.$$
- Euclidean SGD uses $\|\Delta\theta\|_2^2$, which measures **weight-coordinate movement** rather than **geometric change**: rotating the output layer $W_L \mapsto Q W_L$ produces a large $\|\Delta\theta\|_2 > 0$ even though $g(x; \theta)$ has changed by **zero**, whereas a tiny $\Delta\theta$ near a singular layer can distort data-space geometry enormously.
- Measuring the squared DeWitt distance between **before-step** and **after-step** metric fields:
  $$\mathbb{E}_x\!\Big[ d_{\mathrm{AI}}\!\big(g(x; \theta),\, g(x; \theta + \Delta\theta)\big)^2 \Big] = \Delta\theta^T\, G^{\Theta,\mathrm{DeWitt}}(\theta)\, \Delta\theta + O(\|\Delta\theta\|^3)$$
  plays the **exact same role for Riemannian metrics that KL divergence plays for probability distributions** in Amari's Natural Gradient (Notebooks `06`–`09`).

---

### 3. Why Naive Matrix Subtraction $\|g_1 - g_2\|_F$ Breaks on Metrics
Even if you agree we must compare two matrices $g_1, g_2 \in \mathrm{Sym}^+_d$, treating them as flat vectors via Frobenius norm $\|g_1 - g_2\|_F$ fails for three geometric reasons:
1. **Singular Collapse Is at Finite Distance**: The zero matrix $g = 0$ (where space collapses to a point, $\det g = 0$, and curvature $K \propto 1/\det g$ explodes) is only Euclidean distance $\sqrt{2}$ from $I_2$. Under $d_{\mathrm{AI}}$, $\det g = 0$ is at **infinite distance** ($\log 0 = -\infty$), naturally repelling optimization from rank collapse.
2. **Expansion/Contraction Asymmetry**: Stretching space by $10\times$ ($g = 100 I$) and shrinking by $10\times$ ($g = 0.01 I$) are exact geometric inverses, yet $\|100I - I\|_F = 99\sqrt{2}$ vs. $\|0.01I - I\|_F = 0.99\sqrt{2}$ ($100\times$ bias!). Under $d_{\mathrm{AI}}$, $|\log 100| = |\log 0.01|$—expansion and contraction are treated symmetrically.
3. **Coordinate-Rescaling Dependence**: Linearly rescaling input coordinates $x \mapsto A^{-1} x$ transforms metrics by congruence $g \mapsto A^T g A$, scrambling $\|g_1 - g_2\|_F$, whereas $d_{\mathrm{AI}}(A^T g_1 A, A^T g_2 A) = d_{\mathrm{AI}}(g_1, g_2)$ is **strictly invariant**.

---

### Important Contrast: When You *Don't* Need Distances Between Metrics
Not every geometric loss requires comparing two metrics! In the **Geodesic Metric-Barrier** formulation (`geodesic_barrier_loss` in Notebook `08` and `midpoint_polygonal_barrier_loss` in Notebook `09`), we never compare $g(x)$ to another metric $G^*$. Instead, we use the single learned metric $g(\cdot; \theta)$ as a **ruler to measure path lengths between pairs of data points** $(x_i, x_j)$—pulling same-class neighbors together and erecting high-metric walls between different classes. The trade-off is at **test time**: because classes do not have distinct local metric prototypes $G^*_c$, classifying a new point $x^*$ requires measuring geodesic distance to training anchors ($k$-NN), whereas prototype-based DeWitt losses classify $x^*$ from $g(x^*; \theta)$ alone in $O(1)$ without anchors.


---

## Turn 32

### User
in notebook 05 I cannot experiment with a blobs dataset with 5 classes, please fix it

### Assistant
Fixed four issues that prevented multi-class `blobs` (`N_CLASSES = 5`) from working cleanly in [notebooks/05_geometric_loss_ablations_and_synthesis.ipynb](notebooks/05_geometric_loss_ablations_and_synthesis.ipynb) and [src/diffgeom/geometry.py](src/diffgeom/geometry.py):

1. **`centers` vs. `n_classes` Synchronization (`geometry.py` & Notebook `05` Cell 1)**:
   - Synchronized `centers` and `n_classes` in `get_synthetic_dataset` (`key == "blobs"`) and moved `CE_LAYER_DIMS = [*LAYER_DIMS[:-1], max(LAYER_DIMS[-1], N_CLASSES)]` *after* loading `y_train`/`y_test` (`N_CLASSES = max(N_CLASSES, int(max(y_train.max(), y_test.max())) + 1)`), preventing `IndexError: Target out of bounds` in `CrossEntropyLoss` when `centers` and `n_classes` differ.
2. **Restored `scikit-learn`'s Standard `center_box = (-10.0, 10.0)` for `make_blobs` (`geometry.py` L105)**:
   - Previously, `get_synthetic_dataset` overrode `center_box` to `(-3.0, 3.0)`, cramming 5 clusters with `cluster_std = 1.0` on top of one another (`CE` stalled at `~63%` and geometric losses at `~38%–45%`). Restoring `(-10.0, 10.0)` (followed by standard zero-mean/unit-std normalization) produces cleanly separated 5-class clusters.
3. **Class-Count-Adaptive Band Margins, Regularizers & Training Epochs (Notebook `05` Cells 3, 13 & 15)**:
   - Binary margins (`0.60` for Volume, `0.45` for Anisotropy, `0.40` for Curvature) forced extreme outer targets at `N_CLASSES = 5` (e.g., $t_4 = +2.4 \implies \det g \approx 122$). Scaling `margin = max(0.25, 1.50 / N_CLASSES)` (`0.30` for `C = 5`), relaxing isotropic pull-back slightly, and using consistent `row_margins` across training, benchmarking, and the $3 \times 4$ evaluation matrix preserves identical `C = 2` behavior while reaching **`93.33%–95.83%`** accuracy on 5-class `blobs`.
4. **Multi-Class Decision Region & Band-Wall Contour Plotting (Notebook `05` Cells 5 & 15)**:
   - Replaced hardcoded binary `[0.0]` contours (`0.0` is the center of Class `2` when `C = 5`, not a boundary!) and 6-color wraparound with a 10-color palette, a $2 \times 4$ multi-class landscape view (top row: predicted 5-class regions; bottom row: continuous geometric scalar fields with all $C - 1 = 4$ exact band-wall contours $b_c$), and exact multi-threshold contours in the $3 \times 4$ matrix.

### Verified Results on Active `config.yaml` (`blobs`, `5 classes`, `[2, 10, 20, 10, 2]`, `gelu`)
- **Cross-Entropy Baseline**: **`95.83%`**
- **Geometric Volume-Element Loss**: **`95.00%`** (`L4 Gap = +4.155`)
- **Geometric Metric Anisotropy Loss**: **`94.17%`** (`L4 Aniso Gap = +3.620`)
- **Geometric Curvature Margin Loss**: **`93.33%`**


---

## Turn 33

### User
ok, great, check as well notebook 08

### Assistant
Audited, updated, rebuilt, and verified [notebooks/08-symmetric-coupled-multiclass.ipynb](notebooks/08-symmetric-coupled-multiclass.ipynb) (along with [src/diffgeom/geometry.py](src/diffgeom/geometry.py) and [scripts/build_all_notebooks.py](scripts/build_all_notebooks.py)) so that experimenting with 5-class `blobs` (`C = 5`, `gelu`, `[2, 10, 20, 10, 2]`)—as well as arbitrary class counts and architectures in `config.yaml`—works cleanly end-to-end across all five sections:

1. **Synchronized `C_CFG`, Color Palettes, and Decoupled Architecture Shapes (Cell 0)**:
   - Updated `C_CFG` after `get_synthetic_dataset(...)` (`C_CFG = max(C_CFG, int(max(y_cfg_tr.max(), y_cfg_te.max())) + 1)`), expanded `CLASS_COLORS` to 10 distinct colors (eliminating `C >= 5` color collisions), and separated `GEOM_2D_DIMS` (for the fixed 2D triple-junction/tetrahedron tasks) from `CFG_GEOM_DIMS` (for `X_cfg_tr`) so arbitrary `layer_dims` in `config.yaml` never trigger dimension mismatches.
2. **Added Exact $S_C$-Equidistant `DeWitt Fibered (Sym^+_2)^C` to Section 2 & Generalized High-$d_0$ Support (`geometry.py` & Cell 2)**:
   - Because $\dim(\mathrm{Sym}^+_2) = 3$, at most $C = 4$ prototypes can be mutually equidistant in a single 2D metric fiber; for $C = 5$, adding `New: DeWitt Fibered (Sym^+_2)^C` (`fibered_dewitt_simplex_loss` + new `fibered_dewitt_simplex_predict` exported in `diffgeom`, generalized to arbitrary `d_0 >= 2` via `compute_canonical_2frame`) provides exact $S_5$ equidistance alongside single-fiber `Sym^+_2`.
   - Scaled `Prev: 1D Volume Band` margin adaptively with `n_cls` (`max(0.25, 1.50 / n_cls)`), guarded `coupled_subspace_predict` against empty reference-class slices, and increased the `Coupled Subspace Ortho` subset size (`min(140, len(X_tr))`) so all $C \ge 5$ classes are well populated.
3. **Updated Section 4 Visualizations to Display Your Active `config.yaml` Dataset (`blobs`, `C = 5`)**:
   - Previously, Section 4 hardcoded the 3-class triple-junction dataset; when `input_dim == 2`, it now renders the learned geometric representations directly on your configured dataset (`blobs (C=5)`): 5-class `DeWitt Fibered` Voronoi regions overlaid with learned Tissot ellipses, 5-basin `Geodesic Metric-Barrier` Voronoi walls, and the $5 \times 5$ block-diagonal Coupled Grassmannian Kernel matrix $\mathcal{K}_{\Theta \leftrightarrow \mathcal{M}}$ on sorted held-out test points.

### Verified Results in Executed Notebook `08` (`0` errors, all `5` figures rendered)

| Method | 3-Class Blobs Acc | 4-Class Tetra Acc | Configured `blobs (C=5)` Acc |
| :--- | :---: | :---: | :---: |
| **`SGD + Cross-Entropy`** | `99.17%` | `100.00%` | **`96.67%`** |
| **`Prev: 1D Volume Band`** | `97.50%` | `99.17%` | **`94.17%`** |
| **`New: DeWitt Simplex (SGD+FD)`** | `98.33%` | `100.00%` | **`93.33%`** |
| **`New: DeWitt Simplex (Coupled NGD)`** | `98.33%` | `100.00%` | **`94.17%`** |
| **`New: DeWitt Fibered (Sym^+_2)^C`** | `98.33%` | `100.00%` | **`95.83%`** |
| **`New: Geodesic Barrier (Endpoint)`** | `98.33%` | `100.00%` | **`97.50%`** |
| **`New: Coupled Subspace Ortho`** | `97.50%` | `100.00%` | **`95.00%`** |
