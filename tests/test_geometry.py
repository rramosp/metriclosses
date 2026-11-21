"""Unit tests for the diffgeom package."""

from __future__ import annotations

import torch

from diffgeom import (
    MoonsMLP,
    __version__,
    analytic_head_subspace_loss,
    analytic_head_subspace_predict,
    anisotropy_margin_loss,
    build_dewitt_simplex_prototypes,
    build_knn_jet_operator,
    build_log_fiber_prototypes,
    build_metric_barrier_pairs,
    compute_canonical_2frame,
    compute_coupled_dewitt_metric,
    compute_coupled_geometric_metric,
    compute_data_information_matrix,
    compute_dewitt_dist_sq,
    compute_fast_graph_geometry,
    compute_fd_graph_curvature,
    compute_fd_pullback_metric,
    compute_knn_jet_geometry,
    compute_layer_geometry,
    compute_layer_pullback_metric,
    compute_parameter_fim,
    compute_sketched_plane_curvature,
    compute_sketched_pullback_metric,
    compute_uncoupled_coord_metric,
    coupled_subspace_orthogonality_loss,
    coupled_subspace_predict,
    curvature_margin_loss,
    dewitt_simplex_loss,
    dewitt_simplex_predict,
    fibered_dewitt_simplex_loss,
    fibered_dewitt_simplex_predict,
    flatten_params,
    geodesic_barrier_loss,
    geodesic_knn_predict,
    get_synthetic_dataset,
    gram_subspace_orthogonality_loss,
    hyperboloid_dewitt_dist_sq,
    log_fiber_dewitt_loss,
    log_fiber_dewitt_predict,
    midpoint_polygonal_barrier_loss,
    polygonal_barrier_knn_predict,
    scalar_geometric_predict,
    sketched_param_subspace_loss,
    sketched_param_subspace_predict,
    sym2_to_hyperboloid,
    volume_contrastive_loss,
    woodbury_coupled_ngd_direction,
)
from diffgeom.geometry import _christoffel_and_curvature_from_g_single


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_synthetic_datasets_binary_and_multiclass() -> None:
    for ds_name, extra in [
        ("moons", {}),
        ("circles", {"factor": 0.45}),
        ("blobs", {"cluster_std": 0.65}),
        ("classification", {"class_sep": 1.3}),
        ("gaussian_quantiles", {"cov": 1.2}),
    ]:
        for n_cls in (2, 3):
            X_tr, y_tr, X_te, y_te = get_synthetic_dataset(
                dataset_name=ds_name,
                n_train=60,
                n_test=30,
                n_classes=n_cls,
                noise=0.10,
                seed=42,
                **extra,
            )
            assert X_tr.shape == (60, 2)
            assert X_te.shape == (30, 2)
            assert int(y_tr.unique().numel()) == n_cls
            assert int(y_te.unique().numel()) == n_cls


def test_analytical_sphere_and_saddle_curvature() -> None:
    R_s = 2.0
    g_sphere = lambda x: ((2.0 * R_s**2 / (R_s**2 + torch.sum(x**2))) ** 2) * torch.eye(2, dtype=x.dtype)
    _, _, _, K_sphere = _christoffel_and_curvature_from_g_single(g_sphere)(
        torch.tensor([0.8, -0.6], dtype=torch.float64)
    )
    assert abs(K_sphere.item() - 0.25) < 1e-10

    g_saddle = lambda x: torch.eye(2, dtype=x.dtype) + torch.outer(
        torch.stack([2.0 * x[0], -1.5 * x[1]]),
        torch.stack([2.0 * x[0], -1.5 * x[1]]),
    )
    _, _, _, K_saddle = _christoffel_and_curvature_from_g_single(g_saddle)(
        torch.zeros(2, dtype=torch.float64)
    )
    assert abs(K_saddle.item() - (-3.0)) < 1e-10


def test_gauss_equation_and_e2_symmetry_quotient() -> None:
    X_tr, _, _, _ = get_synthetic_dataset("moons", n_train=32, n_test=16, n_classes=2, seed=7)
    model = MoonsMLP(layer_dims=[2, 6, 4, 2], activation="tanh", output_activation="identity", seed=7)

    ref_geom = compute_layer_geometry(model, X_tr, layer_idx=model.num_layers, metric_mode="graph")
    g_fast, K_fast = compute_fast_graph_geometry(model, X_tr)
    assert torch.allclose(g_fast, ref_geom["g"], atol=1e-12)
    assert torch.allclose(K_fast, ref_geom["K"], atol=1e-12)

    # Pure output translation along final bias must lie in exact nullspace of DeWitt metric
    G_dw = compute_coupled_dewitt_metric(model, X_tr, metric_mode="graph", damping=0.0)
    P = flatten_params(dict(model.named_parameters())).numel()
    v_trans = torch.zeros(P, dtype=torch.float64)
    v_trans[-2] = 1.0
    assert abs(float(v_trans @ (G_dw @ v_trans))) < 1e-24


def test_custom_activations_multiclass_losses_and_efficient_operators() -> None:
    X_tr, y_tr, _, _ = get_synthetic_dataset("blobs", n_train=48, n_test=24, n_classes=3, seed=7)
    model = MoonsMLP(
        layer_dims=[2, 6, 8, 3],
        activation="gelu",
        output_activation="tanh",
        seed=7,
    )
    P = flatten_params(dict(model.named_parameters())).numel()

    fim, _ = compute_parameter_fim(model, X_tr[:20])
    dim_mat = compute_data_information_matrix(model, X_tr[:20])
    g_unc = compute_uncoupled_coord_metric(model, X_tr[:20])
    assert fim.shape == (P, P)
    assert dim_mat.shape == (20, 2, 2)
    assert g_unc.shape == (P, P)

    for task in ("curvature", "volume", "anisotropy"):
        G_cp = compute_coupled_geometric_metric(model, X_tr[:20], y_tr[:20], task=task, n_classes=3)
        assert G_cp.shape == (P, P) and torch.isfinite(G_cp).all()

    g_graph, K_graph = compute_fast_graph_geometry(model, X_tr[:20])
    l_curv, _ = curvature_margin_loss(K_graph, y_tr[:20], g_graph, n_classes=3)
    l_curv.backward()
    grad_flat = torch.cat(
        [(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in model.parameters()]
    )
    nat_dir, r_norm = woodbury_coupled_ngd_direction(
        model, X_tr[:16], y_tr[:16], grad_flat, task="curvature", n_classes=3
    )
    assert nat_dir.shape == (P,) and r_norm > 0.0

    g_euc = compute_layer_pullback_metric(model, X_tr[:20], layer_idx=model.num_layers, metric_mode="euclidean")
    l_vol, _ = volume_contrastive_loss(g_euc, y_tr[:20], n_classes=3)
    l_ani, _ = anisotropy_margin_loss(g_graph, y_tr[:20], n_classes=3)
    preds = scalar_geometric_predict(K_graph, n_classes=3)
    assert preds.shape == (20,) and torch.isfinite(l_vol) and torch.isfinite(l_ani)

    g_fd = compute_fd_pullback_metric(model, X_tr[:20])
    g_fdk, K_fdk = compute_fd_graph_curvature(model, X_tr[:20])
    knn_idx, jet_op = build_knn_jet_operator(X_tr)
    g_knn, K_knn = compute_knn_jet_geometry(model, X_tr, knn_idx, jet_op)
    g_sk, _ = compute_sketched_pullback_metric(model, X_tr[:20])
    _, K_sk = compute_sketched_plane_curvature(model, X_tr[:20])
    assert all(torch.isfinite(t).all() for t in (g_fd, g_fdk, K_fdk, g_knn, K_knn, g_sk, K_sk))


def test_symmetric_coupled_multiclass_losses() -> None:
    # 1. Exact pairwise DeWitt equidistance for C = 2, 3, 4
    for c_cnt in (2, 3, 4):
        protos, proto_isq = build_dewitt_simplex_prototypes(c_cnt, tau=0.75)
        dists = []
        for i in range(c_cnt):
            for j in range(i + 1, c_cnt):
                ev = torch.linalg.eigvalsh(proto_isq[i] @ protos[j] @ proto_isq[i])
                dists.append(torch.sqrt((torch.log(ev) ** 2).sum()).item())
        assert max(dists) - min(dists) < 1e-13

    # 2. Exact S_3 label-permutation invariance across all three symmetric losses
    X_tr, y_tr, X_te, _ = get_synthetic_dataset("blobs", n_train=36, n_test=18, n_classes=3, seed=11)
    model = MoonsMLP(layer_dims=[2, 6, 2], activation="tanh", output_activation="identity", seed=7)
    g_graph = compute_fd_pullback_metric(model, X_tr, metric_mode="graph")
    _, proto_isq3 = build_dewitt_simplex_prototypes(3, tau=0.75)

    perm = torch.tensor([2, 0, 1], dtype=torch.long)
    y_perm = perm[y_tr]
    inv_perm = torch.argsort(perm)

    # Permuting y AND prototype assignment identically must preserve loss to machine precision
    l_orig, _ = dewitt_simplex_loss(g_graph, y_tr, proto_isq3)
    l_perm, _ = dewitt_simplex_loss(g_graph, y_perm, proto_isq3[inv_perm])
    assert abs(l_orig.item() - l_perm.item()) < 1e-13
    preds_dw = dewitt_simplex_predict(g_graph, proto_isq3)
    assert preds_dw.shape == (36,)

    # Geodesic barrier loss is strictly invariant under any label permutation without prototype relabeling
    s1, d1, s2, d2 = build_metric_barrier_pairs(X_tr, y_tr, k_same=4, k_diff=5)
    s1p, d1p, s2p, d2p = build_metric_barrier_pairs(X_tr, y_perm, k_same=4, k_diff=5)
    g_euc = compute_fd_pullback_metric(model, X_tr, metric_mode="euclidean")
    lb_orig, _ = geodesic_barrier_loss(g_euc, X_tr, s1, d1, s2, d2)
    lb_perm, _ = geodesic_barrier_loss(g_euc, X_tr, s1p, d1p, s2p, d2p)
    assert abs(lb_orig.item() - lb_perm.item()) < 1e-14
    preds_gb = geodesic_knn_predict(model, X_te, X_tr, y_tr, n_classes=3)
    assert preds_gb.shape == (18,)

    # Coupled subspace orthogonality loss is strictly invariant under any label permutation
    lc_orig, _ = coupled_subspace_orthogonality_loss(model, X_tr, y_tr, use_fd=True)
    lc_perm, _ = coupled_subspace_orthogonality_loss(model, X_tr, y_perm, use_fd=True)
    assert abs(lc_orig.item() - lc_perm.item()) < 1e-14
    preds_cs, _ = coupled_subspace_predict(model, X_te, X_tr, y_tr, n_classes=3, use_fd=True)
    assert preds_cs.shape == (18,)

    # Fibered (Sym^+_2)^C DeWitt simplex loss & predict for large C (e.g. C = 6) and d_0 >= 2
    X6, y6, X6_te, _ = get_synthetic_dataset("blobs", n_train=36, n_test=18, n_classes=6, input_dim=4, seed=19)
    q6 = compute_canonical_2frame(X6)
    m_fib = MoonsMLP(layer_dims=[4, 8, 12], activation="tanh", output_activation="identity", seed=7)
    lf_0, _ = fibered_dewitt_simplex_loss(m_fib, X6, y6, n_classes=6, frame_2d=q6)
    preds_fib = fibered_dewitt_simplex_predict(m_fib, X6_te, n_classes=6, frame_2d=q6)
    assert torch.isfinite(lf_0) and preds_fib.shape == (18,)



def test_scalable_highdim_manyclass_losses() -> None:
    torch.manual_seed(0)
    # 1. Exact Minkowski-hyperboloid DeWitt formula matches 2x2 generalized eigenvalue formula
    a_rand = torch.randn(24, 2, 2, dtype=torch.float64)
    g_spd = torch.eye(2, dtype=torch.float64) + torch.einsum("nai,naj->nij", a_rand, a_rand)
    protos, proto_isq = build_dewitt_simplex_prototypes(4, tau=0.75)
    d2_eig = compute_dewitt_dist_sq(g_spd, proto_isq)
    sg, ug = sym2_to_hyperboloid(g_spd)
    sp, up = sym2_to_hyperboloid(protos)
    up_mink = up * torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64)
    d2_hyp = hyperboloid_dewitt_dist_sq(sg[:, None, None], ug[:, None, None, :], sp[None, :, None], up_mink[None, :, None, :])
    assert (d2_hyp - d2_eig).abs().max().item() < 1e-9

    # 2. Exact Gram trace identity matches explicit O(C^2) off-diagonal class-subspace overlap
    w_raw = torch.randn(30, 3, 8, dtype=torch.float64)
    w_hat = w_raw / torch.sqrt((w_raw * w_raw).sum(dim=(-2, -1), keepdim=True))
    y_lbl = torch.arange(30) % 6
    _, info_gram = gram_subspace_orthogonality_loss(w_hat, y_lbl)
    w_cls = []
    for c in range(6):
        wc = w_hat[y_lbl == c].mean(dim=0)
        w_cls.append(wc / torch.sqrt((wc * wc).sum() + 1e-8))
    w_cls_t = torch.stack(w_cls, dim=0)
    pair_sq = (torch.einsum("car,dbr->cdab", w_cls_t, w_cls_t) ** 2).sum(dim=(-2, -1))
    explicit_off = (pair_sq * (1.0 - torch.eye(6, dtype=torch.float64))).sum().item() / 30.0
    assert abs(info_gram["mean_cross_cls_sq"] - explicit_off) < 1e-13

    # 3. High-d_0 + multiclass forward/backward and prediction smoke tests (d_0 = 16, C = 8)
    X_tr, y_tr, X_te, _ = get_synthetic_dataset("blobs", n_train=48, n_test=24, n_classes=8, input_dim=16, seed=5)
    q0 = compute_canonical_2frame(X_tr)
    assert q0.shape == (16, 2)
    assert torch.allclose(q0.T @ q0, torch.eye(2, dtype=torch.float64), atol=1e-12)

    k_fib, s_p8, u_p8 = build_log_fiber_prototypes(8)
    assert k_fib == 3
    net_lf = MoonsMLP(layer_dims=[16, 12, 2 * k_fib], seed=3)
    loss_lf, _ = log_fiber_dewitt_loss(net_lf, X_tr, y_tr, k_fib, q0, s_p8, u_p8, m_neg=4)
    loss_lf.backward()
    assert log_fiber_dewitt_predict(net_lf, X_te, k_fib, q0, s_p8, u_p8).shape == (24,)

    s_s, d_s, s_d, d_d = build_metric_barrier_pairs(X_tr, y_tr, k_same=3, k_diff=5)
    net_2d = MoonsMLP(layer_dims=[16, 16, 2], seed=3)
    loss_mb, _ = midpoint_polygonal_barrier_loss(net_2d, X_tr, s_s, d_s, s_d, d_d)
    loss_mb.backward()
    assert polygonal_barrier_knn_predict(net_2d, X_te, X_tr, y_tr, n_classes=8).shape == (24,)

    net_2d.zero_grad(set_to_none=True)
    loss_ah, _ = analytic_head_subspace_loss(net_2d, X_tr, y_tr, q0)
    loss_ah.backward()
    assert analytic_head_subspace_predict(net_2d, X_te, X_tr, y_tr, n_classes=8, frame_2d=q0).shape == (24,)

    net_2d.zero_grad(set_to_none=True)
    loss_sp, _ = sketched_param_subspace_loss(net_2d, X_tr, y_tr, q0, r_probes=4)
    loss_sp.backward()
    assert sketched_param_subspace_predict(net_2d, X_te, X_tr, y_tr, n_classes=8, frame_2d=q0, r_probes=6).shape == (24,)



def test_blobs_five_classes_separable_and_synced() -> None:
    # Even when centers=5 is passed with default n_classes=2 or n_classes=5, 5 well-separated classes are returned
    _, y_tr, _, y_te = get_synthetic_dataset(
        "blobs", n_train=150, n_test=50, n_classes=5, centers=5, cluster_std=1.0, seed=42
    )
    assert int(y_tr.max().item()) == 4
    assert int(y_te.max().item()) == 4
    # If n_classes is later changed to 3 while centers=5 remains in kwargs, n_classes=3 takes precedence
    _, y3, _, _ = get_synthetic_dataset(
        "blobs", n_train=90, n_test=30, n_classes=3, centers=5, cluster_std=1.0, seed=42
    )
    assert int(y3.max().item()) == 2
