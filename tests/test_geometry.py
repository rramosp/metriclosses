"""Unit tests for the diffgeom package."""

from __future__ import annotations

import torch

from diffgeom import (
    MoonsMLP,
    __version__,
    anisotropy_margin_loss,
    build_knn_jet_operator,
    compute_coupled_dewitt_metric,
    compute_coupled_geometric_metric,
    compute_data_information_matrix,
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
    curvature_margin_loss,
    flatten_params,
    get_synthetic_dataset,
    scalar_geometric_predict,
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
