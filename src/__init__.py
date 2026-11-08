"""Differential geometry utilities for deep neural networks."""

from .geometry import (
    MoonsMLP,
    anisotropy_margin_loss,
    compute_christoffel_symbols,
    compute_data_information_matrix,
    compute_gaussian_curvature,
    compute_layer_geometry,
    compute_layer_jacobian,
    compute_layer_pullback_metric,
    compute_metric_spectrum,
    compute_parameter_fim,
    compute_volume_element,
    curvature_margin_loss,
    flatten_params,
    get_moons_dataset,
    make_evaluation_grid,
    volume_contrastive_loss,
)

__all__ = [
    "MoonsMLP",
    "get_moons_dataset",
    "make_evaluation_grid",
    "compute_layer_jacobian",
    "compute_layer_pullback_metric",
    "compute_volume_element",
    "compute_metric_spectrum",
    "compute_christoffel_symbols",
    "compute_gaussian_curvature",
    "compute_layer_geometry",
    "compute_parameter_fim",
    "compute_data_information_matrix",
    "curvature_margin_loss",
    "volume_contrastive_loss",
    "anisotropy_margin_loss",
    "flatten_params",
]
