"""Bootstrap LiNGAM skeleton followed by polynomial regression-error orientation."""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from .graph_conversion import binary_adjacency, break_cycles, seed_numeric, validate_data


def bootstrap_skeleton(matrix, *, n_bootstrap: int, edge_threshold: float,
                       weight_floor: float, seed: int, lingam_options: Mapping) -> np.ndarray:
    from causallearn.search.FCMBased.lingam.direct_lingam import DirectLiNGAM
    values = validate_data(matrix)
    if n_bootstrap <= 0 or not 0 <= edge_threshold <= 1 or weight_floor < 0:
        raise ValueError("Invalid bootstrap count, frequency threshold, or weight floor.")
    seed_numeric(seed)
    samples, size = values.shape
    frequency = np.zeros((size, size))
    rng = np.random.default_rng(seed)
    successful = 0
    last_error = None
    for _ in range(n_bootstrap):
        indices = rng.choice(samples, samples, replace=True)
        try:
            model = DirectLiNGAM(**dict(lingam_options))
            model.fit(values[indices])
            weights = np.abs(np.asarray(model.adjacency_matrix_))
            # Symmetrization makes the library's row convention immaterial here.
            skeleton = ((weights + weights.T) > weight_floor).astype(int)
            np.fill_diagonal(skeleton, 0)
            frequency += skeleton
            successful += 1
        except Exception as error:
            last_error = error
    if successful == 0:
        raise RuntimeError("All bootstrap LiNGAM fits failed.") from last_error
    frequency /= successful
    result = (frequency >= edge_threshold).astype(int)
    np.fill_diagonal(result, 0)
    return result


def reci_orient(matrix, skeleton, *, degree: int, regression_options: Mapping) -> np.ndarray:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    values = validate_data(matrix)
    size = values.shape[1]
    skeleton = binary_adjacency(skeleton, size)
    if degree <= 0:
        raise ValueError("Polynomial degree must be positive.")
    adjacency = np.zeros((size, size), dtype=int)
    poly = PolynomialFeatures(degree=degree)
    for i in range(size):
        for j in range(i + 1, size):
            if skeleton[i, j] == 0 and skeleton[j, i] == 0:
                continue
            xi, xj = values[:, i:i + 1], values[:, j:j + 1]
            xi_poly = poly.fit_transform(xi)
            pred_ij = LinearRegression(**dict(regression_options)).fit(xi_poly, xj).predict(xi_poly)
            xj_poly = poly.fit_transform(xj)
            pred_ji = LinearRegression(**dict(regression_options)).fit(xj_poly, xi).predict(xj_poly)
            if mean_squared_error(xj, pred_ij) < mean_squared_error(xi, pred_ji):
                adjacency[i, j] = 1
            else:
                adjacency[j, i] = 1
    return break_cycles(adjacency)


def bootstrap_reci(matrix, *, seed: int, n_bootstrap: int,
                    skeleton_threshold: float, weight_floor: float,
                    reci_degree: int, lingam_options: Mapping,
                    regression_options: Mapping) -> np.ndarray:
    values = validate_data(matrix)
    skeleton = bootstrap_skeleton(values, n_bootstrap=n_bootstrap,
                                  edge_threshold=skeleton_threshold, weight_floor=weight_floor,
                                  seed=seed, lingam_options=lingam_options)
    return reci_orient(values, skeleton, degree=reci_degree, regression_options=regression_options)
