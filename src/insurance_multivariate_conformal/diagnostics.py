"""
Coverage and calibration diagnostics.

These tools help actuaries validate that the conformal predictor is
working correctly on test data — and diagnose problems like conditional
coverage failure (different coverage across risk segments).

Key functions:
- coverage_report(): the main validation tool; call this first.
- width_by_dimension(): compare interval widths across methods.
- calibration_plot(): matplotlib visualisation (optional dep).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .predictor import JointConformalPredictor
from .regions import JointPredictionSet


def coverage_report(
    predictor: JointConformalPredictor,
    X_test: NDArray,
    Y_test: Union[Dict[str, NDArray], NDArray],
    exposure: Optional[NDArray] = None,
) -> Dict[str, Any]:
    """
    Full coverage report on a test set.

    Computes joint and marginal empirical coverage rates, mean interval widths,
    and the mean hyperrectangle volume (efficiency metric).

    Parameters
    ----------
    predictor : JointConformalPredictor
        A calibrated predictor.
    X_test : ndarray, shape (m, p)
    Y_test : dict or ndarray
        True test outcomes.
    exposure : ndarray, shape (m,), optional

    Returns
    -------
    dict with keys:
        joint_coverage : float — should be >= 1 - alpha.
        marginal_coverages : dict[str, float] — per-dimension.
        mean_widths : dict[str, float] — mean interval width per dimension.
        mean_volume : float — mean hyperrectangle volume.
        alpha : float — nominal miscoverage level.
        method : str — method used.
        n_test : int — test set size.
        target_coverage : float — 1 - alpha.
        coverage_gap : float — (joint_coverage - target_coverage). Positive = over-covered.
    """
    joint_set = predictor.predict(X_test, exposure=exposure)
    joint_coverage = joint_set.joint_coverage_check(Y_test)
    marginal_coverages = joint_set.marginal_coverage_rates(Y_test)
    widths = joint_set.marginal_intervals()
    mean_widths = {k: float(np.mean(v)) for k, v in widths.items()}
    mean_volume = float(np.mean(joint_set.volume()))

    return {
        "joint_coverage": joint_coverage,
        "marginal_coverages": marginal_coverages,
        "mean_widths": mean_widths,
        "mean_volume": mean_volume,
        "alpha": predictor.alpha,
        "method": predictor.method,
        "n_test": X_test.shape[0],
        "target_coverage": 1.0 - predictor.alpha,
        "coverage_gap": joint_coverage - (1.0 - predictor.alpha),
    }


def width_by_dimension(
    predictor: JointConformalPredictor,
    X_test: NDArray,
    exposure: Optional[NDArray] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Per-dimension interval width statistics over a test set.

    Returns a dict: dimension -> {mean, std, median, q10, q90} of widths.
    Useful for comparing methods (LWC should give narrower widths than Bonferroni).
    """
    joint_set = predictor.predict(X_test, exposure=exposure)
    widths = joint_set.marginal_intervals()

    stats = {}
    for k, w in widths.items():
        stats[k] = {
            "mean": float(np.mean(w)),
            "std": float(np.std(w)),
            "median": float(np.median(w)),
            "q10": float(np.quantile(w, 0.10)),
            "q90": float(np.quantile(w, 0.90)),
        }
    return stats


def compare_methods(
    models: Dict[str, Any],
    X_cal: NDArray,
    Y_cal: Union[Dict[str, NDArray], NDArray],
    X_test: NDArray,
    Y_test: Union[Dict[str, NDArray], NDArray],
    alpha: float = 0.05,
    exposure_cal: Optional[NDArray] = None,
    exposure_test: Optional[NDArray] = None,
    zero_claim_mask: Optional[NDArray] = None,
    methods: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Calibrate and evaluate multiple methods on the same cal/test split.

    Returns a dict: method -> coverage_report dict.

    Useful for showing LWC efficiency gains over Bonferroni on real data.
    """
    from .predictor import JointConformalPredictor

    if methods is None:
        methods = ["bonferroni", "gwc", "lwc"]

    results = {}
    for method in methods:
        pred = JointConformalPredictor(models=models, alpha=alpha, method=method)
        pred.calibrate(
            X_cal, Y_cal,
            exposure=exposure_cal,
            zero_claim_mask=zero_claim_mask,
        )
        results[method] = coverage_report(pred, X_test, Y_test, exposure=exposure_test)

    return results


def calibration_plot(
    predictor: JointConformalPredictor,
    X_test: NDArray,
    Y_test: Union[Dict[str, NDArray], NDArray],
    exposure: Optional[NDArray] = None,
    figsize: tuple = (10, 5),
) -> "matplotlib.figure.Figure":
    """
    Calibration plot: empirical coverage vs nominal level across alpha values.

    Sweeps alpha from 0.01 to 0.30 and plots empirical joint coverage vs 1-alpha.
    A well-calibrated predictor should lie on or above the diagonal.

    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for calibration_plot()")

    from .predictor import JointConformalPredictor

    alphas = np.linspace(0.01, 0.30, 30)
    joint_coverages = []
    marginal_coverages_by_dim: Dict[str, List[float]] = {
        k: [] for k in predictor.dimensions_
    }

    # Re-calibrate at each alpha level using the same calibration data
    cal_data = predictor.calibrated_scores_
    for a in alphas:
        from .calibration import calibrate as _cal
        new_cal = _cal(
            models=predictor.models,
            X_cal=np.zeros((cal_data.n_cal, 1)),  # placeholder — we recalibrate
            Y_cal={k: np.zeros(cal_data.n_cal) for k in cal_data.dimensions},
            alpha=a,
            method=predictor.method,
        )
        # Actually: we can compute coverage directly from stored residuals
        # by finding new thresholds on the stored residual matrix
        from .methods import lwc_quantile_exact, gwc_quantile, bonferroni_quantile
        res = cal_data.residuals
        n = cal_data.n_cal

        if predictor.method == "lwc":
            thresholds, mu_hat, sigma_hat = lwc_quantile_exact(res, a)
        elif predictor.method == "gwc":
            q, mu_hat, sigma_hat = gwc_quantile(res, a)
            thresholds = np.full(len(cal_data.dimensions), q)
        else:
            thresholds = bonferroni_quantile(res, a)
            mu_hat = cal_data.mu_hat
            sigma_hat = cal_data.sigma_hat

        # Predict on test with these thresholds
        tmp_pred = JointConformalPredictor(
            models=predictor.models, alpha=a, method=predictor.method
        )
        from copy import deepcopy
        from .calibration import CalibratedScores
        tmp_pred.calibrated_scores_ = CalibratedScores(
            dimensions=cal_data.dimensions,
            residuals=cal_data.residuals,
            mu_hat=mu_hat,
            sigma_hat=sigma_hat,
            n_cal=n,
            method=predictor.method,
            alpha=a,
            thresholds=thresholds if predictor.method in ("gwc", "lwc") else None,
            per_dim_quantiles=thresholds if predictor.method in ("bonferroni", "sidak") else None,
        )
        tmp_pred.dimensions_ = cal_data.dimensions

        report = coverage_report(tmp_pred, X_test, Y_test, exposure=exposure)
        joint_coverages.append(report["joint_coverage"])
        for k in predictor.dimensions_:
            marginal_coverages_by_dim[k].append(report["marginal_coverages"][k])

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Joint coverage
    ax = axes[0]
    ax.plot(1 - alphas, joint_coverages, "o-", label="Empirical joint", markersize=4)
    ax.plot([0.7, 1.0], [0.7, 1.0], "k--", alpha=0.5, label="Diagonal (perfect)")
    ax.set_xlabel("Nominal coverage (1 - alpha)")
    ax.set_ylabel("Empirical joint coverage")
    ax.set_title("Joint Coverage Calibration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Marginal coverage per dimension
    ax2 = axes[1]
    for k, cov_list in marginal_coverages_by_dim.items():
        ax2.plot(1 - alphas, cov_list, "o-", label=k, markersize=4)
    ax2.plot([0.7, 1.0], [0.7, 1.0], "k--", alpha=0.5, label="Diagonal")
    ax2.set_xlabel("Nominal coverage (1 - alpha)")
    ax2.set_ylabel("Empirical marginal coverage")
    ax2.set_title("Marginal Coverage by Dimension")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
