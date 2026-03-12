"""
Core conformal calibration: fitting nonconformity scores on calibration data.

Design decision: we always use absolute residuals |y - y_hat| as the
nonconformity score used to construct prediction intervals. This is the
standard split conformal approach and means the interval half-width is
directly in the original units (claims, £) — interpretable and correct.

The deviance score functions in scores.py exist for use in custom workflows
but are not used in the default calibration pipeline, because deviance residuals
are not in prediction units and cannot be added directly to point predictions
to form intervals.

For the GWC/LWC methods: we apply coordinate-wise standardization to absolute
residuals. The standardization handles the Poisson/Gamma scale mismatch
(frequency residuals ~ 0.1-2, severity residuals ~ £200-£3,000). After
standardization both are z-scores; the quantile of the max-score is then
converted back to per-dimension half-widths in original units.

For Bonferroni/Sidak: per-dimension quantile of |y - y_hat| at level 1-alpha/d.
This is exactly the standard split conformal quantile applied per dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from .methods import (
    gwc_quantile,
    lwc_quantile_exact,
    bonferroni_quantile,
    sidak_quantile,
)


ModelLike = Any  # Any object with .predict(X) -> ndarray


def _get_predictions(
    model: ModelLike,
    X: NDArray,
    exposure: Optional[NDArray] = None,
) -> NDArray[np.floating]:
    """
    Get predictions from a model, optionally using exposure.

    Tries predict_with_exposure(X, exposure) first, then predict(X, exposure),
    then predict(X). This allows sklearn-style models and custom wrappers.
    """
    if exposure is not None:
        if hasattr(model, "predict_with_exposure"):
            return np.asarray(model.predict_with_exposure(X, exposure), dtype=float)
        try:
            return np.asarray(model.predict(X, exposure), dtype=float)
        except TypeError:
            pass
    return np.asarray(model.predict(X), dtype=float)


def _normalise_models(
    models: Union[Dict[str, ModelLike], List[ModelLike], ModelLike],
) -> Dict[str, ModelLike]:
    """Normalise models to a dict keyed by dimension name."""
    if isinstance(models, dict):
        return models
    if isinstance(models, (list, tuple)):
        return {str(i): m for i, m in enumerate(models)}
    return {"0": models}


def _normalise_y(
    Y: Union[Dict[str, NDArray], NDArray],
    model_keys: Sequence[str],
) -> Dict[str, NDArray]:
    """Normalise Y to a dict keyed by dimension name."""
    if isinstance(Y, dict):
        return {k: np.asarray(v, dtype=float) for k, v in Y.items()}
    arr = np.asarray(Y, dtype=float)
    if arr.ndim == 1:
        return {model_keys[0]: arr}
    if arr.ndim == 2:
        assert arr.shape[1] == len(model_keys), (
            f"Y has {arr.shape[1]} columns but {len(model_keys)} models"
        )
        return {k: arr[:, i] for i, k in enumerate(model_keys)}
    raise ValueError(f"Y must be 1D or 2D, got shape {arr.shape}")


@dataclass
class CalibratedScores:
    """
    Calibration output: residuals, standardization statistics, and thresholds.

    Attributes
    ----------
    dimensions : list of str
        Dimension names (e.g. ['frequency', 'severity']).
    residuals : ndarray, shape (n, d)
        Absolute calibration residuals |y - y_hat| per dimension.
        These are in original units (claims, £) — used to compute thresholds.
    mu_hat : ndarray, shape (d,)
        Per-dimension mean of calibration residuals (standardization mean).
    sigma_hat : ndarray, shape (d,)
        Per-dimension std of calibration residuals (standardization std).
    n_cal : int
        Number of calibration points.
    method : str
        Conformal method: 'bonferroni', 'sidak', 'gwc', 'lwc'.
    alpha : float
        Joint miscoverage level.
    zero_claim_mask : ndarray of bool or None
        Which calibration observations had zero claims (severity masked to 0).

    Computed thresholds (set during calibrate()):
    thresholds : ndarray, shape (d,) or None
        For GWC/LWC: per-dimension thresholds in standardized units.
        Convert to raw half-widths via: thresholds * sigma_hat + mu_hat.
    per_dim_quantiles : ndarray, shape (d,) or None
        For Bonferroni/Sidak: per-dimension raw quantiles in original units.
        These are directly the half-widths.
    """

    dimensions: List[str]
    residuals: NDArray[np.floating]
    mu_hat: NDArray[np.floating]
    sigma_hat: NDArray[np.floating]
    n_cal: int
    method: str
    alpha: float
    zero_claim_mask: Optional[NDArray[np.bool_]] = None

    thresholds: Optional[NDArray[np.floating]] = field(default=None, repr=False)
    per_dim_quantiles: Optional[NDArray[np.floating]] = field(default=None, repr=False)

    def standardized_residuals(self) -> NDArray[np.floating]:
        """Return (residuals - mu_hat) / sigma_hat, shape (n, d)."""
        return (self.residuals - self.mu_hat) / self.sigma_hat

    def max_scores(self) -> NDArray[np.floating]:
        """Return max standardized score per calibration observation, shape (n,)."""
        return np.max(self.standardized_residuals(), axis=1)

    def interval_half_widths(self) -> NDArray[np.floating]:
        """
        Per-dimension interval half-widths in original prediction units.

        For GWC/LWC: hw_j = threshold_j * sigma_hat_j + mu_hat_j
            (inverts the standardization; threshold_j is in standardized units)
        For Bonferroni/Sidak: hw_j = per_dim_quantiles_j
            (direct quantile of |y - y_hat|; already in original units)

        Returns
        -------
        ndarray, shape (d,)
            Half-widths: prediction interval is [y_pred_j - hw_j, y_pred_j + hw_j].
        """
        if self.method in ("gwc", "lwc"):
            if self.thresholds is None:
                raise RuntimeError("thresholds not set — internal error")
            # Invert standardization: q_std * sigma + mu = raw half-width
            return self.thresholds * self.sigma_hat + self.mu_hat
        else:  # bonferroni or sidak
            if self.per_dim_quantiles is None:
                raise RuntimeError("per_dim_quantiles not set — internal error")
            return self.per_dim_quantiles


def _compute_absolute_residuals(
    models: Dict[str, ModelLike],
    X_cal: NDArray,
    Y_cal: Dict[str, NDArray],
    exposure: Optional[NDArray] = None,
    zero_claim_mask: Optional[NDArray[np.bool_]] = None,
) -> NDArray[np.floating]:
    """
    Compute absolute residuals |y - y_hat| per dimension.

    Returns shape (n, d). Always uses absolute error — correct units for
    interval construction (half-widths added directly to point predictions).

    zero_claim_mask: where True, sets severity residual to 0 (conservative
    masking for unobserved severity on zero-claim policies).
    """
    keys = list(models.keys())
    n = X_cal.shape[0]
    d = len(keys)
    residuals = np.zeros((n, d), dtype=float)

    for j, key in enumerate(keys):
        model = models[key]
        y_true = Y_cal[key]
        y_pred = _get_predictions(model, X_cal, exposure)
        res = np.abs(y_true - y_pred)

        if zero_claim_mask is not None and key == "severity":
            res = res.copy()
            res[zero_claim_mask] = 0.0

        residuals[:, j] = res

    return residuals


def compute_residuals(
    models: Dict[str, ModelLike],
    X_cal: NDArray,
    Y_cal: Dict[str, NDArray],
    score_fn: str = "absolute",
    exposure: Optional[NDArray] = None,
    zero_claim_mask: Optional[NDArray[np.bool_]] = None,
) -> NDArray[np.floating]:
    """
    Compute nonconformity residuals for each dimension.

    score_fn is accepted for API compatibility but the only supported value
    in the main pipeline is 'absolute'. For deviance-based scores, import
    the score functions from scores.py directly.

    Returns shape (n, d).
    """
    return _compute_absolute_residuals(
        models, X_cal, Y_cal,
        exposure=exposure,
        zero_claim_mask=zero_claim_mask,
    )


def calibrate(
    models: Union[Dict[str, ModelLike], List[ModelLike]],
    X_cal: NDArray,
    Y_cal: Union[Dict[str, NDArray], NDArray],
    alpha: float = 0.05,
    method: str = "lwc",
    score_fn: str = "absolute",
    exposure: Optional[NDArray] = None,
    zero_claim_mask: Optional[NDArray[np.bool_]] = None,
) -> CalibratedScores:
    """
    Calibrate a joint conformal predictor on held-out data.

    Parameters
    ----------
    models : dict or list
        Fitted base models with .predict(X) interface.
        Dict: {'frequency': glm, 'severity': gbm}
        List: [model_0, model_1, ...]
    X_cal : ndarray, shape (n, p)
        Calibration feature matrix (held-out from training).
    Y_cal : dict or ndarray
        Calibration targets. Dict keys must match models keys.
    alpha : float
        Joint miscoverage level. 0.05 = 95% joint coverage.
    method : str
        'bonferroni': per-dimension quantile at 1-alpha/d. Valid, conservative.
        'sidak': per-dimension quantile at 1-(1-alpha)^(1/d). Valid under independence only.
        'gwc': global worst-case max-score. Valid, O(dn).
        'lwc': local worst-case max-score. Valid, tightest, O(d^2 n log n).
    score_fn : str
        Accepted but currently only 'absolute' is used for interval construction.
    exposure : ndarray, shape (n,), optional
        Exposure values for Poisson frequency models.
    zero_claim_mask : ndarray of bool, shape (n,), optional
        True where severity is unobserved (N=0 policies).

    Returns
    -------
    CalibratedScores
    """
    models_dict = _normalise_models(models)
    keys = list(models_dict.keys())
    Y_dict = _normalise_y(Y_cal, keys)

    n = X_cal.shape[0]
    for k in keys:
        if k not in Y_dict:
            raise ValueError(f"Y_cal missing key '{k}' (available: {list(Y_dict.keys())})")
        if Y_dict[k].shape[0] != n:
            raise ValueError(
                f"Y_cal['{k}'] has {Y_dict[k].shape[0]} rows, X_cal has {n}"
            )

    if zero_claim_mask is not None:
        zero_claim_mask = np.asarray(zero_claim_mask, dtype=bool)
        if zero_claim_mask.shape[0] != n:
            raise ValueError(
                f"zero_claim_mask has {zero_claim_mask.shape[0]} entries, expected {n}"
            )

    residuals = _compute_absolute_residuals(
        models_dict, X_cal, Y_dict,
        exposure=exposure,
        zero_claim_mask=zero_claim_mask,
    )

    d = len(keys)
    mu_hat = np.mean(residuals, axis=0)
    sigma_hat = np.maximum(np.std(residuals, axis=0), 1e-8)

    if method == "bonferroni":
        per_dim_q = bonferroni_quantile(residuals, alpha, d=d)
        cal = CalibratedScores(
            dimensions=keys,
            residuals=residuals,
            mu_hat=mu_hat,
            sigma_hat=sigma_hat,
            n_cal=n,
            method=method,
            alpha=alpha,
            zero_claim_mask=zero_claim_mask,
            per_dim_quantiles=per_dim_q,
        )
    elif method == "sidak":
        per_dim_q = sidak_quantile(residuals, alpha, d=d)
        cal = CalibratedScores(
            dimensions=keys,
            residuals=residuals,
            mu_hat=mu_hat,
            sigma_hat=sigma_hat,
            n_cal=n,
            method=method,
            alpha=alpha,
            zero_claim_mask=zero_claim_mask,
            per_dim_quantiles=per_dim_q,
        )
    elif method == "gwc":
        q_scalar, mu_hat, sigma_hat = gwc_quantile(residuals, alpha)
        cal = CalibratedScores(
            dimensions=keys,
            residuals=residuals,
            mu_hat=mu_hat,
            sigma_hat=sigma_hat,
            n_cal=n,
            method=method,
            alpha=alpha,
            zero_claim_mask=zero_claim_mask,
            thresholds=np.full(d, q_scalar),
        )
    elif method == "lwc":
        thresholds, mu_hat, sigma_hat = lwc_quantile_exact(residuals, alpha)
        cal = CalibratedScores(
            dimensions=keys,
            residuals=residuals,
            mu_hat=mu_hat,
            sigma_hat=sigma_hat,
            n_cal=n,
            method=method,
            alpha=alpha,
            zero_claim_mask=zero_claim_mask,
            thresholds=thresholds,
        )
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Choose 'bonferroni', 'sidak', 'gwc', 'lwc'."
        )

    return cal
