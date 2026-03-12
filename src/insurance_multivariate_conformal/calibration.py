"""
Core conformal calibration: fitting nonconformity scores on calibration data.

The calibration step takes a fitted set of models, a calibration dataset, and
produces the information needed to form prediction intervals at test time.

Key design choices:
1. Models are passed as a dict (e.g. {'frequency': model, 'severity': model})
   or a list. Each must implement .predict(X) returning shape (n,).
2. Zero-claim masking: for policies with observed claims = 0, severity is
   unobserved. We set the severity residual to 0 for those observations
   (conservative — treats zero-claim obs as perfectly predicted for severity).
3. Exposure offsets: Poisson frequency models often need exposure as an offset.
   We handle this by accepting exposure_cal as an optional array and passing it
   to models that support it via a predict_with_exposure() hook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from .scores import absolute_residual_score, poisson_deviance_score, gamma_deviance_score
from .methods import gwc_quantile, lwc_quantile, lwc_quantile_exact, bonferroni_quantile, sidak_quantile


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
        # Try passing exposure as second positional arg
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
    # Single model
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
        # Single output
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
    Stores calibration residuals and standardization statistics.

    Attributes
    ----------
    dimensions : list of str
        Dimension names (e.g. ['frequency', 'severity']).
    residuals : ndarray, shape (n, d)
        Raw calibration residuals (absolute errors or deviance residuals).
    mu_hat : ndarray, shape (d,)
        Per-dimension mean of calibration residuals (for standardization).
    sigma_hat : ndarray, shape (d,)
        Per-dimension std of calibration residuals (for standardization).
    n_cal : int
        Number of calibration points.
    method : str
        Which conformal method was used: 'bonferroni', 'sidak', 'gwc', 'lwc'.
    alpha : float
        Miscoverage level this calibration targets.
    zero_claim_mask : ndarray of bool or None
        Which calibration observations had zero claims (severity masked).
    score_fn : str
        Score function used: 'absolute', 'poisson_deviance', 'gamma_deviance', 'auto'.

    Internal use: thresholds, q_scalar, per_dim_quantiles — set by compute_thresholds().
    """

    dimensions: List[str]
    residuals: NDArray[np.floating]
    mu_hat: NDArray[np.floating]
    sigma_hat: NDArray[np.floating]
    n_cal: int
    method: str
    alpha: float
    zero_claim_mask: Optional[NDArray[np.bool_]] = None
    score_fn: str = "auto"

    # Set after compute_thresholds()
    thresholds: Optional[NDArray[np.floating]] = field(default=None, repr=False)
    # For bonferroni/sidak: per-dimension raw thresholds
    per_dim_quantiles: Optional[NDArray[np.floating]] = field(default=None, repr=False)

    def standardized_residuals(self) -> NDArray[np.floating]:
        """Return (residuals - mu_hat) / sigma_hat."""
        return (self.residuals - self.mu_hat) / self.sigma_hat

    def max_scores(self) -> NDArray[np.floating]:
        """Return max standardized score per calibration observation, shape (n,)."""
        return np.max(self.standardized_residuals(), axis=1)

    def interval_half_widths(self) -> NDArray[np.floating]:
        """
        Return per-dimension interval half-widths (raw units) for new predictions.

        For GWC/LWC: half_width_j = threshold_j * sigma_hat_j + mu_hat_j.
        For Bonferroni/Sidak: half_width_j = per_dim_quantiles_j.
        """
        if self.method in ("gwc", "lwc"):
            if self.thresholds is None:
                raise RuntimeError("Call compute_thresholds() first")
            return self.thresholds * self.sigma_hat + self.mu_hat
        else:  # bonferroni or sidak
            if self.per_dim_quantiles is None:
                raise RuntimeError("Call compute_thresholds() first")
            return self.per_dim_quantiles


def compute_residuals(
    models: Dict[str, ModelLike],
    X_cal: NDArray,
    Y_cal: Dict[str, NDArray],
    score_fn: str = "auto",
    exposure: Optional[NDArray] = None,
    zero_claim_mask: Optional[NDArray[np.bool_]] = None,
) -> NDArray[np.floating]:
    """
    Compute nonconformity residuals for each dimension.

    Returns residuals as shape (n, d) — always absolute (non-negative).

    score_fn options:
    - 'auto': use poisson_deviance for 'frequency', gamma_deviance for 'severity',
      absolute for anything else.
    - 'absolute': |y - y_hat| for all dimensions.
    - 'poisson_deviance': Poisson deviance for all dimensions.
    - 'gamma_deviance': Gamma deviance for all dimensions.

    zero_claim_mask: boolean array (n,). Where True, the 'severity' dimension
    residual is set to 0 (conservative masking).
    """
    keys = list(models.keys())
    n = X_cal.shape[0]
    d = len(keys)
    residuals = np.zeros((n, d), dtype=float)

    for j, key in enumerate(keys):
        model = models[key]
        y_true = Y_cal[key]
        y_pred = _get_predictions(model, X_cal, exposure)

        if score_fn == "auto":
            if key == "frequency":
                res = poisson_deviance_score(y_true, y_pred)
            elif key == "severity":
                res = gamma_deviance_score(y_true, y_pred)
            else:
                res = absolute_residual_score(y_true, y_pred)
        elif score_fn == "absolute":
            res = absolute_residual_score(y_true, y_pred)
        elif score_fn == "poisson_deviance":
            res = poisson_deviance_score(y_true, y_pred)
        elif score_fn == "gamma_deviance":
            res = gamma_deviance_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown score_fn: {score_fn!r}")

        # Zero-claim masking for severity dimension
        if zero_claim_mask is not None and key == "severity":
            res = res.copy()
            res[zero_claim_mask] = 0.0

        residuals[:, j] = res

    return residuals


def calibrate(
    models: Union[Dict[str, ModelLike], List[ModelLike]],
    X_cal: NDArray,
    Y_cal: Union[Dict[str, NDArray], NDArray],
    alpha: float = 0.05,
    method: str = "lwc",
    score_fn: str = "auto",
    exposure: Optional[NDArray] = None,
    zero_claim_mask: Optional[NDArray[np.bool_]] = None,
) -> CalibratedScores:
    """
    Calibrate a joint conformal predictor.

    Parameters
    ----------
    models : dict or list
        Fitted base models with .predict(X) interface. If dict, keys are used
        as dimension names (e.g. {'frequency': glm, 'severity': gbm}).
    X_cal : ndarray, shape (n, p)
        Calibration feature matrix.
    Y_cal : dict or ndarray
        Calibration targets. Dict keys must match models keys.
    alpha : float
        Joint miscoverage level (e.g. 0.05 for 95% joint coverage).
    method : str
        'bonferroni', 'sidak', 'gwc', or 'lwc'.
    score_fn : str
        'auto', 'absolute', 'poisson_deviance', 'gamma_deviance'.
    exposure : ndarray, shape (n,), optional
        Exposure values for Poisson models.
    zero_claim_mask : ndarray of bool, shape (n,), optional
        True where severity is unobserved (zero claims). Severity residual
        set to 0 for these observations.

    Returns
    -------
    CalibratedScores
        Calibration result with standardization statistics and thresholds.
    """
    models_dict = _normalise_models(models)
    keys = list(models_dict.keys())
    Y_dict = _normalise_y(Y_cal, keys)

    # Validate shapes
    n = X_cal.shape[0]
    for k in keys:
        if k not in Y_dict:
            raise ValueError(f"Y_cal missing key '{k}' (models has {keys})")
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

    residuals = compute_residuals(
        models_dict, X_cal, Y_dict,
        score_fn=score_fn,
        exposure=exposure,
        zero_claim_mask=zero_claim_mask,
    )

    # Compute thresholds
    d = len(keys)
    if method == "bonferroni":
        per_dim_q = bonferroni_quantile(residuals, alpha, d=d)
        mu_hat = np.mean(residuals, axis=0)
        sigma_hat = np.maximum(np.std(residuals, axis=0), 1e-8)
        cal = CalibratedScores(
            dimensions=keys,
            residuals=residuals,
            mu_hat=mu_hat,
            sigma_hat=sigma_hat,
            n_cal=n,
            method=method,
            alpha=alpha,
            zero_claim_mask=zero_claim_mask,
            score_fn=score_fn,
            per_dim_quantiles=per_dim_q,
        )
    elif method == "sidak":
        per_dim_q = sidak_quantile(residuals, alpha, d=d)
        mu_hat = np.mean(residuals, axis=0)
        sigma_hat = np.maximum(np.std(residuals, axis=0), 1e-8)
        cal = CalibratedScores(
            dimensions=keys,
            residuals=residuals,
            mu_hat=mu_hat,
            sigma_hat=sigma_hat,
            n_cal=n,
            method=method,
            alpha=alpha,
            zero_claim_mask=zero_claim_mask,
            score_fn=score_fn,
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
            score_fn=score_fn,
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
            score_fn=score_fn,
            thresholds=thresholds,
        )
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Choose 'bonferroni', 'sidak', 'gwc', 'lwc'."
        )

    return cal
