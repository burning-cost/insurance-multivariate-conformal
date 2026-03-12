"""
Tests for calibration module.
"""

import numpy as np
import pytest

from insurance_multivariate_conformal.calibration import (
    calibrate,
    compute_residuals,
    CalibratedScores,
    _normalise_models,
    _normalise_y,
)
from insurance_multivariate_conformal.datasets import _SimpleExpModel


class TestNormaliseModels:
    def test_dict_passthrough(self):
        m = {"a": 1, "b": 2}
        assert _normalise_models(m) == m

    def test_list_to_dict(self):
        m_list = ["m0", "m1"]
        result = _normalise_models(m_list)
        assert list(result.keys()) == ["0", "1"]
        assert result["0"] == "m0"

    def test_single_model(self):
        result = _normalise_models("model")
        assert result == {"0": "model"}


class TestNormaliseY:
    def test_dict_passthrough(self):
        Y = {"freq": np.array([1.0, 2.0]), "sev": np.array([100.0, 200.0])}
        result = _normalise_y(Y, ["freq", "sev"])
        np.testing.assert_array_equal(result["freq"], Y["freq"])

    def test_1d_array(self):
        Y = np.array([1.0, 2.0, 3.0])
        result = _normalise_y(Y, ["freq"])
        assert "freq" in result
        np.testing.assert_array_equal(result["freq"], Y)

    def test_2d_array(self):
        Y = np.column_stack([np.array([1.0, 2.0]), np.array([10.0, 20.0])])
        result = _normalise_y(Y, ["a", "b"])
        np.testing.assert_array_equal(result["a"], [1.0, 2.0])
        np.testing.assert_array_equal(result["b"], [10.0, 20.0])

    def test_mismatch_raises(self):
        Y = np.ones((10, 3))
        with pytest.raises(AssertionError):
            _normalise_y(Y, ["a", "b"])  # 2 keys but 3 columns


class TestComputeResiduals:
    def test_basic_shape(self, motor_data):
        d = motor_data
        res = compute_residuals(
            d["models"], d["X_cal"], d["Y_cal"], score_fn="absolute"
        )
        assert res.shape == (d["n_cal"], 2)

    def test_nonnegative(self, motor_data):
        d = motor_data
        res = compute_residuals(
            d["models"], d["X_cal"], d["Y_cal"], score_fn="absolute"
        )
        assert np.all(res >= 0)

    def test_auto_score_fn(self, motor_data):
        d = motor_data
        res = compute_residuals(
            d["models"], d["X_cal"], d["Y_cal"], score_fn="auto"
        )
        assert res.shape == (d["n_cal"], 2)
        assert np.all(res >= 0)

    def test_zero_claim_masking(self, motor_data):
        d = motor_data
        mask = d["zero_mask_cal"]
        res_masked = compute_residuals(
            d["models"], d["X_cal"], d["Y_cal"],
            score_fn="auto",
            zero_claim_mask=mask,
        )
        # Severity residuals should be 0 where mask is True
        assert np.all(res_masked[mask, 1] == 0.0)

    def test_zero_claim_masking_frequency_unchanged(self, motor_data):
        d = motor_data
        mask = d["zero_mask_cal"]
        res_no_mask = compute_residuals(
            d["models"], d["X_cal"], d["Y_cal"], score_fn="auto"
        )
        res_masked = compute_residuals(
            d["models"], d["X_cal"], d["Y_cal"],
            score_fn="auto",
            zero_claim_mask=mask,
        )
        # Frequency residuals unaffected by zero-claim mask
        np.testing.assert_array_equal(res_masked[:, 0], res_no_mask[:, 0])

    def test_unknown_score_fn_raises(self, motor_data):
        d = motor_data
        with pytest.raises(ValueError, match="Unknown score_fn"):
            compute_residuals(d["models"], d["X_cal"], d["Y_cal"], score_fn="bad")


class TestCalibrate:
    @pytest.mark.parametrize("method", ["bonferroni", "sidak", "gwc", "lwc"])
    def test_returns_calibrated_scores(self, motor_data, method):
        d = motor_data
        cal = calibrate(
            d["models"], d["X_cal"], d["Y_cal"],
            alpha=0.05, method=method
        )
        assert isinstance(cal, CalibratedScores)
        assert cal.n_cal == d["n_cal"]
        assert cal.alpha == 0.05
        assert cal.method == method

    def test_dimensions_match_model_keys(self, motor_data):
        d = motor_data
        cal = calibrate(d["models"], d["X_cal"], d["Y_cal"])
        assert set(cal.dimensions) == {"frequency", "severity"}

    def test_residuals_shape(self, motor_data):
        d = motor_data
        cal = calibrate(d["models"], d["X_cal"], d["Y_cal"])
        assert cal.residuals.shape == (d["n_cal"], 2)

    def test_sigma_positive(self, motor_data):
        d = motor_data
        cal = calibrate(d["models"], d["X_cal"], d["Y_cal"])
        assert np.all(cal.sigma_hat > 0)

    def test_lwc_thresholds_leq_gwc(self, motor_data):
        d = motor_data
        cal_gwc = calibrate(d["models"], d["X_cal"], d["Y_cal"], method="gwc")
        cal_lwc = calibrate(d["models"], d["X_cal"], d["Y_cal"], method="lwc")
        hw_gwc = cal_gwc.interval_half_widths()
        hw_lwc = cal_lwc.interval_half_widths()
        # LWC should produce <= half-widths than GWC (tighter or equal)
        assert np.all(hw_lwc <= hw_gwc + 1e-10)

    def test_interval_half_widths_positive(self, motor_data):
        d = motor_data
        for method in ["bonferroni", "gwc", "lwc"]:
            cal = calibrate(d["models"], d["X_cal"], d["Y_cal"], method=method)
            hw = cal.interval_half_widths()
            assert np.all(hw > 0), f"Non-positive half-widths for method={method}"

    def test_unknown_method_raises(self, motor_data):
        d = motor_data
        with pytest.raises(ValueError, match="Unknown method"):
            calibrate(d["models"], d["X_cal"], d["Y_cal"], method="bad")

    def test_shape_mismatch_raises(self, motor_data):
        d = motor_data
        Y_bad = {"frequency": np.ones(5), "severity": np.ones(d["n_cal"])}
        with pytest.raises(ValueError, match="rows"):
            calibrate(d["models"], d["X_cal"], Y_bad)

    def test_zero_claim_mask_shape_mismatch_raises(self, motor_data):
        d = motor_data
        bad_mask = np.ones(5, dtype=bool)
        with pytest.raises(ValueError, match="zero_claim_mask"):
            calibrate(d["models"], d["X_cal"], d["Y_cal"], zero_claim_mask=bad_mask)

    def test_list_models_input(self, motor_data):
        d = motor_data
        models_list = list(d["models"].values())
        Y_arr = np.column_stack([d["Y_cal"]["frequency"], d["Y_cal"]["severity"]])
        cal = calibrate(models_list, d["X_cal"], Y_arr)
        assert cal.n_cal == d["n_cal"]

    def test_standardized_residuals(self, motor_data):
        d = motor_data
        cal = calibrate(d["models"], d["X_cal"], d["Y_cal"])
        std = cal.standardized_residuals()
        assert std.shape == (d["n_cal"], 2)

    def test_max_scores(self, motor_data):
        d = motor_data
        cal = calibrate(d["models"], d["X_cal"], d["Y_cal"])
        ms = cal.max_scores()
        assert ms.shape == (d["n_cal"],)
