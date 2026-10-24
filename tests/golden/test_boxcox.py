# =============================================================================
# FILE: tests/golden/test_boxcox.py
# =============================================================================
"""
``stratigraphy.boxcox``（Box–Cox 幂变换）的测试。

断言围绕四条主线组织：

* **λ = 0 ⇒ 对数变换**、**λ = 1 ⇒ 平移后的线性变换**。这两条是变换定义本身，
  真值即 ``ln(x)`` 与 ``x − 1``（Box–Cox 族在 λ = 0 处的连续延拓是
  ``(x^λ − 1)/λ → ln x``）。
* **变换可逆**。``box_cox_inverse(box_cox_transform(x, λ))`` 必须回到原值，
  容差 ``1e-8``。这是本模块存在的理由：只有自己实现变换，才能同时给出 λ
  和可逆的逆变换。
* **λ 的极大似然估计**。正偏态数据（伽马分布）必须给出 ``λ > 0``（幂变换
  压掉右尾），对数正态数据必须给出 ``λ ≈ 0``（对数变换才是正解）。
* **与独立实现的交叉验证**。本模块自己解 λ，但 ``scipy.stats.boxcox`` 解的是
  同一个轮廓似然，两者的 λ 必须一致 —— 这是"独立推导真值"式的对照。

真值来源
--------
* ``λ = 0`` / ``λ = 1`` 的变换值：Box–Cox 族的定义（Box & Cox, 1964）。
* 往返误差：数值恒等式，容差取双精度下的 ``1e-8``。
* λ 的估计：与 ``scipy.stats.boxcox``（MathWorks ``boxcox`` 的同一算法）
  交叉验证，容差 ``1e-3``。
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"box_cox_transform"}


import numpy as np
import pytest
from scipy.stats import boxcox as scipy_boxcox

from stratigraphy.boxcox import box_cox_inverse, box_cox_transform
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 一组简单的正值测试数据。
VALUES = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

#: 往返变换的容差。
ROUNDTRIP_TOLERANCE = 1.0e-8

#: 与 ``scipy.stats.boxcox`` 交叉验证时的 λ 容差。
LAMBDA_CROSS_CHECK_TOLERANCE = 1.0e-3


# =============================================================================
# 辅助函数
# =============================================================================


def _right_skewed(seed: int = 0, n: int = 400) -> np.ndarray:
    """右偏的正值样本（伽马分布，形状 2）。"""
    generator = np.random.default_rng(seed)
    return generator.gamma(shape=2.0, scale=1.0, size=n) + 0.05


def _lognormal(seed: int = 0, n: int = 400) -> np.ndarray:
    """对数正态样本（右偏，但它的正解是对数变换）。"""
    generator = np.random.default_rng(seed)
    return np.exp(generator.normal(size=n))


# =============================================================================
# 变换定义
# =============================================================================


class TestTransformDefinition:
    """λ = 0 与 λ = 1 的行为由定义直接给出。"""

    def test_lambda_zero_is_the_logarithm(self):
        result = box_cox_transform(VALUES, lambda_=0.0)

        assert np.allclose(result.transformed, np.log(VALUES), rtol=1e-12, atol=1e-12)
        assert result.lambda_ == pytest.approx(0.0)
        assert result.offset == pytest.approx(0.0)

    def test_lambda_one_is_the_shifted_linear_transform(self):
        """λ = 1 ⇒ (x¹ − 1)/1 = x − 1（平移后的线性变换）。"""
        result = box_cox_transform(VALUES, lambda_=1.0)

        assert np.allclose(result.transformed, VALUES - 1.0, rtol=1e-12, atol=1e-12)
        assert result.lambda_ == pytest.approx(1.0)

    def test_transformed_shape_matches_the_input(self):
        result = box_cox_transform(VALUES, lambda_=0.5)

        assert result.transformed.shape == VALUES.shape


# =============================================================================
# 往返
# =============================================================================


class TestRoundTrip:
    """变换再逆变换必须回到原值。"""

    @pytest.mark.parametrize("lambda_", [0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0])
    def test_inverse_recovers_the_original_values(self, lambda_: float):
        result = box_cox_transform(VALUES, lambda_=lambda_)

        restored = box_cox_inverse(result.transformed, lambda_, result.offset)

        assert np.allclose(restored, VALUES, rtol=0.0, atol=ROUNDTRIP_TOLERANCE), (
            f"the round trip failed for lambda = {lambda_}: {restored}"
        )

    def test_inverse_recovers_values_for_a_auto_estimated_lambda(self):
        values = _right_skewed()
        result = box_cox_transform(values)

        restored = box_cox_inverse(result.transformed, result.lambda_, result.offset)

        assert np.allclose(restored, values, rtol=0.0, atol=ROUNDTRIP_TOLERANCE)


# =============================================================================
# λ 的极大似然估计
# =============================================================================


class TestLambdaEstimation:
    """λ 必须由数据估出来，且方向正确。"""

    def test_right_skewed_data_gives_a_positive_lambda(self):
        """正偏态数据 ⇒ λ > 0（幂变换压掉右尾）。"""
        result = box_cox_transform(_right_skewed())

        assert result.lambda_ > 0.0, f"a right-skewed sample gave lambda = {result.lambda_:.4g}"

    def test_lognormal_data_gives_a_lambda_near_zero(self):
        """对数正态数据 ⇒ λ ≈ 0（对数变换才是正解）。"""
        result = box_cox_transform(_lognormal())

        assert abs(result.lambda_) < 0.1, f"a lognormal sample gave lambda = {result.lambda_:.4g}"

    def test_estimated_lambda_matches_an_independent_implementation(self):
        """与 ``scipy.stats.boxcox`` 交叉验证：两者解同一个轮廓似然。"""
        values = _right_skewed()

        result = box_cox_transform(values)
        _, scipy_lambda = scipy_boxcox(values)

        assert result.lambda_ == pytest.approx(float(scipy_lambda), abs=LAMBDA_CROSS_CHECK_TOLERANCE)

    def test_estimation_is_deterministic(self):
        values = _right_skewed()

        first = box_cox_transform(values)
        second = box_cox_transform(values)

        assert first.lambda_ == second.lambda_
        assert np.array_equal(first.transformed, second.transformed)

    def test_non_positive_data_get_an_offset_and_stay_invertible(self):
        """数据含非正值时自动加偏移量，偏移量记在结果里。"""
        values = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])

        result = box_cox_transform(values)

        assert result.offset > 0.0
        restored = box_cox_inverse(result.transformed, result.lambda_, result.offset)
        assert np.allclose(restored, values, rtol=0.0, atol=ROUNDTRIP_TOLERANCE)


# =============================================================================
# 结果对象
# =============================================================================


class TestResult:
    """结果对象的基本性质。"""

    def test_summary_is_a_non_empty_string(self):
        result = box_cox_transform(VALUES, lambda_=0.0)

        summary = result.summary()
        assert isinstance(summary, str)
        assert summary.strip()


# =============================================================================
# 边界
# =============================================================================


class TestBoundaries:
    """非法输入必须抛 ``ValidationError``。"""

    def test_non_positive_values_with_an_explicit_lambda_are_rejected(self):
        with pytest.raises(ValidationError):
            box_cox_transform(np.array([-1.0, 1.0, 2.0]), lambda_=0.5)

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValidationError):
            box_cox_transform(np.array([]))

    def test_two_dimensional_input_is_rejected(self):
        with pytest.raises(ValidationError):
            box_cox_transform(np.ones((2, 2)))

    def test_non_finite_values_are_rejected(self):
        with pytest.raises(ValidationError):
            box_cox_transform(np.array([1.0, np.nan, 2.0]))

    def test_non_finite_lambda_is_rejected(self):
        with pytest.raises(ValidationError):
            box_cox_transform(VALUES, lambda_=float("nan"))

    def test_estimating_lambda_needs_at_least_two_values(self):
        with pytest.raises(ValidationError):
            box_cox_transform(np.array([1.0]))

    def test_inverse_with_a_non_positive_base_is_rejected(self):
        """λ·y + 1 ≤ 0 时逆变换没有实数解，必须报错而不是返回 NaN。"""
        with pytest.raises(ValidationError):
            box_cox_inverse(np.array([-10.0]), lambda_=1.0)

    def test_inverse_rejects_non_finite_input(self):
        with pytest.raises(ValidationError):
            box_cox_inverse(np.array([0.0, np.inf]), lambda_=1.0)
