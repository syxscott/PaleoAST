# =============================================================================
# FILE: tests/golden/test_transforms.py
# =============================================================================
"""
``stratigraphy.transforms`` (标准化 / 对数变换 / 简单函数) 的测试。

判据说明
--------
* **z 分数**: 结果的均值恰为 0, **样本标准差** (``ddof=1``) 恰为 1 —— 这是
  z 分数的定义本身, 所以是精确断言; 同时验证它与 ``(x - mean) / std`` 逐点
  相等 (与仓库既有的 ``utils.transformations.zscore_standardize`` 口径一致)。
* **min-max**: 最小值映到 0, 最大值映到 1, 全部落在 ``[0, 1]``。
* **对数变换**: 用的是**不带偏移的 log**, 即 ``log_base(x)``; 因此
  ``log10([1, 10, 100]) == [0, 1, 2]``, 而 0 与负数必须抛
  ``ValidationError`` (而不是被悄悄变成 ``-inf`` / ``NaN``)。
* **简单函数**: ``square``/``abs``/``sqrt``/``exp`` 逐点等于 numpy 的对应
  结果; ``exp`` 溢出时抛 ``ValidationError`` (而不是把 ``inf`` 交给下游)。
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"standardize", "log_transform", "apply_function"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.transforms import apply_function, log_transform, standardize
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 一条非对称、非正态的样本序列 (用来避免"恰好对称"的假通过)。
SAMPLE: npt.NDArray = np.array([3.0, 1.5, 8.0, 2.0, 11.0, 4.5, 0.5, 6.0])


# =============================================================================
# 标准化
# =============================================================================


def test_zscore_has_zero_mean_and_unit_sample_std():
    """判据: z 分数的均值恰为 0, 样本标准差 (ddof=1) 恰为 1。"""
    result = standardize(SAMPLE, method="zscore")

    assert float(np.mean(result)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.std(result, ddof=1)) == pytest.approx(1.0, abs=1e-12)


def test_zscore_matches_the_definition_pointwise():
    """判据: 逐点等于 ``(x - mean) / std`` (ddof=1)。"""
    result = standardize(SAMPLE, method="zscore")
    expected = (SAMPLE - SAMPLE.mean()) / SAMPLE.std(ddof=1)

    assert np.allclose(result, expected)


def test_zscore_is_the_default_method():
    """默认方法就是 zscore。"""
    assert np.allclose(standardize(SAMPLE), standardize(SAMPLE, method="zscore"))


def test_minmax_maps_the_series_onto_zero_one():
    """判据: 最小值映到 0, 最大值映到 1, 全部落在 [0, 1]。"""
    result = standardize(SAMPLE, method="minmax")

    assert float(result.min()) == pytest.approx(0.0)
    assert float(result.max()) == pytest.approx(1.0)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_minmax_is_affine_and_preserves_order():
    """判据: min-max 是单调的仿射映射, 因此保序 (排序结果不变)。"""
    result = standardize(SAMPLE, method="minmax")

    assert np.array_equal(np.argsort(result), np.argsort(SAMPLE))


def test_standardize_rejects_unknown_method():
    """未知方法必须报错。"""
    with pytest.raises(ValidationError):
        standardize(SAMPLE, method="robust")


def test_standardize_rejects_a_constant_series():
    """常数序列没有可用的尺度, 必须报错而不是静默返回原值。"""
    with pytest.raises(ValidationError):
        standardize(np.full(8, 2.5))


def test_standardize_rejects_empty_and_non_finite_input():
    """空输入与含 NaN 的输入必须报错。"""
    with pytest.raises(ValidationError):
        standardize(np.array([]))
    with pytest.raises(ValidationError):
        standardize(np.array([1.0, np.nan, 3.0]))


# =============================================================================
# 对数变换
# =============================================================================


def test_log_transform_base10_is_the_ten_log():
    """判据: ``log10([1, 10, 100]) == [0, 1, 2]``。"""
    values = np.array([1.0, 10.0, 100.0])

    assert np.allclose(log_transform(values), np.array([0.0, 1.0, 2.0]))


def test_log_transform_natural_base_and_base_two():
    """判据: 底数 e 与 2 各自给出对应的对数。"""
    values = np.array([1.0, np.e, np.e**2])

    assert np.allclose(log_transform(values, base=np.e), np.array([0.0, 1.0, 2.0]))
    assert np.allclose(log_transform(values, base=2.0), np.log2(values))


def test_log_transform_rejects_non_positive_values():
    """0 与负数必须报错, 不允许静默变成 NaN / -inf。"""
    with pytest.raises(ValidationError):
        log_transform(np.array([-1.0, 2.0]))
    with pytest.raises(ValidationError):
        log_transform(np.array([0.0, 1.0]))


def test_log_transform_rejects_an_invalid_base():
    """底数必须为正且不等于 1。"""
    with pytest.raises(ValidationError):
        log_transform(np.array([1.0, 2.0]), base=1.0)
    with pytest.raises(ValidationError):
        log_transform(np.array([1.0, 2.0]), base=-10.0)


def test_log_transform_rejects_non_finite_input():
    """含 NaN/inf 的输入必须报错。"""
    with pytest.raises(ValidationError):
        log_transform(np.array([1.0, np.nan]))


# =============================================================================
# 简单函数
# =============================================================================


def test_apply_square_and_abs():
    """判据: 逐点等于 numpy 的对应结果。"""
    values = np.array([-3.0, 0.5, 2.0])

    assert np.allclose(apply_function(values, "square"), np.array([9.0, 0.25, 4.0]))
    assert np.allclose(apply_function(values, "abs"), np.array([3.0, 0.5, 2.0]))


def test_apply_exp_matches_numpy():
    """判据: exp 逐点等于 ``numpy.exp``。"""
    values = np.array([-1.0, 0.0, 1.0])

    assert np.allclose(apply_function(values, "exp"), np.exp(values))


def test_apply_sqrt_matches_numpy_for_non_negative_input():
    """判据: sqrt 逐点等于 ``numpy.sqrt``。"""
    values = np.array([0.0, 1.0, 4.0, 9.0])

    assert np.allclose(apply_function(values, "sqrt"), np.sqrt(values))


def test_apply_exp_overflow_raises_instead_of_returning_inf():
    """判据: exp 溢出必须抛 ``ValidationError``, 而不是返回 ``inf``。"""
    with pytest.raises(ValidationError):
        apply_function(np.array([800.0, 1.0]), "exp")


def test_apply_sqrt_rejects_negative_values():
    """判据: 负数开方必须报错。"""
    with pytest.raises(ValidationError):
        apply_function(np.array([-1.0, 2.0]), "sqrt")


def test_apply_rejects_unknown_function():
    """未知函数名必须报错。"""
    with pytest.raises(ValidationError):
        apply_function(SAMPLE, "cube")
