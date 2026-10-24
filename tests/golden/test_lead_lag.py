# =============================================================================
# FILE: tests/golden/test_lead_lag.py
# =============================================================================
"""
``stratigraphy.lead_lag``（最小 RMSE 的超前/滞后判定）的测试。

本文件是**真值对照**测试：滞后量是人为造出来的，答案因此是已知的，
不需要和任何外部实现比对。

真值怎么造
----------
构造一段长度为 ``n + Δ`` 的信号 ``s``，取

* 第一条序列 = ``s[Δ:]``（记录的是较晚的时刻），
* 第二条序列 = ``s[:n]``（记录的是较早的时刻）。

于是第二条序列的每个特征都比第一条序列**晚 Δ 个样本**出现，也就是
"第二条滞后第一条 Δ 个样本"。按 ``stratigraphy.lead_lag`` 的符号约定
（**正 = 第二条滞后第一条**），正确答案是 ``best_lag == +Δ``；把两条
序列对调，答案就是 ``−Δ``。这两个断言把符号约定钉死，避免将来有人
悄悄改掉方向而测试还照样通过。

判据不止"滞后量对不对"，还包括：

* 最优滞后量上的 RMSE 必须**明显小于**零滞后处的 RMSE（零滞后处两条
  序列形状相同但错位，标准化 RMSE 接近 √2）；
* RMSE 曲线必须在 ``best_lag`` 处取到最小值；
* 重叠样本不足的候选滞后量必须被**跳过**（不能拿两三个点算 RMSE）。

边界与约定
----------
* 样本太少、含非有限值、是常数序列、``max_lag`` 越界 → 抛
  ``utils.exceptions.ValidationError``；
* 本分析是确定性的，没有随机过程，因此不涉及 ``random_seed``。

参考来源
--------
* Thomson, D.J. 1982. Spectrum estimation and harmonic analysis. Proceedings
  of the IEEE 70: 1055-1096.（最小二乘配准与多锥谱的总体框架。）
* Percival, D.B. & Walden, A.T. 1993. Spectral Analysis and its
  Applications. Cambridge University Press.
* Li, M., Kump, L.R., Hinnov, L.A. & Mann, M.E. 2018. Cyclostratigraphic
  correlation of astronomical forcing and Earth system processes. Earth and
  Planetary Science Letters 501: 165-179.
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"lead_lag_rmse"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.lead_lag import MIN_OVERLAP, lead_lag_rmse
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 合成记录的长度（不含人为延迟的那一段）。
N_SAMPLES = 200

#: 人为延迟的样本数。
DELAY = 7

#: 允许的最大滞后量。
MAX_LAG = 40


# =============================================================================
# 辅助函数
# =============================================================================


def _signal(length: int, seed: int = 11) -> npt.NDArray:
    """一段确定性的测试信号：两个周期 + 一点点噪声。"""
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=float)
    return np.sin(2.0 * np.pi * t / 17.0) + 0.5 * np.cos(2.0 * np.pi * t / 5.0) + 0.2 * rng.standard_normal(length)


def _lagged_pair(delay: int = DELAY, n: int = N_SAMPLES, seed: int = 11) -> tuple[npt.NDArray, npt.NDArray]:
    """构造 (第一条, 第二条)：第二条比第一条**滞后** ``delay`` 个样本。

    第一条取 ``s[delay:]``、第二条取 ``s[:n]``，所以第二条的特征比第一条
    晚 ``delay`` 个样本出现（见模块 docstring 的推导）。
    """
    signal = _signal(delay + n, seed=seed)
    return signal[delay:], signal[:n]


def _rmse_at(result, lag: int) -> float:
    """取 RMSE 曲线上某个滞后量处的值。"""
    match = np.flatnonzero(result.lags == lag)
    assert match.size == 1, f"lag {lag} is not in the evaluated grid"
    return float(result.rmse_curve[match[0]])


# =============================================================================
# 真值对照
# =============================================================================


def test_a_known_delay_is_recovered_with_the_documented_sign():
    """第二条滞后第一条 DELAY 个样本 -> best_lag == +DELAY。"""
    first, second = _lagged_pair(DELAY)
    result = lead_lag_rmse(first, second, max_lag=MAX_LAG)
    assert result.best_lag == DELAY, f"expected best_lag == +{DELAY} (second lags first); got {result.best_lag}"


def test_a_known_lead_is_recovered_with_the_opposite_sign():
    """把两条序列对调 -> best_lag == -DELAY（第二条超前第一条）。"""
    first, second = _lagged_pair(DELAY)
    result = lead_lag_rmse(second, first, max_lag=MAX_LAG)
    assert result.best_lag == -DELAY, f"expected best_lag == -{DELAY} (second leads first); got {result.best_lag}"


def test_the_best_rmse_is_far_below_the_rmse_at_zero_lag():
    """最优滞后量上的 RMSE 必须明显小于零滞后处的 RMSE。"""
    first, second = _lagged_pair(DELAY)
    result = lead_lag_rmse(first, second, max_lag=MAX_LAG)
    at_zero = _rmse_at(result, 0)
    assert result.best_rmse < 0.1, f"best_rmse is not small: {result.best_rmse}"
    assert at_zero > 1.0, f"the RMSE at lag 0 should be around sqrt(2) for unaligned series; got {at_zero}"
    assert at_zero > 10.0 * result.best_rmse, (
        f"the zero-lag RMSE ({at_zero}) is not clearly above the best RMSE ({result.best_rmse})"
    )


def test_the_rmse_curve_is_minimised_at_the_best_lag():
    """RMSE 曲线上的最小值必须落在 best_lag 上。"""
    first, second = _lagged_pair(DELAY)
    result = lead_lag_rmse(first, second, max_lag=MAX_LAG)
    assert result.lags.size == result.rmse_curve.size
    assert int(np.argmin(result.rmse_curve)) == int(np.flatnonzero(result.lags == result.best_lag)[0])
    assert np.all(np.diff(result.lags) > 0), "the lag grid must be strictly increasing"
    assert np.all(np.isfinite(result.rmse_curve))


def test_a_delay_beyond_the_search_range_is_not_invented():
    """真实延迟超出搜索范围时，不能凭空报一个很小的滞后量。"""
    first, second = _lagged_pair(DELAY)
    result = lead_lag_rmse(first, second, max_lag=3)
    assert abs(result.best_lag) <= 3
    assert result.best_rmse > _rmse_at(result, 0) * 0.2, "with a too small search range the fit must stay poor"


# =============================================================================
# 重叠不足的候选滞后量
# =============================================================================


def test_lags_without_enough_overlap_are_skipped():
    """重叠样本不足的候选被跳过：不能用两三个点算 RMSE。"""
    length = 10
    first = _signal(length, seed=3)[:length]
    second = _signal(length, seed=3)[:length]
    result = lead_lag_rmse(first, second, max_lag=length - 1)
    assert int(np.max(np.abs(result.lags))) == length - MIN_OVERLAP, (
        "every evaluated lag must keep at least MIN_OVERLAP overlapping samples"
    )
    assert np.all(np.isfinite(result.rmse_curve))


# =============================================================================
# 边界
# =============================================================================


def test_short_series_are_rejected():
    """样本太少 -> ValidationError。"""
    with pytest.raises(ValidationError):
        lead_lag_rmse(np.arange(5.0), np.arange(5.0))


def test_non_finite_values_are_rejected():
    """含 NaN/Inf -> ValidationError（丢掉非有限值会破坏两条序列的对齐）。"""
    first, second = _lagged_pair(DELAY)
    broken = first.copy()
    broken[3] = np.nan
    with pytest.raises(ValidationError):
        lead_lag_rmse(broken, second)


def test_a_constant_series_is_rejected():
    """常数序列没有形状可对齐 -> ValidationError。"""
    with pytest.raises(ValidationError):
        lead_lag_rmse(np.ones(50), np.arange(50.0))


@pytest.mark.parametrize("bad_lag", [0, -1, 200, 500])
def test_out_of_range_max_lag_is_rejected(bad_lag):
    """max_lag 越界 -> ValidationError。"""
    first, second = _lagged_pair(DELAY)
    with pytest.raises(ValidationError):
        lead_lag_rmse(first, second, max_lag=bad_lag)


def test_the_summary_reports_the_best_lag():
    """摘要文本里必须出现最优滞后量。"""
    first, second = _lagged_pair(DELAY)
    result = lead_lag_rmse(first, second, max_lag=MAX_LAG)
    text = result.summary()
    assert str(result.best_lag) in text
    assert "RMSE" in text
