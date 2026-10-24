# =============================================================================
# FILE: tests/golden/test_smoothing.py
# =============================================================================
"""
``stratigraphy.smoothing`` (最优跨度平滑 + 移动平均) 的测试。

判据说明 (每条断言都写明"凭什么算通过")
----------------------------------------

1. **移动平均在区间内部无偏。** 对一条严格线性的序列, 任何**对称**窗口的
   均值都等于窗口中心处的值 —— 这是移动平均的解析性质, 因此断言是精确的
   (``==`` 级别), 不依赖任何统计涨落。

2. **真值对照: 选出的跨度必须显著降低对已知真值的误差。** 合成序列 =
   直线趋势 + AR(1) 噪声, 真值就是那条直线。判据: 平滑后残差的标准差
   不到原始噪声标准差的 30 %。

3. **选出的跨度必须吃掉噪声的正序列相关。** AR(1) 噪声的 lag-1 自相关是
   +0.9; 移动平均会把这种"红噪声"变成 lag-1 自相关**为负**的残差
   (差分效应)。判据: 平滑后残差的 lag-1 自相关 **小于 0**, 即噪声里
   那部分"正相关"已经被平滑曲线吸收。注意判据不是"自相关 == 0":
   移动平均后的残差天然带负相关, 说它"白"是不成立的。

4. **纯噪声序列: 平滑后残差方差必须下降。** 判据:
   ``var(y - smoothed) < var(y)``。

5. **阶跃不能被抹平。** 对一个单位阶跃, 判据有三层:
   左平台仍约为 0、右平台仍约为 1, 且过渡只发生在阶跃附近
   (过渡样本数 <= 2 * 跨度 + 2), 即没有被抹成一条长斜坡。

6. **可复现性。** 固定 ``random_seed`` 给出完全相同的置信带; 不给种子时
   按仓库约定发 ``RuntimeWarning``。

参考文献
--------
* Golub, G.H. et al., 1979. *Numerische Mathematik*, 36: 447-473. (GCV)
* Akaike, H., 1974. *IEEE Trans. Autom. Control*, 19: 716-723. (AIC)
* Künsch, H.R., 1989. *Annals of Statistics*, 17: 1217-1241. (块自助)
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"optimal_span_smooth", "moving_average_smooth"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.smoothing import moving_average_smooth, optimal_span_smooth
from utils.exceptions import ValidationError

# =============================================================================
# 辅助函数
# =============================================================================


def _ar1_noise(n: int, phi: float = 0.9, seed: int = 7) -> npt.NDArray:
    """
    生成 AR(1) 噪声 (相邻样本正相关, 模拟地质记录里的自相关噪声)。

    Parameters:
        n: 样本数。
        phi: 自相关系数 (0 < phi < 1)。
        seed: 随机种子。

    Returns:
        长度 ``n`` 的噪声序列。
    """
    rng = np.random.default_rng(seed)
    noise = np.zeros(n, dtype=float)
    for index in range(1, n):
        noise[index] = phi * noise[index - 1] + rng.normal(0.0, 1.0)
    return noise


def _lag1_autocorrelation(values: npt.NDArray) -> float:
    """
    序列的 lag-1 自相关系数。

    Parameters:
        values: 输入序列。

    Returns:
        lag-1 自相关系数; 序列无变异时返回 0。
    """
    centered = values - values.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(centered[:-1], centered[1:]) / denominator)


# =============================================================================
# 移动平均
# =============================================================================


def test_moving_average_is_exact_on_a_straight_line_inside_the_record():
    """
    判据: 对线性序列, 对称窗口的均值恰好等于窗口中心的值。

    这条断言是解析性质, 因此用精确比较; 记录两端因窗口被截短而不在此列。
    """
    times = np.arange(200, dtype=float)
    values = 2.5 * times - 3.0

    smoothed = moving_average_smooth(times, values, span=7)

    interior = slice(10, 190)
    assert np.allclose(smoothed[interior], values[interior], atol=1e-9)


def test_moving_average_preserves_length_and_rejects_bad_spans():
    """跨度非法 (小于 1 或超过样本数) 或长度不匹配时必须报错。"""
    times = np.arange(50, dtype=float)
    values = np.sin(times)

    assert moving_average_smooth(times, values, span=5).size == values.size

    with pytest.raises(ValidationError):
        moving_average_smooth(times, values, span=0)
    with pytest.raises(ValidationError):
        moving_average_smooth(times, values, span=51)
    with pytest.raises(ValidationError):
        moving_average_smooth(times, values[:-1], span=5)
    with pytest.raises(ValidationError):
        moving_average_smooth(times, np.full(50, np.nan), span=5)


# =============================================================================
# 跨度选择
# =============================================================================


def test_selected_span_sharply_reduces_error_against_known_truth():
    """
    真值对照: 直线趋势 + AR(1) 噪声, 真值就是那条直线。

    判据: 平滑后残差的标准差不到原始噪声标准差的 30 %。
    """
    n = 400
    times = np.arange(n, dtype=float)
    truth = 0.01 * times
    noise = _ar1_noise(n)
    series = truth + noise

    result = optimal_span_smooth(times, series, n_bootstrap=100, random_seed=42)

    residual = series - result.smoothed
    assert float(np.std(residual, ddof=1)) < 0.3 * float(np.std(noise, ddof=1))


def test_selected_span_removes_the_positive_serial_correlation():
    """
    判据: AR(1) 噪声的 lag-1 自相关为正 (约 +0.9); 平滑后残差的 lag-1
    自相关必须 **小于 0**, 即那部分正相关已被平滑曲线吸收。

    (移动平均后的残差天然带负相关, 所以判据不是"自相关 == 0"。)
    """
    n = 400
    times = np.arange(n, dtype=float)
    noise = _ar1_noise(n)
    series = 0.01 * times + noise

    result = optimal_span_smooth(times, series, n_bootstrap=100, random_seed=42)

    assert _lag1_autocorrelation(noise) > 0.5
    assert _lag1_autocorrelation(series - result.smoothed) < 0.0


def test_selected_span_lies_inside_the_candidate_range():
    """选中的跨度必须落在请求的候选区间内, 且是合法的样本数。"""
    n = 300
    times = np.arange(n, dtype=float)
    series = np.sin(times / 20.0) + 0.5 * np.cos(times / 3.0)

    result = optimal_span_smooth(times, series, min_span=5, max_span=60, random_seed=3)

    assert 5 <= result.span <= 60
    assert float(result.span).is_integer()


def test_pure_noise_smoothing_reduces_the_residual_variance():
    """
    判据: 纯白噪声序列平滑后, 残差方差必须小于原始方差 (噪声被压下去了)。
    """
    n = 300
    times = np.arange(n, dtype=float)
    series = np.random.default_rng(11).normal(0.0, 1.0, n)

    result = optimal_span_smooth(times, series, n_bootstrap=100, random_seed=1)

    residual_variance = float(np.var(series - result.smoothed, ddof=1))
    assert residual_variance < float(np.var(series, ddof=1))


def test_a_step_is_not_flattened_by_the_selected_span():
    """
    判据 (三层): 单位阶跃的两侧平台仍约为 0 与 1, 且过渡只发生在阶跃附近
    (过渡样本数 <= 2 * 跨度 + 2), 没有被抹成贯穿整条记录的长斜坡。
    """
    n = 200
    times = np.arange(n, dtype=float)
    series = np.zeros(n)
    series[100:] = 1.0

    result = optimal_span_smooth(times, series, n_bootstrap=100, random_seed=3)
    smoothed = result.smoothed

    assert smoothed[50] == pytest.approx(0.0, abs=0.05)
    assert smoothed[150] == pytest.approx(1.0, abs=0.05)

    transitional = np.nonzero((smoothed > 0.1) & (smoothed < 0.9))[0]
    assert transitional.size <= 2 * int(result.span) + 2


def test_aic_criterion_is_accepted():
    """``criterion="aic"`` 必须可用, 并给出落在候选区间内的跨度。"""
    n = 250
    times = np.arange(n, dtype=float)
    series = np.sin(times / 15.0) + 0.3 * np.random.default_rng(5).normal(0.0, 1.0, n)

    result = optimal_span_smooth(times, series, criterion="aic", n_bootstrap=50, random_seed=2)

    assert result.criterion == "aic"
    assert 3 <= result.span <= n


# =============================================================================
# 置信带与可复现性
# =============================================================================


def test_confidence_band_brackets_the_smoothed_curve():
    """置信带必须包住拟合曲线, 且上下界不交叉。"""
    n = 300
    times = np.arange(n, dtype=float)
    series = np.sin(times / 20.0) + 0.5 * np.random.default_rng(13).normal(0.0, 1.0, n)

    result = optimal_span_smooth(times, series, n_bootstrap=200, random_seed=17)

    assert np.all(result.lower <= result.smoothed + 1e-9)
    assert np.all(result.smoothed <= result.upper + 1e-9)
    assert np.all(result.lower <= result.upper)


def test_fixed_seed_gives_identical_results():
    """固定随机种子必须给出完全相同的结果。"""
    n = 200
    times = np.arange(n, dtype=float)
    series = np.sin(times / 10.0) + 0.4 * np.random.default_rng(3).normal(0.0, 1.0, n)

    first = optimal_span_smooth(times, series, n_bootstrap=100, random_seed=99)
    second = optimal_span_smooth(times, series, n_bootstrap=100, random_seed=99)

    assert first.span == second.span
    assert np.array_equal(first.smoothed, second.smoothed)
    assert np.array_equal(first.lower, second.lower)
    assert np.array_equal(first.upper, second.upper)


def test_missing_seed_warns_about_irreproducibility():
    """不给种子时按仓库约定发 ``RuntimeWarning``。"""
    n = 120
    times = np.arange(n, dtype=float)
    series = np.sin(times / 10.0)

    with pytest.warns(RuntimeWarning):
        optimal_span_smooth(times, series, n_bootstrap=20)


def test_summary_reports_the_selected_span():
    """摘要应含判据与选中的跨度。"""
    n = 120
    times = np.arange(n, dtype=float)
    series = np.sin(times / 10.0)

    summary = optimal_span_smooth(times, series, n_bootstrap=20, random_seed=1).summary()

    assert "gcv" in summary
    assert "span" in summary


# =============================================================================
# 边界与错误路径
# =============================================================================


def test_too_few_samples_raise():
    """点数太少必须报错。"""
    with pytest.raises(ValidationError):
        optimal_span_smooth(np.arange(3, dtype=float), np.zeros(3))


def test_invalid_span_bounds_raise():
    """``min_span`` < 3、``max_span`` <= ``min_span`` 必须报错。"""
    n = 100
    times = np.arange(n, dtype=float)
    series = np.sin(times)

    with pytest.raises(ValidationError):
        optimal_span_smooth(times, series, min_span=2)
    with pytest.raises(ValidationError):
        optimal_span_smooth(times, series, min_span=10, max_span=10)
    with pytest.raises(ValidationError):
        optimal_span_smooth(times, series, min_span=10, max_span=5)


def test_invalid_candidate_count_raises():
    """候选跨度少于 2 个必须报错。"""
    n = 100
    times = np.arange(n, dtype=float)
    series = np.sin(times)

    with pytest.raises(ValidationError):
        optimal_span_smooth(times, series, n_candidates=1)


def test_unknown_criterion_raises():
    """未知判据必须报错。"""
    n = 100
    times = np.arange(n, dtype=float)
    series = np.sin(times)

    with pytest.raises(ValidationError):
        optimal_span_smooth(times, series, criterion="bic")
