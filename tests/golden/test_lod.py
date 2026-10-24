# =============================================================================
# FILE: tests/golden/test_lod.py
# =============================================================================
"""
``stratigraphy.lod`` 的测试: 一日之长（Length of Day, LOD）的米兰科维奇计算。

LOD 的真实含义
~~~~~~~~~~~~~~
Acycle 的 "LOD" 工具是 ``code/guicode/LODGUI.m``。读完之后得到的结论是:
**LOD = Length of Day, 即"一日之长"（小时）, 配套给出"一年有多少天"**,
不是分形维数, 也不是对数-对数谱。所以本文件断言的"真值"全部是**一日之长
的真值**, 而不是任何分形维数。

断言里用到的真值（出处见每条测试的注释）
----------------------------------------
1. **今日锚点 23.934468 h** —— Laskar et al. (2004) 长期解给日长定的标定点。
   它同时可以用一个**独立的天文事实**核对: 现代平太阳日的长度是
   ``86400 / 1.00273790935 = 86164.0905 s = 23.9344696 h``。两者相差不到
   2e-6 h。
2. **500 Ma 处 20.084134 h、1000 Ma 处 16.125858 h** —— 同一篇论文的日长
   长期解（Acycle 的 ``lodla04.m`` 用的就是这条拟合）。
3. **今日年长 366.25002 天** —— 可以**独立推出来**: 一个儒略年是 365.25 个
   平太阳日, 而每个平太阳日里地球自转 1.00273790935 圈, 于是
   ``365.25 * 1.00273790935 = 366.2500214`` 天。这条断言检验的是"日长与年长
   之间的换算", 与拟合式本身无关。
4. **今日日长增速 2.6756 ms/世纪** —— 拟合式的导数; 公认的潮汐耗散导致的
   日长增长量级 (~2.3 ms/世纪, 月球激光测距实测) 相符。

合成的"已知真值"记录
--------------------
:func:`lod_dimension` 用已发表的前向模型（Laskar et al. 2004 的日长曲线）
去拟合观测, 只留一个比例系数。合成记录因此直接取**真值曲线**:
``values = LOD_true(age)``, 估计值必须回到真值上; 再加一点噪声, 估计值仍
必须落在真值的容差内, 并且给出**非零**的不确定度。

参考来源
--------
* Laskar, J., Robutel, P., Joutel, F., Gastineau, M., Correia, A.C.M. &
  Levrard, B., 2004. A long-term numerical solution for the insolation
  quantities of the Earth. *Astronomy & Astrophysics*, 428: 261-285.
* Laskar, J., Gastineau, M., Jounder, E. & Bourgeois, F., 2004. Solution for
  the eccentricity and obliquity of the Earth's orbits over 5 Myr.
  *Astronomy and Astrophysics*, 413: 755-767.
* Acycle: https://github.com/mingsongli/acycle （GPL; 本仓库是独立实现,
  没有翻译其 MATLAB 源码）。
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"lod_dimension"}


import numpy as np
import pytest

from stratigraphy.lod import (
    LODResult,
    lod_day_lengthening_rate,
    lod_days_in_year,
    lod_dimension,
    lod_length_of_day,
)
from utils.exceptions import ValidationError

# =============================================================================
# 真值常量
# =============================================================================

#: Laskar et al. (2004) 长期解给今日 (0 Ma) 日长定的锚点 (小时)。
TRUE_REFERENCE_HOURS = 23.934468

#: 现代平太阳日的长度 (小时), 由 86400 s / 1.00273790935 独立算出。
MODERN_MEAN_SOLAR_DAY_HOURS = 23.9344696

#: 500 Ma 与 1000 Ma 处的日长真值 (小时), 由已发表的长期解给出。
TRUE_DAY_LENGTH_500_MA = 20.084134
TRUE_DAY_LENGTH_1000_MA = 16.125858

#: 今日年长的独立推算: 儒略年的 365.25 平太阳日 × 每日 1.00273790935 自转。
TRUE_DAYS_IN_YEAR = 365.25 * 1.00273790935

#: 今日日长增速 (ms/世纪) 的模型值, 以及公认的潮汐增长量级。
TRUE_RATE_MS_PER_CENTURY = 2.6756
PUBLISHED_TIDAL_RANGE = (2.0, 3.5)

#: 合成记录覆盖的年龄范围 (Ma) 与点数。
SYNTHETIC_SPAN_MA = 500.0
SYNTHETIC_POINTS = 51

#: 噪声合成记录的噪声水平 (小时) 与由此得到的容差。
NOISE_LEVEL_HOURS = 0.01
NOISY_TOLERANCE_HOURS = 0.01


# =============================================================================
# 辅助函数
# =============================================================================


def _synthetic_ages(points: int = SYNTHETIC_POINTS, span_ma: float = SYNTHETIC_SPAN_MA) -> np.ndarray:
    """合成的年龄轴 (Ma), 正值向过去。"""
    return np.linspace(0.0, span_ma, points)


def _true_day_length(ages: np.ndarray) -> np.ndarray:
    """真值日长 (小时) —— 已发表的长期解。"""
    return np.asarray(lod_length_of_day(ages), dtype=float)


# =============================================================================
# 1. 模型的锚点事实
# =============================================================================


class TestPublishedAnchors:
    """Laskar et al. (2004) 的日长解必须复现那些已知事实。"""

    def test_today_day_length_is_the_published_anchor(self):
        """今日的日长必须是 23.934468 h。"""
        value = float(lod_length_of_day(0.0))
        assert value == pytest.approx(TRUE_REFERENCE_HOURS, abs=1e-9), (
            f"the day length at 0 Ma is {value!r} h, expected {TRUE_REFERENCE_HOURS} h"
        )

    def test_today_day_length_agrees_with_the_modern_mean_solar_day(self):
        """今日的日长必须与现代平太阳日 (86400 s / 1.00273790935) 一致。

        这是一条**独立**的天文事实, 与拟合式无关: 平太阳日的长度由地球自转
        速率定, 与 Laskar 的解没有耦合。
        """
        value = float(lod_length_of_day(0.0))
        assert value == pytest.approx(MODERN_MEAN_SOLAR_DAY_HOURS, abs=1e-4), (
            f"the day length at 0 Ma is {value:.6f} h, the modern mean solar day is {MODERN_MEAN_SOLAR_DAY_HOURS:.6f} h"
        )

    def test_day_length_deep_in_the_past_matches_the_published_solution(self):
        """500 Ma 与 1000 Ma 的日长必须落在已发表的长期解上 (±0.001 h)。"""
        at_500 = float(lod_length_of_day(500.0))
        at_1000 = float(lod_length_of_day(1000.0))
        assert at_500 == pytest.approx(TRUE_DAY_LENGTH_500_MA, abs=0.001)
        assert at_1000 == pytest.approx(TRUE_DAY_LENGTH_1000_MA, abs=0.001)

    def test_days_in_year_is_independently_derivable(self):
        """今日的年长必须等于儒略年里的自转圈数 (365.25 × 1.00273790935)。"""
        value = float(lod_days_in_year(0.0))
        assert value == pytest.approx(TRUE_DAYS_IN_YEAR, abs=1e-3), (
            f"the length of the year at 0 Ma is {value:.5f} d, expected {TRUE_DAYS_IN_YEAR:.5f} d"
        )

    def test_year_length_and_day_length_are_consistent(self):
        """年长与日长必须严格互为倒数关系 (秒数恒为 3.15576e7)。"""
        ages = _synthetic_ages()
        years = np.asarray(lod_days_in_year(ages), dtype=float)
        days = np.asarray(lod_length_of_day(ages), dtype=float)
        assert np.allclose(years * days * 3600.0, 3.15576e7)

    def test_day_lengthening_rate_at_the_present(self):
        """今日的日长增速必须是 ~2.68 ms/世纪, 且落在公认的潮汐量级内。"""
        rate = float(lod_day_lengthening_rate(0.0))
        assert rate == pytest.approx(TRUE_RATE_MS_PER_CENTURY, abs=0.01), (
            f"the day-lengthening rate at 0 Ma is {rate:.4f} ms/century, expected {TRUE_RATE_MS_PER_CENTURY} ms/century"
        )
        assert PUBLISHED_TIDAL_RANGE[0] <= rate <= PUBLISHED_TIDAL_RANGE[1], (
            "the modelled rate is outside the published range for tidal day-lengthening"
        )

    def test_day_length_shortens_monotonically_into_the_past(self):
        """向过去走, 日长必须单调变短。"""
        ages = np.linspace(0.0, 1000.0, 200)
        days = np.asarray(lod_length_of_day(ages), dtype=float)
        assert np.all(np.diff(days) < 0.0), "the day length is not monotonically decreasing into the past"
        assert float(days[-1]) == pytest.approx(TRUE_DAY_LENGTH_1000_MA, abs=0.001)

    def test_day_length_stays_physically_plausible(self):
        """0–1000 Ma 的日长必须落在 10–24 h 之间。"""
        ages = np.linspace(0.0, 1000.0, 400)
        days = np.asarray(lod_length_of_day(ages), dtype=float)
        assert np.all(days > 10.0)
        assert np.all(days <= 24.0)


# =============================================================================
# 2. 估计: 合成真值
# =============================================================================


class TestLodDimensionGroundTruth:
    """``lod_dimension`` 必须把合成的已知真值找回来。"""

    def test_recovers_the_true_day_length_from_a_clean_record(self):
        """观测就是真值曲线时, 估计值必须回到真值上。"""
        ages = _synthetic_ages()
        observed = _true_day_length(ages)

        result = lod_dimension(ages, observed)

        assert result.dimension == pytest.approx(TRUE_REFERENCE_HOURS, abs=1e-3), (
            f"the estimated day length is {result.dimension:.5f} h, expected {TRUE_REFERENCE_HOURS} h"
        )
        assert result.reference_age_ma == pytest.approx(0.0)

    def test_recovers_the_true_day_length_from_a_noisy_record(self):
        """加了 0.01 h 噪声的记录, 估计值仍必须落在真值的容差内。"""
        rng = np.random.default_rng(7)
        ages = _synthetic_ages()
        observed = _true_day_length(ages) + NOISE_LEVEL_HOURS * rng.standard_normal(ages.size)

        result = lod_dimension(ages, observed)

        assert result.dimension == pytest.approx(TRUE_REFERENCE_HOURS, abs=NOISY_TOLERANCE_HOURS)
        assert result.uncertainty > 0.0, "a noisy record must produce a non-zero uncertainty"
        assert result.uncertainty < NOISY_TOLERANCE_HOURS, (
            f"the uncertainty {result.uncertainty:.5f} h is larger than the tolerance"
        )

    def test_the_estimated_uncertainty_covers_the_estimation_error(self):
        """估计误差必须落在一个标准误之内。"""
        rng = np.random.default_rng(11)
        ages = _synthetic_ages()
        observed = _true_day_length(ages) + NOISE_LEVEL_HOURS * rng.standard_normal(ages.size)

        result = lod_dimension(ages, observed)

        error = abs(result.dimension - TRUE_REFERENCE_HOURS)
        assert error <= 3.0 * max(result.uncertainty, 1e-9), (
            f"the error {error:.6f} h is much larger than the reported uncertainty {result.uncertainty:.6f} h"
        )

    def test_the_scale_of_the_record_is_recovered(self):
        """观测整体放大两倍时, 估计出的日长也必须放大两倍。

        这检验的是拟合里那个唯一的自由参数 (比例系数) 确实被解出来了, 而
        不是把模型值原样抄了一遍。
        """
        ages = _synthetic_ages()
        observed = 2.0 * _true_day_length(ages)

        result = lod_dimension(ages, observed)

        assert result.dimension == pytest.approx(2.0 * TRUE_REFERENCE_HOURS, abs=1e-3)

    def test_without_observations_the_model_value_is_reported(self):
        """不给观测时, 返回模型在参考年龄上的日长, 不确定度为 0。"""
        ages = _synthetic_ages()

        result = lod_dimension(ages, None)

        assert result.dimension == pytest.approx(TRUE_REFERENCE_HOURS, abs=1e-9)
        assert result.uncertainty == 0.0


# =============================================================================
# 3. 返回值的形状
# =============================================================================


class TestLodResultShape:
    """结果容器必须给出模型曲线与年龄栅格。"""

    def test_the_curve_is_sampled_on_the_requested_grid(self):
        """栅格点数必须等于 ``n_scales``, 且覆盖整条记录。"""
        ages = _synthetic_ages()
        observed = _true_day_length(ages)

        result = lod_dimension(ages, observed, n_scales=13)

        assert isinstance(result, LODResult)
        assert result.scales.shape == (13,)
        assert result.values.shape == (13,)
        assert float(result.scales[0]) == pytest.approx(0.0)
        assert float(result.scales[-1]) == pytest.approx(SYNTHETIC_SPAN_MA)

    def test_the_curve_decreases_into_the_past(self):
        """返回的日长曲线必须向过去单调变短。"""
        ages = _synthetic_ages()
        observed = _true_day_length(ages)

        result = lod_dimension(ages, observed)

        assert np.all(np.diff(result.values) < 0.0)

    def test_the_curve_agrees_with_the_model_at_the_reference_age(self):
        """曲线的参考年龄处必须与 ``dimension`` 相符 (比例系数一致)。"""
        ages = _synthetic_ages()
        observed = 1.5 * _true_day_length(ages)

        result = lod_dimension(ages, observed, min_scale=0)

        assert float(result.values[0]) == pytest.approx(result.dimension, abs=1e-9)

    def test_explicit_age_bounds_are_honoured(self):
        """给了 ``min_scale`` / ``max_scale`` 时, 栅格必须落在给定的年龄界内。"""
        ages = np.linspace(100.0, 300.0, 21)

        result = lod_dimension(ages, None, n_scales=5, min_scale=100, max_scale=300)

        assert float(result.scales[0]) == pytest.approx(100.0)
        assert float(result.scales[-1]) == pytest.approx(300.0)

    def test_the_curve_always_covers_the_record(self):
        """记录比 ``min_scale`` 更年轻时, 栅格仍从记录的最年轻年龄开始。"""
        ages = _synthetic_ages()

        result = lod_dimension(ages, None, n_scales=5, min_scale=100, max_scale=300)

        assert float(result.scales[0]) == pytest.approx(0.0)
        assert float(result.scales[-1]) == pytest.approx(300.0)

    def test_summary_mentions_the_day_length(self):
        """摘要必须能读出日长。"""
        ages = _synthetic_ages()
        result = lod_dimension(ages, _true_day_length(ages))

        text = result.summary()

        assert "LOD" in text
        assert f"{result.dimension:.4f}" in text


# =============================================================================
# 4. 边界
# =============================================================================


class TestBoundaries:
    """非法输入必须抛 ``ValidationError``。"""

    def test_mismatched_lengths_are_rejected(self):
        ages = _synthetic_ages()
        with pytest.raises(ValidationError):
            lod_dimension(ages, _true_day_length(ages)[:-1])

    def test_a_single_age_is_rejected(self):
        with pytest.raises(ValidationError):
            lod_dimension(np.array([0.0]), np.array([23.9]))

    def test_non_finite_ages_are_rejected(self):
        ages = _synthetic_ages()
        broken = ages.copy()
        broken[3] = np.nan
        with pytest.raises(ValidationError):
            lod_dimension(broken, _true_day_length(broken))

    def test_non_finite_observations_are_rejected(self):
        ages = _synthetic_ages()
        observed = _true_day_length(ages)
        observed[2] = np.inf
        with pytest.raises(ValidationError):
            lod_dimension(ages, observed)

    def test_a_two_dimensional_age_axis_is_rejected(self):
        with pytest.raises(ValidationError):
            lod_dimension(np.zeros((4, 2)), np.zeros((4, 2)))

    def test_too_few_grid_points_are_rejected(self):
        ages = _synthetic_ages()
        with pytest.raises(ValidationError):
            lod_dimension(ages, _true_day_length(ages), n_scales=1)

    def test_an_empty_age_grid_is_rejected(self):
        ages = np.linspace(300.0, 400.0, 11)
        with pytest.raises(ValidationError):
            lod_dimension(ages, None, min_scale=400, max_scale=100)
