# =============================================================================
# FILE: tests/golden/test_age_depth_uncertainty.py
# =============================================================================
"""
``stratigraphy.age_depth_uncertainty``（Undatable 风格年龄-深度不确定度模型）
的测试。

测试围绕六条主线组织，每条断言的依据写在各自的 docstring 里：

* **无误差时的退化。** 所有定年点的不确定度为 0 时，年龄-深度带必须**塌缩**
  到给定的年龄-深度曲线（带宽 ≈ 0）。这是本文件最核心的断言：它钉死了
  "带宽只由锚点不确定度驱动"这条契约——如果中间点（或自助）凭空注入宽度，
  这条断言立刻失败。

* **不确定度越大带宽越大。** 把锚点的年龄 1σ 从 0 增到 2、5、10，1σ 带的宽度
  必须**单调增大**。这条断言保证"带"确实反映输入的不确定度，而不是一个
  与输入无关的常数。

* **百分位的单调性。** 1..99 百分位带必须满足 ``p1 <= p2 <= ... <= p99``
  （逐点）。百分位表的行列含义一旦搞反（例如把模拟当成深度），这条断言会
  在随机数据上立刻失败。

* **倒转剔除有效。** 一组**故意让年龄随深度倒转**的锚点必须让
  ``n_accepted < n_simulations``：倒转组合被剔除掉了。反过来，一条年龄随深度
  单调递增的剖面必须**一条模拟都不丢**（``n_accepted == n_simulations``）——
  否则倒转剔除就变成了无差别丢数据。

* **锚点穿过。** 模型给出的年龄-深度曲线在每个锚点深度处必须**穿过**该锚点
  的年龄（落在其 1σ 带内）。这条断言保证模型是"过锚点"的，而不是在锚点
  之间飘过去的。

* **沉积速率的手算对照。** 对**线性**年龄-深度关系，沉积速率是常数，且等于
  ``d(age)/d(depth)`` 的解析值（逐点比对，容差 1e-6）；对二次关系，
  ``numpy.gradient`` 在内点上应当给出解析导数。

可复现性断言固定种子给出完全相同的结果，不同种子给出不同但统计上相近的
结果。边界测试断言抛 ``utils.exceptions.ValidationError``。

参考来源（断言中引用的数值出处）
--------------------------------
* Lougheed, C. M. & Obrochta, G. W. (2019). Uncertainty in age-depth models
  and its impact on paleoclimate reconstructions. *Paleoceanography and
  Paleoclimatology*, 34(7), 122-133. doi:10.1029/2018PA003457.
  （Undatable 的算法论文：蒙特卡洛年龄-深度模型、锚点外推、倒转剔除。）
* Lougheed, C. M. & Obrochta, G. W. (2016). Undatable: A software package
  for objective age-depth modeling. *Journal of Open Research Software*, 4(1),
  e42. doi:10.5334/jors.2016.42.（软件论文。）
* Bronk Ramsey, C. (2008). Radiocarbon calibration: individual ages from
  radiocarbon years. *Radiocarbon*, 50(4), 1303-1315.（¹⁴C 定年校准原理；
  本模块**不**实现校准，只使用校准后的年龄。）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12-22.（Acycle 的 "Undatable" 子功能。）
"""

from __future__ import annotations

#: 本文件覆盖的分析（矩阵以此声明为准）
ANALYSES_COVERED = {"age_depth_uncertainty"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.age_depth_uncertainty import (
    age_depth_density,
    sedimentation_rate_from_age_depth,
    undatable_age_depth_model,
)
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 合成剖面的深度（m）：四个锚点，等间距。
DEPTHS = [0.0, 50.0, 100.0, 150.0]

#: 与 ``DEPTHS`` 对应的年龄：线性关系，每米 10 个年龄单位。
AGES = [0.0, 500.0, 1000.0, 1500.0]

#: 线性年龄-深度关系的解析速率 d(age)/d(depth)。
LINEAR_RATE = 10.0

#: 测试用的蒙特卡洛次数：足够让百分位稳定，又不至于让整个文件跑超时。
NSIM = 300

#: 基准种子。固定种子 ⇒ 结果逐位可复现。
SEED = 42

#: 另一个种子：用来对照"不同种子统计上相近"。
OTHER_SEED = 43

#: 无误差退化断言的容差：此时所有模拟逐位相同，偏差只可能是浮点舍入。
COLLAPSE_TOLERANCE = 1e-9


# =============================================================================
# 辅助函数
# =============================================================================


def _band_width(model) -> npt.NDArray:
    """1σ 带的宽度（逐深度）。"""
    return model.sigma1_age_depth[1] - model.sigma1_age_depth[0]


def _input_range_mask(model) -> npt.NDArray:
    """输出栅格落在输入深度范围内的部分。

    默认的两轮结构会向输入剖面的上下各外推一个锚点，栅格因此比输入深；
    只在输入深度范围内做比较，才是在比较"模型 vs 给定曲线"。
    """
    return (model.depths >= DEPTHS[0]) & (model.depths <= DEPTHS[-1])


def _reference_curve(depths: npt.NDArray) -> npt.NDArray:
    """给定的年龄-深度曲线（锚点之间的分段线性插值）。"""
    return np.interp(depths, DEPTHS, AGES)


def _mean_band_width(sigma: float) -> float:
    """锚点 1σ 为 ``sigma`` 时，1σ 带在输入深度范围内的平均宽度。"""
    model = undatable_age_depth_model(
        DEPTHS,
        AGES,
        [sigma] * len(DEPTHS),
        nsim=NSIM,
        seed=SEED,
    )
    return float(np.nanmean(_band_width(model)[_input_range_mask(model)]))


def _model_at(model, depth: float, values: npt.NDArray) -> float:
    """取模型在给定深度处的值（栅格上线性插值）。"""
    return float(np.interp(depth, model.depths, values))


# =============================================================================
# 无误差时的退化
# =============================================================================


class TestZeroUncertaintyDegeneracy:
    """不确定度为零时，年龄-深度带必须塌缩到给定的曲线。"""

    def test_the_band_collapses_onto_the_given_curve(self):
        """所有锚点 1σ 为 0：1σ 带的宽度必须 ≈ 0。

        依据：带宽的**唯一**来源是锚点自身的不确定度（见模块 docstring）。
        自助也必须关掉（``bootpc=0``）——自助按比例丢掉定年点，被丢掉的点
        所在深度上的曲线只能由邻点插值得到，那是另一种宽度来源，与本条断言
        无关。
        """
        model = undatable_age_depth_model(
            DEPTHS,
            AGES,
            [0.0] * len(DEPTHS),
            nsim=NSIM,
            bootpc=0.0,
            seed=SEED,
        )

        width = _band_width(model)
        assert np.nanmax(width) <= COLLAPSE_TOLERANCE, (
            f"the zero-uncertainty band is {np.nanmax(width):.6g} wide, not collapsed"
        )

    def test_the_percentiles_sit_on_the_given_curve(self):
        """不确定度为零：整条百分位带必须落在给定的分段线性曲线上。"""
        model = undatable_age_depth_model(
            DEPTHS,
            AGES,
            [0.0] * len(DEPTHS),
            nsim=NSIM,
            bootpc=0.0,
            seed=SEED,
        )

        inside = _input_range_mask(model)
        reference = _reference_curve(model.depths[inside])
        deviation = np.nanmax(np.abs(model.age_depth_percentiles[:, inside] - reference[None, :]))
        assert deviation <= COLLAPSE_TOLERANCE, f"the percentile band deviates from the given curve by {deviation:.6g}"

    def test_the_median_passes_through_every_anchor(self):
        """不确定度为零：中位曲线在每个锚点深度处等于该锚点的年龄。"""
        model = undatable_age_depth_model(
            DEPTHS,
            AGES,
            [0.0] * len(DEPTHS),
            nsim=NSIM,
            bootpc=0.0,
            seed=SEED,
        )

        for depth, age in zip(DEPTHS, AGES, strict=True):
            assert _model_at(model, depth, model.median_age_depth) == pytest.approx(age, abs=1e-6)

    def test_the_default_configuration_also_collapses(self):
        """默认参数（``bootpc=0.1``、四个锚点）同样必须塌缩。

        四个锚点、自助比例 0.1 ⇒ 本次实现每次模拟丢掉
        ``round(0.1 * 4) = 0`` 个点，所以自助不贡献宽度；band 应当仍然是 0。
        """
        model = undatable_age_depth_model(DEPTHS, AGES, [0.0] * len(DEPTHS), nsim=NSIM, seed=SEED)

        assert model.n_accepted == model.n_simulations
        assert np.nanmax(_band_width(model)) <= COLLAPSE_TOLERANCE


# =============================================================================
# 不确定度越大带宽越大
# =============================================================================


class TestUncertaintyWidening:
    """1σ 带的宽度必须随锚点不确定度单调增大。"""

    def test_the_band_width_is_monotone_in_the_uncertainty(self):
        """把锚点 1σ 从 0 增到 2、5、10：1σ 带的宽度必须**单调增大**。

        依据：本实现里带宽只由锚点 1σ 驱动（中间点的扰动尺度也是锚点 1σ 的
        倍数），因此宽度应当随 σ 单调增大；σ=0 时宽度必须为 0。
        """
        widths = [_mean_band_width(sigma) for sigma in (0.0, 2.0, 5.0, 10.0)]

        assert widths[0] <= COLLAPSE_TOLERANCE, f"the band is {widths[0]:.6g} wide for an uncertainty-free section"
        assert widths[0] < widths[1] < widths[2] < widths[3], (
            f"the band widths are not increasing with the uncertainty: {widths}"
        )

    def test_the_widest_part_of_the_band_grows_with_the_uncertainty(self):
        """带宽的最大值也必须随 σ 增大（锚点之间那一段最宽）。"""
        widths = []
        for sigma in (0.0, 2.0, 5.0, 10.0):
            model = undatable_age_depth_model(
                DEPTHS,
                AGES,
                [sigma] * len(DEPTHS),
                nsim=NSIM,
                seed=SEED,
            )
            widths.append(float(np.nanmax(_band_width(model))))

        assert widths[0] < widths[1] < widths[2] < widths[3], (
            f"the maximum band widths are not increasing with the uncertainty: {widths}"
        )

    def test_a_bigger_uncertainty_gives_a_wider_band_than_a_smaller_one(self):
        """σ=10 的带必须明显宽于 σ=2 的带。"""
        narrow = undatable_age_depth_model(DEPTHS, AGES, [2.0] * len(DEPTHS), nsim=NSIM, seed=SEED)
        wide = undatable_age_depth_model(DEPTHS, AGES, [10.0] * len(DEPTHS), nsim=NSIM, seed=SEED)

        narrow_width = float(np.nanmean(_band_width(narrow)))
        wide_width = float(np.nanmean(_band_width(wide)))
        assert wide_width > 2.0 * narrow_width, (
            f"the band width grew from {narrow_width:.6g} to {wide_width:.6g}, not by the expected factor"
        )


# =============================================================================
# 百分位的单调性
# =============================================================================


class TestPercentileMonotonicity:
    """1..99 百分位带必须逐点单调不减。"""

    def test_the_percentiles_are_monotone(self):
        """``p1 <= p2 <= ... <= p99`` 必须逐深度成立。"""
        model = undatable_age_depth_model(
            DEPTHS,
            AGES,
            [10.0] * len(DEPTHS),
            nsim=NSIM,
            seed=SEED,
        )

        percentiles = model.age_depth_percentiles
        assert percentiles.shape[0] == 99
        assert np.all(np.diff(percentiles, axis=0) >= -1e-9), (
            "the age-depth percentiles are not monotonically increasing"
        )

    def test_the_median_is_the_fiftieth_percentile(self):
        """中位曲线必须就是第 50 百分位。"""
        model = undatable_age_depth_model(
            DEPTHS,
            AGES,
            [10.0] * len(DEPTHS),
            nsim=NSIM,
            seed=SEED,
        )

        assert np.allclose(model.median_age_depth, model.age_depth_percentiles[49], equal_nan=True)

    def test_the_sigma_bands_are_ordered(self):
        """1σ 带必须嵌在 2σ 带里，中位数必须落在 1σ 带里。"""
        model = undatable_age_depth_model(
            DEPTHS,
            AGES,
            [10.0] * len(DEPTHS),
            nsim=NSIM,
            seed=SEED,
        )

        assert np.all(model.sigma2_age_depth[0] <= model.sigma1_age_depth[0] + 1e-9)
        assert np.all(model.sigma1_age_depth[1] <= model.sigma2_age_depth[1] + 1e-9)
        assert np.all(model.median_age_depth >= model.sigma1_age_depth[0] - 1e-9)
        assert np.all(model.median_age_depth <= model.sigma1_age_depth[1] + 1e-9)


# =============================================================================
# 倒转剔除
# =============================================================================


class TestReversalFilter:
    """年龄随深度倒转的组合必须被剔除。"""

    def test_reversed_dates_are_filtered_out(self):
        """深度递增但年龄递减：``n_accepted`` 必须小于 ``n_simulations``。

        依据：年龄必须随深度递增；抽到倒转组合的模拟会被剔除掉较浅的点，
        剩下的点不足两个时整条模拟作废。这里给了 400 个年龄单位的 1σ，
        所以一部分模拟仍能凑出合法曲线（``n_accepted > 0``），但绝大多数
        必须被剔掉。
        """
        model = undatable_age_depth_model(
            [0.0, 10.0, 20.0, 30.0],
            [4000.0, 3000.0, 2000.0, 1000.0],
            [400.0] * 4,
            nsim=NSIM,
            bootpc=0.0,
            seed=11,
        )

        assert model.n_accepted < model.n_simulations, (
            f"no simulation was filtered: {model.n_accepted} of {model.n_simulations} survived"
        )
        assert model.n_accepted > 0, "every simulation was filtered, so the filter is untested"

    def test_a_monotone_section_keeps_every_simulation(self):
        """年龄随深度单调递增的剖面必须一条模拟都不丢。

        这条断言是上一条的对照：倒转剔除必须只针对"真的倒转"的组合，否则
        它就退化成了无差别丢数据。
        """
        model = undatable_age_depth_model(
            DEPTHS,
            AGES,
            [10.0] * len(DEPTHS),
            nsim=NSIM,
            seed=SEED,
        )

        assert model.n_accepted == model.n_simulations


# =============================================================================
# 锚点穿过
# =============================================================================


class TestAnchorPiercing:
    """模型曲线必须穿过每个锚点的年龄。"""

    def test_the_curve_passes_through_every_anchor(self):
        """每个锚点深度处，给定的年龄必须落在模型的 1σ 带内。

        依据：锚点是年龄-深度曲线的节点，因此模型在该深度处的年龄应当就是
        该锚点的年龄（中位数的偏差只是蒙特卡洛噪声，量级 ``σ/sqrt(nsim)``）。
        """
        model = undatable_age_depth_model(
            DEPTHS,
            AGES,
            [50.0] * len(DEPTHS),
            nsim=NSIM,
            seed=SEED,
        )

        for depth, age in zip(DEPTHS, AGES, strict=True):
            low = _model_at(model, depth, model.sigma1_age_depth[0])
            high = _model_at(model, depth, model.sigma1_age_depth[1])
            assert low <= age <= high, (
                f"the anchor age {age} at depth {depth} is outside the 1sigma band [{low}, {high}]"
            )

    def test_the_median_is_close_to_every_anchor(self):
        """中位曲线在锚点处的偏差必须远小于该锚点的不确定度。"""
        sigma = 50.0
        model = undatable_age_depth_model(
            DEPTHS,
            AGES,
            [sigma] * len(DEPTHS),
            nsim=NSIM,
            seed=SEED,
        )

        for depth, age in zip(DEPTHS, AGES, strict=True):
            median = _model_at(model, depth, model.median_age_depth)
            assert abs(median - age) < sigma, (
                f"the median age at depth {depth} is {median:.6g}, more than one sigma away from {age}"
            )


# =============================================================================
# 沉积速率
# =============================================================================


class TestSedimentationRate:
    """沉积速率必须与 d(age)/d(depth) 的解析值一致。"""

    def test_the_rate_of_a_linear_relation_is_constant(self):
        """线性年龄-深度关系：沉积速率处处相同，且等于解析值。

        解析值：``d(age)/d(depth) = (1500 - 0) / (150 - 0) = 10``。
        容差 1e-6：``numpy.gradient`` 对线性关系是精确的（浮点舍入量级）。
        """
        rates = sedimentation_rate_from_age_depth(DEPTHS, AGES)

        assert rates.shape == (len(DEPTHS),)
        assert np.allclose(rates, LINEAR_RATE, atol=1e-6), f"the sedimentation rate is not constant: {rates}"

    def test_the_model_rate_of_a_linear_relation_is_constant(self):
        """模型给出的中位沉积速率也必须是那个常数。

        依据：不确定度为零时曲线就是锚点之间的分段直线，速率处处相同。
        """
        model = undatable_age_depth_model(
            DEPTHS,
            AGES,
            [0.0] * len(DEPTHS),
            nsim=NSIM,
            bootpc=0.0,
            seed=SEED,
        )

        rates = model.median_sedimentation_rate
        assert np.allclose(rates, LINEAR_RATE, atol=1e-6), (
            f"the median sedimentation rate is not the analytic one: {rates}"
        )

    def test_the_rate_percentiles_are_ordered(self):
        """沉积速率的 1–99 百分位必须逐点单调不减。"""
        model = undatable_age_depth_model(
            DEPTHS,
            AGES,
            [10.0] * len(DEPTHS),
            nsim=NSIM,
            seed=SEED,
        )

        rates = model.sedimentation_rate_percentiles
        assert rates.shape[0] == 99
        assert np.all(np.diff(rates, axis=0) >= -1e-9)

    def test_the_rate_matches_the_analytic_derivative(self):
        """二次年龄-深度关系：内点上的速率必须等于解析导数。

        ``numpy.gradient`` 对二次函数在**内点**上是精确的（中心差分），
        两端用的是单边差分，因此只比较内点。
        """
        depths = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ages = 3.0 * depths**2 + 2.0 * depths + 7.0

        rates = sedimentation_rate_from_age_depth(depths, ages)
        analytic = 6.0 * depths + 2.0

        assert np.allclose(rates[1:-1], analytic[1:-1], atol=1e-9)


# =============================================================================
# 可复现性
# =============================================================================


class TestReproducibility:
    """固定种子必须给出完全相同的结果。"""

    def test_the_same_seed_gives_the_same_result(self):
        """同一个种子：两次调用的每一个数组都必须逐位相同。"""
        first = undatable_age_depth_model(DEPTHS, AGES, [10.0] * len(DEPTHS), nsim=NSIM, seed=SEED)
        second = undatable_age_depth_model(DEPTHS, AGES, [10.0] * len(DEPTHS), nsim=NSIM, seed=SEED)

        assert np.array_equal(first.age_depth_percentiles, second.age_depth_percentiles)
        assert np.array_equal(first.median_age_depth, second.median_age_depth)
        assert np.array_equal(first.age_samples, second.age_samples)
        assert first.n_accepted == second.n_accepted

    def test_different_seeds_differ_but_agree_within_the_band(self):
        """不同种子：结果必须不同，但中位年龄的差异必须小于 1σ 带宽。

        依据：蒙特卡洛的中位数估计量的抽样误差量级是 ``σ/sqrt(nsim)``，
        远小于 1σ 带本身的宽度。
        """
        first = undatable_age_depth_model(DEPTHS, AGES, [10.0] * len(DEPTHS), nsim=NSIM, seed=SEED)
        other = undatable_age_depth_model(DEPTHS, AGES, [10.0] * len(DEPTHS), nsim=NSIM, seed=OTHER_SEED)

        inside = _input_range_mask(first)
        assert not np.allclose(
            first.median_age_depth[inside],
            other.median_age_depth[inside],
            atol=1e-9,
        ), "two different seeds gave identical results"

        difference = np.abs(first.median_age_depth - other.median_age_depth)[inside]
        width = _band_width(first)[inside]
        assert np.all(difference < width), (
            "the median ages of two independent runs differ by more than their own 1sigma band"
        )

    def test_the_uncertainty_is_reported_as_a_warning_without_a_seed(self):
        """不给种子时必须发出 ``RuntimeWarning``（仓库的统一约定）。"""
        with pytest.warns(RuntimeWarning):
            undatable_age_depth_model(DEPTHS, AGES, [10.0] * len(DEPTHS), nsim=50)


# =============================================================================
# 密度云
# =============================================================================


class TestAgeDepthDensity:
    """密度云必须是一个归一化的二维直方图。"""

    def test_the_density_is_normalised(self):
        """密度之和为 1，且处处非负。"""
        model = undatable_age_depth_model(
            DEPTHS,
            AGES,
            [10.0] * len(DEPTHS),
            nsim=NSIM,
            seed=SEED,
        )
        cloud = age_depth_density(model.age_samples.T, model.depths)

        assert cloud.density.shape == (model.depths.size, cloud.age_edges.size - 1)
        assert np.all(cloud.density >= 0.0)
        assert float(np.sum(cloud.density)) == pytest.approx(1.0, abs=1e-12)

    def test_the_density_peaks_at_the_shallowest_anchor(self):
        """最浅锚点所在的那一行密度必须在该锚点的年龄处达到峰值。

        依据：模型在锚点深度处的年龄就是该锚点的年龄，所以那一行的密度
        必然集中在锚点年龄附近。
        """
        model = undatable_age_depth_model(
            DEPTHS,
            AGES,
            [10.0] * len(DEPTHS),
            nsim=NSIM,
            seed=SEED,
        )
        cloud = age_depth_density(model.age_samples.T, model.depths)

        row = int(np.argmin(np.abs(cloud.depth_centers - DEPTHS[0])))
        column = int(np.argmax(cloud.density[row]))
        bin_width = float(cloud.age_edges[1] - cloud.age_edges[0])
        assert abs(float(cloud.age_centers[column]) - AGES[0]) <= bin_width, (
            f"the density row at depth {DEPTHS[0]} peaks at age {cloud.age_centers[column]:.6g}, not at {AGES[0]}"
        )

    def test_the_density_rejects_a_mismatched_depth_axis(self):
        """深度轴长度与样本列数不一致时必须报错。"""
        model = undatable_age_depth_model(
            DEPTHS,
            AGES,
            [10.0] * len(DEPTHS),
            nsim=NSIM,
            seed=SEED,
        )
        with pytest.raises(ValidationError):
            age_depth_density(model.age_samples.T, model.depths[:-1])


# =============================================================================
# 边界
# =============================================================================


class TestBoundaries:
    """非法输入必须抛 ``ValidationError``。"""

    def test_non_increasing_depths_are_rejected(self):
        with pytest.raises(ValidationError):
            undatable_age_depth_model([0.0, 50.0, 50.0, 150.0], AGES)

    def test_decreasing_depths_are_rejected(self):
        with pytest.raises(ValidationError):
            undatable_age_depth_model([0.0, 50.0, 100.0, 99.0], AGES)

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValidationError):
            undatable_age_depth_model(DEPTHS, AGES[:-1])

    def test_non_finite_ages_are_rejected(self):
        broken = list(AGES)
        broken[2] = np.nan
        with pytest.raises(ValidationError):
            undatable_age_depth_model(DEPTHS, broken)

    def test_non_finite_depths_are_rejected(self):
        broken = list(DEPTHS)
        broken[1] = np.inf
        with pytest.raises(ValidationError):
            undatable_age_depth_model(broken, AGES)

    def test_too_few_dates_are_rejected(self):
        with pytest.raises(ValidationError):
            undatable_age_depth_model([0.0], [0.0])

    def test_mismatched_uncertainty_length_is_rejected(self):
        with pytest.raises(ValidationError):
            undatable_age_depth_model(DEPTHS, AGES, [10.0, 10.0])

    def test_negative_uncertainty_is_rejected(self):
        with pytest.raises(ValidationError):
            undatable_age_depth_model(DEPTHS, AGES, [-1.0, 10.0, 10.0, 10.0])

    @pytest.mark.parametrize("nsim", [0, -5])
    def test_invalid_simulation_counts_are_rejected(self, nsim: int):
        with pytest.raises(ValidationError):
            undatable_age_depth_model(DEPTHS, AGES, [10.0] * len(DEPTHS), nsim=nsim)

    @pytest.mark.parametrize("bootpc", [-0.1, 1.0, 1.5, np.nan])
    def test_out_of_range_bootstrapping_is_rejected(self, bootpc: float):
        with pytest.raises(ValidationError):
            undatable_age_depth_model(DEPTHS, AGES, [10.0] * len(DEPTHS), bootpc=bootpc)

    def test_a_negative_uncertainty_factor_is_rejected(self):
        with pytest.raises(ValidationError):
            undatable_age_depth_model(DEPTHS, AGES, [10.0] * len(DEPTHS), xfactor=-1.0)

    def test_an_unknown_anchor_mode_is_rejected(self):
        with pytest.raises(ValidationError):
            undatable_age_depth_model(DEPTHS, AGES, [10.0] * len(DEPTHS), anchor_mode="guess")

    def test_the_sedimentation_rate_helper_rejects_bad_input(self):
        with pytest.raises(ValidationError):
            sedimentation_rate_from_age_depth([0.0, 10.0], [0.0])
        with pytest.raises(ValidationError):
            sedimentation_rate_from_age_depth([10.0, 0.0], [0.0, 10.0])
