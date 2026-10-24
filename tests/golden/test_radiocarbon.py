# =============================================================================
# FILE: tests/golden/test_radiocarbon.py
# =============================================================================
"""
``stratigraphy.radiocarbon``（¹⁴C 定年校准）的测试。

测试围绕七条主线组织，每条断言的依据写在各自的 docstring 里：

* **真值对照（核心）。** 构造一条**人造的校准曲线**——"日历年龄 = ¹⁴C 年龄 +
  常数偏移"的直线，σ 很小——校准一个 ¹⁴C 年龄，得到的**中位与众数日历年龄
  必须落回真值**（真值 = ¹⁴C 年龄 + 偏移，容差 ±2 年）。这条断言同时钉死了
  几件事：校准方向（曲线是"日历年龄 = ¹⁴C 年龄 + 偏移"，不是反过来）、年度
  栅格（真值落在整数年上）、以及中位/众数的定义。

* **HPD 区间的覆盖。** 区间由**后验密度**给出而不是等宽区间：每个置信水平上
  的 HPD 区间必须**包含众数年龄**，68.2% 区间必须比 95.4% 区间**窄**，而且
  区间覆盖的概率质量必须 ≥ 请求的置信水平。

* **单调性。** 曲线越"陡"（σ 越小、或 ¹⁴C→日历的斜率越大），校准区间越窄。

* **水库年龄与 ΔR。** 施加储库校正后，日历年龄的位移必须**恰好等于** R(t)
  与 ΔR（平方和传播 ΔR 的不确定度）；ΔR 的不确定度增大 → 日历年龄的不确定度
  增大。

* **输入校验。** 负的 ¹⁴C 年龄、非正的误差、空曲线 / 含非有限值的曲线 / 列数
  不对的曲线、置信水平不在 ``(0, 1)`` 内、``ΔR`` 的 σ 为负 → 全部抛
  ``utils.exceptions.ValidationError``。

* **批量校准。** 批量结果与逐点单点校准**逐位一致**。

* **可复现性。** 同一个种子两次调用给出完全相同的结果。

* **曲线文件读取。** 写一个**临时**的曲线文本文件（落在 ``%TEMP%``，**不写进
  仓库**），读回来后与写入的数组逐点相等；表头行数与文件内容不符时报
  ``ValidationError``。

* **门面。** ``RadiocarbonCalibrationAnalyzer``（供参数表单驱动）的结果必须与
  直接调用 :func:`calibrate_radiocarbon` **完全一致**；曲线文件不存在或格式
  不对时必须抛 ``ValidationError``。

参考来源（断言中引用的数值出处）
--------------------------------
* Christen, J. A. & Litton, C. D. (1995). Bayesian calibration of radiocarbon
  dates. *Radiocarbon*, 37(2), 211-218.（似然 × 先验的贝叶斯框架与 HPD 区间
  的构造。）
* Bronk Ramsey, C. (2008). Radiocarbon calibration: individual ages from
  radiocarbon years. *Radiocarbon*, 50(4), 1303-1315.（概率法：单个年龄的
  后验密度。）
* Reimer, P. J. et al. (2020). IntCal20 and Marine20 radiocarbon age
  calibration. *Radiocarbon*, 62(4), 725-757.（曲线格式与曲线 σ 的传播。）
* Lougheed, B. C. & Obrochta, S. P. (2016). MatCal: open source Bayesian ¹⁴C
  age calibration in Matlab. *Journal of Open Research Software*, 4(1), e42.
  （Acycle 的 MatCal 所实现的算法。）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12-22.（Acycle 的 "MatCal" 子功能。）
"""

from __future__ import annotations

#: 本文件覆盖的分析（矩阵以此声明为准）
ANALYSES_COVERED = {"radiocarbon_calibration"}


from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.radiocarbon import (
    RadiocarbonCalibrationAnalyzer,
    calibrate_radiocarbon,
    calibrate_radiocarbon_batch,
    read_calibration_curve,
)
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 合成校准曲线覆盖的日历年龄范围（cal yr BP）。
CAL_MIN = 0.0
CAL_MAX = 1000.0

#: 合成校准曲线的节点数（间隔 5 年）。
N_CURVE_POINTS = 201

#: 合成校准曲线的偏移：日历年龄 = ¹⁴C 年龄 + OFFSET。
OFFSET = 137.0

#: 合成校准曲线的 ¹⁴C 不确定度（¹⁴C yr）。
CURVE_SIGMA = 2.0

#: 被校准的 ¹⁴C 年龄（¹⁴C yr BP）与其 1σ。
LAB_AGE = 500.0
LAB_ERROR = 20.0

#: 真值日历年龄 = LAB_AGE + OFFSET。
TRUE_CALENDAR_AGE = LAB_AGE + OFFSET

#: 真值对照的容差（年）。栅格是年度的，后验关于真值对称，因此容差只需覆盖
#: 栅格量化（半个到一年）加上离散的取整误差。
TRUTH_TOLERANCE = 2.0

#: 默认的置信水平。
LEVEL_68 = 0.6827
LEVEL_95 = 0.9545


# =============================================================================
# 辅助函数
# =============================================================================


def _linear_curve(
    offset: float = OFFSET,
    sigma: float = CURVE_SIGMA,
    slope: float = 1.0,
    cal_min: float = CAL_MIN,
    cal_max: float = CAL_MAX,
    n_points: int = N_CURVE_POINTS,
) -> npt.NDArray:
    """人造的校准曲线：``calendar_age = slope * 14C_age + offset``。

    列顺序与模块接口一致：``[radiocarbon_age, calendar_age, sigma]``。

    ``slope`` 是日历年龄对 ¹⁴C 年龄的斜率：``slope = 1`` 是常规情形，
    ``slope > 1`` 表示曲线更陡（同样的 ¹⁴C 误差对应更窄的日历年龄区间）。

    Returns:
        npt.NDArray: ``(n_points, 3)`` 的曲线数组。
    """
    calendar = np.linspace(cal_min, cal_max, n_points)
    radiocarbon = (calendar - offset) / slope
    return np.column_stack([radiocarbon, calendar, np.full(n_points, sigma)])


def _hpd_width(result, level: float) -> float:
    """某个置信水平上 HPD 区间的总宽度（外包络）。"""
    lower, upper = result.hpd_bounds(level)
    return upper - lower


def _curve_file_text(
    calendar: npt.NDArray,
    radiocarbon: npt.NDArray,
    sigma: npt.NDArray,
    header_lines: int = 11,
) -> str:
    """生成一个 IntCal 风格的 ``.14c`` 文本内容。

    数据行按**由老到新**排列（真实曲线文件就是这个顺序），每行 5 列，
    第四、五列是 ``ΔR`` 与其误差（填 0）。
    """
    lines = [f"# comment line {index}" for index in range(header_lines)]
    for cal, r14, sig in zip(calendar, radiocarbon, sigma, strict=True):
        lines.append(f"{cal:.1f},{r14:.1f},{sig:.1f},0,0")
    return "\n".join(lines) + "\n"


# =============================================================================
# 真值对照
# =============================================================================


class TestGroundTruth:
    """校准必须把 ¹⁴C 年龄放回它真正对应的日历年龄。"""

    def test_the_mode_age_falls_back_on_the_truth(self):
        """众数日历年龄必须落在真值上。

        真值的来源：曲线的定义是"日历年龄 = ¹⁴C 年龄 + 137"，所以 500 ¹⁴C yr BP
        对应的日历年龄是 ``500 + 137 = 637`` cal yr BP。后验密度在这一点上最大，
        因此众数必须落在这里（容差 ±2 年，覆盖年度栅格的量化）。

        这条断言同时钉死了校准的**方向**：如果实现把曲线反过来读（把日历年龄
        当成 ¹⁴C 年龄），众数会落在 500 − 137 = 363 上，被稳稳抓住。
        """
        result = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, _linear_curve())

        assert abs(result.mode_age - TRUE_CALENDAR_AGE) <= TRUTH_TOLERANCE, (
            f"the mode age is {result.mode_age:.1f}, not the true {TRUE_CALENDAR_AGE:.1f} cal yr BP"
        )

    def test_the_median_age_falls_back_on_the_truth(self):
        """中位日历年龄必须落在真值上。

        后验关于真值对称（正态似然 × 均匀先验），因此中位与众数重合。
        """
        result = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, _linear_curve())

        assert abs(result.median_age - TRUE_CALENDAR_AGE) <= TRUTH_TOLERANCE, (
            f"the median age is {result.median_age:.1f}, not the true {TRUE_CALENDAR_AGE:.1f} cal yr BP"
        )

    def test_the_posterior_is_centred_on_the_truth(self):
        """后验密度必须在真值处达到最大。

        这是"真值对照"的另一半：不看汇总数字，直接看密度本身。
        """
        curve = _linear_curve()
        result = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, curve)

        peak = float(result.calendar_ages[int(np.argmax(result.probability_density))])
        assert abs(peak - TRUE_CALENDAR_AGE) <= TRUTH_TOLERANCE, (
            f"the posterior peaks at {peak:.1f}, not at the true {TRUE_CALENDAR_AGE:.1f}"
        )

    def test_the_posterior_is_a_normalised_density(self):
        """后验必须归一化、非负、处处有限。"""
        result = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, _linear_curve())

        assert np.all(np.isfinite(result.probability_density))
        assert np.all(result.probability_density >= 0.0)
        assert float(np.sum(result.probability_density)) == pytest.approx(1.0, abs=1e-12)
        assert result.calendar_ages.size == result.probability_density.size

    def test_the_grid_is_annual_and_covers_the_curve(self):
        """后验栅格必须是**逐年**的，并且覆盖整条曲线。

        年度栅格来自 Acycle 的 ``interpres = 1`` 约定，也让 HPD 区间的边界落在
        整年上。
        """
        curve = _linear_curve()
        result = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, curve)

        assert np.all(np.diff(result.calendar_ages) == 1.0)
        assert float(result.calendar_ages[0]) == pytest.approx(CAL_MIN)
        assert float(result.calendar_ages[-1]) == pytest.approx(CAL_MAX)


# =============================================================================
# HPD 区间
# =============================================================================


class TestHpdIntervals:
    """HPD 区间必须来自**后验密度**，而不是等宽区间。"""

    def test_the_interval_contains_the_mode(self):
        """每个置信水平的 HPD 区间都必须包含众数年龄。

        HPD 区间的定义就是"后验密度最高的那些点"，众数是密度最大的点，因此
        它必然落在区间里。区间算错（比如退化成中位数附近的等宽区间且截断了
        众数）时这条断言会失败。
        """
        result = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, _linear_curve())

        for level in (LEVEL_68, LEVEL_95):
            intervals = result.hpd(level)
            assert intervals, f"no HPD interval at the {level} level"
            for interval in intervals:
                assert interval.lower <= result.mode_age <= interval.upper, (
                    f"the mode {result.mode_age} is outside the {level} HPD interval "
                    f"[{interval.lower}, {interval.upper}]"
                )

    def test_the_68_interval_is_narrower_than_the_95_interval(self):
        """68.2% 区间必须比 95.4% 区间窄。

        区间由密度给出，所以更低的置信水平给出**更窄**的区间；如果实现返回
        的是"中位数 ± k·σ"这样的等宽区间并且两档用了同一个 k，这条断言同样会
        失败。
        """
        result = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, _linear_curve())

        narrow = _hpd_width(result, LEVEL_68)
        wide = _hpd_width(result, LEVEL_95)

        assert narrow < wide, (
            f"the 68.2% interval is {narrow:.1f} years wide and the 95.4% interval is {wide:.1f} years wide"
        )

    def test_the_interval_mass_covers_the_requested_confidence(self):
        """区间覆盖的概率质量必须 ≥ 请求的置信水平。

        密度超水平集的质量天然 ≥ 置信水平（边界节点被整体计入，所以可能略高）。
        """
        result = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, _linear_curve())

        for level in (LEVEL_68, LEVEL_95):
            mass = sum(interval.probability for interval in result.hpd(level))
            assert mass >= level - 1e-9, f"the {level} HPD set only carries {mass:.4f}"
            assert mass <= 1.0 + 1e-9

    def test_a_bimodal_posterior_gives_two_intervals(self):
        """多峰后验必须给出多个 HPD 区间，而不是把中间的低谷也括进去。

        构造一条"倒 V"曲线：``r14(cal) = 2000 - |cal - 2000| / 10``。它在
        日历年龄 500 与 3500 处给出同一个 ¹⁴C 年龄 1850，中间（2000 cal yr
        BP）则是 ¹⁴C 年龄的极大值 —— 于是 1850 ± 10 的后验在两个日历年龄
        上各有一个峰，中间是一个深谷。HPD 集合必须被切成两段。
        """
        calendar = np.arange(0.0, 4001.0, 1.0)
        radiocarbon = 2000.0 - np.abs(calendar - 2000.0) / 10.0
        curve = np.column_stack([radiocarbon, calendar, np.full(calendar.size, 1.0)])

        result = calibrate_radiocarbon(1850.0, 10.0, curve)

        assert len(result.hpd(LEVEL_95)) >= 2, "a posterior with a deep valley must not be reported as one interval"
        # 两个峰分别在 500 与 3500 cal yr BP 附近。
        peaks = sorted((interval.lower + interval.upper) / 2.0 for interval in result.hpd(LEVEL_95))
        assert any(abs(peak - 500.0) < 200.0 for peak in peaks)
        assert any(abs(peak - 3500.0) < 200.0 for peak in peaks)


# =============================================================================
# 单调性
# =============================================================================


class TestMonotonicity:
    """曲线越陡，区间越窄。"""

    def test_a_smaller_curve_uncertainty_gives_a_narrower_interval(self):
        """曲线的 σ 变小 → 日历年龄区间变窄。

        似然的方差是 ``σ_lab² + σ_c²``（平方和），因此曲线的 σ 变小必然让
        后验变窄。这条断言钉死了"曲线的 σ 参与传播"这件事。
        """
        wide = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, _linear_curve(sigma=20.0))
        middle = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, _linear_curve(sigma=5.0))
        narrow = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, _linear_curve(sigma=0.0))

        assert _hpd_width(narrow, LEVEL_68) < _hpd_width(middle, LEVEL_68)
        assert _hpd_width(middle, LEVEL_68) < _hpd_width(wide, LEVEL_68)

    def test_the_calendar_interval_scales_with_the_curve_slope(self):
        """日历年龄区间的宽度随曲线斜率线性变化。

        曲线的定义是 ``calendar = slope · 14C + offset``，于是似然的残差
        ``r14_meas - (cal - offset)/slope`` 在 ``|cal - cal*| = slope · σ_14C``
        处达到阈值——**日历年龄的区间宽度与斜率成正比**。斜率 2 的曲线给出
        两倍宽的区间，斜率 0.5 的曲线给出的一半宽的区间。

        （注意方向的含义：斜率越大 = 日历年对 ¹⁴C 年越"敏感"，同样的 ¹⁴C
        误差铺开在更多的日历年份上，因此区间更宽；真正让区间变窄的是曲线
        **更精确**（σ 更小），见上一条断言。）
        """
        flat = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, _linear_curve(slope=1.0))
        steep = calibrate_radiocarbon(
            LAB_AGE,
            LAB_ERROR,
            _linear_curve(slope=2.0, cal_max=2000.0),
        )
        shallow = calibrate_radiocarbon(
            LAB_AGE,
            LAB_ERROR,
            _linear_curve(slope=0.5, cal_max=600.0),
        )

        assert _hpd_width(steep, LEVEL_68) > _hpd_width(flat, LEVEL_68)
        assert _hpd_width(shallow, LEVEL_68) < _hpd_width(flat, LEVEL_68)
        # 宽度与斜率成正比（栅格量化允许一点点出入）。
        assert _hpd_width(steep, LEVEL_68) == pytest.approx(2.0 * _hpd_width(flat, LEVEL_68), rel=0.1)
        assert _hpd_width(shallow, LEVEL_68) == pytest.approx(0.5 * _hpd_width(flat, LEVEL_68), rel=0.1)

    def test_a_steeper_curve_still_recovers_the_truth(self):
        """斜率改变时真值对照依然成立（真值 = slope · ¹⁴C 年龄 + 偏移）。"""
        result = calibrate_radiocarbon(
            LAB_AGE,
            LAB_ERROR,
            _linear_curve(slope=2.0, cal_max=2000.0),
        )

        assert abs(result.mode_age - (2.0 * LAB_AGE + OFFSET)) <= TRUTH_TOLERANCE


# =============================================================================
# 水库年龄与 ΔR
# =============================================================================


class TestReservoirAndDeltaR:
    """储库校正必须被正确地施加。

    **方向**：本实现（Acycle ``udmatcal.m`` 的 ``labdet = labdet - resage``）
    把 ``R(t)`` 与 ``ΔR`` 以**减量**施加在 ¹⁴C 年龄上。储库效应使样品的 ¹⁴C
    年龄偏老，减掉它之后查曲线得到的是**更年轻**的日历年龄。因此正的储库
    校正让日历年龄**变年轻** R（在单调递增的曲线上）。这一点与"Acycle 的
    减法约定"和"储库效应的物理"一致。
    """

    def test_the_reservoir_age_shifts_the_calendar_age(self):
        """加上水库年龄 R，日历年龄必须位移**恰好** R（方向：变年轻）。

        位移量由"减掉 R 之后的 ¹⁴C 年龄查同一条曲线"唯一决定：曲线的斜率是
        1，所以日历年龄的位移就是 R 的相反数。
        """
        plain = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, _linear_curve())
        corrected = calibrate_radiocarbon(
            LAB_AGE,
            LAB_ERROR,
            _linear_curve(),
            reservoir_age=200.0,
        )

        assert corrected.corrected_lab_age == pytest.approx(LAB_AGE - 200.0)
        assert corrected.median_age == pytest.approx(plain.median_age - 200.0, abs=TRUTH_TOLERANCE)
        assert corrected.mode_age == pytest.approx(plain.mode_age - 200.0, abs=TRUTH_TOLERANCE)

    def test_the_delta_r_shifts_the_calendar_age(self):
        """ΔR 与 R(t) 的作用方式相同（都是 ¹⁴C 年龄上的减量）。"""
        plain = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, _linear_curve())
        corrected = calibrate_radiocarbon(
            LAB_AGE,
            LAB_ERROR,
            _linear_curve(),
            delta_r=100.0,
        )

        assert corrected.corrected_lab_age == pytest.approx(LAB_AGE - 100.0)
        assert corrected.median_age == pytest.approx(plain.median_age - 100.0, abs=TRUTH_TOLERANCE)

    def test_the_reservoir_age_and_delta_r_add_up(self):
        """R(t) 与 ΔR 一起施加时位移是两者之和。"""
        plain = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, _linear_curve())
        corrected = calibrate_radiocarbon(
            LAB_AGE,
            LAB_ERROR,
            _linear_curve(),
            reservoir_age=200.0,
            delta_r=50.0,
        )

        assert corrected.corrected_lab_age == pytest.approx(LAB_AGE - 250.0)
        assert corrected.median_age == pytest.approx(plain.median_age - 250.0, abs=TRUTH_TOLERANCE)

    def test_the_delta_r_uncertainty_widens_the_interval(self):
        """ΔR 的不确定度越大，日历年龄的不确定度越大。

        ``σ_ΔR`` 以平方和并入似然的方差（高斯先验下这是精确的卷积），因此
        68.2% 的 HPD 宽度必然随 ``σ_ΔR`` 单调增大。
        """
        narrow = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, _linear_curve(), delta_r=0.0)
        wide = calibrate_radiocarbon(
            LAB_AGE,
            LAB_ERROR,
            _linear_curve(),
            delta_r=0.0,
            delta_r_sigma=150.0,
        )

        assert _hpd_width(wide, LEVEL_68) > _hpd_width(narrow, LEVEL_68)
        assert wide.total_lab_error > narrow.total_lab_error

    def test_the_delta_r_uncertainty_does_not_move_the_median(self):
        """ΔR 的**不确定度**只加宽区间，不移动中位数（它是对称的）。"""
        narrow = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, _linear_curve(), delta_r=0.0)
        wide = calibrate_radiocarbon(
            LAB_AGE,
            LAB_ERROR,
            _linear_curve(),
            delta_r=0.0,
            delta_r_sigma=150.0,
        )

        assert wide.median_age == pytest.approx(narrow.median_age, abs=TRUTH_TOLERANCE)


# =============================================================================
# 输入校验
# =============================================================================


class TestValidation:
    """非法输入必须抛 ``ValidationError``。"""

    def test_a_negative_14c_age_is_rejected(self):
        with pytest.raises(ValidationError):
            calibrate_radiocarbon(-10.0, LAB_ERROR, _linear_curve())

    def test_a_non_positive_uncertainty_is_rejected(self):
        with pytest.raises(ValidationError):
            calibrate_radiocarbon(LAB_AGE, 0.0, _linear_curve())
        with pytest.raises(ValidationError):
            calibrate_radiocarbon(LAB_AGE, -5.0, _linear_curve())

    def test_a_non_finite_age_is_rejected(self):
        with pytest.raises(ValidationError):
            calibrate_radiocarbon(np.nan, LAB_ERROR, _linear_curve())

    def test_an_empty_curve_is_rejected(self):
        with pytest.raises(ValidationError):
            calibrate_radiocarbon(LAB_AGE, LAB_ERROR, np.zeros((0, 3)))

    def test_a_single_point_curve_is_rejected(self):
        with pytest.raises(ValidationError):
            calibrate_radiocarbon(LAB_AGE, LAB_ERROR, np.array([[500.0, 637.0, 2.0]]))

    def test_a_curve_with_non_finite_values_is_rejected(self):
        broken = _linear_curve()
        broken[10, 0] = np.nan
        with pytest.raises(ValidationError):
            calibrate_radiocarbon(LAB_AGE, LAB_ERROR, broken)

    def test_a_curve_with_the_wrong_number_of_columns_is_rejected(self):
        with pytest.raises(ValidationError):
            calibrate_radiocarbon(LAB_AGE, LAB_ERROR, np.zeros((10, 2)))
        with pytest.raises(ValidationError):
            calibrate_radiocarbon(LAB_AGE, LAB_ERROR, np.zeros((10, 4)))

    def test_a_curve_with_negative_uncertainties_is_rejected(self):
        broken = _linear_curve()
        broken[3, 2] = -1.0
        with pytest.raises(ValidationError):
            calibrate_radiocarbon(LAB_AGE, LAB_ERROR, broken)

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.2, 1.5])
    def test_confidence_levels_outside_the_open_unit_interval_are_rejected(self, level: float):
        with pytest.raises(ValidationError):
            calibrate_radiocarbon(LAB_AGE, LAB_ERROR, _linear_curve(), confidence=(level,))

    def test_an_empty_confidence_tuple_is_rejected(self):
        with pytest.raises(ValidationError):
            calibrate_radiocarbon(LAB_AGE, LAB_ERROR, _linear_curve(), confidence=())

    def test_a_negative_delta_r_uncertainty_is_rejected(self):
        with pytest.raises(ValidationError):
            calibrate_radiocarbon(LAB_AGE, LAB_ERROR, _linear_curve(), delta_r_sigma=-1.0)

    def test_a_negative_extra_curve_uncertainty_is_rejected(self):
        with pytest.raises(ValidationError):
            calibrate_radiocarbon(LAB_AGE, LAB_ERROR, _linear_curve(), curve_sigma=-1.0)

    def test_a_non_finite_reservoir_age_is_rejected(self):
        with pytest.raises(ValidationError):
            calibrate_radiocarbon(LAB_AGE, LAB_ERROR, _linear_curve(), reservoir_age=np.nan)


# =============================================================================
# 批量校准
# =============================================================================


class TestBatch:
    """批量校准必须与逐点单点校准一致。"""

    def test_the_batch_matches_the_single_calibrations(self):
        ages = [400.0, 500.0, 900.0]
        errors = [15.0, 20.0, 35.0]
        curve = _linear_curve()

        results = calibrate_radiocarbon_batch(ages, errors, curve)

        assert len(results) == len(ages)
        for age, error, result in zip(ages, errors, results, strict=True):
            single = calibrate_radiocarbon(age, error, curve)
            assert np.allclose(result.probability_density, single.probability_density)
            assert result.median_age == single.median_age
            assert result.mode_age == single.mode_age

    def test_the_batch_applies_the_reservoir_correction_to_every_point(self):
        curve = _linear_curve()
        results = calibrate_radiocarbon_batch(
            [400.0, 500.0],
            [15.0, 20.0],
            curve,
            reservoir_age=200.0,
            delta_r=50.0,
        )

        for result in results:
            assert result.corrected_lab_age == pytest.approx(result.lab_age - 250.0)

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValidationError):
            calibrate_radiocarbon_batch([400.0, 500.0], [15.0], _linear_curve())

    def test_a_single_date_batch_is_rejected(self):
        with pytest.raises(ValidationError):
            calibrate_radiocarbon_batch([400.0], [15.0], _linear_curve())


# =============================================================================
# 可复现性
# =============================================================================


class TestReproducibility:
    """同一个种子必须给出完全相同的结果。"""

    def test_two_calls_with_the_same_seed_agree(self):
        curve = _linear_curve()
        first = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, curve, seed=2024)
        second = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, curve, seed=2024)

        assert np.allclose(first.probability_density, second.probability_density)
        assert first.median_age == second.median_age
        assert first.mode_age == second.mode_age
        assert first.seed == second.seed == 2024

    def test_the_batch_is_reproducible(self):
        curve = _linear_curve()
        first = calibrate_radiocarbon_batch([400.0, 500.0], [15.0, 20.0], curve, seed=7)
        second = calibrate_radiocarbon_batch([400.0, 500.0], [15.0, 20.0], curve, seed=7)

        for left, right in zip(first, second, strict=True):
            assert np.allclose(left.probability_density, right.probability_density)


# =============================================================================
# 曲线文件读取
# =============================================================================


class TestCurveFile:
    """``.14c`` 曲线文件的读取。"""

    def test_the_file_is_read_back_point_by_point(self, tmp_path):
        """读回来的数组必须与写入的数组**逐点相等**。

        文件按真实曲线的习惯"由老到新"排列，读取后应当按日历年龄升序。
        """
        calendar = np.array([300.0, 200.0, 100.0, 0.0])
        radiocarbon = calendar - OFFSET
        sigma = np.full(calendar.size, CURVE_SIGMA)

        target = tmp_path / "TestCurve.14c"
        target.write_text(
            _curve_file_text(calendar, radiocarbon, sigma),
            encoding="utf-8",
        )

        read_r14, read_cal, read_sigma = read_calibration_curve(target)

        assert np.allclose(read_cal, np.sort(calendar))
        assert np.allclose(read_r14, np.sort(radiocarbon))
        assert np.allclose(read_sigma, sigma)

    def test_the_curve_can_be_calibrated_right_after_reading(self, tmp_path):
        """读出来的曲线可以直接送进校准函数（两种形式都可以）。"""
        calendar = np.arange(0.0, 1001.0, 5.0)
        radiocarbon = calendar - OFFSET
        sigma = np.full(calendar.size, CURVE_SIGMA)

        target = tmp_path / "TestCurve.14c"
        target.write_text(
            _curve_file_text(calendar, radiocarbon, sigma),
            encoding="utf-8",
        )

        read_r14, read_cal, read_sigma = read_calibration_curve(target)
        from_tuple = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, (read_r14, read_cal, read_sigma))
        from_array = calibrate_radiocarbon(
            LAB_AGE,
            LAB_ERROR,
            np.column_stack([read_r14, read_cal, read_sigma]),
        )

        assert from_tuple.mode_age == pytest.approx(TRUE_CALENDAR_AGE, abs=TRUTH_TOLERANCE)
        assert np.allclose(from_tuple.probability_density, from_array.probability_density)

    def test_a_wrong_header_count_is_rejected(self, tmp_path):
        """表头行数与文件内容不符时必须报错（而不是读出一堆垃圾）。"""
        calendar = np.arange(0.0, 101.0, 5.0)
        radiocarbon = calendar - OFFSET
        sigma = np.full(calendar.size, CURVE_SIGMA)

        target = tmp_path / "TestCurve.14c"
        target.write_text(
            _curve_file_text(calendar, radiocarbon, sigma),
            encoding="utf-8",
        )

        with pytest.raises(ValidationError):
            read_calibration_curve(target, header_lines=500)

    def test_a_non_numeric_row_is_rejected(self, tmp_path):
        target = tmp_path / "Broken.14c"
        target.write_text(
            _curve_file_text(
                np.arange(0.0, 21.0, 5.0),
                np.arange(0.0, 21.0, 5.0),
                np.full(5, CURVE_SIGMA),
            ).replace("0.0,0.0,2.0", "oops,oops,oops"),
            encoding="utf-8",
        )

        with pytest.raises(ValidationError):
            read_calibration_curve(target)

    def test_a_missing_file_is_rejected(self, tmp_path):
        with pytest.raises(ValidationError):
            read_calibration_curve(tmp_path / "NoSuchCurve.14c")

    def test_the_temporary_curve_file_stays_outside_the_repository(self, tmp_path):
        """测试用的曲线文件必须落在仓库之外（``%TEMP%``）。

        ``tmp_path`` 由 pytest 建在系统临时目录下；这条断言把它钉死，免得
        以后有人改成往仓库里写文件。
        """
        repo_root = Path(__file__).resolve().parents[2]
        assert not Path(tmp_path).resolve().is_relative_to(repo_root), (
            "the temporary curve file must not be written inside the repository"
        )


# =============================================================================
# 门面（供参数表单驱动）
# =============================================================================


class TestAnalyzerFacade:
    """门面必须与直接调用给出同一个答案。

    门面的存在理由是**槽契约**：目录里声明的槽必须是表格的一列，而校准曲线
    是三列的数组（¹⁴C 年龄、日历年龄、σ），装不进一列。因此表格给两列
    （¹⁴C 年龄与其 1σ），曲线由使用者给出**文件路径**，由门面读取。
    """

    def test_the_facade_matches_the_direct_calibration(self, tmp_path):
        """门面的结果必须与直接调用 :func:`calibrate_radiocarbon` 一致。

        同一个曲线文件、同一组年龄：后验密度、HPD 区间、中位与众数年龄都
        必须逐点相同。门面只是把"表格的一列 + 曲线文件"翻译成函数调用，
        多一层就多一个能悄悄改掉数字的地方。
        """
        calendar = np.arange(0.0, 1001.0, 5.0)
        radiocarbon = calendar - OFFSET
        sigma = np.full(calendar.size, CURVE_SIGMA)

        target = tmp_path / "TestCurve.14c"
        target.write_text(
            _curve_file_text(calendar, radiocarbon, sigma),
            encoding="utf-8",
        )

        analyzer = RadiocarbonCalibrationAnalyzer()
        through = analyzer.analyze(np.array([LAB_AGE]), np.array([LAB_ERROR]), str(target))
        direct = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, read_calibration_curve(target))

        assert np.allclose(through.probability_density, direct.probability_density)
        assert through.median_age == direct.median_age
        assert through.mode_age == direct.mode_age
        assert through.hpd_bounds(LEVEL_68) == direct.hpd_bounds(LEVEL_68)
        assert through.hpd_bounds(LEVEL_95) == direct.hpd_bounds(LEVEL_95)

    def test_the_facade_uses_the_first_row_of_the_columns(self, tmp_path):
        """表格里有多行时，门面校准的是**第一行**。"""
        calendar = np.arange(0.0, 1001.0, 5.0)
        radiocarbon = calendar - OFFSET
        sigma = np.full(calendar.size, CURVE_SIGMA)

        target = tmp_path / "TestCurve.14c"
        target.write_text(
            _curve_file_text(calendar, radiocarbon, sigma),
            encoding="utf-8",
        )

        analyzer = RadiocarbonCalibrationAnalyzer()
        through = analyzer.analyze(
            np.array([LAB_AGE, 900.0]),
            np.array([LAB_ERROR, 35.0]),
            str(target),
        )
        direct = calibrate_radiocarbon(LAB_AGE, LAB_ERROR, read_calibration_curve(target))

        assert through.mode_age == direct.mode_age

    def test_a_missing_curve_file_is_rejected(self, tmp_path):
        """曲线文件不存在时必须抛 ``ValidationError``（而不是别的异常）。"""
        analyzer = RadiocarbonCalibrationAnalyzer()

        with pytest.raises(ValidationError):
            analyzer.analyze(
                np.array([LAB_AGE]),
                np.array([LAB_ERROR]),
                str(tmp_path / "NoSuchCurve.14c"),
            )

    def test_a_malformed_curve_file_is_rejected(self, tmp_path):
        """曲线文件格式不对时必须抛 ``ValidationError``。"""
        target = tmp_path / "Broken.14c"
        target.write_text(
            _curve_file_text(
                np.arange(0.0, 21.0, 5.0),
                np.arange(0.0, 21.0, 5.0),
                np.full(5, CURVE_SIGMA),
            ).replace("0.0,0.0,2.0", "oops,oops,oops"),
            encoding="utf-8",
        )

        analyzer = RadiocarbonCalibrationAnalyzer()
        with pytest.raises(ValidationError):
            analyzer.analyze(np.array([LAB_AGE]), np.array([LAB_ERROR]), str(target))

    def test_an_empty_curve_path_is_rejected(self):
        """曲线路径为空时必须报错（表单里的文本框可以留空）。"""
        analyzer = RadiocarbonCalibrationAnalyzer()

        with pytest.raises(ValidationError):
            analyzer.analyze(np.array([LAB_AGE]), np.array([LAB_ERROR]), "")

    def test_columns_of_different_lengths_are_rejected(self, tmp_path):
        """两列长度不一致时必须报错（年龄列与误差列配错了）。"""
        analyzer = RadiocarbonCalibrationAnalyzer()

        with pytest.raises(ValidationError):
            analyzer.analyze(
                np.array([LAB_AGE, 900.0]),
                np.array([LAB_ERROR]),
                str(tmp_path / "TestCurve.14c"),
            )
