# =============================================================================
# FILE: tests/golden/test_astronomy.py
# =============================================================================
"""
``stratigraphy.astronomy`` 的测试: 天文解加载、插值、日照量与米兰科维奇周期。

覆盖:

1. 内置 La04 天文解可加载, 各数组等长、有限、无 NaN, 偏心率与倾角落在
   合理区间;
2. :meth:`AstronomicalSolution.at` 的线性插值: 长度、端点值;
3. :func:`milankovitch_periods` 的周期落在文献量级 (岁差 ~21 kyr、
   倾角 ~41 kyr、偏心率 ~100 kyr, ±30% 宽容度);
4. :func:`daily_mean_insolation` 的纬度依赖 (赤道最高、极点最低) 与
   极昼/极夜的方向;
5. :func:`astronomical_target_curve` 的周期由 FFT 主峰验证;
6. 现代 (0 Ma) 的锚点事实: 偏心率 ≈ 0.0167、倾角 ≈ 23.44°、
   近日点黄经 ≈ 102.9°。

注: 任务书给出的经验区间 (e ∈ [0.01, 0.06]、epsilon ∈ [22°, 24°]、
90° 年平均日照量 < 150 W/m²) 与 Berger (1978) 定义下的实际值略有出入,
本文件按物理正确的量断言并在相应位置注明偏差:

* La04 覆盖 249 Myr, e 的全域范围是 0.0001–0.0677, 倾角是 21.83–24.48°;
  区间型断言因此改用 5–95 百分位 + 中位数。
* 极点的年平均日日照量由 Berger (1978) 公式给出 ≈ 169 W/m²
  (= S0·sin(epsilon)/pi), 并非 < 150 W/m²: 极地半年全日照, 半年全黑夜。
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"astronomical_solution", "astronomical_target_curve", "insolation", "milankovitch_periods"}

import numpy as np
import pytest

from stratigraphy.astronomy import (
    AstronomicalSolution,
    astronomical_target_curve,
    daily_mean_insolation,
    insolation,
    load_astronomical_solution,
    milankovitch_periods,
)
from utils.exceptions import ValidationError

#: 现代 (0 Ma) 的锚点事实。
MODERN_ECCENTRICITY = 0.0167
MODERN_OBLIQUITY = 23.44
MODERN_LONGITUDE_PERIHELION = 102.9

#: 岁差/倾角/偏心率的经典周期 (kyr), 用于 ±30% 的量级检查。
CLASSIC_PRECESSION_KYR = 21.0
CLASSIC_OBLIQUITY_KYR = 41.0
CLASSIC_ECCENTRICITY_KYR = 100.0

#: 相对宽容度。
PERIOD_TOLERANCE = 0.30


@pytest.fixture(scope="module")
def solution() -> AstronomicalSolution:
    """模块级加载一次内置天文解 (249001 个采样点, 读取约 7 MB)。"""
    return load_astronomical_solution("la04")


# =============================================================================
# 1. 数据加载与自洽性
# =============================================================================


def test_solution_loads_and_arrays_are_consistent(solution: AstronomicalSolution) -> None:
    """La04 可加载; 五个数组等长、全为有限值、无 NaN。"""
    arrays = (
        solution.times_ma,
        solution.eccentricity,
        solution.obliquity,
        solution.longitude_perihelion,
        solution.longitude_ascending_node,
    )
    lengths = {len(array) for array in arrays}
    assert len(lengths) == 1, "all astronomical solution arrays must share one length"
    assert lengths.pop() > 1000

    for array in arrays:
        assert np.all(np.isfinite(array)), "astronomical solution must not contain NaN/Inf"
        assert not np.any(np.isnan(array))


def test_solution_time_axis_is_ordered_oldest_first(solution: AstronomicalSolution) -> None:
    """年龄轴由老到新 (最老在前), 与本仓库其它地层表一致。"""
    assert np.all(np.diff(solution.times_ma) < 0.0)
    assert solution.times_ma[0] > solution.times_ma[-1]
    assert solution.times_ma[-1] == pytest.approx(0.0, abs=1e-9)


def test_solution_eccentricity_in_physical_range(solution: AstronomicalSolution) -> None:
    """偏心率为正, 且绝大部分落在 0.01–0.06 的物理区间。"""
    ecc = solution.eccentricity
    assert np.all(ecc > 0.0), "eccentricity must be strictly positive"
    assert np.all(ecc < 0.2), "eccentricity must stay far below unity"

    median = float(np.median(ecc))
    assert 0.015 <= median <= 0.040, f"median eccentricity out of range: {median}"

    low, high = np.percentile(ecc, [5, 95])
    assert 0.005 <= low <= 0.060
    assert 0.010 <= high <= 0.065


def test_solution_obliquity_in_physical_range(solution: AstronomicalSolution) -> None:
    """黄赤交角落在 22–24° 附近 (全域 21.8–24.5°, 主体在 22–24°)。"""
    obl = solution.obliquity
    assert np.all((obl > 20.0) & (obl < 26.0))

    median = float(np.median(obl))
    assert 22.5 <= median <= 23.5, f"median obliquity out of range: {median}"

    low, high = np.percentile(obl, [5, 95])
    assert 22.0 <= low <= 22.6
    assert 23.4 <= high <= 24.2


def test_solution_longitudes_are_degrees(solution: AstronomicalSolution) -> None:
    """黄经以度给出, 且落在 [0, 360)。"""
    for array in (solution.longitude_perihelion, solution.longitude_ascending_node):
        assert np.all((array >= 0.0) & (array < 360.0))


# =============================================================================
# 2. 插值
# =============================================================================


def test_at_interpolates_to_requested_times(solution: AstronomicalSolution) -> None:
    """at() 返回与请求年龄等长的解, 且端点值与原数组一致。"""
    new_times = np.array([0.0, 0.5, 12.25, 100.0, 249.0])
    sampled = solution.at(new_times)

    assert len(sampled.times_ma) == len(new_times)
    for array in (
        sampled.eccentricity,
        sampled.obliquity,
        sampled.longitude_perihelion,
        sampled.longitude_ascending_node,
    ):
        assert len(array) == len(new_times)
        assert np.all(np.isfinite(array))

    # 端点: 0 Ma 与 249 Ma 处的值应与原数组几乎一致
    assert sampled.eccentricity[0] == pytest.approx(solution.eccentricity[-1], rel=1e-9)
    assert sampled.obliquity[0] == pytest.approx(solution.obliquity[-1], rel=1e-9)
    assert sampled.longitude_perihelion[0] == pytest.approx(solution.longitude_perihelion[-1], abs=1e-6)
    assert sampled.eccentricity[-1] == pytest.approx(solution.eccentricity[0], rel=1e-9)
    assert sampled.obliquity[-1] == pytest.approx(solution.obliquity[0], rel=1e-9)


def test_at_interpolates_linearly_between_samples(solution: AstronomicalSolution) -> None:
    """相邻采样点之间的插值与手工线性插值一致。"""
    idx = 1000
    t0 = float(solution.times_ma[idx])
    t1 = float(solution.times_ma[idx + 1])
    midpoint = 0.5 * (t0 + t1)

    sampled = solution.at(np.array([midpoint]))
    weight = (midpoint - t0) / (t1 - t0)
    expected = (1.0 - weight) * solution.eccentricity[idx] + weight * solution.eccentricity[idx + 1]
    assert float(sampled.eccentricity[0]) == pytest.approx(float(expected), rel=1e-9)


def test_at_rejects_ages_outside_the_solution(solution: AstronomicalSolution) -> None:
    """超出解的时间范围的年龄应抛出 ValidationError, 而不是静默外推。"""
    with pytest.raises(ValidationError):
        solution.at(np.array([solution.times_ma.max() + 1.0]))
    with pytest.raises(ValidationError):
        solution.at(np.array([-0.5]))


# =============================================================================
# 3. 米兰科维奇周期
# =============================================================================


def test_milankovitch_periods_match_literature() -> None:
    """周期表中的岁差/倾角/偏心率落在文献量级 (±30%)。"""
    periods = milankovitch_periods()
    assert isinstance(periods, dict)
    assert {"precession_21", "obliquity_41", "eccentricity_100"}.issubset(periods)

    for name, classic in (
        ("precession_21", CLASSIC_PRECESSION_KYR),
        ("obliquity_41", CLASSIC_OBLIQUITY_KYR),
        ("eccentricity_100", CLASSIC_ECCENTRICITY_KYR),
    ):
        value = periods[name]
        assert classic * (1.0 - PERIOD_TOLERANCE) <= value <= classic * (1.0 + PERIOD_TOLERANCE)

    # 岁差族的三个分支都在 19–23 kyr 附近
    for name in ("precession_19", "precession_21", "precession_23"):
        assert 17.0 <= periods[name] <= 25.0


def test_period_kyr_lookup_matches_table(solution: AstronomicalSolution) -> None:
    """AstronomicalSolution.period_kyr 与周期表一致, 未知谐波报错。"""
    periods = milankovitch_periods()
    assert solution.period_kyr("obliquity_41") == periods["obliquity_41"]
    assert solution.period_kyr("precession_21") == pytest.approx(21.0)

    with pytest.raises(ValidationError):
        solution.period_kyr("not_a_harmonic")


# =============================================================================
# 4. 日照量
# =============================================================================


def test_daily_mean_insolation_latitude_dependence(solution: AstronomicalSolution) -> None:
    """年平均日日照量随纬度升高而降低; 赤道最高、极点最低。"""
    times = np.array([0.0, 10.0, 50.0])
    equator = daily_mean_insolation(0.0, solution, times)
    mid = daily_mean_insolation(60.0, solution, times)
    pole = daily_mean_insolation(90.0, solution, times)

    assert len(equator) == len(times)
    assert np.all(np.isfinite(equator))

    # 赤道 300–450 W/m^2, 60° 100–250 W/m^2
    assert np.all((equator >= 300.0) & (equator <= 450.0)), equator
    assert np.all((mid >= 100.0) & (mid <= 250.0)), mid

    # 90° 明显更低。Berger (1978) 下极点的年平均日日照量为
    # S0 * sin(epsilon) / pi ≈ 169 W/m^2 (半年全日照、半年全黑夜),
    # 并不低于 150 W/m^2。
    assert np.all(pole < 200.0), pole
    assert np.all(pole < 0.9 * mid), (pole, mid)


def test_daily_mean_insolation_decreases_monotonically_with_latitude(
    solution: AstronomicalSolution,
) -> None:
    """从赤道到极点, 年平均日日照量单调下降。"""
    times = np.array([0.0, 25.0])
    values = [float(daily_mean_insolation(lat, solution, times)[0]) for lat in (0, 15, 30, 45, 60, 75, 90)]
    assert values == sorted(values, reverse=True), values
    assert values[0] > values[-1]


def test_insolation_polar_day_and_night(solution: AstronomicalSolution) -> None:
    """高纬度冬至几乎无日照, 夏至日照极多 (极昼)。"""
    times = np.array([0.0])
    summer = float(insolation(75.0, 172, solution, times)[0])
    winter = float(insolation(75.0, 355, solution, times)[0])

    assert summer > 450.0, summer
    assert winter == pytest.approx(0.0, abs=1e-6)

    # 60° 冬至仍有日照但远少于夏至
    assert float(insolation(60.0, 355, solution, times)[0]) < 60.0
    assert float(insolation(60.0, 172, solution, times)[0]) > 400.0


def test_insolation_spectrum_carries_orbital_bands(solution: AstronomicalSolution) -> None:
    """65°N 夏至日照量的频谱应含岁差带、倾角带与偏心率带的能量。

    判据用 "带内能量 / 相邻带能量" 的信噪比, 以免被该解中异常强烈的
    岁差调制 (Acycle 的 La04 把近日点-春分点角距编码成 ±90° 的三角波)
    所淹没。
    """
    times = np.arange(0.0, 200.0, 0.001)  # 1 kyr 步长, 覆盖 0–200 Ma
    values = insolation(65.0, 172, solution, times)

    power = np.abs(np.fft.rfft(values - values.mean())) ** 2
    freqs = np.fft.rfftfreq(len(values), d=1.0)  # cycles/kyr
    total = float(power[1:].sum())

    def band_power(low_kyr: float, high_kyr: float) -> float:
        mask = (freqs >= 1.0 / high_kyr) & (freqs <= 1.0 / low_kyr)
        return float(power[mask].sum()) / total

    # 岁差带 (19–23 kyr) 是该解中最强的信号
    assert band_power(17.0, 25.0) > 0.5

    # 倾角带 (41 kyr) 明显强于其相邻的 45–55 kyr
    assert band_power(38.0, 45.0) > 5.0 * band_power(45.0, 55.0)

    # 偏心率带 (~100 kyr) 明显强于其相邻的 80–90 kyr
    assert band_power(90.0, 110.0) > 5.0 * band_power(80.0, 90.0)


# =============================================================================
# 5. 理论目标曲线
# =============================================================================


@pytest.mark.parametrize("period", [19.0, 21.0, 23.0, 41.0, 100.0])
def test_target_curve_fft_peak_matches_period(period: float) -> None:
    """目标曲线的 FFT 主峰落在给定周期上 (1 kyr 采样, 峰位误差 < 0.5%)。"""
    times = np.arange(0.0, 5.0, 0.001)  # 1 kyr 步长
    curve = astronomical_target_curve(period, times)

    assert len(curve) == len(times)
    assert np.all(np.isfinite(curve))

    spectrum = np.abs(np.fft.rfft(curve - curve.mean()))
    freqs = np.fft.rfftfreq(len(curve), d=1.0)
    peak_period = 1.0 / freqs[int(np.argmax(spectrum[1:])) + 1]
    assert peak_period == pytest.approx(period, rel=0.005)


def test_target_curve_amplitude_and_shape() -> None:
    """振幅参数生效, 曲线为余弦 (t = 0 处取最大值)。"""
    times = np.linspace(0.0, 1.0, 101)
    curve = astronomical_target_curve(100.0, times, amplitude=3.0)
    assert float(np.max(np.abs(curve))) == pytest.approx(3.0, rel=1e-6)
    assert float(curve[0]) == pytest.approx(3.0, rel=1e-9)


def test_target_curve_rejects_invalid_period() -> None:
    """非正周期应报错。"""
    with pytest.raises(ValidationError):
        astronomical_target_curve(0.0, np.array([0.0, 1.0]))
    with pytest.raises(ValidationError):
        astronomical_target_curve(-41.0, np.array([0.0, 1.0]))


# =============================================================================
# 6. 现代锚点事实
# =============================================================================


def test_modern_orbital_anchor_values(solution: AstronomicalSolution) -> None:
    """0 Ma 处: e ≈ 0.0167、epsilon ≈ 23.44°、varpi ≈ 102.9°。"""
    present = solution.at(np.array([0.0]))

    assert float(present.eccentricity[0]) == pytest.approx(MODERN_ECCENTRICITY, abs=0.0005)
    assert float(present.obliquity[0]) == pytest.approx(MODERN_OBLIQUITY, abs=0.05)
    assert float(present.longitude_perihelion[0]) == pytest.approx(MODERN_LONGITUDE_PERIHELION, abs=0.5)


def test_modern_anchor_consistent_with_january_perihelion(solution: AstronomicalSolution) -> None:
    """近日点黄经应对应 1 月初的近日点 (太阳黄经 ≈ 283°)。"""
    perihelion_longitude = float(solution.at(np.array([0.0])).longitude_perihelion[0])
    sun_longitude_at_perihelion = (perihelion_longitude + 180.0) % 360.0
    assert sun_longitude_at_perihelion == pytest.approx(282.9, abs=1.5)


# =============================================================================
# 错误处理
# =============================================================================


def test_unknown_solution_name_raises() -> None:
    """未知的天文解名称应抛出 ValidationError。"""
    with pytest.raises(ValidationError):
        load_astronomical_solution("no_such_solution")


def test_invalid_insolation_arguments_raise(solution: AstronomicalSolution) -> None:
    """越界的纬度/日序应抛出 ValidationError。"""
    with pytest.raises(ValidationError):
        insolation(95.0, 172, solution, np.array([0.0]))
    with pytest.raises(ValidationError):
        insolation(0.0, 0, solution, np.array([0.0]))
    with pytest.raises(ValidationError):
        insolation(0.0, 400, solution, np.array([0.0]))
    with pytest.raises(ValidationError):
        daily_mean_insolation(-100.0, solution, np.array([0.0]))
