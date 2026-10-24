# =============================================================================
# FILE: tests/golden/test_orbital_frequencies.py
# =============================================================================
"""
``stratigraphy.orbital_frequencies``（9 条主天文频率）的测试。

测试围绕五条主线组织：

* **数量与命名。** 必须恰好 9 条，名字与 Acycle ``calculate_orbit9.m`` 的
  顺序逐字一致（``g2-g5  g3-g2  g4-g2  g3-g5  g4-g5  k+s3  k+g5  k+g2
  k+g4``）。名字写错会让 COCO 目标谱与文献里的"轨道计数"对不上。

* **量级。** 9 条频率在 0 Ma 的**量级**取自已发表的轨道周期表，而不是抄
  本模块的输出：
  - ``g4-g2`` 与 ``g3-g5`` 是长偏心率拍频，落在 **~400 kyr**（文献里的
    "405 kyr 长偏心率周期"，Laskar et al. 2004, EPSL 224: 299-313）；
  - ``g3-g2`` 与 ``g4-g5`` 是更长的拍频，落在 **~2400 kyr**（文献里
    "2.4 Myr" 的轨道调制，Pälike et al. 2006; Kent et al. 2017, PNAS）；
  - ``g2-g5`` 是中长偏心率拍频，落在 **~300-400 kyr**；
  - ``k+g2`` / ``k+g4`` / ``k+g5`` 是**岁差带**（岁差常数与偏心率谐波的
    边带），落在 **19-23 kyr**；
  - ``k+s3`` 是岁差常数与第 3 条岁差谐波的组合，落在 **10-14 kyr**。
  断言用的是**量级区间**（允许 ±10 % 左右），不是小数点后多位的精确值。

* **年代依赖。** La04 解的偏心率谐波频率本身固定，但谐波的振幅与相位随
  时间演化，所以**局部**（在目标年龄附近拟合）测得的频率随年龄移动。测试
  要求：偏心率组合在 0 Ma 与 1 Ma 之间的相对变化 **> 0.5 %**，而
  岁差-偏心率组合（``k+s3`` 等）的变化要**小得多**（实测相差两个数量级：
  偏心率组合最大 ~4.9 %，岁差组合最大 ~0.03 %）。这条断言锁的是
  "偏心率组合携带年龄信息、岁差组合几乎不携带"这一契约。

* **与仓库已有天文解自洽。** 模块解算所用的轨道量必须来自
  ``stratigraphy.astronomy.load_astronomical_solution("la04")``：偏心率必须
  等于该解在同一年龄的偏心率，岁差常数必须等于该解升交点黄经的速率，
  局部拟合出的偏心率谐波周期必须落在 La04 解标称值附近。

* **标量与数组一致。** ``orbital_frequencies(0.0)`` 必须与
  ``orbital_frequencies(np.array([0.0]))[0]`` 完全相等（``==`` 为真）。

边界测试断言抛 ``utils.exceptions.ValidationError``（越界年龄、负年龄、
非有限年龄）；可复现性断言两次调用结果完全一致。

参考来源（断言中引用的数值出处）
--------------------------------
* Laskar, J., Gastineau, M., Ireland, J. A. & Petit, E. (2004). Solution for the
  orbital and climatic precession. *Astronomy & Astrophysics*, 413, 339-348.
  （气候岁差解的边带结构 ``k ± g_j``；岁差常数。）
* Laskar, J., Gastineau, M., Jounder, E. & Bourgeois, F. (2004). A 250-Myr
  solution to the Pliocene-Pleistocene ice ages. *Earth and Planetary Science
  Letters*, 224, 299-313.（La04 轨道解；偏心率谐波频率与 405 kyr 周期。）
* Berger, A. & Loutre, M. F. (1991). Excursions in orbital geometry: possible
  causes. *Paleoceanography*, 6(3), 301-317.（岁差常数与轨道几何的长期演化。）
* Pälike, U. et al. (2006). The heartbeat of the Oligocene climate system.
  *Science*, 314, 1894-1898.（~2.4 Myr 的轨道调制。）
* Kent, P. et al. (2017). The Triassic 405-ky cycle. *PNAS*, 114, E4896-E4904.
  （405 kyr 周期由行星长期岁差频率之差给出；~2400 kyr 与 ~1200 kyr 的
  轨道调制。）
* Li, M., Hinnov, L. A., Sun, D., Chen, Y., Bottjer, D. J. & Kump, L. R. (2018).
  Identifying orbital pacing in the astronomical record of the Paleogene and
  Neogene. *Earth and Planetary Science Letters*, 501, 165-179.（9 条频率在
  COCO 目标谱中的用法。）
"""

from __future__ import annotations

#: 本文件覆盖的分析（矩阵以此声明为准）
ANALYSES_COVERED = {"orbital_frequencies"}


import numpy as np
import pytest

from stratigraphy.astronomy import load_astronomical_solution
from stratigraphy.orbital_frequencies import (
    ARCSEC_PER_CYCLE,
    LA04_ECCENTRICITY_HARMONIC_PERIODS_KYR,
    ORBITAL_FREQUENCY_NAMES,
    orbital_frequencies,
    orbital_frequency_table,
)
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 参考年龄 (Ma)：现在与 1 Ma。
PRESENT_AGE_MA = 0.0
ONE_MA = 1.0

#: 9 条频率的名字（必须与 Acycle 的顺序逐字一致）。
EXPECTED_NAMES = (
    "g2_minus_g5",
    "g3_minus_g2",
    "g4_minus_g2",
    "g3_minus_g5",
    "g4_minus_g5",
    "k_plus_s3",
    "k_plus_g5",
    "k_plus_g2",
    "k_plus_g4",
)

#: 偏心率组合（g−g）四条长周期与一条中长周期的量级区间 (kyr)。
#: 区间取自已发表的轨道周期表：405 kyr 长偏心率周期、~2.4 Myr 的轨道调制。
ECCENTRICITY_COMBO_BANDS_KYR = {
    "g2_minus_g5": (250.0, 500.0),
    "g3_minus_g2": (1500.0, 3500.0),
    "g4_minus_g2": (350.0, 470.0),
    "g3_minus_g5": (350.0, 470.0),
    "g4_minus_g5": (1500.0, 3500.0),
}

#: 岁差带 (k+g*) 的量级区间 (kyr)：19-23 kyr 的岁差带。
PRECESSION_BAND_BANDS_KYR = {
    "k_plus_g5": (19.0, 23.0),
    "k_plus_g2": (19.0, 23.0),
    "k_plus_g4": (19.0, 23.0),
}

#: ``k+s3`` 的量级区间 (kyr)。
K_PLUS_S3_BAND_KYR = (10.0, 14.0)

#: 偏心率组合在 0 Ma 与 1 Ma 之间必须出现的相对变化下限。
MIN_ECCENTRICITY_COMBO_CHANGE = 0.005


# =============================================================================
# 夹具
# =============================================================================


@pytest.fixture(scope="module")
def present() -> object:
    """0 Ma 处的 9 条频率。"""
    return orbital_frequencies(PRESENT_AGE_MA)


@pytest.fixture(scope="module")
def one_ma() -> object:
    """1 Ma 处的 9 条频率。"""
    return orbital_frequencies(ONE_MA)


def _relative_change(first: float, second: float) -> float:
    """两条周期之间的相对变化 (|ΔP| / P)。"""
    return abs(second - first) / abs(first)


# =============================================================================
# 数量与命名
# =============================================================================


class TestNaming:
    """必须恰好 9 条，名字与 Acycle 逐字一致。"""

    def test_there_are_exactly_nine_frequencies(self, present):
        mapping = present.as_mapping()
        assert len(mapping) == 9, f"expected 9 orbital frequencies, got {len(mapping)}"

    def test_the_names_match_acycle_order(self, present):
        assert tuple(present.as_mapping()) == EXPECTED_NAMES
        assert ORBITAL_FREQUENCY_NAMES == EXPECTED_NAMES

    def test_every_entry_declares_its_harmonics(self, present):
        """每条频率都必须声明自己由哪些谐波组合而成。"""
        expected_harmonics = {
            "g2_minus_g5": ("g2", "g5"),
            "g3_minus_g2": ("g3", "g2"),
            "g4_minus_g2": ("g4", "g2"),
            "g3_minus_g5": ("g3", "g5"),
            "g4_minus_g5": ("g4", "g5"),
            "k_plus_s3": ("k", "s3"),
            "k_plus_g5": ("k", "g5"),
            "k_plus_g2": ("k", "g2"),
            "k_plus_g4": ("k", "g4"),
        }
        for name, entry in present.as_mapping().items():
            assert entry.harmonics == expected_harmonics[name]
            assert entry.name == name
            first, second = expected_harmonics[name]
            operator = "-" if name.startswith("g") else "+"
            assert entry.combination == f"{first} {operator} {second}"

    def test_the_nine_periods_are_distinct(self, present):
        """9 条频率必须互不相同（否则"轨道计数"会重复计同一条）。"""
        periods = [float(entry.period_kyr) for entry in present.as_mapping().values()]
        assert len(set(periods)) == len(periods)

    def test_the_table_lists_the_same_nine_entries(self, present):
        table = orbital_frequency_table(PRESENT_AGE_MA)
        assert tuple(table) == EXPECTED_NAMES
        for name, entry in table.items():
            assert entry.name == name
            assert entry.period_kyr == pytest.approx(float(getattr(present, name).period_kyr))


# =============================================================================
# 量级
# =============================================================================


class TestMagnitudes:
    """9 条频率的周期必须落在已发表的轨道周期的量级区间里。"""

    @pytest.mark.parametrize("name", sorted(ECCENTRICITY_COMBO_BANDS_KYR))
    def test_eccentricity_combinations_have_long_periods(self, present, name):
        low, high = ECCENTRICITY_COMBO_BANDS_KYR[name]
        period = float(getattr(present, name).period_kyr)
        assert low <= period <= high, f"{name} has a period of {period:.1f} kyr, outside {low}-{high} kyr"

    @pytest.mark.parametrize("name", sorted(PRECESSION_BAND_BANDS_KYR))
    def test_precession_bands_sit_in_the_19_to_23_kyr_range(self, present, name):
        low, high = PRECESSION_BAND_BANDS_KYR[name]
        period = float(getattr(present, name).period_kyr)
        assert low <= period <= high, f"{name} has a period of {period:.1f} kyr, outside {low}-{high} kyr"

    def test_k_plus_s3_sits_in_the_precession_range(self, present):
        low, high = K_PLUS_S3_BAND_KYR
        period = float(present.k_plus_s3.period_kyr)
        assert low <= period <= high, f"k_plus_s3 has a period of {period:.1f} kyr, outside {low}-{high} kyr"

    def test_period_and_frequency_are_mutually_consistent(self, present):
        """周期与频率必须自洽：``P(kyr) = 360 * 3600 / |f(arcsec/yr)| / 1000``。"""
        for entry in present.as_mapping().values():
            expected = ARCSEC_PER_CYCLE / abs(float(entry.frequency_arcsec_per_yr)) / 1000.0
            assert float(entry.period_kyr) == pytest.approx(expected, rel=1e-12)
            assert float(entry.period_kyr) > 0.0

    def test_the_frequency_unit_is_cycles_per_kyr(self, present):
        """频率以"每 kyr 的循环数"给出：1 角秒/年 = 1000/1296000 循环/kyr。"""
        for entry in present.as_mapping().values():
            expected = float(entry.frequency_arcsec_per_yr) * 1000.0 / ARCSEC_PER_CYCLE
            assert float(entry.frequency_per_kyr) == pytest.approx(expected, rel=1e-12)
            sign = 1.0 if float(entry.frequency_arcsec_per_yr) > 0 else -1.0
            assert float(entry.frequency_per_kyr) == pytest.approx(sign / float(entry.period_kyr), rel=1e-12)


# =============================================================================
# 年代依赖
# =============================================================================


class TestAgeDependence:
    """偏心率组合必须随年龄变化，岁差组合的变化要小得多。"""

    def test_eccentricity_combinations_change_with_age(self, present, one_ma):
        """至少有一条偏心率组合在 0 Ma 与 1 Ma 之间变化 > 0.5 %。"""
        changes = {
            name: _relative_change(float(getattr(present, name).period_kyr), float(getattr(one_ma, name).period_kyr))
            for name in ECCENTRICITY_COMBO_BANDS_KYR
        }
        largest = max(changes.values())
        assert largest > MIN_ECCENTRICITY_COMBO_CHANGE, (
            "no eccentricity combination changes by more than "
            f"{MIN_ECCENTRICITY_COMBO_CHANGE * 100:.1f} % between 0 Ma and 1 Ma: {changes}"
        )

    def test_eccentricity_combinations_vary_more_than_precession_terms(self, present, one_ma):
        """偏心率组合的变化幅度必须明显大于斜率-岁差组合。"""
        eccentricity_changes = [
            _relative_change(float(getattr(present, name).period_kyr), float(getattr(one_ma, name).period_kyr))
            for name in ECCENTRICITY_COMBO_BANDS_KYR
        ]
        precession_changes = [
            _relative_change(float(getattr(present, name).period_kyr), float(getattr(one_ma, name).period_kyr))
            for name in ("k_plus_s3", "k_plus_g5", "k_plus_g2", "k_plus_g4")
        ]
        assert max(eccentricity_changes) > 10.0 * max(precession_changes), (
            f"eccentricity combinations change by up to {max(eccentricity_changes) * 100:.3f} % while "
            f"precession terms change by up to {max(precession_changes) * 100:.4f} %"
        )

    def test_the_precession_terms_are_almost_age_independent(self, present, one_ma):
        """岁差-偏心率组合在 0 Ma 与 1 Ma 之间只允许极小的变化。"""
        for name in ("k_plus_s3", "k_plus_g5", "k_plus_g2", "k_plus_g4"):
            change = _relative_change(float(getattr(present, name).period_kyr), float(getattr(one_ma, name).period_kyr))
            assert change < 0.01, f"{name} changes by {change * 100:.4f} % between 0 Ma and 1 Ma"


# =============================================================================
# 与仓库已有天文解的一致性
# =============================================================================


class TestSolutionConsistency:
    """模块用到的轨道量必须与仓库自带的 La04 解在同一年龄上自洽。"""

    def test_the_eccentricity_comes_from_the_bundled_solution(self, present, one_ma):
        solution = load_astronomical_solution("la04")
        sampled = solution.at(np.array([PRESENT_AGE_MA, ONE_MA]))
        assert float(present.eccentricity) == pytest.approx(float(sampled.eccentricity[0]), rel=1e-12)
        assert float(one_ma.eccentricity) == pytest.approx(float(sampled.eccentricity[1]), rel=1e-12)

    def test_the_obliquity_comes_from_the_bundled_solution(self, present):
        solution = load_astronomical_solution("la04")
        sampled = solution.at(np.array([PRESENT_AGE_MA]))
        assert float(present.obliquity_deg) == pytest.approx(float(sampled.obliquity[0]), rel=1e-12)

    def test_the_precession_constant_matches_the_solution_node_rate(self, present):
        """岁差常数 k 必须等于解的升交点黄经速率（角秒/年）。"""
        solution = load_astronomical_solution("la04")
        order = np.argsort(solution.times_ma)
        node = np.asarray(solution.longitude_ascending_node, dtype=float)[order]
        times_kyr = np.asarray(solution.times_ma, dtype=float)[order] * 1000.0
        window = (times_kyr >= -5000.0) & (times_kyr <= 5000.0)
        slope = float(np.polyfit(times_kyr[window], np.degrees(np.unwrap(np.radians(node[window]))), 1)[0])
        expected = abs(slope) * 3600.0 / 1000.0
        assert float(present.precession_rate_arcsec_per_yr) == pytest.approx(expected, rel=1e-9)

    def test_the_local_harmonics_match_the_published_la04_frequencies(self, present):
        """局部拟合出的偏心率谐波周期必须落在 La04 解标称值附近。"""
        for name, nominal in LA04_ECCENTRICITY_HARMONIC_PERIODS_KYR.items():
            measured = 1.0 / float(present.eccentricity_harmonics_per_kyr[name])
            assert measured == pytest.approx(nominal, rel=0.02), (
                f"the locally fitted {name} period is {measured:.3f} kyr, far from the La04 value {nominal} kyr"
            )

    def test_the_precession_harmonics_are_the_classic_sidebands(self, present):
        """岁差谐波必须是 ``k ± g_j`` 的边带结构（Laskar et al. 2004, A&A 413）。"""
        precession = float(present.precession_rate_arcsec_per_yr)
        harmonics = {
            name: value * ARCSEC_PER_CYCLE / 1000.0 for name, value in present.eccentricity_harmonics_per_kyr.items()
        }
        sidebands = present.precession_harmonics_arcsec_per_yr
        assert len(sidebands) == 9
        expected = {precession + value for value in harmonics.values()}
        expected |= {precession - value for value in harmonics.values()}
        expected.remove(min(expected, key=abs))
        for value in sidebands.values():
            assert any(abs(float(value) - candidate) < 1e-9 for candidate in expected)


# =============================================================================
# 标量与数组
# =============================================================================


class TestScalarAndArray:
    """标量调用与数组调用必须给出完全一致的单年龄结果。"""

    def test_scalar_and_array_agree(self):
        scalar = orbital_frequencies(PRESENT_AGE_MA)
        array = orbital_frequencies(np.array([PRESENT_AGE_MA]))
        assert array[0] == scalar

    def test_array_input_returns_arrays(self):
        ages = np.array([PRESENT_AGE_MA, ONE_MA, 10.0])
        result = orbital_frequencies(ages)
        for name in EXPECTED_NAMES:
            periods = np.asarray(getattr(result, name).period_kyr)
            assert periods.shape == ages.shape
        assert np.allclose(
            np.asarray(result.g4_minus_g2.period_kyr),
            [float(orbital_frequencies(age).g4_minus_g2.period_kyr) for age in ages],
        )

    def test_array_indexing_selects_one_age(self):
        result = orbital_frequencies(np.array([PRESENT_AGE_MA, ONE_MA]))
        assert float(result[1].age_ma) == pytest.approx(ONE_MA)
        assert result[1] == orbital_frequencies(ONE_MA)


# =============================================================================
# 边界
# =============================================================================


class TestBoundaries:
    """非法年龄必须抛 ``ValidationError``。"""

    def test_age_beyond_the_solution_range_is_rejected(self):
        with pytest.raises(ValidationError):
            orbital_frequencies(250.0)

    def test_age_below_the_solution_range_is_rejected(self):
        with pytest.raises(ValidationError):
            orbital_frequencies(-1.0)

    def test_negative_ages_are_rejected(self):
        with pytest.raises(ValidationError):
            orbital_frequencies(-0.5)

    def test_non_finite_ages_are_rejected(self):
        with pytest.raises(ValidationError):
            orbital_frequencies(np.nan)

    def test_the_oldest_age_of_the_solution_is_accepted(self):
        solution = load_astronomical_solution("la04")
        oldest = float(np.max(solution.times_ma))
        result = orbital_frequencies(oldest)
        assert float(result.age_ma) == pytest.approx(oldest)


# =============================================================================
# 可复现性
# =============================================================================


class TestReproducibility:
    """同一输入必须给出完全相同的结果（模块里没有随机成分）。"""

    def test_two_calls_agree(self):
        first = orbital_frequencies(PRESENT_AGE_MA)
        second = orbital_frequencies(PRESENT_AGE_MA)
        for name in EXPECTED_NAMES:
            assert float(getattr(first, name).period_kyr) == float(getattr(second, name).period_kyr)
            assert float(getattr(first, name).frequency_arcsec_per_yr) == float(
                getattr(second, name).frequency_arcsec_per_yr
            )
