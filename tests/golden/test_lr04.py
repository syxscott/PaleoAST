# =============================================================================
# FILE: tests/golden/test_lr04.py
# =============================================================================
"""
``stratigraphy.lr04`` (LR04 底栖 δ¹⁸O 复合栈) 的测试。

本文件是**真值对照** (golden test): 断言用的 δ¹⁸O 数值不是从本模块的输出里
抄来的, 而是 **Lisiecki & Raymo (2005) 发表的 LR04 栈**的公开数值 —— 同一
数据集在 NOAA/NCEI 古气候数据库 (study 5847) 里逐行相同, 例如:

===========  ==========
年龄 (ka)    δ¹⁸O (‰)
===========  ==========
0.0          3.23
10.0         3.52
20.0         4.99
30.0         4.62
40.0         4.55
50.0         4.34
===========  ==========

因此"在已知年龄处插值得到的值 == 已发表值"是一条**独立于实现**的真值断言。

其余断言覆盖结构性性质: 年龄轴单调且**由老到新**、覆盖范围符合数据本身、
δ¹⁸O 值域落在已发表栈的范围内, 以及越界年龄抛
``utils.exceptions.ValidationError``。

参考来源
--------
* Lisiecki, L.E. & Raymo, M.E., 2005. A Pliocene-Pleistocene stack of 57
  globally distributed benthic δ¹⁸O records. *Paleoceanography*, 20(2):
  269-305. doi:10.1029/2004PA001071
* NOAA/NCEI Paleoclimatology: LR04 Global Pliocene-Pleistocene Benthic δ¹⁸O
  Stack, study 5847.
  https://www.ncei.noaa.gov/access/paleo-search/study/5847
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"lr04_stack"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.lr04 import load_lr04_stack, lr04_target_curve
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 已发表的 LR04 栈采样点数 (NOAA/NCEI study 5847 与 Acycle 分发的文件一致)。
EXPECTED_SAMPLES = 2115

#: 栈覆盖的年龄范围 (Ma)。
EXPECTED_MIN_AGE_MA = 0.0
EXPECTED_MAX_AGE_MA = 5.32

#: 已发表 LR04 栈的 δ¹⁸O 值域 (‰ VPDB)。
EXPECTED_MIN_D18O = 2.65
EXPECTED_MAX_D18O = 5.08

#: 已发表的 LR04 栈真值: 年龄 (ka) -> 底栖 δ¹⁸O (‰, VPDB)。
PUBLISHED_LR04: dict[float, float] = {
    0.0: 3.23,
    10.0: 3.52,
    20.0: 4.99,
    30.0: 4.62,
    40.0: 4.55,
    50.0: 4.34,
}


# =============================================================================
# 载入与结构
# =============================================================================


def test_stack_loads_with_published_sample_count():
    """栈应含 2115 个采样点 (与已发表的数据文件一致)。"""
    stack = load_lr04_stack()

    assert stack.ages_ma.shape == (EXPECTED_SAMPLES,)
    assert stack.d18o.shape == (EXPECTED_SAMPLES,)


def test_age_axis_is_monotonic_and_oldest_first():
    """年龄轴必须严格单调, 且按仓库约定**由老到新** (数值递减)。"""
    stack = load_lr04_stack()

    differences = np.diff(stack.ages_ma)
    assert np.all(differences < 0.0), "ages must decrease from oldest to newest"
    assert stack.ages_ma[0] == pytest.approx(EXPECTED_MAX_AGE_MA)
    assert stack.ages_ma[-1] == pytest.approx(EXPECTED_MIN_AGE_MA)


def test_age_coverage_matches_published_stack():
    """覆盖范围应为 0 – 5.32 Ma。"""
    stack = load_lr04_stack()

    assert float(stack.ages_ma.max()) == pytest.approx(EXPECTED_MAX_AGE_MA)
    assert float(stack.ages_ma.min()) == pytest.approx(EXPECTED_MIN_AGE_MA)


def test_d18o_values_are_within_the_published_range():
    """
    δ¹⁸O 值域应与已发表的 LR04 栈一致 (2.65 – 5.08 ‰ VPDB)。

    注意底栖 δ¹⁸O 的**绝对值**落在 2.6–5.1 ‰: 冰期高、间冰期低, 0 附近
    并非该代理指标的真实取值区间。
    """
    stack = load_lr04_stack()
    values = stack.d18o

    assert float(values.min()) == pytest.approx(EXPECTED_MIN_D18O, abs=1e-9)
    assert float(values.max()) == pytest.approx(EXPECTED_MAX_D18O, abs=1e-9)
    assert np.all((values > -1.0) & (values < 6.0))


# =============================================================================
# 真值对照
# =============================================================================


@pytest.mark.parametrize(("age_ka", "expected"), sorted(PUBLISHED_LR04.items()))
def test_interpolated_values_match_published_truth(age_ka: float, expected: float):
    """
    真值对照: 在已发表的采样年龄处, 插值结果必须等于已发表的 δ¹⁸O。

    采样点上的线性插值返回该点本身的值, 所以这里是**精确**比较
    (容差只留给浮点表示)。
    """
    stack = load_lr04_stack()

    value = stack.at(np.array([age_ka / 1000.0]))

    assert float(value[0]) == pytest.approx(expected, abs=1e-9)


def test_target_curve_agrees_with_the_stack():
    """目标曲线 (线性插值) 与栈的插值必须一致。"""
    stack = load_lr04_stack()
    ages = np.array([0.0, 0.02, 0.5, 2.0, 5.32])

    curve = lr04_target_curve(ages)

    assert np.allclose(curve, stack.at(ages))


def test_target_curve_nearest_uses_the_published_sample_values():
    """最近邻取样应落在已发表的采样值上 (含 LGM 的 4.99 ‰)。"""
    curve = lr04_target_curve(np.array([0.0201]), method="nearest")

    assert float(curve[0]) == pytest.approx(PUBLISHED_LR04[20.0], abs=1e-9)


# =============================================================================
# 边界与错误路径
# =============================================================================


def test_at_out_of_range_age_raises():
    """超出 0 – 5.32 Ma 的年龄必须报错, 而不是被外推。"""
    stack = load_lr04_stack()

    with pytest.raises(ValidationError):
        stack.at(np.array([6.0]))
    with pytest.raises(ValidationError):
        stack.at(np.array([-0.1]))


def test_at_rejects_non_finite_ages():
    """含 NaN/inf 的年龄轴必须报错。"""
    stack = load_lr04_stack()

    with pytest.raises(ValidationError):
        stack.at(np.array([np.nan, 1.0]))
    with pytest.raises(ValidationError):
        stack.at(np.array([np.inf]))


def test_target_curve_rejects_unknown_method():
    """未知的取样方式必须报错。"""
    with pytest.raises(ValidationError):
        lr04_target_curve(np.array([0.1]), method="spline")


def test_target_curve_out_of_range_raises():
    """目标曲线同样不外推。"""
    with pytest.raises(ValidationError):
        lr04_target_curve(np.array([9.0]))


# =============================================================================
# 摘要
# =============================================================================


def test_summary_reports_the_coverage():
    """摘要应含样本数与年龄覆盖。"""
    summary = load_lr04_stack().summary()

    assert str(EXPECTED_SAMPLES) in summary
    assert "5.32" in summary
    assert "oldest first" in summary


def test_values_and_ages_are_finite_and_paired():
    """两个数组必须等长、有限, 且年龄严格递减。"""
    stack = load_lr04_stack()
    ages: npt.NDArray = stack.ages_ma
    values: npt.NDArray = stack.d18o

    assert ages.size == values.size
    assert np.all(np.isfinite(ages))
    assert np.all(np.isfinite(values))
    assert np.all(np.diff(ages) < 0.0)
