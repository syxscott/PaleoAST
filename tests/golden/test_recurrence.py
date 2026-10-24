# =============================================================================
# FILE: tests/golden/test_recurrence.py
# =============================================================================
"""
``stratigraphy.recurrence``（递归图与递归定量分析）的测试。

测试围绕五条主线组织：

* **暴力真值对照（最重要）。** 构造一条长度 12 的可手工枚举的轨迹，在固定
  半径下，测试里**独立地**按定义枚举所有点对（``d(x_i, x_j) <= radius``），
  再逐条对角线 / 逐列扫描出极大连续复发点段，算出 DET 与 LAM，与模块结果
  **逐位精确比对**（不是近似）。这条断言同时钉死了四件事：距离度量用的是
  哪个范数、复发点的判等是 ``<= radius``、Theiler 窗剔除了哪些点对、
  以及"线结构"指的是极大连续段而不是别的切法。

* **Theiler 窗与 LAM / LAMIN 的关系。** ``theiler_window = 0`` 时不剔除任何
  点对，LAM 与 LAMIN 必须相等；``theiler_window > 0`` 时窗口会把垂直线
  切断，于是 ``LAM <= LAMIN``（Marwan et al. 2007 的 "laminarity without
  considering the Theiler window"）。

* **对角线与对称性。** 递归矩阵的主对角线恒为 ``True``（自身距离为 0），
  且矩阵对称（自递归图）。

* **白噪声 vs 周期信号。** 同样长度、同样半径下，白噪声的 DET 必须明显
  小于周期信号的 DET——周期信号把复发点排成一条条对角带，DET 高；白噪声
  的复发点是散点，DET 低。

* **置换不变性。** 递归矩阵对样本顺序是**等变**的（``R' = P R P^T``），
  因此复发点总数与复发率（RR）在置换下**精确不变**；DET 的分子是对角线
  结构，它**依赖样本顺序**（Marwan et al. 2007 的定义如此），所以只有当
  递归结构里没有非平凡对角线时 DET 才在置换下精确不变。

边界测试断言抛 ``utils.exceptions.ValidationError``；可复现性断言重复调用
给出完全相同的结果。

参考来源（断言中引用的数值/定义出处）
------------------------------------
* Eckmann, J.-P., Oliffson Kamphorst, S. & Ruelle, D. (1987). Recurrence Plots
  for Dynamical Systems. *Europhysics Letters*, 4(9), 417-427.（递归点定义）
* Marwan, N., Kurths, J. & Herrmann, W. (2007). Recurrence plots for the
  analysis of complex systems. *Physics Reports*, 439(1-2), 1-74.（DET / MDL /
  LAM 的公式、Theiler 窗、最小线长 ``v_min``、LAMIN 的定义）
* Small, M. (2012). *Nonlinear Time Series Analysis.* Cambridge University
  Press.（RQA 的标准参考）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12-22.（Acycle 的 "recurrence plot" 子功能；本模块为独立实现）
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"recurrence_plot"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.recurrence import recurrence_plot, recurrence_quantification
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 暴力真值对照用的轨迹（12 个点，可手工枚举）：含重复值，因此既会产生
#: 非平凡的复发点，也会产生非平凡的对角线与垂直线结构。
TRAJECTORY_SMALL = np.array([0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 3.0, 2.0, 1.0, 0.0, 2.0, 1.0])

#: 第二条暴力对照轨迹（20 个点），用来排除"只对一条轨迹成立"的巧合。
TRAJECTORY_LONG = np.array(
    [
        0.0,
        2.0,
        4.0,
        3.0,
        1.0,
        0.0,
        2.0,
        5.0,
        5.0,
        3.0,
        1.0,
        0.0,
        1.0,
        4.0,
        6.0,
        4.0,
        2.0,
        1.0,
        3.0,
        2.0,
    ]
)

#: 暴力对照使用的半径（绝对距离）。
BRUTE_FORCE_RADIUS = 1.0

#: 暴力对照使用的 Theiler 窗宽。
BRUTE_FORCE_WINDOW = 2

#: 对比实验的记录长度。
N_POINTS = 200

#: 周期信号的周期（样本数）。
SIGNAL_PERIOD = 20

#: 对比实验统一使用的绝对半径（白噪声与周期信号用同一个半径才公平）。
CONTRAST_RADIUS = 0.1

#: 对比实验里周期信号的 DET 必须高于此值。
SIGNAL_DET_FLOOR = 0.5

#: 对比实验里白噪声的 DET 必须低于此值。
NOISE_DET_CEILING = 0.3

#: Theiler 窗生效时 LAM 不得大于 LAMIN。
LAMIN_FLOOR_MARGIN = 0.0


# =============================================================================
# 辅助函数：暴力枚举（独立于被测模块的实现）
# =============================================================================


def _distance(first: npt.NDArray, second: npt.NDArray, metric: str = "euclidean") -> float:
    """按定义计算两点距离（欧氏 / 1-范数 / 无穷范数）。"""
    difference = np.abs(first - second)
    if metric == "manhattan":
        return float(difference.sum())
    if metric == "chebyshev":
        return float(difference.max())
    return float(np.sqrt((difference**2).sum()))


def _brute_force_matrix(
    values: npt.NDArray,
    radius: float,
    theiler_window: int = 0,
    metric: str = "euclidean",
) -> list[list[bool]]:
    """按定义枚举全部点对，返回递归矩阵（纯 Python 双重循环）。

    判据：``d(x_i, x_j) <= radius`` 即为复发点；``theiler_window > 0`` 时
    ``|i - j| <= theiler_window`` 的点对不计入（被置为 ``False``）。
    """
    size = values.shape[0]
    matrix = [[False] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if _distance(values[i], values[j], metric) <= radius:
                if theiler_window > 0 and abs(i - j) <= theiler_window:
                    continue
                matrix[i][j] = True
    return matrix


def _brute_force_diagonal_lines(matrix: list[list[bool]], min_line_length: int = 2) -> list[int]:
    """逐条对角线扫描出极大连续复发点段的长度。"""
    size = len(matrix)
    lengths: list[int] = []
    for offset in range(-(size - 1), size):
        run = 0
        for j in range(size):
            i = j + offset
            if 0 <= i < size and matrix[i][j]:
                run += 1
            else:
                if run >= min_line_length:
                    lengths.append(run)
                run = 0
        if run >= min_line_length:
            lengths.append(run)
    return lengths


def _brute_force_vertical_lines(matrix: list[list[bool]], min_line_length: int = 2) -> list[int]:
    """逐列扫描出极大连续复发点段的长度。"""
    size = len(matrix)
    lengths: list[int] = []
    for j in range(size):
        run = 0
        for i in range(size):
            if matrix[i][j]:
                run += 1
            else:
                if run >= min_line_length:
                    lengths.append(run)
                run = 0
        if run >= min_line_length:
            lengths.append(run)
    return lengths


def _brute_force_det(matrix: list[list[bool]], min_line_length: int = 2) -> float:
    """按定义暴力计算 DET。

    分子 = 所有长度 ``>= min_line_length`` 的对角线长度之和；分母 = 矩阵里
    的复发点总数。
    """
    total = sum(1 for row in matrix for value in row if value)
    diagonal_sum = sum(_brute_force_diagonal_lines(matrix, min_line_length))
    return diagonal_sum / total if total else 0.0


def _brute_force_mdl(matrix: list[list[bool]], min_line_length: int = 2) -> float:
    """按定义暴力计算 MDL（对角线长度的平均值）。"""
    lengths = _brute_force_diagonal_lines(matrix, min_line_length)
    return sum(lengths) / len(lengths) if lengths else 0.0


def _brute_force_lam(matrix: list[list[bool]], min_line_length: int = 2) -> float:
    """按定义暴力计算 LAM（垂直线长度的平均值）。"""
    lengths = _brute_force_vertical_lines(matrix, min_line_length)
    return sum(lengths) / len(lengths) if lengths else 0.0


def _brute_force_entropy(matrix: list[list[bool]], min_line_length: int = 2) -> float:
    """按定义暴力计算垂直线长度分布的 Shannon 熵（以 2 为底）。"""
    lengths = _brute_force_vertical_lines(matrix, min_line_length)
    if not lengths:
        return 0.0
    total = float(len(lengths))
    entropy = 0.0
    for length in set(lengths):
        probability = sum(1 for value in lengths if value == length) / total
        entropy -= probability * np.log2(probability)
    return float(entropy)


def _periodic_signal(n_points: int = N_POINTS) -> npt.NDArray:
    """周期信号（周期 ``SIGNAL_PERIOD`` 个样本）。"""
    index = np.arange(n_points)
    return np.sin(2.0 * np.pi * index / SIGNAL_PERIOD)


def _white_noise(n_points: int = N_POINTS, seed: int = 5) -> npt.NDArray:
    """白噪声（固定种子，保证可复现）。"""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n_points)


# =============================================================================
# 暴力真值对照
# =============================================================================


class TestBruteForceGroundTruth:
    """模块结果必须与按定义暴力枚举的真值**逐位**一致。"""

    @pytest.mark.parametrize("trajectory", [TRAJECTORY_SMALL, TRAJECTORY_LONG])
    def test_det_matches_brute_force_enumeration(self, trajectory: npt.NDArray):
        """DET 必须与暴力枚举的结果精确相等（无 Theiler 窗）。

        真值：分子 = 所有长度 ``>= v_min`` 的对角线长度之和，分母 = 复发点
        总数（Marwan et al. 2007 的 DET 定义）。
        """
        result = recurrence_plot(trajectory, radius=BRUTE_FORCE_RADIUS)
        reference = _brute_force_matrix(trajectory, BRUTE_FORCE_RADIUS)

        expected = _brute_force_det(reference)
        assert result.det == expected, f"DET is {result.det!r} but the brute-force enumeration gives {expected!r}"

    @pytest.mark.parametrize("trajectory", [TRAJECTORY_SMALL, TRAJECTORY_LONG])
    def test_lam_matches_brute_force_enumeration(self, trajectory: npt.NDArray):
        """LAM 必须与暴力枚举的垂直线平均长度精确相等。"""
        result = recurrence_plot(trajectory, radius=BRUTE_FORCE_RADIUS)
        reference = _brute_force_matrix(trajectory, BRUTE_FORCE_RADIUS)

        expected = _brute_force_lam(reference)
        assert result.lam == expected, f"LAM is {result.lam!r} but the brute-force enumeration gives {expected!r}"

    def test_mdl_and_entropy_match_brute_force_enumeration(self):
        """MDL 与 ENTROPY 也必须与暴力枚举一致。"""
        result = recurrence_plot(TRAJECTORY_SMALL, radius=BRUTE_FORCE_RADIUS)
        reference = _brute_force_matrix(TRAJECTORY_SMALL, BRUTE_FORCE_RADIUS)

        assert result.mdl == _brute_force_mdl(reference)
        assert result.entropy == pytest.approx(_brute_force_entropy(reference), rel=1e-12, abs=1e-12)

    def test_theiler_windowed_det_matches_brute_force_enumeration(self):
        """带 Theiler 窗时 DET 必须与暴力枚举（同样的剔除规则）精确相等。

        剔除规则：``|i - j| <= theiler_window`` 的点对既不计入分子也不计入
        分母，并且会切断穿过该带的线结构。
        """
        result = recurrence_plot(
            TRAJECTORY_SMALL,
            radius=BRUTE_FORCE_RADIUS,
            theiler_window=BRUTE_FORCE_WINDOW,
        )
        reference = _brute_force_matrix(
            TRAJECTORY_SMALL,
            BRUTE_FORCE_RADIUS,
            theiler_window=BRUTE_FORCE_WINDOW,
        )

        expected = _brute_force_det(reference)
        assert result.det == expected, f"DET is {result.det!r} but the brute-force enumeration gives {expected!r}"

    def test_theiler_windowed_lam_matches_brute_force_enumeration(self):
        """带 Theiler 窗时 LAM 必须与暴力枚举精确相等。"""
        result = recurrence_plot(
            TRAJECTORY_SMALL,
            radius=BRUTE_FORCE_RADIUS,
            theiler_window=BRUTE_FORCE_WINDOW,
        )
        reference = _brute_force_matrix(
            TRAJECTORY_SMALL,
            BRUTE_FORCE_RADIUS,
            theiler_window=BRUTE_FORCE_WINDOW,
        )

        expected = _brute_force_lam(reference)
        assert result.lam == expected, f"LAM is {result.lam!r} but the brute-force enumeration gives {expected!r}"

    def test_recurrence_points_match_brute_force_enumeration(self):
        """复发点总数必须与暴力枚举一致（DET 的分母）。"""
        result = recurrence_plot(TRAJECTORY_SMALL, radius=BRUTE_FORCE_RADIUS)
        reference = _brute_force_matrix(TRAJECTORY_SMALL, BRUTE_FORCE_RADIUS)

        expected = sum(1 for row in reference for value in row if value)
        assert result.n_recurrence_points == expected
        assert result.recurrence_rate == pytest.approx(expected / float(TRAJECTORY_SMALL.size**2))

    def test_quantification_reproduces_the_plot_metrics(self):
        """``recurrence_quantification`` 必须重现 ``recurrence_plot`` 的指标。

        这是"可以对已有矩阵重算"的契约：同一个矩阵 + 同一个半径 + 同一个
        Theiler 窗 -> 同一组指标。
        """
        result = recurrence_plot(TRAJECTORY_SMALL, radius=BRUTE_FORCE_RADIUS, theiler_window=1)
        recomputed = recurrence_quantification(
            result.matrix,
            result.radii,
            1,
        )

        assert recomputed.det == result.det
        assert recomputed.mdl == result.mdl
        assert recomputed.lam == result.lam
        assert recomputed.laminarity == result.laminarity
        assert recomputed.entropy == result.entropy
        assert recomputed.radii == result.radii


# =============================================================================
# 递归矩阵本身的性质
# =============================================================================


class TestRecurrenceMatrix:
    """递归矩阵必须满足对称性、主对角线全 True 等结构性质。"""

    def test_the_main_diagonal_is_all_true(self):
        """主对角线必须是复发点：自身距离为 0，永远 ``<= radius``。"""
        result = recurrence_plot(TRAJECTORY_SMALL, radius=BRUTE_FORCE_RADIUS)

        assert result.matrix.shape == (TRAJECTORY_SMALL.size, TRAJECTORY_SMALL.size)
        assert result.matrix.dtype == np.bool_
        assert np.all(np.diag(result.matrix))

    def test_the_matrix_is_symmetric(self):
        """自递归图必须对称（``d(x_i, x_j) = d(x_j, x_i)``）。"""
        result = recurrence_plot(_periodic_signal(), radius=CONTRAST_RADIUS)

        assert np.array_equal(result.matrix, result.matrix.T)

    def test_a_zero_radius_keeps_only_identical_values(self):
        """半径为 0 时，只有取值完全相同的点才互为复发点。"""
        values = np.array([0.0, 1.0, 1.0, 2.0])
        result = recurrence_plot(values, radius=0.0)

        expected = np.array(
            [
                [True, False, False, False],
                [False, True, True, False],
                [False, True, True, False],
                [False, False, False, True],
            ]
        )
        assert np.array_equal(result.matrix, expected)


# =============================================================================
# Theiler 窗
# =============================================================================


class TestTheilerWindow:
    """Theiler 窗必须只剔除主对角线附近的点对。"""

    def test_the_window_excludes_the_main_diagonal(self):
        """窗宽 ``>= 1`` 时主对角线上的点对不计入 DET 的分母。"""
        without = recurrence_plot(TRAJECTORY_SMALL, radius=BRUTE_FORCE_RADIUS)
        with_window = recurrence_plot(
            TRAJECTORY_SMALL,
            radius=BRUTE_FORCE_RADIUS,
            theiler_window=1,
        )

        assert without.n_recurrence_points > with_window.n_recurrence_points

    def test_a_zero_window_excludes_nothing(self):
        """窗宽为 0 时不剔除任何点对（含主对角线本身）。"""
        result = recurrence_plot(TRAJECTORY_SMALL, radius=BRUTE_FORCE_RADIUS, theiler_window=0)

        assert result.n_recurrence_points == sum(1 for row in result.matrix for value in row if value)

    def test_lam_is_not_greater_than_lamin_with_a_window(self):
        """有 Theiler 窗时 ``LAM <= LAMIN``。

        窗会把穿过它的垂直线**切断**，而 LAMIN 用的是没有被切断的完整矩阵，
        所以 LAMIN 的垂直线平均长度不小于 LAM（Marwan et al. 2007）。
        """
        values = _periodic_signal() + 0.05 * np.random.default_rng(11).standard_normal(N_POINTS)
        result = recurrence_plot(values, radius=CONTRAST_RADIUS, theiler_window=2)

        assert result.lam <= result.laminarity + LAMIN_FLOOR_MARGIN, (
            f"LAM {result.lam:.6f} is larger than LAMIN {result.laminarity:.6f}"
        )

    def test_lam_equals_lamin_without_a_window(self):
        """没有 Theiler 窗时 LAM 与 LAMIN 必须相等（分析矩阵就是完整矩阵）。"""
        result = recurrence_plot(_periodic_signal(), radius=CONTRAST_RADIUS, theiler_window=0)

        assert result.lam == result.laminarity

    def test_the_window_changes_the_metrics(self):
        """Theiler 窗必须真的改变指标（否则它没有生效）。"""
        values = _periodic_signal()
        without = recurrence_plot(values, radius=CONTRAST_RADIUS, theiler_window=0)
        with_window = recurrence_plot(values, radius=CONTRAST_RADIUS, theiler_window=2)

        assert without.det != with_window.det
        assert without.n_recurrence_points != with_window.n_recurrence_points


# =============================================================================
# 白噪声 vs 周期信号
# =============================================================================


class TestNoiseAgainstPeriodicity:
    """周期信号必须比白噪声"更确定"。"""

    def test_periodic_signal_has_a_higher_det_than_white_noise(self):
        """同样长度、同样半径下，白噪声的 DET 必须明显小于周期信号。

        周期信号把复发点排成一条条对角带（轨迹回到同一状态），所以复发点
        大多落在对角线上；白噪声的复发点是散点，绝大多数不构成对角线结构。
        """
        signal = recurrence_plot(_periodic_signal(), radius=CONTRAST_RADIUS)
        noise = recurrence_plot(_white_noise(), radius=CONTRAST_RADIUS)

        assert signal.det > SIGNAL_DET_FLOOR, f"the periodic signal only reaches DET {signal.det:.4f}"
        assert noise.det < NOISE_DET_CEILING, f"the white noise reaches DET {noise.det:.4f}"
        assert noise.det < 0.5 * signal.det, (
            f"the white noise DET {noise.det:.4f} is not clearly below the periodic {signal.det:.4f}"
        )

    def test_the_contrast_survives_other_noise_realisations(self):
        """换一个噪声种子，结论必须依然成立（这个定性差异是稳健的）。"""
        signal = recurrence_plot(_periodic_signal(), radius=CONTRAST_RADIUS)

        for seed in range(6):
            noise = recurrence_plot(_white_noise(seed=seed), radius=CONTRAST_RADIUS)
            assert noise.det < 0.5 * signal.det, f"the white noise (seed {seed}) reaches DET {noise.det:.4f}"


# =============================================================================
# 置换不变性
# =============================================================================


class TestPermutation:
    """样本顺序只应改变结构，不应改变距离分布。"""

    def test_the_matrix_is_permutation_equivariant(self):
        """打乱顺序后，递归矩阵必须满足 ``R' = P R P^T``（精确相等）。

        这是"置换不改变距离分布"的最强形式：由定义，``d(x'_i, x'_j)`` 就是
        原矩阵某个元素的重排。
        """
        values = _periodic_signal()
        result = recurrence_plot(values, radius=CONTRAST_RADIUS)

        order = np.random.default_rng(3).permutation(values.size)
        shuffled = recurrence_plot(values[order], radius=CONTRAST_RADIUS)

        permutation = np.eye(values.size)[order]
        assert np.array_equal(shuffled.matrix, permutation @ result.matrix @ permutation.T)

    def test_the_recurrence_count_is_permutation_invariant(self):
        """复发点总数（DET 的分母）在置换下必须精确不变。"""
        values = _periodic_signal()
        result = recurrence_plot(values, radius=CONTRAST_RADIUS)

        order = np.random.default_rng(3).permutation(values.size)
        shuffled = recurrence_plot(values[order], radius=CONTRAST_RADIUS)

        assert shuffled.n_recurrence_points == result.n_recurrence_points
        assert shuffled.recurrence_rate == result.recurrence_rate

    def test_det_is_permutation_invariant_without_off_diagonal_structures(self):
        """没有非平凡对角线结构时，DET 在置换下必须精确不变。

        取严格递增的序列且半径小于最小间隔：递归矩阵就是单位阵，任何置换
        都给出同一个矩阵，DET 因此不变。
        """
        values = np.arange(12.0)
        result = recurrence_plot(values, radius=0.5)

        order = np.random.default_rng(4).permutation(values.size)
        shuffled = recurrence_plot(values[order], radius=0.5)

        assert shuffled.det == result.det
        assert shuffled.mdl == result.mdl

    def test_det_depends_on_the_order_of_the_samples(self):
        """DET 的分子是对角线结构，因此**依赖**样本顺序。

        这条断言记录 Marwan et al. (2007) 的 DET 的真实语义：在一般情况下
        打乱顺序会改变 DET（只有"距离分布"是不变的，见上面两条断言）。把它
        显式钉在这里，是为了避免有人误以为 DET 是置换不变量。
        """
        alternating = recurrence_plot(np.tile([0.0, 1.0], 8), radius=0.5)
        blocked = recurrence_plot(np.repeat([0.0, 1.0], 8), radius=0.5)

        assert alternating.det != blocked.det


# =============================================================================
# 半径与度量
# =============================================================================


class TestRadiusAndMetric:
    """半径与距离度量的口径必须自洽。"""

    def test_radius_fraction_matches_the_derived_absolute_radius(self):
        """相对半径必须与由它换算出的绝对半径给出**完全相同**的结果。

        一维轨迹的最大点对距离就是 ``max - min``（三种范数在一维上一致），
        测试自己算出这个值再换算，因此不依赖被测模块的内部实现。
        """
        values = _periodic_signal()
        fraction = 0.05
        absolute = fraction * float(values.max() - values.min())

        relative = recurrence_plot(values, radius_fraction=fraction)
        direct = recurrence_plot(values, radius=absolute)

        assert relative.radii == absolute
        assert np.array_equal(relative.matrix, direct.matrix)
        assert relative.det == direct.det
        assert relative.mdl == direct.mdl
        assert relative.lam == direct.lam
        assert relative.laminarity == direct.laminarity
        assert relative.entropy == direct.entropy

    def test_the_reported_radius_is_the_one_actually_used(self):
        """返回的 ``radii`` 必须是实际采用的半径。"""
        values = _periodic_signal()

        assert recurrence_plot(values, radius=0.25).radii == 0.25
        assert recurrence_plot(values, radius_fraction=0.25).radii == pytest.approx(
            0.25 * float(values.max() - values.min())
        )

    def test_the_three_metrics_agree_for_univariate_input(self):
        """一维输入时三种范数给出相同的距离（这是数学事实，不是巧合）。"""
        values = _periodic_signal()
        results = [
            recurrence_plot(values, radius=CONTRAST_RADIUS, metric=metric)
            for metric in ("euclidean", "manhattan", "chebyshev")
        ]

        for result in results[1:]:
            assert np.array_equal(result.matrix, results[0].matrix)
            assert result.det == results[0].det

    def test_the_three_metrics_differ_for_multivariate_input(self):
        """多维输入时三种范数必须给出不同的结果（否则度量选项是摆设）。"""
        rng = np.random.default_rng(8)
        points = rng.standard_normal((60, 3))

        results = {
            metric: recurrence_plot(points, radius_fraction=0.2, metric=metric)
            for metric in ("euclidean", "manhattan", "chebyshev")
        }

        assert not np.array_equal(results["euclidean"].matrix, results["manhattan"].matrix)
        assert not np.array_equal(results["euclidean"].matrix, results["chebyshev"].matrix)


# =============================================================================
# 可复现性与规模
# =============================================================================


class TestReproducibility:
    """算法必须没有随机成分。"""

    def test_repeated_calls_are_identical(self):
        """同一输入重复调用必须给出完全相同的结果。"""
        values = _periodic_signal()

        first = recurrence_plot(values, radius=CONTRAST_RADIUS, theiler_window=2)
        second = recurrence_plot(values, radius=CONTRAST_RADIUS, theiler_window=2)

        assert np.array_equal(first.matrix, second.matrix)
        assert first.det == second.det
        assert first.mdl == second.mdl
        assert first.lam == second.lam
        assert first.laminarity == second.laminarity
        assert first.entropy == second.entropy
        assert first.radii == second.radii

    def test_two_thousand_points_are_handled(self):
        """``N = 2000`` 的 O(N^2) 距离矩阵必须能在合理时间内算完。"""
        values = _periodic_signal(n_points=2000)
        result = recurrence_plot(values, radius_fraction=0.05, theiler_window=2)

        assert result.matrix.shape == (2000, 2000)
        assert 0.0 <= result.det <= 1.0
        assert result.det > 0.0


# =============================================================================
# 边界
# =============================================================================


class TestBoundaries:
    """非法输入必须抛 ``ValidationError``。"""

    def test_too_few_points_are_rejected(self):
        with pytest.raises(ValidationError):
            recurrence_plot(np.array([1.0, 2.0]), radius=0.5)

    def test_non_finite_values_are_rejected(self):
        values = _periodic_signal().copy()
        values[10] = np.nan
        with pytest.raises(ValidationError):
            recurrence_plot(values, radius=CONTRAST_RADIUS)

    def test_infinite_values_are_rejected(self):
        values = _periodic_signal().copy()
        values[3] = np.inf
        with pytest.raises(ValidationError):
            recurrence_plot(values, radius=CONTRAST_RADIUS)

    def test_a_missing_radius_is_rejected(self):
        with pytest.raises(ValidationError):
            recurrence_plot(_periodic_signal())

    def test_both_radius_and_radius_fraction_are_rejected(self):
        with pytest.raises(ValidationError):
            recurrence_plot(_periodic_signal(), radius=0.1, radius_fraction=0.1)

    def test_a_negative_radius_is_rejected(self):
        with pytest.raises(ValidationError):
            recurrence_plot(_periodic_signal(), radius=-1.0)

    @pytest.mark.parametrize("fraction", [-0.1, 1.5, np.inf, np.nan])
    def test_an_invalid_radius_fraction_is_rejected(self, fraction: float):
        with pytest.raises(ValidationError):
            recurrence_plot(_periodic_signal(), radius_fraction=fraction)

    def test_an_unknown_metric_is_rejected(self):
        with pytest.raises(ValidationError):
            recurrence_plot(_periodic_signal(), radius=0.1, metric="cosine")

    def test_a_negative_theiler_window_is_rejected(self):
        with pytest.raises(ValidationError):
            recurrence_plot(_periodic_signal(), radius=0.1, theiler_window=-1)

    def test_an_invalid_min_line_length_is_rejected(self):
        with pytest.raises(ValidationError):
            recurrence_plot(_periodic_signal(), radius=0.1, min_line_length=0)

    def test_quantification_rejects_a_non_square_matrix(self):
        with pytest.raises(ValidationError):
            recurrence_quantification(np.ones((4, 5), dtype=bool), radii=0.1)

    def test_quantification_rejects_a_too_small_matrix(self):
        with pytest.raises(ValidationError):
            recurrence_quantification(np.ones((2, 2), dtype=bool), radii=0.1)

    def test_quantification_rejects_several_radii(self):
        with pytest.raises(ValidationError):
            recurrence_quantification(np.ones((5, 5), dtype=bool), radii=[0.1, 0.2])
