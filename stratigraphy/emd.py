# =============================================================================
# FILE: stratigraphy/emd.py
# =============================================================================
"""
经验模态分解（EMD）与带噪经验模态分解（EEMD）。

本模块为古生物地层/同位素时间序列提供两条互补的分解路径：

* :func:`emd` —— 经典的 Huang et al. (1998) 经验模态分解。把一条记录逐级
  拆成若干**本征模态函数**（IMF）与一条残差（趋势项）。IMF 是窄带的、
  幅度包络近似恒定的振荡成分，其频率由高到低排列，适合回答“这条记录里
  有几个尺度、每个尺度有多强”这类问题。
* :func:`eemd` —— Wu & Huang (2009) 的集合经验模态分解。在原序列上加白
  噪声做多次独立分解，再把同名 IMF 归一化后取平均。白噪声在每一次试验
  中对 IMF 的扰动是随机的，集合平均可以把这种扰动（噪声残留）压低，
  同时保留稳定的周期性成分。

算法要点（按论文独立实现，未翻译 Acycle 的 GPL 源码）：

1. **包络**：用三次样条（CubicSpline, natural 边界条件）分别穿过全部
   局部极大值点与局部极小值点，得到上包络与下包络；两者均值
   ``m(t) = (upper + lower) / 2`` 即包络均值。
2. **抽筛（sifting）**：``h(t) = x(t) - m(t)``，反复执行。
3. **停止准则**：von Neumann 判据
   ``SD = sum(h_prev(t) * h(t)) / sum(h_prev(t) ** 2)``，当 ``SD`` 低于
   ``sd_tolerance``（默认 0.2）时认为本次抽筛可以收束；同时要求 h 满足
   Huang et al. (1998) 给出的 IMF 条件（见 :func:`_is_imf`）。若抽筛已经
   数值收敛（相邻两次迭代的相对变化可忽略）也会提前结束，以免在已经
   稳定的 IMF 上白白消耗迭代次数。
4. **IMF 条件**：极值点个数与过零次数之差不超过 1，且包络均值相对振幅
   可忽略。
5. **残差**：原序列减去全部 IMF。

单位约定：

* 所有**周期**以 **kyr** 为单位，与 Acycle 一致；本模块本身不做任何
  周期换算，EMD 是时域算法，不关心时间轴单位。
* **时间轴**单位由调用方决定（:func:`emd` 与 :func:`eemd` 接受可选的
  ``times`` 参数，仅用于随结果返回，本模块不做任何单位假设）。

参考文献：

    1. Huang, N.E., Shen, Z., Long, M.C., Wu, M.C., Shih, H.H., Zheng, Q.N.,
       Yen, N.C., Liu, C.C., Liu, S.S., Lu, Q.R., Chen, W.Y., Zhang, Y.G. &
       Yang, F.R. 1998. The empirical mode decomposition and the Hilbert
       spectrum in nonlinear analysis. Proceedings of the Royal Society of
       London A 454: 903-995.
    2. Wu, Z. & Huang, N.E. 2009. A study of the characteristics of white
       noise using the empirical mode decomposition method. Advances in Data
       Science and Engineering 1: 1-41.
    3. Torres, M.A., Colominas, M.A. & Schlotterer, G. 2001. A new approach
       to empirical mode decomposition. (EEMD 的集合平均思想。)
    4. Wu, Z. & Huang, N.E. 2003. Ensemble empirical mode decomposition: a
       noise-assisted data analysis method. Advances in Adaptive Data
       Analysis 1: 1-41.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

from config.i18n import _
from utils.exceptions import DataValidationError
from utils.statistics_core import make_rng

logger = logging.getLogger(__name__)

#: 经验模态分解要求的最少采样点数。少于该点数无法构造上下包络。
MIN_SAMPLES = 8

#: 判定“包络均值已可忽略”的相对阈值（相对于 IMF 的峰谷振幅）。
ENVELOPE_MEAN_TOL = 0.05

#: 判定抽筛已数值收敛的相对变化阈值。
CONVERGENCE_TOL = 1e-10


@dataclass(frozen=True)
class EMDResult:
    """EMD / EEMD 分解结果。

    Attributes:
        imfs: 本征模态函数矩阵，形状 ``(n_imfs, n)``。第 0 行是频率最高的
            IMF，依次向低频排列，行数可以为 0（序列不可分解时）。
        trend: 残差（趋势项），形状 ``(n,)``。
        times: 时间轴，可为 ``None``。单位由调用方决定。
        n_sift: 抽筛迭代总次数（所有 IMF 累计）。
    """

    imfs: npt.NDArray
    trend: npt.NDArray
    times: npt.NDArray | None = None
    n_sift: int = 0

    def summary(self) -> str:
        """生成摘要文本。"""
        lines = [
            _("Empirical Mode Decomposition"),
            "=" * 50,
            _("IMF count: {0}").format(self.imfs.shape[0]),
            _("Sifting iterations: {0}").format(self.n_sift),
            _("Samples: {0}").format(self.trend.size),
        ]
        for index, imf in enumerate(self.imfs):
            lines.append(_("  IMF {0}: sd = {1}").format(index + 1, f"{float(np.std(imf)):.6g}"))
        lines.append(_("Trend sd: {0}").format(f"{float(np.std(self.trend)):.6g}"))
        return "\n".join(lines)


# =============================================================================
# 内部工具函数
# =============================================================================


def _as_series(values: npt.NDArray, name: str = "values") -> npt.NDArray:
    """把输入整理成一维有限实数序列，并检查最基本的可用性。

    Args:
        values: 输入序列。
        name: 报错信息中使用的参数名。

    Returns:
        形状 ``(n,)`` 的 ``float64`` 数组。

    Raises:
        DataValidationError: 输入不是一维、点数过少、含非有限值或为常数序列。
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise DataValidationError(
            _("{0} must be a one-dimensional array").format(name),
            details={"ndim": int(array.ndim)},
        )
    if array.size < MIN_SAMPLES:
        raise DataValidationError(
            _("EMD needs at least {0} samples; got {1}").format(MIN_SAMPLES, array.size),
            details={"n_samples": int(array.size), "minimum": MIN_SAMPLES},
        )
    if not np.all(np.isfinite(array)):
        raise DataValidationError(
            _("{0} must contain only finite values").format(name),
            details={"n_nonfinite": int(np.count_nonzero(~np.isfinite(array)))},
        )
    if float(np.ptp(array)) <= 0.0:
        raise DataValidationError(
            _("{0} is constant; EMD needs an oscillatory series").format(name),
            details={"n_samples": int(array.size)},
        )
    return array


def _extrema_indices(x: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """返回局部极大值与局部极小值的索引。

    两端点不计入极值（与 Acycle / MATLAB ``emd`` 的 ``find_extrema`` 一致），
    水平平台由 :func:`scipy.signal.find_peaks` 取平台中点。

    Args:
        x: 输入序列。

    Returns:
        ``(max_idx, min_idx)``，两个整数数组。
    """
    max_idx, _ = find_peaks(x)
    min_idx, _ = find_peaks(-x)
    return max_idx, min_idx


def _count_extrema(x: npt.NDArray) -> int:
    """统计内部极值点总数（极大值 + 极小值）。"""
    max_idx, min_idx = _extrema_indices(x)
    return int(max_idx.size + min_idx.size)


def _count_zero_crossings(x: npt.NDArray) -> bool | int:
    """统计过零次数，恰好为零的点被跳过。"""
    nonzero = x[x != 0.0]
    if nonzero.size < 2:
        return 0
    return int(np.count_nonzero(np.signbit(nonzero[1:]) != np.signbit(nonzero[:-1])))


def _envelopes(x: npt.NDArray) -> tuple[npt.NDArray | None, npt.NDArray | None]:
    """用三次样条插值求出上包络与下包络。

    Args:
        x: 输入序列。

    Returns:
        ``(upper, lower)``；当极大值点或极小值点少于 2 个、无法构造样条时
        返回 ``(None, None)``。
    """
    max_idx, min_idx = _extrema_indices(x)
    if max_idx.size < 2 or min_idx.size < 2:
        return None, None
    positions = np.arange(x.size, dtype=float)
    upper = CubicSpline(max_idx.astype(float), x[max_idx], bc_type="natural")(positions)
    lower = CubicSpline(min_idx.astype(float), x[min_idx], bc_type="natural")(positions)
    return upper, lower


def _is_imf(h: npt.NDArray) -> bool:
    """判断 ``h`` 是否满足 Huang et al. (1998) 的 IMF 条件。

    两个条件分别是：

    1. 极值点个数与过零次数之差不超过 1；
    2. 包络均值相对振幅可忽略（即包络均值处处接近零）。

    Args:
        h: 待检验的候选 IMF。

    Returns:
        满足两个条件时返回 ``True``。
    """
    if abs(_count_extrema(h) - _count_zero_crossings(h)) > 1:
        return False
    amplitude = float(np.ptp(h))
    if amplitude <= 0.0:
        return True
    upper, lower = _envelopes(h)
    if upper is None or lower is None:
        # 没有足够的极值点构造包络，此时无法检验包络均值条件。
        return True
    mean_envelope = 0.5 * (upper + lower)
    return bool(float(np.max(np.abs(mean_envelope))) <= ENVELOPE_MEAN_TOL * amplitude)


def _sift(residual: npt.NDArray, max_sift: int, sd_tolerance: float) -> tuple[npt.NDArray, int]:
    """对残差做抽筛，得到一个 IMF。

    每次迭代计算上下包络均值并从当前迭代量中减去；停止条件是

    * ``SD = sum(h_prev * h) / sum(h_prev ** 2) < sd_tolerance``（von Neumann
      判据），且
    * ``h`` 满足 IMF 条件；

    或者抽筛已经数值收敛（相邻两次迭代的相对变化小于
    :data:`CONVERGENCE_TOL`）。

    Args:
        residual: 待分解的残差。
        max_sift: 最大抽筛迭代次数。
        sd_tolerance: von Neumann 判据阈值。

    Returns:
        ``(imf, n_sift)``，其中 ``n_sift`` 是实际执行的迭代次数（0 表示
        残差已无法继续分解）。
    """
    h = residual.copy()
    h_prev = residual.copy()
    n_sift = 0
    for _ in range(max_sift):
        upper, lower = _envelopes(h)
        if upper is None or lower is None:
            break
        mean_envelope = 0.5 * (upper + lower)
        h_new = h - mean_envelope

        denominator = float(np.dot(h_prev, h_prev))
        sd = float(np.dot(h_prev, h_new) / denominator) if denominator > 0.0 else 0.0

        h_prev = h
        h = h_new
        n_sift += 1

        if not _is_imf(h):
            # IMF 条件尚未满足，继续抽筛。
            continue

        if sd < sd_tolerance:
            break

        norm_h = float(np.linalg.norm(h))
        if norm_h > 0.0 and float(np.linalg.norm(h - h_prev)) / norm_h < CONVERGENCE_TOL:
            # 抽筛已收敛，再迭代也不会改变结果。
            break
    return h, n_sift


def _extract_imfs(
    residual: npt.NDArray,
    max_imfs: int,
    max_sift: int,
    sd_tolerance: float,
) -> tuple[npt.NDArray, npt.NDArray, int]:
    """从残差中逐级提取 IMF。

    Args:
        residual: 待分解序列。
        max_imfs: 最多提取的 IMF 个数。
        max_sift: 每个 IMF 的最大抽筛迭代次数。
        sd_tolerance: von Neumann 判据阈值。

    Returns:
        ``(imfs, trend, n_sift)``，``imfs`` 形状 ``(n_imfs, n)``，
        ``trend`` 为去掉全部 IMF 之后的残差。
    """
    imfs: list[npt.NDArray] = []
    trend = residual.copy()
    n_sift_total = 0
    reference_scale = float(np.std(residual))

    for _ in range(max_imfs):
        max_idx, min_idx = _extrema_indices(trend)
        if max_idx.size < 2 or min_idx.size < 2:
            break
        imf, n_sift = _sift(trend, max_sift, sd_tolerance)
        n_sift_total += n_sift
        if n_sift == 0:
            break
        if float(np.std(imf)) <= 0.0:
            break
        imfs.append(imf)
        trend = trend - imf
        if reference_scale > 0.0 and float(np.std(trend)) <= reference_scale * 1e-12:
            break

    stacked = np.asarray(imfs, dtype=float) if imfs else np.empty((0, residual.size), dtype=float)
    return stacked, trend, n_sift_total


# =============================================================================
# 公开接口
# =============================================================================


def emd(
    values: npt.NDArray,
    max_imfs: int = 10,
    max_sift: int = 50,
    sd_tolerance: float = 0.2,
    times: npt.NDArray | None = None,
) -> EMDResult:
    """经验模态分解（EMD）。

    把一条记录拆成若干本征模态函数与一条残差（趋势项）。IMF 按频率由高到低
    排列，符合 Huang et al. (1998) 的经典结果。

    Args:
        values: 输入序列，一维有限实数，长度不少于 :data:`MIN_SAMPLES`。
        max_imfs: 最多提取的 IMF 个数。
        max_sift: 每个 IMF 的最大抽筛迭代次数。
        sd_tolerance: von Neumann 停止判据阈值。
        times: 可选时间轴，仅随结果返回，不参与计算，单位由调用方决定。

    Returns:
        EMDResult: 分解结果，``imfs`` 形状 ``(n_imfs, n)``，
        ``trend = values - sum(imfs)``。

    Raises:
        DataValidationError: 输入不合法，或参数取值非法。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.emd import emd
        >>> t = np.arange(0.0, 500.0, 0.5)
        >>> signal = np.sin(2 * np.pi * t / 100.0)
        >>> result = emd(signal)
        >>> result.imfs.shape[0] >= 1
        True
    """
    series = _as_series(values)
    if max_imfs < 1:
        raise DataValidationError(_("max_imfs must be at least 1"), details={"max_imfs": int(max_imfs)})
    if max_sift < 1:
        raise DataValidationError(_("max_sift must be at least 1"), details={"max_sift": int(max_sift)})
    if not 0.0 < sd_tolerance <= 1.0:
        raise DataValidationError(
            _("sd_tolerance must lie in (0, 1]; got {0}").format(sd_tolerance),
            details={"sd_tolerance": float(sd_tolerance)},
        )

    imfs, trend, n_sift = _extract_imfs(series, int(max_imfs), int(max_sift), float(sd_tolerance))
    return EMDResult(imfs=imfs, trend=trend, times=times, n_sift=n_sift)


def eemd(
    values: npt.NDArray,
    n_trials: int = 50,
    noise_amplitude: float = 0.2,
    max_imfs: int = 10,
    random_seed: int | None = None,
    max_sift: int = 50,
    sd_tolerance: float = 0.2,
    times: npt.NDArray | None = None,
) -> EMDResult:
    """带噪经验模态分解（EEMD）。

    对原序列反复加白噪声做独立分解，再把同名 IMF 归一化后取集合平均。
    白噪声对每次 IMF 的扰动方向随机，集合平均可以压低噪声残留，使
    周期性成分稳定下来（Wu & Huang 2009）。

    具体做法：

    * 第 ``i`` 次试验抽取一组高斯白噪声 ``w_i``，标准差为
      ``noise_amplitude * std(values)``；
    * 用 ``+w_i`` 与 ``-w_i`` 两个版本分别做 EMD（一对噪声的贡献在集合
      平均中相互抵消，这是 Wu & Huang 的对称噪声配对做法）；
    * 每个 IMF 先做零均值、单位标准差的归一化，再按出现次数取平均，最后
      乘回参与者的平均标准差以恢复幅度尺度；
    * 趋势项取各次试验残差的平均。

    Args:
        values: 输入序列，一维有限实数，长度不少于 :data:`MIN_SAMPLES`。
        n_trials: 集合平均的试验次数。
        noise_amplitude: 噪声标准差与输入序列标准差之比。
        max_imfs: 每个试验最多提取的 IMF 个数。
        random_seed: 随机种子。``None`` 时结果不可复现，
            :func:`utils.statistics_core.make_rng` 会发出 ``RuntimeWarning``。
        max_sift: 每个 IMF 的最大抽筛迭代次数。
        sd_tolerance: von Neumann 停止判据阈值。
        times: 可选时间轴，仅随结果返回，不参与计算，单位由调用方决定。

    Returns:
        EMDResult: 集合平均后的分解结果。

    Raises:
        DataValidationError: 输入不合法，或参数取值非法。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.emd import eemd
        >>> t = np.arange(0.0, 500.0, 0.5)
        >>> signal = np.sin(2 * np.pi * t / 100.0)
        >>> result = eemd(signal, n_trials=5, random_seed=42)
        >>> result.imfs.shape[0] >= 1
        True
    """
    series = _as_series(values)
    if n_trials < 1:
        raise DataValidationError(_("n_trials must be at least 1"), details={"n_trials": int(n_trials)})
    if noise_amplitude < 0.0:
        raise DataValidationError(
            _("noise_amplitude must not be negative"),
            details={"noise_amplitude": float(noise_amplitude)},
        )
    if max_imfs < 1:
        raise DataValidationError(_("max_imfs must be at least 1"), details={"max_imfs": int(max_imfs)})

    rng = make_rng(random_seed, context="EEMD")
    noise_scale = float(noise_amplitude) * float(np.std(series))
    n_samples = series.size

    sum_normalised = np.zeros((int(max_imfs), n_samples), dtype=float)
    sum_scale = np.zeros(int(max_imfs), dtype=float)
    count_members = np.zeros(int(max_imfs), dtype=int)
    count_scales = np.zeros(int(max_imfs), dtype=int)
    trend_accumulator = np.zeros(n_samples, dtype=float)
    n_sift_total = 0

    # NOTE: the loop variable must not be named ``_``: ``_`` is the i18n
    # translator imported above, and a local ``_`` would shadow it for the
    # whole function -- every error path below would then raise
    # UnboundLocalError instead of DataValidationError.
    for _trial in range(int(n_trials)):
        noise = rng.normal(0.0, noise_scale, size=n_samples)
        pair_trend = np.zeros(n_samples, dtype=float)
        for sign in (1.0, -1.0):
            imfs, trend, n_sift = _extract_imfs(
                series + sign * noise,
                int(max_imfs),
                int(max_sift),
                float(sd_tolerance),
            )
            n_sift_total += n_sift
            pair_trend = pair_trend + 0.5 * trend
            for index in range(imfs.shape[0]):
                imf = imfs[index]
                scale = float(np.std(imf))
                if scale <= 0.0:
                    continue
                normalised = (imf - float(np.mean(imf))) / scale
                sum_normalised[index] += normalised
                count_members[index] += 1
                sum_scale[index] += scale
                count_scales[index] += 1
        trend_accumulator += pair_trend

    imfs = np.zeros((0, n_samples), dtype=float)
    # 只有真正出现过成员的 IMF 才有意义：某次试验没能分解出这么多层时，
    # 剩下的行会是一整行零，留在结果里会让下游把“零 IMF”当成一个成分。
    populated = np.flatnonzero(count_members > 0)
    if populated.size:
        n_rows = int(populated.max()) + 1
        mean_normalised = np.zeros((n_rows, n_samples), dtype=float)
        mean_scale = np.zeros(n_rows, dtype=float)
        for index in range(n_rows):
            if count_members[index] > 0:
                mean_normalised[index] = sum_normalised[index] / count_members[index]
            if count_scales[index] > 0:
                mean_scale[index] = sum_scale[index] / count_scales[index]
        imfs = mean_normalised * mean_scale[:, None]

    trend = trend_accumulator / float(n_trials)
    return EMDResult(imfs=imfs, trend=trend, times=times, n_sift=n_sift_total)
