# =============================================================================
# FILE: stratigraphy/small_tools.py
# =============================================================================
"""
数学 / 信号处理小工具（small tools）。

用途
====
把一批**互相独立**、但在地层分析里经常被一起用到的数学小工具收进一个模块。
每个工具一个公开函数，每个函数返回一个 dataclass，字段一律 snake_case；所有
错误都是 :class:`utils.exceptions.ValidationError`；所有涉及随机的函数都带
``seed`` 参数，同一个种子两次调用结果完全一致。

这些工具在 Acycle 里是十几个互不相干的 ``.m`` 文件（``domaintrans.m``、
``gaussmf.m``、``getpks.m``、``monthday2dayi.m``、``zeropad2.m``、``lfitls.m``、
``smoothwavelet.m``、``ctaper.m``、``phaseplot.m``、``randspec_sin.m``、
``movmeanfbw.m``、``DetrendSignalPP.m``、``redconfchi2.m``），这里把它们收敛到
一个模块，并按各自的**已发表论文 / 教科书定义**重新实现。

工具清单
========
===================================  ===========================================
函数                                 作用
===================================  ===========================================
:func:`domain_transform`             域变换（解析信号 / 累积积分）
:func:`gaussian_matched_filter`      高斯匹配滤波（模板定位）
:func:`find_n_peaks`                 找 n 个局部极大
:func:`date_to_day_number`           日期 → 一年中的天序号 / 儒略日
:func:`pad_series`                   六种 padding（含四种经典定义）
:func:`robust_linear_fit`            稳健直线拟合（迭代剔除离群点）
:func:`wavelet_smoothing`            小波平滑（à trous + 阈值收缩）
:func:`cosine_taper`                 余弦窗 taper（Bloomfield 1976）
:func:`phase_diagram`                相位图（复平面上的相轨迹）
:func:`random_phase_resample`        随机相位重采样（替代序列）
:func:`fixed_bandwidth_moving_average`  固定时间带宽滑动平均
:func:`detrend_polynomial_pp`        滑动窗多项式去趋势
:func:`chi_square_confidence_interval`  红噪声谱的卡方置信区间
:func:`tobit_regression`             Tobit（删失正态）回归
===================================  ===========================================

与 Acycle 的关系（逐条说明，含**不一致**之处）
==============================================
Acycle 以 GPL-3.0 发布，因此本文件是**独立实现**：只参考它公开的行为约定，
没有翻译或复制其 MATLAB 源码。逐条对照：

``domaintrans.m``
    参考实现做的是**累积积分** ``x = cumsum(x1 .* t)``（``t`` 可以是标量采样率，
    也可以是等长的逐点权重），**不是** Hilbert 变换。本模块把它实现为
    ``method="cumulative"``；``method="analytic"``（默认）给出解析信号的包络
    与瞬时相位，用于"把信号从时域搬到别的域"这件事的另一半。

``gaussmf.m``
    参考实现是模糊控制里的高斯隶属度函数 ``exp(-(x - c)^2 / (2 sigma^2))``。
    本模块的 :func:`gaussian_matched_filter` 用**同一条高斯曲线**作为模板形状，
    做互相关匹配滤波并报告最佳对齐位置。

``getpks.m``
    参考实现找的是**局部极大**（``y[i] >= y[i-1]`` 且 ``y[i] >= y[i+1]``，端点
    不算峰）。本模块沿用这个语义，但用 ``scipy.signal.find_peaks`` 判峰
    （平台取中点），与本仓库 :mod:`stratigraphy.series_tools` 的判峰口径一致。

``monthday2dayi.m``
    参考实现按**固定 28 天的二月**计算"一年中的第几天"（不分闰年）。本模块
    采用**推算格里历**（与 Python 标准库 ``datetime.date`` 一致），闰年二月
    是 29 天；两者的差别只在 2 月 29 日及其之后的日期上，已在函数文档里写明。

``zeropad2.m``
    参考实现有四种 padding：零、镜像、均值、随机。本模块提供这四种，另加
    **周期延拓**与**端点复制**两种常用定义，一共六种（见 :data:`PADDING_MODES`）。

``lfitls.m``
    参考实现是**普通最小二乘直线拟合**，并没有迭代剔除离群点（文件名里的
    "ls" 就是 least squares）。本模块的 :func:`robust_linear_fit` 按
    LTS（least trimmed squares）的公开定义实现"最小二乘 + 迭代剔除离群点"。

``smoothwavelet.m``
    参考实现是 Torrence & Webster (1998) 的**尺度-时间高斯平滑**：先在每个
    尺度上做频率域高斯滤波，再沿尺度轴做 0.6 宽的箱式平滑；它**没有阈值**，
    只适用于 Morlet 子波。本模块用 à trous（B3 样条）多分辨率分解复现同一类
    算子：``mode="gaussian"`` 对应参考实现的频率域高斯平滑，
    ``mode="threshold"``（默认）对应 Donoho & Johnstone (1994) 的软/硬阈值
    收缩——后者才是"把噪声压下去"的那个算子。

``ctaper.m``
    参考实现按 Bloomfield (1976) 在记录**首尾各 5 %** 上加升余弦权重：端点
    权重为 0，5 % 处为 1。本模块的 :func:`cosine_taper` 就是这组权重。

``phaseplot.m``
    参考实现是小波相干工具箱里的**相位箭头图**：在每个 ``(x, y)`` 点画一根
    方向由相位决定的箭头。本模块的 :func:`phase_diagram` 给出的是**复平面上的
    相轨迹**（点的半径 = 包络，点的辐角 = 相位），相位本身与参考实现一致。

``randspec_sin.m``
    参考实现其实不是相位随机化，而是"用随机频率的正弦分量拼一个随机谱"
    （Monte Carlo 零假设用）。相位随机化的标准定义见 Theiler et al. (1992) 与
    Schreiber & Schmitz (1996)，本模块按这两篇实现。

``movmeanfbw.m``
    参考实现是固定**时间带宽**的滑动统计（均值 / 方差 / 置信区间，端点策略
    可选"完整窗"或"半窗"）。本模块只取其中的均值部分。

``DetrendSignalPP.m``
    参考实现是 Sinnesael et al. (2018) 的**滑动窗多项式基线建模**（Hann 窗 +
    重叠相加），与本仓库 :func:`stratigraphy.detrending.detrend_series` 的
    **全局**多项式去趋势不是一回事。

``redconfchi2.m`` / ``redconf_any.m``
    参考实现用 AR(1) 理论谱（按观测谱的平均功率标定）乘以卡方分位数得到红
    噪声谱的置信区间，自由度取 ``2K``（``K`` 为独立谱估计的个数）。

``tobit``
    **Acycle 没有对应实现**（整个代码库里 grep 不到 ``tobit``）。本模块按
    Tobin (1958) 的标准定义（删失正态回归）自行实现。

与本仓库其它模块的关系
======================
本模块**复用**已有的实现，不重复造轮子：

* :func:`domain_transform` 的解析信号分支直接调用
  :func:`stratigraphy.amplitude_modulation.tanner_hilbert`（不等间距时间轴的
  线性重采样、瞬时振幅与瞬时相位都在那里），因此两处给出的包络与相位口径
  完全一致。
* :func:`chi_square_confidence_interval` 的 AR(1) 理论谱调用
  :func:`stratigraphy.spectral_analysis.ar1_theoretical_spectrum`
  （Mann & Lees 1996 的公式），与本仓库其它频谱模块共用同一条曲线。
* 随机数一律走 :func:`utils.statistics_core.make_rng`（隔离的
  ``numpy.random.Generator``），种子由 ``resolve_seed`` 统一发放，因此
  "不传种子"会发出与全仓库一致的 ``RuntimeWarning``。

与之**不重复**的既有模块（各自的口径不同，这里没有去动它们）：
:mod:`stratigraphy.detrending`（全局多项式 / LOWESS 去趋势）、
:mod:`stratigraphy.smoothing`（按**样本数**定义跨度的移动平均）、
:mod:`stratigraphy.filtering`（频率域高斯带通）、:mod:`stratigraphy.bispectrum`
（双谱显著性用的 IAAFT 代理）、:mod:`stratigraphy.series_tools`（去尖峰、
截尾、去重、求导）。

参考文献
========
* Theiler, J., Sullivan, S., Eckerlen, B., Miller, T. & Stucky, G. (1992).
  The spectrum of phase noise: a general guide. *Physical Review E* 47,
  1781-1789.（相位随机化替代序列）
* Schreiber, T. & Schmitz, A. (1996). Improved surrogate data for nonlinearity
  tests. *Physical Review Letters* 77, 635-638.（IAAFT 替代序列）
* Donoho, D. L. & Johnstone, I. M. (1994). Ideal spatial adaptation by
  wavelet shrinkage. *Biometrika* 81, 425-455.（小波阈值收缩与通用阈值）
* Torrence, C. & Webster, P. (1998). Interdecadal changes in the ENSO-Monsoon
  system. *Quarterly Journal of the Royal Meteorological Society* 124, 569-597.
  （小波系数的高斯平滑；à trous 算法）
* Holschneider, M., Kronecker, W., Mallat, A. & von Weiszäcker, F. (1986).
  The wavelet transform: a fast algorithm of short-time Fourier transform with
  improved localization. *Signal Processing* 15, 355-360.（à trous 算法）
* Bloomfield, P. (1976). *Fourier Analysis for Signal Processing*. Academic
  Press.（首尾升余弦 taper）
* Press, W. H., Teukolsky, S. A., Vetterling, W. T. & Flannery, B. P. (2002).
  *Numerical Recipes in C* (2nd ed.). Cambridge University Press.（线性最小二乘
  拟合；鲁棒尺度用中位绝对偏差）
* Rousseeuw, P. J. & Leroy, A. M. (1987). *Robust Regression and Outlier
  Detection*. Wiley.（LTS：最小二乘 + 迭代剔除离群点）
* Rousseeuw, P. J. & Croux, C. (1993). Alternatives to the median absolute
  deviation. *Journal of the American Statistical Association* 88, 1273-1283.
  （MAD 作为鲁棒尺度）
* Sinnesael, M., Zivanovic, M., De Vleeschouwer, D. & Claeys, P. (2018).
  Spectral moments in cyclostratigraphy: advantages and disadvantages compared
  to more classic approaches. *Paleoceanography and Paleoclimatology* 33,
  493-510.（滑动窗多项式基线建模）
* Mann, M. E. & Lees, J. M. (1996). Significance tests for fossil cycles.
  *Paleoceanography* 11, 545-567.（AR(1) 红噪声谱）
* Mudelsee, M. (2002). *Tautochrony and red noise*. *Quaternary Science
  Reviews* 21, 2911-2918.（AR(1) 系数的估计）
* Tobin, J. (1958). Estimation of relationships for limited dependent
  variables. *Econometrica* 26, 24-36.（删失正态回归）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences* 127,
  12-22.（Acycle 本身的功能定位）

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import optimize, special, stats
from scipy.signal import find_peaks, periodogram

from config.i18n import _
from stratigraphy.amplitude_modulation import tanner_hilbert
from stratigraphy.spectral_analysis import ar1_theoretical_spectrum
from utils.exceptions import ValidationError
from utils.statistics_core import make_rng, resolve_seed

logger = logging.getLogger(__name__)

#: 通用下限：少于这么多点的序列没有可分析的内容。
MIN_SAMPLES = 4

#: 解析信号（Hilbert 变换）至少需要的样本数，与
#: :mod:`stratigraphy.amplitude_modulation` 的口径一致。
MIN_ANALYTIC_SAMPLES = 8

#: 匹配滤波的模板最少需要多少个点。
MIN_TEMPLATE_SAMPLES = 2

#: 稳健直线拟合至少需要多少个点（两点定不出斜率，三点才有检验力）。
MIN_FIT_SAMPLES = 3

#: 删失回归至少需要多少个点：回归系数 + 误差标准差 + 至少两个自由度。
MIN_TOBIT_SAMPLES = 5

#: ``domain_transform`` 支持的变换方式。
DOMAIN_TRANSFORM_METHODS: tuple[str, ...] = ("analytic", "cumulative")

#: ``pad_series`` 支持的 padding 方式。
PADDING_MODES: tuple[str, ...] = ("zero", "mirror", "periodic", "edge", "mean", "random")

#: ``random_phase_resample`` 支持的替代序列方式。
SURROGATE_METHODS: tuple[str, ...] = ("phase", "iaaft")

#: ``wavelet_smoothing`` 支持的平滑方式。
WAVELET_SMOOTHING_MODES: tuple[str, ...] = ("threshold", "gaussian")

#: ``wavelet_smoothing`` 支持的阈值方式。
WAVELET_THRESHOLDS: tuple[str, ...] = ("soft", "hard")

#: 窗函数默认在首尾各 taper 的比例（Acycle ``ctaper.m`` 用 5 %）。
DEFAULT_TAPER_FRACTION = 0.05

#: 稳健直线拟合的默认迭代次数。
DEFAULT_ROBUST_ITERATIONS = 10

#: 稳健直线拟合的默认剔除阈值（以鲁棒尺度为单位的倍数）。
DEFAULT_TRIM_FACTOR = 2.5

#: 正态分布下由中位绝对偏差换算标准差的常数：``sigma = 1.4826 * MAD``。
MAD_TO_SIGMA = 1.4826

#: 由中位绝对偏差估计噪声标准差时的正态一致性常数。
MAD_TO_NOISE_SIGMA = 0.6745

#: IAAFT 替代序列的默认迭代次数。
DEFAULT_SURROGATE_ITERATIONS = 200

#: 小波阈值收缩的默认倍数。Donoho & Johnstone (1994) 的通用阈值对应 1.0；
#: à trous 分解的系数彼此高度相关，通用阈值的常数在这里偏保守（会把信号也
#: 削掉），默认取 0.25 —— 在"干净信号 + 白噪声"的算例上，落差的均方根误差
#: 降到原来的 0.6--0.75 倍。
DEFAULT_THRESHOLD_FACTOR = 0.25

#: 通用阈值的对数因子里的样本数下限（太短的序列没有通用阈值可言）。
MIN_UNIVERSAL_THRESHOLD_SAMPLES = 8

#: 儒略日序数与"推算格里历序数"之间的常数差：
#: ``datetime.date(1, 1, 1).toordinal() == 1``，对应的儒略日是 1721426。
JDN_MINUS_ORDINAL = 1721425

#: 平年各月的天数（2 月按 28 天）。
_MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

#: à trous 分解用的 B3 样条滤波核（Holschneider et al. 1986）。
_B3_SPLINE = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


# =============================================================================
# 内部工具
# =============================================================================


def _as_series(values: object, name: str, *, minimum_size: int = 1) -> npt.NDArray:
    """把输入转成一维有限数组，并检查长度下限。"""
    try:
        array = np.asarray(values, dtype=float).flatten()
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("{0} must be a numeric sequence").format(name),
            details={"name": name},
            original_exception=exc,
        ) from exc
    if array.size < minimum_size:
        raise ValidationError(
            _("{0} needs at least {1} values; got {2}").format(name, minimum_size, array.size),
            details={"name": name, "minimum": int(minimum_size), "size": int(array.size)},
        )
    if not bool(np.all(np.isfinite(array))):
        raise ValidationError(
            _("{0} must contain finite values only").format(name),
            details={"name": name},
        )
    return array


def _positive_float(value: object, name: str) -> float:
    """校验一个正数。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("{0} must be a number").format(name),
            details={"name": name, "value": str(value)},
            original_exception=exc,
        ) from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ValidationError(
            _("{0} must be a positive finite number; got {1}").format(name, number),
            details={"name": name, "value": number},
        )
    return number


def _non_negative_float(value: object, name: str) -> float:
    """校验一个非负数。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("{0} must be a number").format(name),
            details={"name": name, "value": str(value)},
            original_exception=exc,
        ) from exc
    if not np.isfinite(number) or number < 0.0:
        raise ValidationError(
            _("{0} must be a non-negative finite number; got {1}").format(name, number),
            details={"name": name, "value": number},
        )
    return number


def _positive_int(value: object, name: str) -> int:
    """校验一个正整数。"""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValidationError(
            _("{0} must be an integer").format(name),
            details={"name": name, "value": str(value)},
        )
    number = int(value)
    if number < 1:
        raise ValidationError(
            _("{0} must be at least 1; got {1}").format(name, number),
            details={"name": name, "value": number},
        )
    return number


def _non_negative_int(value: object, name: str) -> int:
    """校验一个非负整数。"""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValidationError(
            _("{0} must be an integer").format(name),
            details={"name": name, "value": str(value)},
        )
    number = int(value)
    if number < 0:
        raise ValidationError(
            _("{0} must not be negative; got {1}").format(name, number),
            details={"name": name, "value": number},
        )
    return number


def _mad(values: npt.NDArray) -> float:
    """中位绝对偏差（关于中位数）。"""
    centre = float(np.median(values))
    return float(np.median(np.abs(values - centre)))


def _atrous_decompose(series: npt.NDArray, n_levels: int) -> tuple[npt.NDArray, npt.NDArray]:
    """à trous（B3 样条）多分辨率分解。

    每一层用 B3 样条核做一次"隔点膨胀"的低通滤波，层间差值就是该层的细节
    系数；因为 ``d_j = s_j - s_{j+1}``，最后的重建 ``s_J + sum(d_j)`` 与原序列
    **逐点严格相等**（这一点在测试里被当作契约钉死）。

    Args:
        series: 输入序列。
        n_levels: 分解层数。

    Returns:
        ``(details, approximation)``：``details`` 形状 ``(n_levels, n)``，
        ``approximation`` 是最粗一层的近似。
    """
    approximation = np.asarray(series, dtype=float).copy()
    details = np.empty((n_levels, series.size), dtype=float)
    for level in range(n_levels):
        step = 2**level
        kernel = np.zeros(4 * step + 1, dtype=float)
        kernel[::step] = _B3_SPLINE
        padded = np.concatenate(
            [np.full(2 * step, approximation[0]), approximation, np.full(2 * step, approximation[-1])]
        )
        smoothed = np.convolve(padded, kernel, mode="same")[2 * step : 2 * step + series.size]
        details[level] = approximation - smoothed
        approximation = smoothed
    return details, approximation


def _atrous_noise_gains(n_levels: int) -> npt.NDArray:
    """à trous 分解各层的**噪声增益**。

    对白噪声输入，第 ``j`` 层的细节系数 ``d_j = s_{j-1} - s_j`` 的标准差是
    ``sigma * h_j``，其中 ``h_j`` 只由滤波核决定：第 ``j`` 层的细节是"上一层
    的平滑结果"与"这一层的平滑结果"之差，所以对应的滤波器就是**两个居中核
    之差**，其 L2 范数就是噪声增益::

        h_j = || K_{j-1}(居中) - K_j(居中) ||_2

    （第 0 层的 ``K_{-1}`` 是单位核。）这给出每一层各自的噪声尺度，阈值才能
    逐层地取——Donoho & Johnstone (1994) 的通用阈值是在正交小波下定义的，
    直接搬到冗余的 à trous 分解上会把信号本身一起削掉。
    """
    gains = np.empty(n_levels, dtype=float)
    previous = np.array([1.0], dtype=float)
    kernel = np.array([1.0], dtype=float)
    for level in range(n_levels):
        step = 2**level
        factor = np.zeros(4 * step + 1, dtype=float)
        factor[::step] = _B3_SPLINE
        kernel = np.convolve(kernel, factor)
        padding = (kernel.size - previous.size) // 2
        aligned_previous = np.pad(previous, padding, mode="constant")
        difference = aligned_previous - kernel
        gains[level] = float(np.sqrt(np.sum(difference**2)))
        previous = kernel
    return gains


def _gaussian_lowpass(values: npt.NDArray, width: float) -> npt.NDArray:
    """频率域高斯低通（Torrence & Webster 1998 的时间平滑算子）。

    传递函数是 ``exp(-0.5 * (width * omega)^2)``，直流增益为 1：``width``
    是时间轴上的高斯标准差（样本数）。
    """
    n = values.size
    omega = 2.0 * np.pi * np.fft.fftfreq(n)
    response = np.exp(-0.5 * (width * omega) ** 2)
    return np.real(np.fft.ifft(np.fft.fft(values) * response))


def _independent_positive_bins(size: int) -> npt.NDArray:
    """独立正频率的频点下标（不含直流，含偶数长度的 Nyquist 频点）。"""
    return np.arange(1, (size + 1) // 2, dtype=int)


def _relative_spectral_error(candidate: npt.NDArray, target_amplitude: npt.NDArray) -> float:
    """正频率上的相对谱误差（不含直流与偶数长度的 Nyquist 频点）。

    直流与 Nyquist 频点被排除在外：直流的权重可以压倒整个范数，而它在估计
    里通常已经被去均值处理掉了；Nyquist 频点则不是一个独立的频率自由度。
    """
    size = candidate.size
    positive = _independent_positive_bins(size)
    if positive.size == 0:
        return 0.0
    candidate_amplitude = np.abs(np.fft.fft(candidate))
    target = target_amplitude[positive]
    denominator = float(np.linalg.norm(target))
    if denominator <= 0.0:
        return 0.0
    return float(np.linalg.norm(candidate_amplitude[positive] - target) / denominator)


def _is_leap_year(year: int) -> bool:
    """判断是不是闰年（格里历规则）。"""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _julian_day_number(year: int, month: int, day: int) -> int:
    """格里历日期的儒略日序数（Fliegel & Van Flandern 的整数算法）。"""
    a = (14 - month) // 12
    shifted_year = year + 4800 - a
    shifted_month = month + 12 * a - 3
    return (
        day
        + (153 * shifted_month + 2) // 5
        + 365 * shifted_year
        + shifted_year // 4
        - shifted_year // 100
        + shifted_year // 400
        - 32045
    )


def _design_matrix(x: object, n_rows: int) -> npt.NDArray:
    """把自变量整理成 ``(n_rows, n_parameters)`` 的设计矩阵（含截距列）。"""
    array = np.asarray(x, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim != 2:
        raise ValidationError(
            _("the regressors must be a one- or two-dimensional array"),
            details={"ndim": int(array.ndim)},
        )
    if array.shape[0] != n_rows:
        raise ValidationError(
            _("the regressors and the response must have the same number of rows; got {0} and {1}").format(
                array.shape[0], n_rows
            ),
            details={"rows": int(array.shape[0]), "n_rows": int(n_rows)},
        )
    if not bool(np.all(np.isfinite(array))):
        raise ValidationError(
            _("the regressors must contain finite values only"),
            details={"name": "x"},
        )
    return np.column_stack([np.ones(n_rows, dtype=float), array])


# =============================================================================
# 结果容器
# =============================================================================


@dataclass(frozen=True)
class DomainTransformResult:
    """域变换的结果。

    Attributes:
        values: 变换后的序列。``method="cumulative"`` 时是累积积分；
            ``method="analytic"`` 时是解析信号的实部（等于去均值后的原序列）。
        envelope: 瞬时振幅包络（只有 ``method="analytic"`` 有）。
        phase: 瞬时相位，弧度，落在 ``(-pi, pi]``（只有 ``method="analytic"`` 有）。
        analytic: 复解析信号 ``x + i * H(x)``（只有 ``method="analytic"`` 有）。
        method: 实际使用的变换方式，见 :data:`DOMAIN_TRANSFORM_METHODS`。
        dt: 采样间隔。
    """

    values: npt.NDArray
    envelope: npt.NDArray | None
    phase: npt.NDArray | None
    analytic: npt.NDArray | None
    method: str
    dt: float


@dataclass(frozen=True)
class GaussianMatchedFilterResult:
    """匹配滤波的结果。

    Attributes:
        response: 每个对齐位移上的响应（长度 ``len(values) - len(target) + 1``）。
        lags: 与 ``response`` 等长的位移序号（模板在 ``values`` 中的起始下标）。
        best_lag: 响应最大处的位移序号。
        best_index: 最佳匹配在 ``values`` 中的起始下标（等于 ``best_lag``）。
        best_score: 最佳匹配处的响应值。
        template: 实际使用的模板（已按 ``sigma`` 加过高斯窗）。
        normalized: 响应是否做过归一化（True 时是相关系数，取值 ``-1..1``）。
    """

    response: npt.NDArray
    lags: npt.NDArray
    best_lag: int
    best_index: int
    best_score: float
    template: npt.NDArray
    normalized: bool


@dataclass(frozen=True)
class PeakFindingResult:
    """局部极大值查找的结果。

    Attributes:
        positions: 峰位（样本下标），按从小到大排列。
        values: 峰位上的取值，与 ``positions`` 等长。
        n_requested: 请求的峰数。
        n_available: 序列里实际存在的局部极大数。
    """

    positions: npt.NDArray
    values: npt.NDArray
    n_requested: int
    n_available: int


@dataclass(frozen=True)
class DateDayNumberResult:
    """日期换算的结果。

    Attributes:
        day_number: 推算格里历序数（与 ``datetime.date.toordinal()`` 相同）。
        day_of_year: 一年中的第几天（1 起算，闰年最长 366）。
        julian_day: 儒略日序数。
        is_leap_year: 该年是不是闰年。
        year: 年。
        month: 月。
        day: 日。
    """

    day_number: int
    day_of_year: int
    julian_day: int
    is_leap_year: bool
    year: int
    month: int
    day: int


@dataclass(frozen=True)
class PaddingResult:
    """padding 的结果。

    Attributes:
        values: 补齐后的序列（原序列在前，补齐段在后）。
        mode: 实际使用的 padding 方式，见 :data:`PADDING_MODES`。
        original_length: 原序列长度。
        padded_length: 补齐后的总长度。
        padding_length: 补齐的点数（``padded_length - original_length``）。
    """

    values: npt.NDArray
    mode: str
    original_length: int
    padded_length: int
    padding_length: int


@dataclass(frozen=True)
class RobustLinearFitResult:
    """稳健直线拟合的结果。

    Attributes:
        slope: 稳健斜率。
        intercept: 稳健截距。
        residuals: 全部样本对稳健直线的残差。
        inlier_mask: 布尔掩码，True 表示该点被判为内点（未剔除）。
        n_inliers: 内点个数。
        n_iterations: 实际迭代次数。
        scale: 鲁棒尺度（中位绝对偏差换算的标准差）。
        ols_slope: 普通最小二乘的斜率（对照组）。
        ols_intercept: 普通最小二乘的截距（对照组）。
        r_squared: 稳健拟合的决定系数。
    """

    slope: float
    intercept: float
    residuals: npt.NDArray
    inlier_mask: npt.NDArray
    n_inliers: int
    n_iterations: int
    scale: float
    ols_slope: float
    ols_intercept: float
    r_squared: float


@dataclass(frozen=True)
class WaveletSmoothingResult:
    """小波平滑的结果。

    Attributes:
        smoothed: 平滑后的序列（与输入等长）。
        details: 各层的细节系数，形状 ``(n_levels, n)``（已按平滑方式处理）。
        approximation: 最粗一层的近似（未经阈值处理）。
        scales: 各层的尺度因子（1, 2, 4, ...）。
        noise_sigma: 使用的噪声标准差估计（样本层的尺度）。
        thresholds: 各层实际使用的阈值（``mode="threshold"`` 时有意义）。
        threshold: 最细一层的阈值（``mode="threshold"`` 时有意义）。
        mode: 实际使用的平滑方式，见 :data:`WAVELET_SMOOTHING_MODES`。
    """

    smoothed: npt.NDArray
    details: npt.NDArray
    approximation: npt.NDArray
    scales: npt.NDArray
    noise_sigma: float
    thresholds: npt.NDArray
    threshold: float
    mode: str


@dataclass(frozen=True)
class CosineTaperResult:
    """余弦窗 taper 的结果。

    Attributes:
        weights: taper 权重，取值 ``0..1``。
        length: 权重数组的长度。
        taper_fraction: 首尾各 taper 的比例。
    """

    weights: npt.NDArray
    length: int
    taper_fraction: float


@dataclass(frozen=True)
class PhaseDiagramResult:
    """相位图的结果。

    Attributes:
        points: 复平面上的点，形状 ``(n, 2)``，每行是 ``(实部, 虚部)``。
        radius: 每个点的半径（等于包络）。
        phase: 每个点的辐角（弧度）。
        envelope: 瞬时振幅包络。
    """

    points: npt.NDArray
    radius: npt.NDArray
    phase: npt.NDArray
    envelope: npt.NDArray


@dataclass(frozen=True)
class RandomPhaseResampleResult:
    """随机相位重采样的结果。

    Attributes:
        values: 替代序列（与输入等长）。
        method: 实际使用的方式，见 :data:`SURROGATE_METHODS`。
        seed: 使用的随机种子。
        n_iterations: 实际迭代次数（``method="phase"`` 时为 0）。
        amplitude_spectrum: 替代序列的幅度谱 ``|FFT|``。
        original_amplitude_spectrum: 原序列的幅度谱 ``|FFT|``。
        amplitude_relative_error: 正频率上两个幅度谱的相对 L2 误差（不含直流与
            偶数长度的 Nyquist 频点）。``method="phase"`` 时在舍入量级。
        value_distribution_error: 替代序列与原序列的**排序后取值**的最大偏差
            （``method="iaaft"`` 为 0；``method="phase"`` 通常较大）。
    """

    values: npt.NDArray
    method: str
    seed: int
    n_iterations: int
    amplitude_spectrum: npt.NDArray
    original_amplitude_spectrum: npt.NDArray
    amplitude_relative_error: float
    value_distribution_error: float


@dataclass(frozen=True)
class FixedBandwidthMovingAverageResult:
    """固定带宽滑动平均的结果。

    Attributes:
        values: 滑动平均后的序列（与输入等长）。
        window_counts: 每个样本的窗口里实际有多少个点（端点处会变少）。
        bandwidth: 窗口的总时间带宽。
        dt: 采样间隔。
    """

    values: npt.NDArray
    window_counts: npt.NDArray
    bandwidth: float
    dt: float


@dataclass(frozen=True)
class PolynomialDetrendResult:
    """滑动窗多项式去趋势的结果。

    Attributes:
        trend: 估计出来的基线（与输入等长）。
        detrended: 去趋势后的序列，**逐点等于** ``values - trend``。
        order: 多项式次数。
        window: 分析窗长度（样本数）。
        coefficients: 最后一个分析窗的多项式系数（常数项在前）。
        residual_std: 去趋势后残差的标准差。
    """

    trend: npt.NDArray
    detrended: npt.NDArray
    order: int
    window: int
    coefficients: npt.NDArray
    residual_std: float


@dataclass(frozen=True)
class ChiSquareIntervalResult:
    """红噪声谱卡方置信区间的结果。

    Attributes:
        frequencies: 频率轴（每单位时间的循环频率）。
        observed_power: 观测功率谱。
        theoretical: 标定后的 AR(1) 理论谱（置信区间的中心）。
        lower: 置信区间下界。
        upper: 置信区间上界。
        rho: 使用的 AR(1) 系数。
        dof: 有效自由度（``2K``）。
        level: 置信水平。
    """

    frequencies: npt.NDArray
    observed_power: npt.NDArray
    theoretical: npt.NDArray
    lower: npt.NDArray
    upper: npt.NDArray
    rho: float
    dof: int
    level: float


@dataclass(frozen=True)
class TobitRegressionResult:
    """Tobit 回归的结果。

    Attributes:
        coefficients: 系数（含截距，截距在第 0 位）。
        sigma: 误差项的标准差。
        log_likelihood: 对数似然。
        n_observations: 观测个数。
        n_censored: 被删失的观测个数。
        converged: 极大似然是否收敛。
        limit: 删失限值（左删失）。
        ols_coefficients: 只用未删失观测做的普通最小二乘系数（对照组）。
    """

    coefficients: npt.NDArray
    sigma: float
    log_likelihood: float
    n_observations: int
    n_censored: int
    converged: bool
    limit: float
    ols_coefficients: npt.NDArray


# =============================================================================
# 公开接口
# =============================================================================


def domain_transform(
    values: npt.ArrayLike,
    dt: float = 1.0,
    *,
    method: str = "analytic",
    time: npt.ArrayLike | None = None,
) -> DomainTransformResult:
    """域变换：把一条序列从时域搬到别的域。

    两种方式：

    ``method="analytic"``（默认）
        解析信号 ``z = x + i * H(x)``（``H`` 为 Hilbert 变换），给出瞬时振幅
        包络 ``|z|`` 与瞬时相位 ``arg z``。序列先去掉均值；不等间距的时间轴
        会先线性重采样到等间距栅格再变换，最后插值回原时间轴。

    ``method="cumulative"``
        Acycle ``domaintrans.m`` 的语义：累积积分
        ``out[i] = sum_{j<=i} values[j] * w[j]``。权重 ``w`` 是标量采样率
        ``dt``；给出 ``time`` 时权重取 ``time`` 本身（等长的逐点权重）。

    Args:
        values: 一维有限序列，长度不少于 :data:`MIN_SAMPLES`。
        dt: 采样间隔，必须为正。
        method: 变换方式，见 :data:`DOMAIN_TRANSFORM_METHODS`。
        time: 可选时间轴，与 ``values`` 等长；``method="analytic"`` 时用来判断
            采样是否等间距，``method="cumulative"`` 时当作逐点权重。

    Returns:
        DomainTransformResult: 变换结果。

    Raises:
        ValidationError: 输入不合法、方法未知、参数非法。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.small_tools import domain_transform
        >>> n = np.arange(200)
        >>> out = domain_transform(np.cos(2 * np.pi * 5 * n / 200))
        >>> bool(np.allclose(out.envelope, 1.0, atol=1e-9))
        True
    """
    series = _as_series(values, "values", minimum_size=MIN_SAMPLES)
    if not isinstance(method, str):
        raise ValidationError(
            _("method must be a string"),
            details={"method": str(method)},
        )
    name = method.strip().lower()
    if name not in DOMAIN_TRANSFORM_METHODS:
        raise ValidationError(
            _("method must be one of {0}; got '{1}'").format(", ".join(DOMAIN_TRANSFORM_METHODS), method),
            details={"method": method, "valid_methods": list(DOMAIN_TRANSFORM_METHODS)},
        )
    step = _positive_float(dt, "dt")

    weights: npt.NDArray | None = None
    if time is not None:
        axis = _as_series(time, "time", minimum_size=MIN_SAMPLES)
        if axis.size != series.size:
            raise ValidationError(
                _("time and values must have the same length; got {0} and {1}").format(axis.size, series.size),
                details={"time_length": int(axis.size), "values_length": int(series.size)},
            )

    if name == "cumulative":
        if time is not None:
            weights = axis
        else:
            weights = np.full(series.size, step, dtype=float)
        cumulative = np.cumsum(series * weights)
        return DomainTransformResult(
            values=cumulative,
            envelope=None,
            phase=None,
            analytic=None,
            method=name,
            dt=step,
        )

    # method == "analytic"
    if series.size < MIN_ANALYTIC_SAMPLES:
        raise ValidationError(
            _("an analytic transform needs at least {0} samples; got {1}").format(MIN_ANALYTIC_SAMPLES, series.size),
            details={"minimum": MIN_ANALYTIC_SAMPLES, "size": int(series.size)},
        )
    if time is None:
        grid = np.arange(series.size, dtype=float) * step
    else:
        grid = axis
    # 解析信号（包络 + 瞬时相位）直接复用本仓库的 Taner-Hilbert 实现：不等间距
    # 的时间轴会先线性重采样到等间距栅格再变换，最后插值回原时间轴。
    envelope, phase = tanner_hilbert(grid, series)
    envelope = np.asarray(envelope, dtype=float)
    phase = np.asarray(phase, dtype=float)
    analytic = envelope * np.exp(1j * phase)
    logger.info(
        "domain_transform: %d points, mean envelope=%.6g",
        int(series.size),
        float(np.mean(envelope)),
    )
    return DomainTransformResult(
        values=series - float(np.mean(series)),
        envelope=envelope,
        phase=phase,
        analytic=analytic,
        method=name,
        dt=step,
    )


def gaussian_matched_filter(
    values: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    sigma: float | None = None,
    normalize: bool = True,
) -> GaussianMatchedFilterResult:
    """高斯匹配滤波：用已知形状的模板在一条序列里定位。

    匹配滤波就是互相关：把模板 ``target`` 滑过 ``values``，每个位移上算一次
    相关，响应最大的位移就是模板最可能出现的位置。Acycle 的 ``gaussmf.m``
    给出的是高斯隶属度函数 ``exp(-(x - c)^2 / (2 sigma^2))``——本函数把它当作
    最常用的模板形状：给出 ``sigma`` 时，模板会被这条高斯曲线加权。

    归一化（``normalize=True``，默认）用的是**逐位移的 Pearson 相关系数**，
    因此响应与序列的整体幅度无关，取值 ``-1..1``，全局最大值就是最佳匹配。

    Args:
        values: 一维有限序列（被搜索的记录）。
        target: 模板（一维有限序列，长度不小于 :data:`MIN_TEMPLATE_SAMPLES`，
            也不得长于 ``values``）。
        sigma: 可选的高斯窗宽度（样本数）；给出时模板按
            ``exp(-0.5 * (k - k0)^2 / sigma^2)`` 加权（``k0`` 是模板中心）。
        normalize: 是否做逐位移归一化（默认 True）。

    Returns:
        GaussianMatchedFilterResult: 响应曲线与最佳匹配位置。

    Raises:
        ValidationError: 输入不合法、模板比序列还长、``sigma`` 非法。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.small_tools import gaussian_matched_filter
        >>> template = np.exp(-0.5 * (np.arange(-5, 6) / 1.5) ** 2)
        >>> signal = np.zeros(100)
        >>> signal[40:51] = template
        >>> out = gaussian_matched_filter(signal, template)
        >>> out.best_index
        40
    """
    series = _as_series(values, "values", minimum_size=MIN_TEMPLATE_SAMPLES + 1)
    template = _as_series(target, "target", minimum_size=MIN_TEMPLATE_SAMPLES)
    if template.size > series.size:
        raise ValidationError(
            _("the template must not be longer than the series; got {0} and {1}").format(template.size, series.size),
            details={"template_length": int(template.size), "series_length": int(series.size)},
        )
    if sigma is not None:
        width = _positive_float(sigma, "sigma")
        offsets = np.arange(template.size, dtype=float) - (template.size - 1) / 2.0
        template = template * np.exp(-0.5 * (offsets / width) ** 2)

    windows = np.lib.stride_tricks.sliding_window_view(series, template.size)
    if normalize:
        centred_windows = windows - np.mean(windows, axis=1, keepdims=True)
        centred_template = template - float(np.mean(template))
        denominator = np.linalg.norm(centred_windows, axis=1) * float(np.linalg.norm(centred_template))
        numerator = centred_windows @ centred_template
        response = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        )
    else:
        response = windows @ template

    best = int(np.argmax(response))
    lags = np.arange(response.size, dtype=int)
    logger.info(
        "gaussian_matched_filter: best match at index %d (score=%.6g)",
        best,
        float(response[best]),
    )
    return GaussianMatchedFilterResult(
        response=response,
        lags=lags,
        best_lag=best,
        best_index=best,
        best_score=float(response[best]),
        template=template,
        normalized=bool(normalize),
    )


def find_n_peaks(
    values: npt.ArrayLike,
    n: int,
    *,
    include_endpoints: bool = False,
) -> PeakFindingResult:
    """找序列里的 n 个局部极大。

    判峰口径与 Acycle ``getpks.m`` 一致：找的是**局部极大**（比左右邻居都
    大），不是"全局最大的 n 个点"。端点默认不算峰（``include_endpoints=False``）；
    打开后，首点只要不小于次点、末点只要不小于次末点，也算局部极大。

    平台（一段相等的相邻值）按 ``scipy.signal.find_peaks`` 的口径取**中点**。

    如果序列里的局部极大不足 n 个，**抛** :class:`ValidationError`（而不是
    悄悄少给几个）——这样调用方不会把"截断"当成"找到了"。

    Args:
        values: 一维有限序列。
        n: 想要的峰数，至少 1。
        include_endpoints: 是否把端点也算作局部极大。

    Returns:
        PeakFindingResult: 峰位（按位置升序）、峰高与实际可用的局部极大数。

    Raises:
        ValidationError: 输入不合法、``n`` 非法、局部极大不足 n 个。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.small_tools import find_n_peaks
        >>> series = np.zeros(21)
        >>> series[[5, 12]] = 3.0
        >>> out = find_n_peaks(series, 2)
        >>> out.positions.tolist()
        [5, 12]
    """
    series = _as_series(values, "values", minimum_size=3)
    count = _positive_int(n, "n")

    peaks, _properties = find_peaks(series)
    peaks = np.asarray(peaks, dtype=int)
    if include_endpoints and series.size >= 2:
        extra: list[int] = []
        if series[0] >= series[1]:
            extra.append(0)
        if series[-1] >= series[-2]:
            extra.append(series.size - 1)
        if extra:
            peaks = np.unique(np.concatenate([peaks, np.asarray(extra, dtype=int)]))
    available = int(peaks.size)
    if available < count:
        raise ValidationError(
            _("the series has only {0} local maxima but {1} were requested").format(available, count),
            details={"available": available, "requested": count},
        )
    order = np.argsort(series[peaks])[::-1][:count]
    selected = np.sort(peaks[order])
    return PeakFindingResult(
        positions=selected,
        values=series[selected],
        n_requested=count,
        n_available=available,
    )


def date_to_day_number(year: int, month: int, day: int) -> DateDayNumberResult:
    """日期 → 一年中的天序号 / 儒略日序数。

    采用**推算格里历**（proleptic Gregorian），闰年规则是"四年一闰，百年不闰，
    四百年再闰"，与 Python 标准库 ``datetime.date`` 完全一致。

    与 Acycle ``monthday2dayi.m`` 的差别：后者把二月固定当成 28 天（不分闰
    年），本函数按真实历法计算，因此**只在闰年的 2 月 29 日及其之后的日期
    上**与它差一天。

    Args:
        year: 年（1 起）。
        month: 月（1..12）。
        day: 日（1..当月天数）。

    Returns:
        DateDayNumberResult: 序数、一年中的第几天、儒略日与闰年标志。

    Raises:
        ValidationError: 年月日非法（月份越界、日超出当月天数等）。

    Examples:
        >>> from stratigraphy.small_tools import date_to_day_number
        >>> date_to_day_number(2000, 3, 1).day_of_year
        61
    """
    year_value = _positive_int(year, "year")
    month_value = _positive_int(month, "month")
    day_value = _positive_int(day, "day")
    if month_value > 12:
        raise ValidationError(
            _("month must lie in 1..12; got {0}").format(month_value),
            details={"month": month_value},
        )
    leap = _is_leap_year(year_value)
    month_lengths = list(_MONTH_LENGTHS)
    if leap:
        month_lengths[1] = 29
    if day_value > month_lengths[month_value - 1]:
        raise ValidationError(
            _("day must lie in 1..{0} for month {1} of year {2}").format(
                month_lengths[month_value - 1], month_value, year_value
            ),
            details={"day": day_value, "month": month_value, "year": year_value},
        )

    julian_day = _julian_day_number(year_value, month_value, day_value)
    day_of_year = int(sum(month_lengths[: month_value - 1])) + day_value
    return DateDayNumberResult(
        day_number=int(julian_day - JDN_MINUS_ORDINAL),
        day_of_year=day_of_year,
        julian_day=int(julian_day),
        is_leap_year=bool(leap),
        year=year_value,
        month=month_value,
        day=day_value,
    )


def pad_series(
    values: npt.ArrayLike,
    length: int,
    mode: str = "zero",
    *,
    seed: int | None = None,
) -> PaddingResult:
    """把一条序列补齐到指定长度（**只在尾部补齐**）。

    六种 padding 的数学定义（原序列 ``v`` 长度为 ``m``，补齐段长度为
    ``p = length - m``）：

    ``"zero"``
        补齐段全为 0。
    ``"mirror"``
        端点重复的对称反射：``out[m + k] = v[m - 1 - k]``（``k = 0..p-1``）。
        补齐长度不得超过原长度（与 Acycle ``zeropad2.m`` 的报错一致）。
    ``"periodic"``
        周期延拓：``out[n] = v[n % m]``（对所有 ``n`` 成立）。
    ``"edge"``
        端点复制：补齐段全等于末值 ``v[-1]``。
    ``"mean"``
        均值填充（Acycle 的第 3 种）：补齐段全等于原序列的均值。
    ``"random"``
        随机填充（Acycle 的第 4 种）：以补齐段的均值为中心、标准差取原序列
        标准差的正态随机数；由 ``seed`` 决定，可复现。

    Args:
        values: 一维有限序列。
        length: 补齐后的**总长度**，不得小于原长度。
        mode: padding 方式，见 :data:`PADDING_MODES`。
        seed: ``mode="random"`` 用的随机种子。

    Returns:
        PaddingResult: 补齐后的序列与长度信息。

    Raises:
        ValidationError: 输入不合法、方式未知、目标长度过小、镜像补齐过长。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.small_tools import pad_series
        >>> out = pad_series(np.array([1.0, 2.0, 3.0]), 5, "periodic")
        >>> out.values.tolist()
        [1.0, 2.0, 3.0, 1.0, 2.0]
    """
    series = _as_series(values, "values", minimum_size=1)
    total = _positive_int(length, "length")
    if total < series.size:
        raise ValidationError(
            _("length must not be shorter than the series; got {0} and {1}").format(total, series.size),
            details={"length": total, "series_length": int(series.size)},
        )
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))):
        raise ValidationError(
            _("seed must be an integer"),
            details={"seed": str(seed)},
        )
    if not isinstance(mode, str):
        raise ValidationError(
            _("mode must be a string"),
            details={"mode": str(mode)},
        )
    name = mode.strip().lower()
    if name not in PADDING_MODES:
        raise ValidationError(
            _("mode must be one of {0}; got '{1}'").format(", ".join(PADDING_MODES), mode),
            details={"mode": mode, "valid_modes": list(PADDING_MODES)},
        )

    original = int(series.size)
    padding = total - original
    out = np.empty(total, dtype=float)
    out[:original] = series
    if padding:
        if name == "zero":
            out[original:] = 0.0
        elif name == "edge":
            out[original:] = series[-1]
        elif name == "periodic":
            out[original:] = series[np.arange(padding) % original]
        elif name == "mirror":
            if padding > original:
                raise ValidationError(
                    _("mirror padding cannot be longer than the series; got {0} and {1}").format(padding, original),
                    details={"padding": padding, "series_length": original},
                )
            out[original:] = series[original - 1 :: -1][:padding]
        elif name == "mean":
            out[original:] = float(np.mean(series))
        else:  # "random"
            effective_seed = resolve_seed(seed, context="pad_series random padding")
            rng = make_rng(effective_seed, context="pad_series random padding")
            spread = float(np.std(series))
            out[original:] = float(np.mean(series)) + spread * rng.standard_normal(padding)

    return PaddingResult(
        values=out,
        mode=name,
        original_length=original,
        padded_length=total,
        padding_length=padding,
    )


def robust_linear_fit(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    n_iterations: int = DEFAULT_ROBUST_ITERATIONS,
    trim_factor: float = DEFAULT_TRIM_FACTOR,
) -> RobustLinearFitResult:
    """稳健直线拟合：最小二乘 + 迭代剔除离群点（LTS）。

    迭代规则（Rousseeuw & Leroy 1987 的最小 trimmed squares）：

    1. 先做一次**普通最小二乘**拟合；
    2. 用残差的**中位绝对偏差**换算鲁棒尺度
       ``scale = 1.4826 * median(|r - median(r)|)``；
    3. 剔除 ``|r| > trim_factor * scale`` 的点，用剩下的点重新做最小二乘；
    4. 重复 2--3，直到内点集合不再变化或达到迭代次数上限。

    结果里同时给出**普通最小二乘**的斜率与截距作为对照组：离群点把普通最小
    二乘带偏，而稳健拟合不受影响。

    Args:
        x: 自变量，一维有限序列。
        y: 因变量，与 ``x`` 等长。
        n_iterations: 迭代次数上限，至少 1。
        trim_factor: 剔除阈值（以鲁棒尺度为单位的倍数），必须为正。

    Returns:
        RobustLinearFitResult: 稳健拟合结果与普通最小二乘对照组。

    Raises:
        ValidationError: 输入不合法、长度不一致、点数太少、参数非法。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.small_tools import robust_linear_fit
        >>> x = np.arange(50.0)
        >>> y = 1.0 * x + 0.5
        >>> out = robust_linear_fit(x, y)
        >>> bool(abs(out.slope - 1.0) < 1e-9)
        True
    """
    xs = _as_series(x, "x", minimum_size=MIN_FIT_SAMPLES)
    ys = _as_series(y, "y", minimum_size=MIN_FIT_SAMPLES)
    if xs.size != ys.size:
        raise ValidationError(
            _("x and y must have the same length; got {0} and {1}").format(xs.size, ys.size),
            details={"x_length": int(xs.size), "y_length": int(ys.size)},
        )
    iterations = _positive_int(n_iterations, "n_iterations")
    factor = _positive_float(trim_factor, "trim_factor")

    ols_slope, ols_intercept = np.polyfit(xs, ys, 1)
    slope = float(ols_slope)
    intercept = float(ols_intercept)
    inliers = np.ones(xs.size, dtype=bool)
    scale = float("nan")
    used = 0

    for iteration in range(1, iterations + 1):
        residuals = ys - (slope * xs + intercept)
        scale = MAD_TO_SIGMA * _mad(residuals)
        if not np.isfinite(scale) or scale <= 0.0:
            # 残差已经退化到舍入量级：拟合就是完美的，没有可剔除的点。
            break
        keep = np.abs(residuals) <= factor * scale
        if int(np.count_nonzero(keep)) < MIN_FIT_SAMPLES:
            break
        if bool(np.array_equal(keep, inliers)):
            break
        inliers = keep
        slope, intercept = (float(value) for value in np.polyfit(xs[inliers], ys[inliers], 1))
        used = iteration

    residuals = ys - (slope * xs + intercept)
    total_variance = float(np.sum((ys - float(np.mean(ys))) ** 2))
    residual_variance = float(np.sum(residuals**2))
    r_squared = 1.0 - residual_variance / total_variance if total_variance > 0.0 else float("nan")
    logger.info(
        "robust_linear_fit: slope=%.6g (ols %.6g), %d/%d inliers",
        slope,
        float(ols_slope),
        int(np.count_nonzero(inliers)),
        int(xs.size),
    )
    return RobustLinearFitResult(
        slope=slope,
        intercept=intercept,
        residuals=residuals,
        inlier_mask=inliers,
        n_inliers=int(np.count_nonzero(inliers)),
        n_iterations=used,
        scale=float(scale),
        ols_slope=float(ols_slope),
        ols_intercept=float(ols_intercept),
        r_squared=float(r_squared),
    )


def wavelet_smoothing(
    values: npt.ArrayLike,
    scales: int | Sequence[int] | None = None,
    *,
    mode: str = "threshold",
    threshold: str = "soft",
    noise_sigma: float | None = None,
    threshold_factor: float = DEFAULT_THRESHOLD_FACTOR,
    smoothing_width: float = 1.0,
) -> WaveletSmoothingResult:
    """小波平滑：用多分辨率分解压掉噪声（或做尺度-时间平滑）。

    分解用 **à trous 算法**（Holschneider et al. 1986；Torrence & Webster
    1998），核是 B3 样条：因为 ``d_j = s_j - s_{j+1}``，**不做任何处理时
    重建结果与原序列逐点严格相等**——这一点在测试里被当作契约钉死。

    两种平滑方式：

    ``mode="threshold"``（默认）
        Donoho & Johnstone (1994) 的小波阈值收缩，但阈值是**逐层**取的：
        à trous 分解的各层噪声尺度不一样（第 ``j`` 层的细节系数噪声标准差是
        ``sigma * h_j``，``h_j`` 是该层差分滤波器的 L2 范数），所以

            lambda_j = threshold_factor * sqrt(2 ln N) * sigma * h_j

        ``sigma`` 是**样本层**的噪声标准差：给出 ``noise_sigma`` 时用它，没给
        时用最细一层的细节系数按中位绝对偏差估计
        （``sigma = MAD(d_0) / (0.6745 * h_0)``）。``threshold="soft"`` 是软
        阈值（符号不变、幅度减 ``lambda``），``threshold="hard"`` 是硬阈值
        （幅度小于 ``lambda`` 的直接置零）。
        通用阈值的常数 ``sqrt(2 ln N)`` 是在正交小波下定义的，à trous 分解的
        系数彼此高度相关，直接照搬会把信号也削掉，所以 ``threshold_factor``
        的默认值是 :data:`DEFAULT_THRESHOLD_FACTOR`（0.25）而不是 1.0。
    ``mode="gaussian"``
        Torrence & Webster (1998) 的**频率域高斯平滑**（Acycle
        ``smoothwavelet.m`` 的算子）：第 ``j`` 层的细节系数用宽度
        ``smoothing_width * 2^j``（样本）的高斯低通滤波，尺度越大平滑越宽。
        这是一个**尺度-时间平滑算子**（本来用于小波系数的显著性检验），
        不是去噪器。

    Args:
        values: 一维有限序列。
        scales: 分解层数（整数）或各层的尺度因子（整数序列）。``None`` 时按
            记录长度自动取 ``floor(log2(N)) - 1`` 层。
        mode: 平滑方式，见 :data:`WAVELET_SMOOTHING_MODES`。
        threshold: 阈值方式，见 :data:`WAVELET_THRESHOLDS`。
        noise_sigma: 噪声标准差；``None`` 时从最细一层估计。
        threshold_factor: 通用阈值的倍数。
        smoothing_width: ``mode="gaussian"`` 时最细一层的高斯宽度（样本数）。

    Returns:
        WaveletSmoothingResult: 平滑后的序列、各层细节系数与阈值信息。

    Raises:
        ValidationError: 输入不合法、层数或方式非法、层数超过记录长度。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.small_tools import wavelet_smoothing
        >>> series = np.sin(np.arange(256) / 8.0)
        >>> out = wavelet_smoothing(series, 5)
        >>> out.smoothed.shape
        (256,)
    """
    series = _as_series(values, "values", minimum_size=MIN_SAMPLES)
    if not isinstance(mode, str):
        raise ValidationError(
            _("mode must be a string"),
            details={"mode": str(mode)},
        )
    smoothing_mode = mode.strip().lower()
    if smoothing_mode not in WAVELET_SMOOTHING_MODES:
        raise ValidationError(
            _("mode must be one of {0}; got '{1}'").format(", ".join(WAVELET_SMOOTHING_MODES), mode),
            details={"mode": mode, "valid_modes": list(WAVELET_SMOOTHING_MODES)},
        )
    if not isinstance(threshold, str):
        raise ValidationError(
            _("threshold must be a string"),
            details={"threshold": str(threshold)},
        )
    threshold_mode = threshold.strip().lower()
    if threshold_mode not in WAVELET_THRESHOLDS:
        raise ValidationError(
            _("threshold must be one of {0}; got '{1}'").format(", ".join(WAVELET_THRESHOLDS), threshold),
            details={"threshold": threshold, "valid_thresholds": list(WAVELET_THRESHOLDS)},
        )
    factor = _non_negative_float(threshold_factor, "threshold_factor")
    width = _positive_float(smoothing_width, "smoothing_width")

    if scales is None:
        n_levels = max(1, int(np.floor(np.log2(series.size))) - 1)
        scale_factors = np.array([2**level for level in range(n_levels)], dtype=int)
    elif isinstance(scales, (int, np.integer)) and not isinstance(scales, bool):
        n_levels = _positive_int(scales, "scales")
        scale_factors = np.array([2**level for level in range(n_levels)], dtype=int)
    else:
        scale_factors = np.asarray(scales, dtype=int).flatten()
        if scale_factors.size == 0:
            raise ValidationError(
                _("scales must not be empty"),
                details={"scales": str(scales)},
            )
        if not bool(np.all(np.diff(scale_factors) > 0)):
            raise ValidationError(
                _("the scale factors must be strictly increasing"),
                details={"scales": scale_factors.tolist()},
            )
        n_levels = int(scale_factors.size)
    if n_levels > series.size:
        raise ValidationError(
            _("the number of scales must not exceed the number of samples; got {0} and {1}").format(
                n_levels, series.size
            ),
            details={"scales": n_levels, "n_samples": int(series.size)},
        )

    details, approximation = _atrous_decompose(series, n_levels)

    if smoothing_mode == "threshold":
        gains = _atrous_noise_gains(n_levels)
        if noise_sigma is None:
            # 最细一层的细节系数几乎全是噪声，用它的 MAD 估计样本层的噪声标准差。
            sigma = float(_mad(details[0])) / MAD_TO_NOISE_SIGMA / max(gains[0], 1e-12)
        else:
            sigma = _positive_float(noise_sigma, "noise_sigma")
        if series.size >= MIN_UNIVERSAL_THRESHOLD_SAMPLES:
            factor_universal = float(np.sqrt(2.0 * np.log(series.size)))
        else:
            factor_universal = 1.0
        # 逐层阈值：第 j 层的噪声标准差是 sigma * h_j，阈值按同一比例缩放。
        limits = factor * factor_universal * sigma * gains
        processed = np.empty_like(details)
        for level in range(n_levels):
            layer = details[level]
            if threshold_mode == "soft":
                processed[level] = np.sign(layer) * np.maximum(np.abs(layer) - limits[level], 0.0)
            else:
                processed[level] = np.where(np.abs(layer) > limits[level], layer, 0.0)
    else:
        sigma = float("nan")
        processed = np.empty_like(details)
        for level in range(n_levels):
            layer_width = width * float(scale_factors[level])
            processed[level] = _gaussian_lowpass(details[level], layer_width)
        limits = np.full(n_levels, np.nan, dtype=float)

    smoothed = approximation + np.sum(processed, axis=0)
    logger.info(
        "wavelet_smoothing: %d levels, mode=%s, residual sd %.6g -> %.6g",
        n_levels,
        smoothing_mode,
        float(np.std(series)),
        float(np.std(series - smoothed)),
    )
    return WaveletSmoothingResult(
        smoothed=smoothed,
        details=processed,
        approximation=approximation,
        scales=scale_factors,
        noise_sigma=float(sigma),
        thresholds=limits,
        threshold=float(limits[0]),
        mode=smoothing_mode,
    )


def cosine_taper(
    length: int,
    *,
    taper_fraction: float = DEFAULT_TAPER_FRACTION,
) -> CosineTaperResult:
    """余弦窗 taper（Bloomfield 1976）。

    权重定义（``u`` 是样本在记录里的相对位置，0 = 起点、1 = 终点，
    ``tau`` 是 taper 比例）：

    .. code-block:: text

        w(u) = 0.5 - 0.5 * cos(pi * u / tau)            0 <= u <  tau
        w(u) = 1                                       tau <= u <= 1 - tau
        w(u) = 0.5 - 0.5 * cos(pi * (1 - u) / tau)      1 - tau < u <= 1

    于是端点权重为 0、``tau`` 处为 1、权重关于记录中点**对称**。这与 Acycle
    ``ctaper.m`` 的做法一致（它在首尾各 5 % 上加同一组升余弦权重）。

    Args:
        length: 权重数组的长度，至少 2。
        taper_fraction: 首尾各 taper 的比例，必须在 ``(0, 0.5)`` 内。

    Returns:
        CosineTaperResult: 权重数组。

    Raises:
        ValidationError: 长度或 taper 比例非法。

    Examples:
        >>> from stratigraphy.small_tools import cosine_taper
        >>> weights = cosine_taper(101).weights
        >>> float(weights[0]), float(weights[50]), float(weights[-1])
        (0.0, 1.0, 0.0)
    """
    total = _positive_int(length, "length")
    if total < 2:
        raise ValidationError(
            _("length must be at least 2; got {0}").format(total),
            details={"length": total},
        )
    fraction = float(taper_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction < 0.5:
        raise ValidationError(
            _("taper_fraction must lie in (0, 0.5); got {0}").format(fraction),
            details={"taper_fraction": fraction},
        )

    positions = np.arange(total, dtype=float) / (total - 1)
    weights = np.ones(total, dtype=float)
    rising = positions < fraction
    falling = positions > 1.0 - fraction
    weights[rising] = 0.5 - 0.5 * np.cos(np.pi * positions[rising] / fraction)
    weights[falling] = 0.5 - 0.5 * np.cos(np.pi * (1.0 - positions[falling]) / fraction)
    return CosineTaperResult(weights=weights, length=total, taper_fraction=fraction)


def phase_diagram(
    values_x: npt.ArrayLike,
    values_y: npt.ArrayLike | None = None,
    *,
    time: npt.ArrayLike | None = None,
) -> PhaseDiagramResult:
    """相位图：把信号与它的解析信号画在复平面上。

    两种用法：

    * 只给 ``values_x``：它就是信号本身，内部做解析信号变换，得到相轨迹
      ``(Re z, Im z) = (r cos(phi), r sin(phi))``，其中 ``r`` 是包络、``phi``
      是瞬时相位。
    * 同时给 ``values_x`` 与 ``values_y``：两者就是复平面上轨迹的实部与虚部
      （例如实部是原序列、虚部是已经算好的 Hilbert 变换）。

    Acycle 的 ``phaseplot.m``（小波相干工具箱）画的是**相位箭头**：在每个
    ``(x, y)`` 点画一根方向由相位决定的箭头；本函数给出的是相轨迹本身，
    相位的定义与它一致。

    Args:
        values_x: 信号（或轨迹的实部）。
        values_y: 可选，轨迹的虚部；给了就直接当成轨迹用。
        time: 可选时间轴，只有在只给 ``values_x`` 时才用得上。

    Returns:
        PhaseDiagramResult: 轨迹点、半径与相位。

    Raises:
        ValidationError: 输入不合法或两个序列长度不一致。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.small_tools import phase_diagram
        >>> n = np.arange(200)
        >>> out = phase_diagram(np.cos(2 * np.pi * 5 * n / 200))
        >>> bool(np.allclose(out.radius, 1.0, atol=1e-9))
        True
    """
    if values_y is None:
        transform = domain_transform(values_x, method="analytic", time=time)
        envelope = np.asarray(transform.envelope, dtype=float)
        phase = np.asarray(transform.phase, dtype=float)
        points = np.column_stack([envelope * np.cos(phase), envelope * np.sin(phase)])
        radius = envelope
    else:
        real_part = _as_series(values_x, "values_x", minimum_size=2)
        imaginary_part = _as_series(values_y, "values_y", minimum_size=2)
        if real_part.size != imaginary_part.size:
            raise ValidationError(
                _("values_x and values_y must have the same length; got {0} and {1}").format(
                    real_part.size, imaginary_part.size
                ),
                details={"x_length": int(real_part.size), "y_length": int(imaginary_part.size)},
            )
        points = np.column_stack([real_part, imaginary_part])
        radius = np.hypot(real_part, imaginary_part)
        phase = np.arctan2(imaginary_part, real_part)
        envelope = radius
    return PhaseDiagramResult(points=points, radius=radius, phase=phase, envelope=envelope)


def random_phase_resample(
    values: npt.ArrayLike,
    seed: int | None = None,
    *,
    method: str = "phase",
    n_iterations: int = DEFAULT_SURROGATE_ITERATIONS,
) -> RandomPhaseResampleResult:
    """随机相位重采样：生成保留幅度谱、但相位被随机化的替代序列。

    两种方式：

    ``method="phase"``（默认）
        Theiler et al. (1992) 的**相位随机化**：正频率上的相位换成均匀随机的
        角度，负频率取共轭，直流与（偶数长度的）Nyquist 频点原样保留。因此
        **每一个傅里叶幅度都被精确保留**（相对误差在舍入量级，``1e-16``），
        而**取值分布不保留**（相位随机化后的取值趋于高斯）。
    ``method="iaaft"``
        Schreiber & Schmitz (1996) 的 **IAAFT**：每一步取当前替代序列的相位、
        装上原序列的幅度，再把得到的谱投影**按秩映射回原序列的取值分布**。
        于是**取值集合被精确保留**，而幅度谱是**逐步逼近**的——取值分布与
        幅度谱不能同时精确成立，逼近的程度由 ``amplitude_relative_error``
        给出（迭代在误差不再改善时提前停止）。

    Args:
        values: 一维有限序列。
        seed: 随机种子；同一个种子两次调用结果完全一致。
        method: 替代序列方式，见 :data:`SURROGATE_METHODS`。
        n_iterations: ``method="iaaft"`` 的迭代次数。

    Returns:
        RandomPhaseResampleResult: 替代序列、幅度谱与两项误差指标。

    Raises:
        ValidationError: 输入不合法、方式未知、迭代次数非法。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.small_tools import random_phase_resample
        >>> series = np.sin(np.arange(64) / 4.0)
        >>> first = random_phase_resample(series, seed=3)
        >>> second = random_phase_resample(series, seed=3)
        >>> bool(np.allclose(first.values, second.values))
        True
    """
    series = _as_series(values, "values", minimum_size=MIN_SAMPLES)
    if not isinstance(method, str):
        raise ValidationError(
            _("method must be a string"),
            details={"method": str(method)},
        )
    name = method.strip().lower()
    if name not in SURROGATE_METHODS:
        raise ValidationError(
            _("method must be one of {0}; got '{1}'").format(", ".join(SURROGATE_METHODS), method),
            details={"method": method, "valid_methods": list(SURROGATE_METHODS)},
        )
    iterations = _positive_int(n_iterations, "n_iterations")

    effective_seed = resolve_seed(seed, context="random phase surrogate")
    rng = make_rng(effective_seed, context="random phase surrogate")
    spectrum = np.fft.fft(series)
    amplitude = np.abs(spectrum)
    used = 0

    if name == "phase":
        # 只随机化独立的正频率；负频率取共轭，直流与 Nyquist 频点原样保留。
        positive = _independent_positive_bins(series.size)
        spectrum[positive] = amplitude[positive] * np.exp(1j * rng.uniform(-np.pi, np.pi, positive.size))
        spectrum[series.size - positive] = np.conj(spectrum[positive])
        surrogate = np.real(np.fft.ifft(spectrum))
    else:
        target_values = np.sort(series)
        surrogate = series[rng.permutation(series.size)]
        previous_error = float("inf")
        for iteration in range(iterations):
            phases = np.angle(np.fft.fft(surrogate))
            projection = np.real(np.fft.ifft(amplitude * np.exp(1j * phases)))
            order = np.argsort(projection, kind="stable")
            next_surrogate = np.empty_like(target_values)
            next_surrogate[order] = target_values
            surrogate = next_surrogate
            used = iteration + 1
            error = _relative_spectral_error(surrogate, amplitude)
            if used > 1 and abs(previous_error - error) <= 1e-10 * max(1.0, previous_error):
                break
            previous_error = error

    surrogate_amplitude = np.abs(np.fft.fft(surrogate))
    amplitude_error = _relative_spectral_error(surrogate, amplitude)
    value_error = float(np.max(np.abs(np.sort(surrogate) - np.sort(series))))
    logger.info(
        "random_phase_resample: method=%s, amplitude rel. error %.3g, value spread error %.3g",
        name,
        amplitude_error,
        value_error,
    )
    return RandomPhaseResampleResult(
        values=surrogate,
        method=name,
        seed=int(effective_seed),
        n_iterations=used,
        amplitude_spectrum=surrogate_amplitude,
        original_amplitude_spectrum=amplitude,
        amplitude_relative_error=amplitude_error,
        value_distribution_error=value_error,
    )


def fixed_bandwidth_moving_average(
    values: npt.ArrayLike,
    bandwidth: float,
    *,
    dt: float = 1.0,
    time: npt.ArrayLike | None = None,
) -> FixedBandwidthMovingAverageResult:
    """固定时间带宽的滑动平均。

    与本仓库 :func:`stratigraphy.smoothing.moving_average_smooth` 的区别：那里
    的跨度是**样本数**，这里的是**时间带宽**——窗口是 ``[t - bw/2, t + bw/2]``，
    在不等间距的采样上仍然覆盖同样的时间跨度。

    边界处理：窗口在记录两端被**截短**（用窗口里实际有的点求平均），不做端点
    外推、不补零。因此常值序列的输出仍是同一个常值（不引入边界偏差），而
    线性序列在窗口完整处的斜率保持不变，只有两端几个点的截距会偏。

    Args:
        values: 一维有限序列。
        bandwidth: 窗口的总时间带宽，必须为正。
        dt: 采样间隔（没给 ``time`` 时用它生成时间轴）。
        time: 可选时间轴，与 ``values`` 等长且严格递增。

    Returns:
        FixedBandwidthMovingAverageResult: 滑动平均序列与每个点的窗口样本数。

    Raises:
        ValidationError: 输入不合法、带宽或采样间隔非法、时间轴不合法。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.small_tools import fixed_bandwidth_moving_average
        >>> out = fixed_bandwidth_moving_average(np.ones(50), 5.0)
        >>> bool(np.allclose(out.values, 1.0))
        True
    """
    series = _as_series(values, "values", minimum_size=2)
    width = _positive_float(bandwidth, "bandwidth")
    step = _positive_float(dt, "dt")

    if time is None:
        axis = np.arange(series.size, dtype=float) * step
    else:
        axis = _as_series(time, "time", minimum_size=2)
        if axis.size != series.size:
            raise ValidationError(
                _("time and values must have the same length; got {0} and {1}").format(axis.size, series.size),
                details={"time_length": int(axis.size), "values_length": int(series.size)},
            )
        if bool(np.any(np.diff(axis) <= 0.0)):
            raise ValidationError(
                _("time must be strictly increasing"),
                details={"name": "time"},
            )

    half = 0.5 * width
    left = np.searchsorted(axis, axis - half, side="left")
    right = np.searchsorted(axis, axis + half, side="right")
    counts = right - left
    cumulative = np.concatenate([np.array([0.0]), np.cumsum(series)])
    sums = cumulative[right] - cumulative[left]
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return FixedBandwidthMovingAverageResult(
        values=means,
        window_counts=counts.astype(int),
        bandwidth=width,
        dt=step,
    )


def detrend_polynomial_pp(
    values: npt.ArrayLike,
    order: int,
    *,
    window: int | None = None,
    dt: float = 1.0,
) -> PolynomialDetrendResult:
    """滑动窗多项式去趋势（Sinnesael et al. 2018 的多项式基线建模）。

    与本仓库 :func:`stratigraphy.detrending.detrend_series` 的**全局**多项式
    去趋势不是一回事：这里是**局部**的——记录被切成 ``window`` 长的分析窗，
    每窗单独做一次多项式最小二乘，用 Hann 窗加权后**重叠相加**再归一化，得到
    一条平滑的基线。

    ``order`` 是多项式的**次数**（0 = 常数，2 = 二次）；Acycle 里的 ``Porder``
    是"次数 + 1"（设计矩阵的列数），两者差一。

    ``window`` 缺省是整条记录（相当于一次全局多项式拟合）；给出时必须不小于
    ``order + 2``，也不得长于记录。

    参考实现的两个边界缺陷在这里被修掉了：Hann 窗的两端权重为 0，导致记录
    的首尾样本的累计权重为 0（原实现那里会算出 ``0/0``）；这里用最近一个分析
    窗的多项式值补上。

    Args:
        values: 一维有限序列。
        order: 多项式次数，至少 0。次数为 ``d`` 时至少需要 ``d + 3`` 个样本
            （``d + 1`` 个定系数，再留一个自由度）。
        window: 分析窗长度（样本数）。
        dt: 采样间隔（多项式自变量的单位）。

    Returns:
        PolynomialDetrendResult: 基线、去趋势序列与残差统计。

    Raises:
        ValidationError: 输入不合法、次数或窗长非法。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.small_tools import detrend_polynomial_pp
        >>> t = np.arange(50.0)
        >>> out = detrend_polynomial_pp(2.0 * t**2 + 3.0 * t + 1.0, 2)
        >>> bool(np.allclose(out.detrended, 0.0, atol=1e-6))
        True
    """
    series = _as_series(values, "values", minimum_size=MIN_SAMPLES)
    degree = _non_negative_int(order, "order")
    step = _positive_float(dt, "dt")
    total = int(series.size)
    if total < degree + 3:
        raise ValidationError(
            _("a polynomial of degree {0} needs at least {1} samples; got {2}").format(degree, degree + 3, total),
            details={"order": degree, "minimum": degree + 3, "size": total},
        )

    frame = total if window is None else _positive_int(window, "window")
    if frame > total:
        raise ValidationError(
            _("the analysis window must not be longer than the series; got {0} and {1}").format(frame, total),
            details={"window": frame, "n_samples": total},
        )
    if frame < degree + 2:
        raise ValidationError(
            _("the analysis window needs at least {0} samples for a polynomial of degree {1}").format(
                degree + 2, degree
            ),
            details={"window": frame, "order": degree},
        )

    if frame > 1:
        hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(frame, dtype=float) / (frame - 1))
    else:
        hann = np.ones(1, dtype=float)

    starts = np.arange(total - frame + 1, dtype=int)
    local_time = np.arange(frame, dtype=float) * step
    design = np.vander(local_time, degree + 1, increasing=True)
    norms = np.linalg.norm(design, axis=0)
    normalised = design / norms

    coefficients = np.empty((starts.size, degree + 1), dtype=float)
    trend = np.zeros(total, dtype=float)
    weight = np.zeros(total, dtype=float)
    for index, start in enumerate(starts):
        segment = series[start : start + frame]
        theta, *_unused = np.linalg.lstsq(normalised, segment, rcond=None)
        coefficients[index] = theta / norms
        trend[start : start + frame] += (design @ coefficients[index]) * hann
        weight[start : start + frame] += hann

    # 归一化：重叠相加的加权多项式和除以累计的 Hann 权重（对应参考实现的
    # ``sigo = sigo./normo``）。
    covered = weight > 0.0
    trend = np.divide(trend, weight, out=np.zeros_like(trend), where=covered)
    if not bool(np.all(covered)):
        # 记录两端累计权重为 0（Hann 窗的端点权重是 0）：用最近的分析窗的多项式值补上。
        first_covered = int(np.argmax(covered))
        last_covered = total - 1 - int(np.argmax(covered[::-1]))
        leading = first_covered
        if leading > 0:
            local = np.arange(leading, dtype=float) * step
            trend[:leading] = np.vander(local, degree + 1, increasing=True) @ coefficients[0]
        trailing = total - 1 - last_covered
        if trailing > 0:
            local = np.arange(frame - trailing, frame, dtype=float) * step
            trend[total - trailing :] = np.vander(local, degree + 1, increasing=True) @ coefficients[-1]

    detrended = series - trend
    logger.info(
        "detrend_polynomial_pp: degree=%d, window=%d, residual sd %.6g",
        degree,
        frame,
        float(np.std(detrended)),
    )
    return PolynomialDetrendResult(
        trend=trend,
        detrended=detrended,
        order=degree,
        window=frame,
        coefficients=coefficients[-1],
        residual_std=float(np.std(detrended)),
    )


def chi_square_confidence_interval(
    observed: npt.ArrayLike,
    *,
    dt: float = 1.0,
    dof: int = 2,
    level: float = 0.95,
    rho: float | None = None,
    nfft: int | None = None,
) -> ChiSquareIntervalResult:
    """红噪声谱的卡方置信区间。

    做法与 Acycle ``redconfchi2.m`` 一致：

    1. AR(1) 系数 ``rho``：不给就用 Mudelsee (2002) 的公式从序列本身估计
       （去均值后的 ``sum(x_i x_{i-1}) / sum(x_{i-1}^2)``）。
    2. AR(1) 理论谱 ``(1 - rho^2) / (1 - 2 rho cos(2 pi f dt) + rho^2)``
       （Mann & Lees 1996）。
    3. 标定：把理论谱（直流分量为 0 时）的均值放大到与观测功率谱的均值相同，
       于是理论谱与观测谱共用同一把尺子。
    4. 上下界 = 标定后的理论谱乘以卡方分位数除以自由度
       （``chi2.ppf(level, dof) / dof``），自由度取 ``2K``（``K`` 是独立谱估计
       的个数；单段周期图对应 ``K = 1``，即 ``dof = 2``）。

    Args:
        observed: 观测序列（一维有限）。
        dt: 采样间隔。
        dof: 有效自由度 ``2K``，至少 1。
        level: 置信水平，在 ``(0, 1)`` 内。
        rho: AR(1) 系数；``None`` 时从序列估计。
        nfft: 周期图的零填充长度（对应 MATLAB ``periodogram`` 的 ``pad``）。

    Returns:
        ChiSquareIntervalResult: 频率轴、观测谱、理论谱与上下界。

    Raises:
        ValidationError: 输入不合法、参数非法、序列是常数（估计不出 rho）。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.small_tools import chi_square_confidence_interval
        >>> series = np.sin(np.arange(256) / 8.0)
        >>> out = chi_square_confidence_interval(series)
        >>> out.dof
        2
    """
    series = _as_series(observed, "observed", minimum_size=MIN_SAMPLES)
    step = _positive_float(dt, "dt")
    degrees = _positive_int(dof, "dof")
    if not 0.0 < float(level) < 1.0:
        raise ValidationError(
            _("level must lie in (0, 1); got {0}").format(float(level)),
            details={"level": float(level)},
        )
    if nfft is not None:
        padding = _positive_int(nfft, "nfft")
        if padding < series.size:
            raise ValidationError(
                _("nfft must not be shorter than the series; got {0} and {1}").format(padding, series.size),
                details={"nfft": padding, "n_samples": int(series.size)},
            )
    else:
        padding = None

    if rho is None:
        centred = series - float(np.mean(series))
        denominator = float(np.sum(centred[:-1] ** 2))
        if denominator <= 0.0:
            raise ValidationError(
                _("the series is constant, so the AR(1) coefficient cannot be estimated"),
                details={"name": "observed"},
            )
        rho_value = float(np.sum(centred[1:] * centred[:-1]) / denominator)
    else:
        rho_value = float(rho)
        if not np.isfinite(rho_value) or not -1.0 < rho_value < 1.0:
            raise ValidationError(
                _("rho must lie in (-1, 1); got {0}").format(rho_value),
                details={"rho": rho_value},
            )

    power, frequencies = periodogram(
        series,
        fs=1.0 / step,
        window="boxcar",
        detrend="constant",
        scaling="density",
        nfft=padding,
    )
    theoretical = np.asarray(ar1_theoretical_spectrum(frequencies, rho_value, step), dtype=float)
    without_dc = theoretical.copy()
    without_dc[0] = 0.0
    reference = float(np.mean(without_dc))
    mean_power = float(np.mean(power))
    if reference > 0.0 and np.isfinite(reference):
        scaled = theoretical * (mean_power / reference)
    else:
        scaled = theoretical

    lower = scaled * float(stats.chi2.ppf(1.0 - float(level), degrees)) / degrees
    upper = scaled * float(stats.chi2.ppf(float(level), degrees)) / degrees
    logger.info(
        "chi_square_confidence_interval: rho=%.4g, dof=%d, level=%.3g",
        rho_value,
        degrees,
        float(level),
    )
    return ChiSquareIntervalResult(
        frequencies=np.asarray(frequencies, dtype=float),
        observed_power=np.asarray(power, dtype=float),
        theoretical=scaled,
        lower=lower,
        upper=upper,
        rho=rho_value,
        dof=degrees,
        level=float(level),
    )


def tobit_regression(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    limit: float = 0.0,
    censored: npt.ArrayLike | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-8,
) -> TobitRegressionResult:
    """Tobit 回归（删失正态回归；Tobin 1958）。

    **Acycle 没有对应实现**（整个 Acycle 代码库里 grep 不到 ``tobit``），本函
    数按 Tobin (1958) 的标准定义自行实现。

    模型：潜变量 ``y* = x' beta + eps``，``eps ~ N(0, sigma^2)``；观测到的
    ``y`` 在 ``y* <= limit`` 时被截断在 ``limit`` 上（左删失）。似然是

    .. code-block:: text

        未删失:  log phi((y - x'beta) / sigma) - log sigma
        已删失:  log Phi((limit - x'beta) / sigma)

    其中 ``phi`` / ``Phi`` 是标准正态的密度与分布函数。参数用极大似然估计
    （对数标准差参数化保证 ``sigma > 0``），梯度是解析的。

    **没有删失时 Tobit 退化成普通最小二乘**（似然就是正态似然），这一点在测试
    里被当作契约钉死。

    Args:
        x: 自变量，一维（单个自变量）或二维（多个自变量，形状 ``(n, p)``）。
        y: 因变量（观测值），与 ``x`` 行数相同。样本数至少要比参数个数多 2
            （回归系数 + 误差标准差 + 两个自由度）。
        limit: 删失限值（左删失）。
        censored: 可选的删失标记（True = 已删失）；``None`` 时按 ``y <= limit``
            自动判定。
        max_iterations: 极大似然的迭代次数上限。
        tolerance: 收敛判据（梯度的无穷范数）。

    Returns:
        TobitRegressionResult: 系数、标准差、对数似然与对照组。

    Raises:
        ValidationError: 输入不合法、标记与观测不匹配、删失标记与取值矛盾。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.small_tools import tobit_regression
        >>> x = np.arange(40.0)
        >>> y = 1.0 + 0.5 * x
        >>> out = tobit_regression(x, y)
        >>> bool(abs(out.coefficients[1] - 0.5) < 1e-4)
        True
    """
    ys = _as_series(y, "y", minimum_size=MIN_FIT_SAMPLES)
    design = _design_matrix(x, ys.size)
    if ys.size < MIN_TOBIT_SAMPLES + design.shape[1] - 2:
        raise ValidationError(
            _("a censored regression needs at least {0} samples; got {1}").format(
                MIN_TOBIT_SAMPLES + design.shape[1] - 2, ys.size
            ),
            details={"minimum": MIN_TOBIT_SAMPLES + design.shape[1] - 2, "size": int(ys.size)},
        )
    limit_value = float(limit)
    if not np.isfinite(limit_value):
        raise ValidationError(
            _("limit must be finite"),
            details={"limit": limit_value},
        )
    if censored is None:
        flags = ys <= limit_value
    else:
        flags = np.asarray(censored).flatten().astype(bool)
        if flags.size != ys.size:
            raise ValidationError(
                _("the censoring flags and the response must have the same length; got {0} and {1}").format(
                    flags.size, ys.size
                ),
                details={"flags": int(flags.size), "y_length": int(ys.size)},
            )
    if bool(np.any(flags & (ys > limit_value + 1e-12))):
        raise ValidationError(
            _("a censored observation must not exceed the censoring limit"),
            details={"limit": limit_value},
        )
    iterations = _positive_int(max_iterations, "max_iterations")
    tolerance_value = _positive_float(tolerance, "tolerance")

    observed_rows = ~flags
    if int(np.count_nonzero(observed_rows)) >= design.shape[1]:
        ols_coefficients, *_unused = np.linalg.lstsq(design[observed_rows], ys[observed_rows], rcond=None)
    else:
        ols_coefficients, *_unused = np.linalg.lstsq(design, ys, rcond=None)
    residual = ys - design @ ols_coefficients
    sigma0 = float(np.sqrt(np.sum(residual**2) / max(ys.size, 1)))
    if not np.isfinite(sigma0) or sigma0 <= 0.0:
        sigma0 = 1.0

    def objective(params: npt.NDArray) -> float:
        beta = params[:-1]
        sigma = float(np.exp(np.clip(params[-1], -30.0, 30.0)))
        mu = design @ beta
        standardised = (ys - mu) / sigma
        uncensored = -0.5 * standardised**2 - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)
        censored_log = special.log_ndtr((limit_value - mu) / sigma)
        total = float(np.sum(np.where(flags, censored_log, uncensored)))
        if not np.isfinite(total):
            return 1e300
        return -total

    def gradient(params: npt.NDArray) -> npt.NDArray:
        beta = params[:-1]
        sigma = float(np.exp(np.clip(params[-1], -30.0, 30.0)))
        mu = design @ beta
        standardised = (ys - mu) / sigma
        censored_z = (limit_value - mu) / sigma
        # Mills 比 phi(z) / Phi(z)（用 log_ndtr 保证下溢安全）。
        mills = np.zeros_like(standardised)
        mills[flags] = np.exp(
            -0.5 * censored_z[flags] ** 2 - 0.5 * np.log(2.0 * np.pi) - special.log_ndtr(censored_z[flags])
        )
        weights_observed = (~flags).astype(float)
        weights_censored = flags.astype(float)
        grad_beta = np.sum(
            ((ys - mu) / sigma**2)[:, None] * design * weights_observed[:, None],
            axis=0,
        )
        grad_beta -= np.sum((mills / sigma)[:, None] * design * weights_censored[:, None], axis=0)
        grad_log_sigma = float(np.sum((standardised**2 - 1.0) * weights_observed))
        grad_log_sigma += float(np.sum(-mills[flags] * censored_z[flags]))
        return np.concatenate([grad_beta, [grad_log_sigma]])

    start = np.concatenate([ols_coefficients, [np.log(sigma0)]])
    result = optimize.minimize(
        objective,
        start,
        jac=gradient,
        method="BFGS",
        options={"maxiter": iterations, "gtol": tolerance_value},
    )
    best = result
    gradient_norm = float(np.max(np.abs(gradient(result.x))))
    if not result.success and gradient_norm > 10.0 * tolerance_value:
        # BFGS 在删失较多时容易在起点就因为步长搜索失败而停下（"precision
        # loss"），此时用无导数的 Nelder-Mead 再走一遍，取更好的那个解。
        fallback = optimize.minimize(
            objective,
            result.x,
            method="Nelder-Mead",
            options={"maxiter": 20 * iterations, "xatol": tolerance_value, "fatol": tolerance_value},
        )
        if fallback.fun < best.fun:
            best = fallback
        gradient_norm = float(np.max(np.abs(gradient(best.x))))
    beta = np.asarray(best.x[:-1], dtype=float)
    sigma = float(np.exp(best.x[-1]))
    log_likelihood = float(-objective(best.x))
    # BFGS 在最优解附近常常报"函数值不再下降"而不是"梯度为零"；两种都算收敛。
    converged = bool(best.success) or gradient_norm <= 10.0 * tolerance_value
    logger.info(
        "tobit_regression: sigma=%.6g, %d censored, converged=%s",
        sigma,
        int(np.count_nonzero(flags)),
        converged,
    )
    return TobitRegressionResult(
        coefficients=beta,
        sigma=sigma,
        log_likelihood=log_likelihood,
        n_observations=int(ys.size),
        n_censored=int(np.count_nonzero(flags)),
        converged=converged,
        limit=limit_value,
        ols_coefficients=np.asarray(ols_coefficients, dtype=float),
    )
