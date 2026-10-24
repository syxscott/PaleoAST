# =============================================================================
# FILE: stratigraphy/bispectrum.py
# =============================================================================
"""
双谱（bispectrum）与双相干（bicoherence）：三阶频谱与二次相位耦合。

用途
====
功率谱只看"每个频率上有多少能量"，相干性只看"两条序列在某个频率上是否同步"。
两者都停留在**二阶**统计量上，无法回答"三个频率是不是耦合在一起的"。
双谱补的正是这一块：对三个频率 ``f1``、``f2``、``f1+f2`` 的**三阶**矩做统计，
因此能识别"两个周期通过非线性过程（调制、阈值、混沌、波动方程的非线性项）
耦合成第三个周期"的现象——在功率谱里 ``f1``、``f2``、``f3`` 看上去只是三个
各自独立的谱峰，双谱却能说明它们之间的相位关系。

在古环境学里这对应经典的**三波共振**（three-wave interaction）判据，也是
Swaminathan, Bruno & Pierce (1999) 用双谱分析湍流耗散场的方法。

定义
====
设 ``X(f)`` 是信号 ``x(t)`` 的复傅里叶谱（分段平均后的谱），则

    B(f1, f2) = E[ X(f1) * X(f2) * conj(X(f1 + f2)) ]

称为**双谱**（bispectrum）。它有三个读法：

* **幅度** ``|B(f1, f2)|``：三元组耦合有多强。
* **相位** ``arg B(f1, f2)``：三个分量的相位关系。**这是双谱真正的物理内容**——
  只有相位才区分"耦合"与"巧合"。经典的零相位差三波共振给出
  ``arg B = 0``（``f1``、``f2`` 两个波以零相位差合成出 ``f3``）。
* **双相干** ``b(f1, f2)``：把幅度按其理论上限归一化后的无量纲量，取值 ``0..1``。

归一化
======
本模块采用 Kim & Powers (1992) 的归一化（与 Acycle 的 ``bicoherenceSquared``
一致）::

    b(f1, f2) = |B(f1, f2)|^2 / ( E[|X(f1) X(f2)|^2] * E[|X(f1 + f2)|^2] )

分母是两个二阶矩的乘积，由 Cauchy--Schwarz 不等式该比值**恒在 ``0..1`` 之间**。
文献里还有只除以 ``E|X(f1)^2 X(f2)^2|`` 的简化写法；那一项的量纲是
``f1+f2`` 处的功率，取值不落在 ``0..1``，因此本模块保留带
``E|X(f1+f2)|^2`` 因子的完整形式。

估计量
======
双谱是**三阶**量，单条记录的一次实现对单个三元组只给一个复数，没有零分布。
本模块沿用相干性分析（:mod:`stratigraphy.coherence`）的 Welch 分段（WOSA）
做法：

1. 记录被切成若干**重叠 50%** 的等长分段，每段做一次 FFT 得到复谱 ``X``；
2. 每个分段内部做**线性去趋势**（去均值 + 去斜率）并加 **Hann 窗**；
3. 三阶矩由三个谱因子的 **Hadamard（逐元素）乘积**合成：

       T(f1, f2) = X(f1) * X(f2) * conj(X(f1 + f2))

   再在分段之间平均得到 ``B``。这正是三阶 DFT 的定义式实现（等价于对时域
   三阶乘积做二维变换，本模块用频域定义直接实现，以免引入额外的插值假设）。
4. 实信号只需要取正频段：正频段的复谱与解析信号谱只差一个正实倍数（2 的
   三次方），该倍数在归一化的双相干里完全抵消。

**为什么必须分段平均**：单次实现下 ``|X(f1) X(f2) conj(X(f1+f2))|^2`` 与
``|X(f1) X(f2)|^2 |X(f1+f2)|^2`` 恒等，双相干退化成常数 1，毫无信息。
分段平均既让它非退化，也是零分布的来源。

频率轴
======
频率轴是**每单位时间**的循环频率（cycles per unit time），与
:mod:`stratigraphy.spectral_analysis` 的口径一致：第 ``k`` 个频点的频率是

    f_k = k / (L * dt)

其中 ``L`` 是实际使用的分段（FFT）长度，``dt`` 是采样间隔。频率轴只覆盖
``f1 + f2`` 仍然落在轴内的**主域**（principal domain）；主域之外的
``(f1, f2)`` 组合没有第三个频率，双谱与双相干取 ``NaN``。

显著性
======
显著性由 **IAAFT 代理数据**（iterative amplitude adjusted Fourier transform，
Schreiber & Schmitz 1996）给出：

1. 初始代理 = 原序列的一个随机排列；
2. 迭代若干次：先用原序列的幅度谱与当前代理的相位做**谱投影**（保持功率谱），
   再把投影结果的取值按秩换回原序列的取值集合（保持边缘分布）。

收敛后的代理同时具备原序列的取值分布与幅度谱，但**三阶相位耦合被系统性地
抹掉**——这正是要检验的零假设。对每个代理算一遍同样的双相干，取其 95%
分位数作为该 ``(f1, f2)`` 的显著性阈值：``bicoherence[i, j] > significance[i, j]``
表示该三元组的相位耦合在 5% 水平上显著。

统计量选的是**双相干**而不是 ``|B|``：``|B|`` 的绝对值由三个分量的幅度决定，
代理数据保留了幅度谱，``|B|`` 因此几乎不变（对确定性的三波系统尤其如此），
而**双相干**才是"三阶矩在分段之间稳定存在"的度量。

**IAAFT 而不是普通相位随机化**：相位随机化会破坏原序列的边缘分布（非高斯
的地层记录里这会引入额外的偏差），IAAFT 把分布也保住，因此零假设更干净。

**这套检验能测什么、不能测什么**：代理数据检验的是"三阶矩在分段之间**统计
稳定地**存在"，因此它能识别湍流、地质过程这类**随机**记录里的持续相位耦合。
对于一个**确定性的**三波系统（三个纯正弦、``f3 = f1 + f2``），相位随机化的
代理本身也是完美的三波系统——幅度谱被保留，``|B|`` 与双相干几乎不变，只有
双谱**相位**从 0 变成随机值。因此确定性的三波共振要靠"双相干接近 1 且相位
接近 0"的**物理读数**来识别，而不是靠这里的代理阈值。

Acycle 以 GPL-3.0 发布，因此本文件是**独立实现**：只参考它公开的行为约定
（主域掩膜、Kim--Powers 归一化、IAAFT 代理数据与逐点阈值），未翻译或复制其
MATLAB 源码。

参考文献
========
* Swaminathan, J. K., Bruno, R. S. & Pierce, J. D. (1999). Multifractal analysis
  of the turbulent energy dissipation field. *Physical Review Letters*, 83,
  2216.（双谱分析湍流耗散场的经典方法）
* Kim, M. C. & Powers, E. J. (1992). Measurement of turbulent energy
  dissipation statistics using the bispectrum. *Journal of Fluid Mechanics*,
  248, 51--68.（双相干的 Kim--Powers 归一化）
* Schreiber, T. & Schmitz, A. (1996). Improved surrogate data for nonlinearity
  tests. *Physical Review Letters*, 77, 635--638.（IAAFT 代理数据）
* Riedel, S. & Sidorin, J. (2008). Bispectrum analysis of paleoclimatic time
  series. *Quaternary Science Reviews*, 27, 1958.（古环境双谱应用）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12-22.（Acycle 的 "bispectral" 子功能）

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ValidationError
from utils.statistics_core import make_rng

logger = logging.getLogger(__name__)

#: 每个 Welch 分段至少需要的样本数（少于这个数，三阶量的估计噪声过大）。
MIN_SEGMENT_SAMPLES = 32

#: 双谱分析至少需要的样本数（保证至少两个分段，估计量才非退化）。
MIN_SAMPLES = 2 * MIN_SEGMENT_SAMPLES

#: 允许的最多分段数。
MAX_SEGMENTS = 16

#: 分段之间的重叠比例（Welch 的 50% 重叠）。
SEGMENT_OVERLAP = 0.5

#: IAAFT 代理序列的默认迭代次数（Schreiber & Schmitz 建议 20--100 次）。
IAAFT_ITERATIONS = 100

#: 点态显著性水平：代理数据 ``|B|`` 的 ``1 - alpha`` 分位数作为阈值。
DEFAULT_ALPHA = 0.05


# =============================================================================
# Result container
# =============================================================================


@dataclass(frozen=True)
class BispectrumResult:
    """
    双谱与双相干分析的结果。

    Attributes:
        frequencies: 第一个频率轴（每单位时间的循环频率，严格递增）。
        frequencies2: 第二个频率轴（每单位时间的循环频率）。当前与
            ``frequencies`` 相同；分开保留是为了让 "``f1`` 轴" 与 "``f2`` 轴"
            在未来的非对称扩展中不必改接口。
        bispectrum: 复双谱 ``B(f1, f2)``，形状 ``(n_f1, n_f2)``。主域之外
            （``f1 + f2`` 超出频率轴）的组合为 ``NaN``。
        bicoherence: 归一化双相干，取值 ``0..1``；主域之外为 ``NaN``。
        phase: 双谱相位 ``arg B(f1, f2)``（弧度，``-pi..pi``）；主域之外为
            ``NaN``。
        significance: 显著性阈值（双相干的量纲，无量纲 ``0..1``），由代理数据
            给出；形状 ``(n_f1, n_f2)``。``bicoherence[i, j] > significance[i, j]``
            表示该三元组的相位耦合在 5% 水平上显著。没有做显著性检验时为
            ``None``。
    """

    frequencies: npt.NDArray
    frequencies2: npt.NDArray
    bispectrum: npt.NDArray
    bicoherence: npt.NDArray
    phase: npt.NDArray
    significance: npt.NDArray | None = None

    def summary(self) -> str:
        """生成结果摘要文本。"""
        magnitude = np.abs(self.bispectrum)
        finite = np.isfinite(magnitude)
        lines = [
            _("Bispectrum and bicoherence"),
            "=" * 48,
            _("Grid: {0} x {1} (f1, f2)").format(self.frequencies.size, self.frequencies2.size),
            _("Frequency range: {0} to {1} per unit time").format(
                f"{self.frequencies[0]:.6g}", f"{self.frequencies[-1]:.6g}"
            ),
        ]
        if np.any(finite):
            masked = np.where(finite, magnitude, -np.inf)
            row, column = np.unravel_index(int(np.argmax(masked)), masked.shape)
            lines.append(
                _("Peak |B|: {0} at ({1}, {2}) per unit time").format(
                    f"{magnitude[row, column]:.6g}",
                    f"{self.frequencies[row]:.6g}",
                    f"{self.frequencies2[column]:.6g}",
                )
            )
            lines.append(_("Phase at peak: {0} degrees").format(f"{np.degrees(self.phase[row, column]):.2f}"))
        coherence = self.bicoherence[np.isfinite(self.bicoherence)]
        if coherence.size:
            lines.append(_("Max bicoherence: {0}").format(f"{float(np.max(coherence)):.4f}"))
        if self.significance is not None:
            threshold = np.where(np.isfinite(self.significance), self.significance, np.inf)
            coherence = np.where(np.isfinite(self.bicoherence), self.bicoherence, np.nan)
            significant = np.isfinite(coherence) & (coherence > threshold)
            lines.append(
                _("Significant triads (95%): {0} of {1}").format(
                    int(significant.sum()), int(np.isfinite(coherence).sum())
                )
            )
        return "\n".join(lines)


# =============================================================================
# Internal helpers
# =============================================================================


def _as_vector(values: object, name: str, *, minimum_size: int) -> npt.NDArray:
    """把输入转换为一维有限数组，并检查长度下限。"""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("{0} must be numeric").format(name),
            details={"parameter": name},
            original_exception=exc,
        ) from exc
    if array.ndim != 1:
        raise ValidationError(
            _("{0} must be a one-dimensional array").format(name),
            details={"parameter": name, "ndim": int(array.ndim)},
        )
    if array.size < minimum_size:
        raise ValidationError(
            _("{0} needs at least {1} points; got {2}").format(name, minimum_size, int(array.size)),
            details={"parameter": name, "length": int(array.size), "minimum": minimum_size},
        )
    if not np.all(np.isfinite(array)):
        raise ValidationError(
            _("{0} must contain only finite values").format(name),
            details={"parameter": name},
        )
    return array


def _validate_dt(dt: object) -> float:
    """检查采样间隔：必须是正的有限数。"""
    try:
        value = float(dt)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("dt must be a positive number"),
            details={"parameter": "dt"},
            original_exception=exc,
        ) from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValidationError(
            _("dt must be a positive finite number"),
            details={"parameter": "dt", "dt": dt},
        )
    return value


def _validate_nfft(nfft: object) -> int:
    """检查 FFT 长度：必须是至少 :data:`MIN_SEGMENT_SAMPLES` 的整数。"""
    try:
        value = int(nfft)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("nfft must be an integer"),
            details={"parameter": "nfft"},
            original_exception=exc,
        ) from exc
    if value != nfft or value < MIN_SEGMENT_SAMPLES:
        raise ValidationError(
            _("nfft must be an integer of at least {0}").format(MIN_SEGMENT_SAMPLES),
            details={"parameter": "nfft", "nfft": nfft, "minimum": MIN_SEGMENT_SAMPLES},
        )
    return value


def _validate_n_surrogates(n_surrogates: object) -> int:
    """检查代理序列个数：必须是至少 1 的整数。"""
    try:
        value = int(n_surrogates)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("n_surrogates must be an integer"),
            details={"parameter": "n_surrogates"},
            original_exception=exc,
        ) from exc
    if value != n_surrogates or value < 1:
        raise ValidationError(
            _("n_surrogates must be an integer of at least 1"),
            details={"parameter": "n_surrogates", "n_surrogates": n_surrogates},
        )
    return value


def _validate_normalize(normalize: object) -> bool:
    """检查归一化开关：必须是布尔值。"""
    if not isinstance(normalize, (bool, np.bool_)):
        raise ValidationError(
            _("normalize must be a boolean"),
            details={"parameter": "normalize", "normalize": normalize},
        )
    return bool(normalize)


def _plan_segments(n_samples: int, nfft: int) -> tuple[int, list[int]]:
    """规划 Welch 分段：返回 ``(分段长度, 分段起点列表)``。

    分段长度取 ``nfft``；记录太短（放不下两个分段）时自动缩短到不超过
    ``n_samples / 2`` 的最大 2 的幂，保证至少两个分段——单分段会让双相干
    退化成常数。
    """
    if n_samples >= 2 * nfft:
        length = nfft
    else:
        length = 1 << int(np.floor(np.log2(max(n_samples // 2, 1))))
    if length < MIN_SEGMENT_SAMPLES:
        raise ValidationError(
            _("values needs at least {0} points for two segments of {1}; got {2}").format(
                2 * MIN_SEGMENT_SAMPLES, MIN_SEGMENT_SAMPLES, n_samples
            ),
            details={"n_samples": int(n_samples), "minimum": 2 * MIN_SEGMENT_SAMPLES},
        )
    step = max(1, round(length * SEGMENT_OVERLAP))
    starts = list(range(0, n_samples - length + 1, step))
    if len(starts) > MAX_SEGMENTS:
        picks = np.linspace(0, n_samples - length, MAX_SEGMENTS)
        starts = [round(value) for value in picks]
    return length, starts


def _detrend_segment(segment: npt.NDArray) -> npt.NDArray:
    """对单个分段做线性去趋势（去均值 + 去斜率）。"""
    x = np.arange(segment.size, dtype=float)
    centered_x = x - x.mean()
    centered = segment - segment.mean()
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 0.0:
        return centered
    slope = float(np.dot(centered_x, centered)) / denominator
    return centered - slope * centered_x


def _segment_spectra(series: npt.NDArray, length: int, starts: list[int]) -> npt.NDArray:
    """逐分段做 FFT，返回形状 ``(length, n_segments)`` 的复谱矩阵。

    每段先做线性去趋势，再乘 Hann 窗，最后做一次 FFT。
    """
    window = np.hanning(length)
    spectra = np.empty((length, len(starts)), dtype=complex)
    for index, start in enumerate(starts):
        segment = _detrend_segment(series[start : start + length])
        spectra[:, index] = np.fft.fft(segment * window)
    return spectra


def _estimate_arrays(
    series: npt.NDArray,
    length: int,
    starts: list[int],
    bins: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """计算双谱、双相干与相位。

    Returns:
        tuple: ``(双谱, 双相干, 相位, 主域掩膜)``；主域之外的三者都是 ``NaN``。
    """
    spectra = _segment_spectra(series, length, starts)
    n_segments = spectra.shape[1]
    grid = bins.size
    valid = (bins[:, None] + bins[None, :]) <= bins[-1]

    sum_triple = np.zeros((grid, grid), dtype=complex)
    sum_product = np.zeros((grid, grid), dtype=float)
    sum_third = np.zeros((grid, grid), dtype=float)

    positive = spectra[bins]
    third_index = bins[:, None] + bins[None, :]
    for index in range(n_segments):
        column = positive[:, index]
        product = np.outer(column, column)
        third = spectra[third_index, index]
        sum_triple += product * np.conj(third)
        sum_product += np.abs(product) ** 2
        sum_third += np.abs(third) ** 2

    bispectrum_values = sum_triple / n_segments
    denominator = sum_product * sum_third
    with np.errstate(divide="ignore", invalid="ignore"):
        bicoherence = np.where(
            denominator > 0.0,
            np.abs(sum_triple) ** 2 / np.where(denominator > 0.0, denominator, 1.0),
            np.nan,
        )
    bicoherence = np.clip(np.real(bicoherence), 0.0, 1.0)
    phase = np.angle(bispectrum_values)

    bispectrum_values = np.where(valid, bispectrum_values, np.nan)
    bicoherence = np.where(valid, bicoherence, np.nan)
    phase = np.where(valid, phase, np.nan)
    return bispectrum_values, bicoherence, phase, valid


def _phase_randomized(series: npt.NDArray, rng: np.random.Generator) -> npt.NDArray:
    """相位随机化代理：保留幅度谱，相位被独立随机的相位替换。

    只在正频段（含直流）抽取独立的均匀随机相位，负频段用**共轭镜像**补齐，
    因此代理仍然是实序列，幅度谱逐点保持不变。
    """
    spectrum = np.fft.fft(series)
    magnitudes = np.abs(spectrum)
    size = spectrum.size
    half = size // 2
    phases = np.zeros(size, dtype=float)
    phases[: half + 1] = rng.uniform(-np.pi, np.pi, half + 1)
    phases[0] = 0.0
    if size % 2 == 0:
        phases[half] = 0.0
    mirror = (size - np.arange(half + 1, size)) % size
    phases[half + 1 :] = -phases[mirror]
    randomized = magnitudes * np.exp(1j * phases)
    return np.fft.ifft(randomized).real


def _iaaft(
    series: npt.NDArray,
    rng: np.random.Generator,
    n_iterations: int = IAAFT_ITERATIONS,
) -> npt.NDArray:
    """迭代幅度自适应傅里叶变换（IAAFT）代理序列。

    迭代步骤（Schreiber & Schmitz 1996）：

    1. **谱投影**：用原序列的幅度谱与当前代理的相位重建信号（保持功率谱）；
    2. **秩映射**：把重建结果的取值按秩换回原序列的取值集合（保持边缘分布）。

    收敛后的代理与原序列共享取值分布与幅度谱，但三阶相位耦合被抹掉。
    """
    sorted_values = np.sort(series)
    amplitudes = np.abs(np.fft.fft(series))
    surrogate = series[rng.permutation(series.size)]
    for iteration in range(n_iterations):
        phases = np.angle(np.fft.fft(surrogate))
        projection = np.fft.ifft(amplitudes * np.exp(1j * phases)).real
        ranks = np.argsort(projection)
        surrogate = np.empty_like(sorted_values)
        surrogate[ranks] = sorted_values
    return surrogate


# =============================================================================
# Public interface
# =============================================================================


def bispectrum(
    values: npt.NDArray,
    dt: float = 1.0,
    nfft: int = 256,
    normalize: bool = True,
) -> BispectrumResult:
    """计算序列的双谱、归一化双相干与双谱相位。

    双谱的定义是三个频率的三阶矩::

        B(f1, f2) = E[ X(f1) * X(f2) * conj(X(f1 + f2)) ]

    其中 ``X`` 是分段平均后的复傅里叶谱。归一化双相干按 Kim & Powers (1992)
    给出，取值 ``0..1``::

        b(f1, f2) = |B(f1, f2)|^2 / ( E[|X(f1) X(f2)|^2] * E[|X(f1 + f2)|^2] )

    双谱相位 ``arg B(f1, f2)`` 给出三个分量的相位关系，是双谱的物理意义所在。

    Parameters:
        values: 一维实值序列（地层代用指标等），至少 :data:`MIN_SAMPLES` 个点。
        dt: 采样间隔（每单位时间的步长）。频率轴是**每单位时间**的循环频率。
        nfft: FFT（分段）长度。记录太短放不下两个分段时会自动缩短。
        normalize: 是否计算归一化双相干。为 ``False`` 时 ``bicoherence``
            全部为 ``NaN``（原始幅度由 ``bispectrum`` 给出）。

    Returns:
        BispectrumResult: 复双谱、``0..1`` 的双相干、相位与频率轴。

    Raises:
        ValidationError: 输入非法（长度不足、非一维、含非有限值、常数序列）、
            ``dt`` 非正、或 ``nfft`` 不是至少 32 的整数。

    Notes:
        频率轴只覆盖**主域**（``f1 + f2`` 仍落在轴内）；主域之外的组合取
        ``NaN``。显著性检验见 :func:`bicoherence_significance`。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.bispectrum import bispectrum
        >>> time = np.arange(884, dtype=float)
        >>> signal = np.cos(2 * np.pi * 17 / 221 * time) + np.cos(2 * np.pi * 13 / 221 * time)
        >>> result = bispectrum(signal, dt=1.0, nfft=221)
        >>> bool(result.bispectrum.shape == (result.frequencies.size, result.frequencies2.size))
        True
    """
    series = _as_vector(values, "values", minimum_size=MIN_SAMPLES)
    sample_interval = _validate_dt(dt)
    fft_length = _validate_nfft(nfft)
    want_normalization = _validate_normalize(normalize)
    if float(np.std(series)) <= 0.0:
        raise ValidationError(
            _("values must not be constant"),
            details={"parameter": "values"},
        )

    length, starts = _plan_segments(series.size, fft_length)
    bins = np.arange(1, length // 2)
    bispectrum_values, bicoherence, phase, valid = _estimate_arrays(series, length, starts, bins)
    frequencies = bins / (length * sample_interval)

    logger.info(
        "Bispectrum complete: %d points, %d segments of %d, grid %dx%d (%d valid triads), dt = %.6g",
        int(series.size),
        len(starts),
        length,
        int(bins.size),
        int(bins.size),
        int(valid.sum()),
        sample_interval,
    )
    return BispectrumResult(
        frequencies=frequencies,
        frequencies2=frequencies.copy(),
        bispectrum=bispectrum_values,
        bicoherence=bicoherence if want_normalization else np.full_like(bicoherence, np.nan),
        phase=phase,
        significance=None,
    )


def bicoherence_significance(
    values: npt.NDArray,
    dt: float = 1.0,
    nfft: int = 256,
    n_surrogates: int = 100,
    iaaft: bool = True,
    random_seed: int | None = None,
) -> BispectrumResult:
    """用代理数据给出双相干的显著性阈值。

    零假设是"三个频率之间没有相位耦合"。代理数据用 **IAAFT**（迭代幅度自适应
    傅里叶变换）生成：它保留原序列的取值分布与幅度谱，但把相位打乱，因此三阶
    耦合被系统性地抹掉。``iaaft=False`` 时退化为普通相位随机化代理（只保留
    幅度谱）。

    对每个代理算一遍同样的双相干，取其 95% 分位数作为该 ``(f1, f2)`` 的阈值：
    ``bicoherence[i, j] > significance[i, j]`` 表示该三元组的相位耦合在 5% 水平
    上显著。

    Parameters:
        values: 一维实值序列，至少 :data:`MIN_SAMPLES` 个点。
        dt: 采样间隔（每单位时间的步长）。
        nfft: FFT（分段）长度。
        n_surrogates: 代理序列条数，至少 1。
        iaaft: 是否使用 IAAFT 代理（``True``，默认）还是相位随机化代理。
        random_seed: 随机种子。给定种子时结果完全可复现；不给种子时调用
            ``utils.statistics_core.make_rng(None, ...)``，它会发出
            ``RuntimeWarning`` 提示结果不可复现。

    Returns:
        BispectrumResult: 与 :func:`bispectrum` 相同的字段，另加
        ``significance``（双相干量纲的阈值矩阵）。

    Raises:
        ValidationError: 输入非法、``n_surrogates`` 不是至少 1 的整数。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.bispectrum import bicoherence_significance
        >>> signal = np.random.default_rng(0).standard_normal(256)
        >>> result = bicoherence_significance(signal, nfft=64, n_surrogates=4, random_seed=1)
        >>> result.significance is not None
        True
    """
    series = _as_vector(values, "values", minimum_size=MIN_SAMPLES)
    sample_interval = _validate_dt(dt)
    fft_length = _validate_nfft(nfft)
    n_surrogates = _validate_n_surrogates(n_surrogates)
    if float(np.std(series)) <= 0.0:
        raise ValidationError(
            _("values must not be constant"),
            details={"parameter": "values"},
        )

    length, starts = _plan_segments(series.size, fft_length)
    bins = np.arange(1, length // 2)
    bispectrum_values, bicoherence, phase, valid = _estimate_arrays(series, length, starts, bins)
    frequencies = bins / (length * sample_interval)

    rng = make_rng(random_seed, context="bispectrum IAAFT surrogates")
    surrogates = np.zeros((n_surrogates, bins.size, bins.size), dtype=float)
    for index in range(n_surrogates):
        if iaaft:
            surrogate = _iaaft(series, rng)
        else:
            surrogate = _phase_randomized(series, rng)
        # The throwaway names are spelled out: a bare "_" would shadow the
        # translation function this module uses, and the error paths would
        # raise UnboundLocalError instead of the ValidationError they are
        # written to raise.
        _spectrum, surrogate_bicoherence, _phase_forward, _phase_backward = _estimate_arrays(
            surrogate, length, starts, bins
        )
        finite_coherence = np.where(np.isfinite(surrogate_bicoherence), surrogate_bicoherence, 0.0)
        surrogates[index] = np.where(valid, finite_coherence, 0.0)

    threshold = np.quantile(surrogates, 1.0 - DEFAULT_ALPHA, axis=0)
    threshold = np.where(valid, threshold, np.nan)

    logger.info(
        "Bispectrum significance complete: %d surrogates, %d segments of %d, grid %dx%d",
        int(n_surrogates),
        len(starts),
        length,
        int(bins.size),
        int(bins.size),
    )
    return BispectrumResult(
        frequencies=frequencies,
        frequencies2=frequencies.copy(),
        bispectrum=bispectrum_values,
        bicoherence=bicoherence,
        phase=phase,
        significance=threshold,
    )
