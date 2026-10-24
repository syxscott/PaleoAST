# =============================================================================
# FILE: stratigraphy/pda.py
# =============================================================================
"""
功率分解分析（Power Decomposition Analysis, PDA）。

PDA 把一条古环境/同位素记录的**功率**按若干“特征周期带”拆开，回答“这个记录
里有多少功率落在 100 kyr、41 kyr、23 kyr 这几个天文周期上，各占多少”。它与
功率谱（spectrum）的区别在于：谱分析给出的是“一个频率上有多少功率”，PDA 给出
的是“**若干个预先选定的带**各拿走了总功率的多少”，剩下的部分作为**残差**
（既不是这几个带、也没有被任何其它成分解释的部分）单独报告。

方法依据
~~~~~~~~
Li, M., Huang, C., Hinnov, L., Ogg, J., Chen, Z.-Q. & Zhang, Y. 2016.
Power decomposition analysis for the detection of cyclostratigraphic
cycles. *Geology* 44: 623-626. doi:10.1130/G37970.1.

论文的做法是：先算记录的总功率谱，再把预先选定的周期带（例如 100 ± 5 kyr、
41 ± 3 kyr、23 ± 2 kyr）从总谱里逐个“扣除”，最后剩下的谱就是残差谱；每个带
的功率占比即该带功率占总功率的比例。本模块按该思路独立实现（未翻译 Acycle 的
GPL 源码），做法如下：

1. **重采样与去趋势**：时间轴若不是严格等间隔，先线性插值到等间隔网格
   （步长取平均间隔），再对数值做最小二乘直线去趋势。趋势属于最低频率成分，
   不去趋势会通过谱泄漏污染最长的周期带。
2. **带通滤波**：每个带用 ``scipy.signal.butter`` 设计 4 阶带通滤波器，以
   二阶节（SOS）形式做**零相位**滤波（``sosfiltfilt``，即 ``filtfilt`` 的
   二阶节形式；窄带降阶滤波器直接展开成传递函数多项式会数值病态，故用 SOS）。
   窄带滤波器的冲激响应很长，直接滤波会在记录两端留下很强的振铃，因此滤波
   前把记录**镜像延拓**若干个周期、滤完再裁回。滤波在等间隔网格上完成，再
   插值回原始时间轴，使返回的 ``filtered`` 与输入逐点对齐。
3. **逐带分解**：按给定顺序依次在**当前残差**上滤波，每得到一个成分就从残差
   里减掉，因此后一个带看到的是前面各带都拿不走的那部分功率。这是“分解”
   （decomposition）而不是“分别滤波”。
4. **功率占比**：总功率取去趋势序列的均方功率；某个带的功率取该带**被滤出的
   成分**的均方功率（与论文里“在带内积分功率谱”等价：带限信号的均方功率就
   是它在带内的谱功率）；占比为二者之比。各带成分位于互不重叠的频带上，近似
   互不相关，因此“各带占比之和 + 残差占比 ≈ 1”。
5. **不确定度**：对分解残差做**循环分块重采样**（block bootstrap），把重采样
   后的残差加回“已解释部分”得到重抽样信号，对每个重抽样信号重复同一套分解。
   该分布回答的是：若残差其实是噪声，它还能往各个带里“塞”进多少功率。噪声
   只会给带内增加功率，所以区间位于观测占比之上，观测占比是区间的下界。

单位约定
~~~~~~~~
* ``time`` 与所有以 ``_kyr`` 结尾的参数单位一致，本模块统一按 **kyr** 处理。
* 带的写法是 ``(中心周期, 半宽)``，例如 ``(100, 5)`` 表示 95–105 kyr。

参考文献：
    1. Li, M., Huang, C., Hinnov, L., Ogg, J., Chen, Z.-Q. & Zhang, Y. 2016.
       Power decomposition analysis for the detection of cyclostratigraphic
       cycles. Geology 44: 623-626.
    2. Hinnov, L. A. & Hilvar, R. A. 2012. Cyclostratigraphy. In Handbook of
       Geology (2nd ed.), Academic Press. (轨道周期带的选择依据。)
    3. Li, M., Hinnov, L. & Kump, L. 2019. Acycle: time-series analysis
       software for paleoclimate projects and education. Computers &
       Geosciences. (Acycle 软件本身，其 PDA 工具是本模块的对照实现。)

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import hann

from config.i18n import _
from utils.exceptions import DataValidationError
from utils.statistics_core import make_rng

logger = logging.getLogger(__name__)

#: 默认的特征周期带：``(中心周期 kyr, 半宽 kyr)``，对应 100/41/23 kyr 轨道周期。
DEFAULT_BANDS: tuple[tuple[float, float], ...] = ((100.0, 5.0), (41.0, 3.0), (23.0, 2.0))

#: 记录至少要有这么多个样本，滤波（零相位 filtfilt 的补零长度）才成立。
MIN_SAMPLES = 32

#: 带通滤波器的阶数（``butter`` 的 ``N``；带通时总阶数为 2N）。
FILTER_ORDER = 4

#: 滤波前的镜像延拓长度，以带内频率宽度 ``delta_f`` 的倒数衡量：延拓
#: ``EDGE_PAD_WIDTHS * 1 / delta_f`` 个样本，足以让窄带滤波器的边缘振铃衰减。
EDGE_PAD_WIDTHS = 8.0

#: 功率谱的零填充倍数：只增加谱线的采样密度，不改变总功率（Parseval）。
SPECTRUM_PADDING = 8

#: 记录时间跨度至少要覆盖多少个中心周期，否则带内信号无法被可靠识别。
MIN_CYCLES_PER_BAND = 2.0

#: bootstrap 置信区间的分位数。
CI_LOW_PERCENTILE = 2.5
CI_HIGH_PERCENTILE = 97.5


# =============================================================================
# Result containers
# =============================================================================


@dataclass(frozen=True)
class PDABand:
    """单个特征周期带的功率分解结果。

    Attributes:
        centre_kyr: 带中心周期，单位 kyr。
        half_width_kyr: 带半宽，单位 kyr；带宽为 ``2 * half_width_kyr``。
        power_fraction: 该带功率占记录总功率的比例。
        filtered: 从序列中滤出的该带成分，与输入 ``values`` 逐点对齐。
        power_fraction_ci: 功率占比的 bootstrap 区间 ``(下界, 上界)``；
            未做 bootstrap 时为 ``None``。
        power_fraction_sd: 功率占比的 bootstrap 标准差；未做 bootstrap 时为 ``None``。
        peak_period_kyr: 该带成分的主周期（谱峰位置），单位 kyr。
    """

    centre_kyr: float
    half_width_kyr: float
    power_fraction: float
    filtered: npt.NDArray
    power_fraction_ci: tuple[float, float] | None = None
    power_fraction_sd: float | None = None
    peak_period_kyr: float | None = None

    @property
    def low_period_kyr(self) -> float:
        """带的最短周期（高频边），单位 kyr。"""
        return self.centre_kyr - self.half_width_kyr

    @property
    def high_period_kyr(self) -> float:
        """带的最长周期（低频边），单位 kyr。"""
        return self.centre_kyr + self.half_width_kyr


@dataclass(frozen=True)
class PDAResult:
    """功率分解的总体结果。

    Attributes:
        bands: 各特征周期带的结果，顺序与输入 ``bands`` 一致。
        residual: 各带都没有解释掉的部分，与输入 ``values`` 逐点对齐
            （含被去趋势掉的直线趋势项）。
        total_power: 去趋势序列的总功率（均方功率）。
        residual_power_fraction: 残差功率占总功率的比例。
        n_bootstrap: 实际执行的 bootstrap 重复次数。
        random_seed: 本次使用的随机种子；``None`` 表示未固定，结果不可复现。
    """

    bands: list[PDABand] = field(default_factory=list)
    residual: npt.NDArray | None = None
    total_power: float | None = None
    residual_power_fraction: float | None = None
    n_bootstrap: int = 0
    random_seed: int | None = None

    def summary(self) -> str:
        """生成结果摘要文本。"""
        lines = [
            _("Power Decomposition Analysis"),
            "=" * 50,
            _("Bands: {0}").format(len(self.bands)),
        ]
        for band in self.bands:
            interval = ""
            if band.power_fraction_ci is not None:
                interval = _(" [{0:.3f}, {1:.3f}]").format(*band.power_fraction_ci)
            lines.append(
                _("  {0} +/- {1} kyr: power fraction {2:.4f}{3}").format(
                    band.centre_kyr,
                    band.half_width_kyr,
                    band.power_fraction,
                    interval,
                )
            )
        if self.residual_power_fraction is not None:
            lines.append(_("  residual: power fraction {0:.4f}").format(self.residual_power_fraction))
        return "\n".join(lines)


# =============================================================================
# Validation helpers
# =============================================================================


def _as_float_array(values: npt.NDArray, name: str) -> npt.NDArray:
    """把输入转成一维有限 ``float64`` 数组。"""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            _("{0} must be numeric; got {1}").format(name, type(values).__name__),
            details={"name": name},
        ) from exc
    if array.ndim != 1:
        raise DataValidationError(
            _("{0} must be a one-dimensional array").format(name),
            details={"name": name, "ndim": int(array.ndim)},
        )
    if not np.all(np.isfinite(array)):
        raise DataValidationError(
            _("{0} must contain only finite values").format(name),
            details={"name": name},
        )
    return array


def _validate_bands(bands: object) -> list[tuple[float, float]]:
    """校验带参数，返回 ``[(中心周期, 半宽), ...]``。"""
    if bands is None:
        return [(float(centre), float(half)) for centre, half in DEFAULT_BANDS]
    try:
        items = list(bands)  # type: ignore[call-overload]
    except TypeError as exc:
        raise DataValidationError(
            _("bands must be an iterable of (centre_kyr, half_width_kyr) pairs"),
            details={"type": type(bands).__name__},
        ) from exc
    if not items:
        raise DataValidationError(_("bands must contain at least one band"), details={"n_bands": 0})

    validated: list[tuple[float, float]] = []
    for index, item in enumerate(items):
        try:
            centre, half_width = (float(part) for part in item)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                _("band {0} must be a (centre_kyr, half_width_kyr) pair").format(index),
                details={"index": index},
            ) from exc
        if not np.isfinite(centre) or not np.isfinite(half_width):
            raise DataValidationError(
                _("band {0} must have finite centre and half width").format(index),
                details={"index": index},
            )
        if centre <= 0.0:
            raise DataValidationError(
                _("band {0} centre must be positive; got {1}").format(index, centre),
                details={"index": index, "centre_kyr": centre},
            )
        if half_width <= 0.0:
            raise DataValidationError(
                _("band {0} half width must be positive; got {1}").format(index, half_width),
                details={"index": index, "half_width_kyr": half_width},
            )
        if centre - half_width <= 0.0:
            raise DataValidationError(
                _("band {0} half width must be smaller than its centre period").format(index),
                details={"index": index, "centre_kyr": centre, "half_width_kyr": half_width},
            )
        validated.append((centre, half_width))
    return validated


def _band_frequencies(centre_kyr: float, half_width_kyr: float) -> tuple[float, float]:
    """把带参数换算成频率边（低频在前，高频在后），单位 1/kyr。"""
    low_frequency = 1.0 / (centre_kyr + half_width_kyr)
    high_frequency = 1.0 / (centre_kyr - half_width_kyr)
    return low_frequency, high_frequency


# =============================================================================
# Signal preparation
# =============================================================================


def _prepare(time: npt.NDArray, values: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, float]:
    """校验时间轴与数值，并重采样到等间隔网格。

    Args:
        time: 严格递增的时间轴，单位 kyr。
        values: 与 ``time`` 等长的数值序列。

    Returns:
        ``(uniform_time, uniform_values, dt)``，``dt`` 为平均采样间隔。

    Raises:
        DataValidationError: 输入不合法。
    """
    t = _as_float_array(time, "time")
    v = _as_float_array(values, "values")
    if t.size != v.size:
        raise DataValidationError(
            _("time and values must have the same length; got {0} and {1}").format(t.size, v.size),
            details={"n_time": int(t.size), "n_values": int(v.size)},
        )
    if t.size < MIN_SAMPLES:
        raise DataValidationError(
            _("PDA needs at least {0} samples; got {1}").format(MIN_SAMPLES, t.size),
            details={"n_samples": int(t.size), "minimum": MIN_SAMPLES},
        )
    spacing = np.diff(t)
    if np.any(spacing <= 0.0):
        raise DataValidationError(
            _("time must be strictly increasing"),
            details={"n_decreasing": int(np.count_nonzero(spacing <= 0.0))},
        )
    dt = float((t[-1] - t[0]) / (t.size - 1))
    if not np.isfinite(dt) or dt <= 0.0:
        raise DataValidationError(
            _("time must advance through a finite positive span"),
            details={"dt": dt},
        )
    uniform_time = np.linspace(t[0], t[-1], t.size)
    uniform_values = np.interp(uniform_time, t, v)
    return uniform_time, uniform_values, dt


def _linear_detrend(values: npt.NDArray) -> npt.NDArray:
    """去掉最小二乘直线趋势，返回残差。"""
    index = np.arange(values.size, dtype=float)
    centred = index - float(index.mean())
    denominator = float(np.dot(centred, centred))
    if denominator <= 0.0:
        return values - float(values.mean())
    slope = float(np.dot(centred, values - float(values.mean())) / denominator)
    return values - (float(values.mean()) + slope * centred)


def _mean_square(values: npt.NDArray) -> float:
    """均方功率（除以点数，零均值的功率与方差同量纲）。"""
    return float(np.dot(values, values) / values.size)


def _bandpass(values: npt.NDArray, low_frequency: float, high_frequency: float, dt: float) -> npt.NDArray:
    """零相位带通滤波。

    滤波器由 ``scipy.signal.butter`` 设计成 4 阶带通，再拆成二阶节（SOS）做
    ``sosfiltfilt``：窄带降阶滤波器直接展开成传递函数多项式会数值病态，SOS
    形式在相同设计下稳定得多，零相位特性与 ``filtfilt`` 一致。

    窄带滤波器的冲激响应很长，直接滤波会在记录两端留下很强的振铃，因此滤波
    前把记录镜像延拓 ``EDGE_PAD_WIDTHS / delta_f`` 个样本（``delta_f`` 是带内
    频率宽度），滤完再裁回。

    Args:
        values: 输入序列（等间隔网格上）。
        low_frequency: 带边低频，单位 1/kyr。
        high_frequency: 带边高频，单位 1/kyr。
        dt: 采样间隔，单位 kyr。

    Returns:
        滤波后的序列。

    Raises:
        DataValidationError: 带边超出奈奎斯特频率或记录长度不足以滤波。
    """
    nyquist = 0.5 / dt
    if not 0.0 < low_frequency < high_frequency < nyquist:
        raise DataValidationError(
            _("band frequencies must satisfy 0 < low < high < Nyquist"),
            details={
                "low_frequency": float(low_frequency),
                "high_frequency": float(high_frequency),
                "nyquist": float(nyquist),
            },
        )
    sos = butter(FILTER_ORDER, [low_frequency, high_frequency], btype="bandpass", fs=1.0 / dt, output="sos")
    padlen = 3 * (2 * sos.shape[0] + 1)
    if values.size <= padlen:
        raise DataValidationError(
            _("record is too short for zero-phase band-pass filtering; need more than {0} samples").format(padlen),
            details={"n_samples": int(values.size), "minimum": int(padlen) + 1},
        )
    width = high_frequency - low_frequency
    pad = int(min(values.size, np.ceil(EDGE_PAD_WIDTHS / (width * dt))))
    padded = np.pad(values, pad, mode="reflect") if pad > 0 else values
    filtered = np.asarray(sosfiltfilt(sos, padded), dtype=float)
    if pad > 0:
        filtered = filtered[pad : pad + values.size]
    return filtered


def _spectrum(values: npt.NDArray, dt: float) -> tuple[npt.NDArray, npt.NDArray]:
    """计算单边功率谱（加 Hann 窗并零填充，零填充不改变总功率）。"""
    size = values.size
    nfft = int(SPECTRUM_PADDING * size)
    tapered = values * hann(size)
    spectrum = np.fft.rfft(tapered, nfft)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(nfft, d=dt)
    return frequencies, power


def _peak_frequency(frequencies: npt.NDArray, power: npt.NDArray, low: float, high: float) -> float:
    """在给定频率区间内找谱峰，用抛物线插值细化到亚 bin 精度。"""
    mask = (frequencies >= low) & (frequencies <= high)
    if not np.any(mask):
        return float("nan")
    indices = np.flatnonzero(mask)
    best = int(indices[int(np.argmax(power[indices]))])
    if best <= 0 or best >= power.size - 1:
        return float(frequencies[best])
    left, centre, right = np.log(np.maximum(power[best - 1 : best + 2], 1e-300))
    denominator = left - 2.0 * centre + right
    offset = 0.0 if denominator == 0.0 else 0.5 * (left - right) / denominator
    offset = float(np.clip(offset, -1.0, 1.0))
    spacing = float(frequencies[1] - frequencies[0])
    return float(frequencies[best] + offset * spacing)


# =============================================================================
# Decomposition
# =============================================================================


def _decompose(
    values: npt.NDArray,
    band_specs: list[tuple[float, float]],
    dt: float,
    reference_power: float | None = None,
) -> tuple[list[npt.NDArray], list[float], list[float]]:
    """在等间隔网格上做一次逐带分解。

    Args:
        values: 已去趋势的序列。
        band_specs: ``[(中心周期, 半宽), ...]``。
        dt: 采样间隔，单位 kyr。
        reference_power: 计算占比时的分母；``None`` 表示用 ``values`` 自身的
            总功率（观测估计用固定参考值，使 bootstrap 各次可比）。

    Returns:
        ``(components, fractions, band_powers)``：各带滤出的成分、各带功率
        占比、各带功率。
    """
    total_power = _mean_square(values) if reference_power is None else float(reference_power)

    components: list[npt.NDArray] = []
    fractions: list[float] = []
    band_powers: list[float] = []

    residual = values.copy()
    for centre, half_width in band_specs:
        low_frequency, high_frequency = _band_frequencies(centre, half_width)
        component = _bandpass(residual, low_frequency, high_frequency, dt)
        band_power = _mean_square(component)
        components.append(component)
        band_powers.append(band_power)
        fractions.append(band_power / total_power if total_power > 0.0 else 0.0)
        residual = residual - component

    return components, fractions, band_powers


def _circular_block_resample(values: npt.NDArray, rng: np.random.Generator, block_length: int) -> npt.NDArray:
    """循环分块重采样（circular moving-block bootstrap）。"""
    size = values.size
    if size == 0:
        return values.copy()
    length = max(1, min(int(block_length), size))
    n_blocks = int(np.ceil(size / length))
    starts = rng.integers(0, size, size=n_blocks)
    offsets = np.arange(length)
    indices = (starts[:, None] + offsets[None, :]) % size
    return values[indices.ravel()[:size]]


# =============================================================================
# Public interface
# =============================================================================


def power_decomposition(
    time: npt.NDArray,
    values: npt.NDArray,
    bands: tuple[tuple[float, float], ...] | None = None,
    n_bootstrap: int = 100,
    random_seed: int | None = None,
) -> PDAResult:
    """功率分解分析（Power Decomposition Analysis）。

    把记录的总功率按若干特征周期带拆开，报告各带的功率占比、被滤出的成分，
    以及所有带都没有解释掉的残差。

    Args:
        time: 时间轴，严格递增，单位 kyr。若不是严格等间隔，先线性插值到
            等间隔网格。
        values: 与 ``time`` 等长的数值序列。
        bands: 特征周期带，``(中心周期 kyr, 半宽 kyr)`` 的序列；``None`` 时用
            默认的 ``(100, 5)``、``(41, 3)``、``(23, 2)``。界面里写成
            ``"100:5, 41:3, 23:2"`` 这样的文本。
        n_bootstrap: bootstrap 重复次数，用于功率占比的不确定度；``0`` 表示
            不做 bootstrap（此时没有区间）。
        random_seed: 随机种子。``None`` 时结果不可复现，
            :func:`utils.statistics_core.make_rng` 会发出 ``RuntimeWarning``。

    Returns:
        PDAResult: 分解结果。

    Raises:
        DataValidationError: 输入或参数不合法。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.pda import power_decomposition
        >>> t = np.arange(0.0, 8000.0, 1.0)
        >>> signal = np.sin(2 * np.pi * t / 100.0) + np.sin(2 * np.pi * t / 41.0)
        >>> result = power_decomposition(t, signal, bands=[(100, 5), (41, 3)], n_bootstrap=0)
        >>> len(result.bands)
        2
    """
    band_specs = _validate_bands(bands)
    if n_bootstrap < 0:
        raise DataValidationError(
            _("n_bootstrap must not be negative; got {0}").format(n_bootstrap),
            details={"n_bootstrap": int(n_bootstrap)},
        )

    uniform_time, uniform_values, dt = _prepare(time, values)

    span = float(uniform_time[-1] - uniform_time[0])
    nyquist = 0.5 / dt
    resolution = 1.0 / (uniform_values.size * dt)
    for index, (centre, half_width) in enumerate(band_specs):
        if span < MIN_CYCLES_PER_BAND * centre:
            raise DataValidationError(
                _("record span {0:.4g} kyr covers fewer than {1} cycles of band {2} (centre {3} kyr)").format(
                    span, MIN_CYCLES_PER_BAND, index, centre
                ),
                details={"index": index, "span_kyr": span, "centre_kyr": centre},
            )
        low_frequency, high_frequency = _band_frequencies(centre, half_width)
        if high_frequency >= nyquist:
            raise DataValidationError(
                _("band {0} reaches above the Nyquist frequency; shorten the sampling interval").format(index),
                details={"index": index, "high_frequency": float(high_frequency), "nyquist": float(nyquist)},
            )
        if (high_frequency - low_frequency) < resolution:
            raise DataValidationError(
                _("band {0} is narrower than the frequency resolution of the record").format(index),
                details={
                    "index": index,
                    "band_width": float(high_frequency - low_frequency),
                    "resolution": float(resolution),
                },
            )

    detrended = _linear_detrend(uniform_values)
    # NOTE: the unused results must not be named ``_``: ``_`` is the i18n
    # translator imported above, and a local ``_`` would shadow it for the
    # whole function -- every error path below would then raise
    # UnboundLocalError instead of DataValidationError.
    components, fractions, _band_powers = _decompose(detrended, band_specs, dt)

    residual_uniform = detrended.copy()
    for component in components:
        residual_uniform = residual_uniform - component
    residual_power = _mean_square(residual_uniform)
    total_power = _mean_square(detrended)
    residual_fraction = residual_power / total_power if total_power > 0.0 else 0.0

    # 把各带成分插值回原始时间轴，保证 filtered 与输入逐点对齐。
    filtered_columns = [np.interp(time, uniform_time, component) for component in components]
    residual = np.asarray(values, dtype=float).copy()
    for column in filtered_columns:
        residual = residual - column

    # Bootstrap：对残差做循环分块重采样，得到重抽样信号，重复同一套分解。
    ci_low: npt.NDArray | None = None
    ci_high: npt.NDArray | None = None
    sd: npt.NDArray | None = None
    n_used = 0
    if n_bootstrap >= 1:
        rng = make_rng(random_seed, context="PDA")
        block_length = max(2, round(float(uniform_values.size) ** (1.0 / 3.0)))
        explained = detrended - residual_uniform
        replicates = np.zeros((int(n_bootstrap), len(band_specs)), dtype=float)
        for index in range(int(n_bootstrap)):
            resampled = _circular_block_resample(residual_uniform, rng, block_length)
            _replicate_components, replicate_fractions, _replicate_powers = _decompose(
                explained + resampled,
                band_specs,
                dt,
                reference_power=total_power,
            )
            replicates[index] = replicate_fractions
        ci_low = np.percentile(replicates, CI_LOW_PERCENTILE, axis=0)
        ci_high = np.percentile(replicates, CI_HIGH_PERCENTILE, axis=0)
        sd = np.std(replicates, axis=0)
        n_used = int(n_bootstrap)

    result_bands: list[PDABand] = []
    for index, (centre, half_width) in enumerate(band_specs):
        low_frequency, high_frequency = _band_frequencies(centre, half_width)
        frequencies, component_power = _spectrum(components[index], dt)
        peak_frequency = _peak_frequency(frequencies, component_power, low_frequency, high_frequency)
        peak_period = 1.0 / peak_frequency if peak_frequency > 0.0 else None
        interval = (float(ci_low[index]), float(ci_high[index])) if ci_low is not None else None
        result_bands.append(
            PDABand(
                centre_kyr=float(centre),
                half_width_kyr=float(half_width),
                power_fraction=float(fractions[index]),
                filtered=filtered_columns[index],
                power_fraction_ci=interval,
                power_fraction_sd=float(sd[index]) if sd is not None else None,
                peak_period_kyr=float(peak_period) if peak_period is not None else None,
            )
        )

    logger.debug(
        "PDA: %d bands, total power %.6g, residual fraction %.4f",
        len(result_bands),
        total_power,
        residual_fraction,
    )

    return PDAResult(
        bands=result_bands,
        residual=residual,
        total_power=total_power,
        residual_power_fraction=float(residual_fraction),
        n_bootstrap=n_used,
        random_seed=random_seed,
    )
