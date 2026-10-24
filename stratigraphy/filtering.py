# =============================================================================
# FILE: stratigraphy/filtering.py
# =============================================================================
"""
时间序列滤波：高斯带通、Taner 加权带通与通用带通滤波。

本模块把 Acycle (https://github.com/mingsongli/acycle) 的滤波功能移植为
Python，实现三个滤波器：

* :func:`gaussian_filter` —— Berger & Streck (1994) 与 Acycle 的高斯滤波。
  频域响应是偶对称的高斯

      H(f) = exp(-2 * pi**2 * sigma**2 * (|f| - f0)**2)

  其中 ``f0 = 1 / T`` 是目标周期 ``T`` 对应的中心频率，``sigma`` 是高斯核在
  时间轴上的标准差。目标周期给定后，高斯宽度按约定

      sigma = cutoff_period_kyr / (2 * pi * 1000)

  计算（单位 Ma，见下面的单位约定）。目标周期处增益为 1，直流处增益为
  ``exp(-1/2) ~ 0.607``，所以该滤波是**带通**：它保留目标周期，同时压制
  线性趋势这种最低频率成分。
* :func:`taner_filter` —— Kodama & Hinnov (2015) 的加权带通。核权重取自
  带边频率的 sinc 函数，即理想带通的时间域冲激响应
  ``h(k) = 2*fh*sinc(2*fh*k) - 2*fl*sinc(2*fl*k)``，并用 Blackman-Harris
  窗抑制 Gibbs 振铃。
* :func:`bandpass_filter` —— Acycle 式的带通，两条带边取 -3 dB（半功率）点，
  响应形状同样是高斯。

单位约定（务必先读）
~~~~~~~~~~~~~~~~~~~~
* ``dt`` 与 ``sigma`` 的单位是 **Ma**（时间轴单位，与仓库其它模块一致）。
* 所有以 ``_kyr`` 结尾的参数单位是 **kyr**；内部统一换算成 Ma。
  换算关系：1 Ma = 1000 kyr。
* 频率一律用“每 Ma 的周期数”表示，``f = 1 / T``。
* 周期参数的“长/短”按**周期长度**命名：``low_period_kyr`` 是带内最短的周期
  （对应最高频率），``high_period_kyr`` 是带内最长的周期（对应最低频率）。

举例：``dt = 0.001``（即 1 kyr 一个样本）、``cutoff_period_kyr = 20`` 表示
目标是 20 个样本（20 kyr）的周期。

不等间距时间轴
~~~~~~~~~~~~~~
三个滤波器都接受可选的 ``time`` 参数（单位 Ma，严格递增，长度与 ``values``
相同）。给出 ``time`` 时滤波器按**实际时间差**工作，而不是按样本序号：记录先
被线性插值到一张以实际时间差中位数为间距的均匀网格上，滤波之后再插值回原始
时间点。于是“不等间距”不会被悄悄当成“等间距”处理，带边频率与核滞后量都用
的是时间而不是序号。``time`` 为 None 时退化为等间距情形，由 ``dt`` 决定间距。

与 Acycle 的差异
~~~~~~~~~~~~~~~~
1. Acycle 的 ``gaussfilter`` 要求用户同时给出中心频率与两条带边频率，本模块
   用“目标周期”一个参数确定中心频率，用 ``sigma`` 控制带宽。
2. Acycle 的 Taner-Hilbert 在频域用广义高斯（超高斯）滚降实现；本模块按
   Kodama & Hinnov (2015) 的**加权**形式，在时域用 sinc 核做加权带通，权重
   由目标周期的 sinc 函数给出。
3. Acycle 的滤波全部在频域零填充后完成，因此记录两端有环绕伪影；本模块的
   时域滤波用镜像延拓（``pad``）压低边缘效应。
4. Acycle 的滤波窗口提供“显示滤波结果让用户决定是否继续”的交互流程；本模块
   是纯计算层，不含界面。调用方应先去趋势（Acycle 的 detrending 工具正是
   为此存在），再滤波。

参考：
    1. Berger, A. & Streck, M. 1994. Paleoceanographic significance of the
       "Damped Oscillation" in climate records of the last 250 ka. In
       Paleoclimate and Climate Change (NATO ASI Series I/26). (高斯滤波。)
    2. Kodama, K.P. & Hinnov, L. 2015. Rock Magnetic Cyclostratigraphy.
       Wiley-Blackwell. (Taner 加权带通滤波。)
    3. Taner, T. 2000. Attributes Revisited. Rock Solid Images, Inc.
       (Taner 滤波的原始描述。)
    4. Li, M., Hinnov, L. & Kump, L. 2019. Acycle: time-series analysis
       software for paleoclimate projects and education. Computers &
       Geosciences. (Acycle 软件本身。)

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.signal import fftconvolve
from scipy.signal.windows import blackmanharris

from config.i18n import _
from utils.exceptions import DataValidationError

logger = logging.getLogger(__name__)

#: 滤波器接受的最短记录长度；Acycle 的带通核同样要求至少 8 行。
MIN_SAMPLES = 8

#: kyr 与 Ma 的换算因子：1 Ma = 1000 kyr。
KYR_PER_MA = 1000.0

#: Acycle 的高斯带通响应为 exp(-ln(sqrt(2)) * ((f - fc) / half_width) ** 2)，
#: 于是两条带边正好是 -3 dB（半功率）点。
HALF_POWER_COEFFICIENT = float(np.log(np.sqrt(2.0)))

#: Taner 加权核的半宽（以样本计）至少要覆盖的低频振荡数。
TANER_KERNEL_OCTAVES = 8

#: Taner 加权核的半宽（以样本计）至少要覆盖的低频瓣数。
TANER_KERNEL_LOBES = 4


# =============================================================================
# Result container
# =============================================================================


@dataclass(frozen=True)
class FilterResult:
    """
    滤波结果。

    Attributes:
        filtered: 滤波后的序列，与 ``original`` 等长。
        original: 滤波前的原始序列副本。
        method: 滤波方法名，取 ``"gaussian"`` 或 ``"taner"``。
        parameters: 本次滤波实际使用的参数，便于复现与登记。
    """

    filtered: npt.NDArray
    original: npt.NDArray
    method: str
    parameters: dict[str, Any]

    def summary(self) -> str:
        """生成结果摘要文本。"""
        details = ", ".join(f"{key}={value}" for key, value in self.parameters.items())
        return (
            f"{self.method} filter\n"
            f"{'=' * 48}\n"
            f"Samples: {int(self.filtered.size)}\n"
            f"Parameters: {details}\n"
            f"Original range: [{self.original.min():.6g}, {self.original.max():.6g}]\n"
            f"Filtered range: [{self.filtered.min():.6g}, {self.filtered.max():.6g}]"
        )


# =============================================================================
# Validation helpers
# =============================================================================


def _as_series(values: npt.NDArray, name: str = "values") -> npt.NDArray:
    """把输入转成一维有限数组，并检查长度下限。"""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            _("filter input must be numeric; got {0}").format(type(values).__name__),
            details={"name": name},
        ) from exc
    if array.ndim != 1:
        raise DataValidationError(
            _("filter input must be one-dimensional; got shape {0}").format(array.shape),
            details={"name": name, "shape": array.shape},
        )
    if array.size < MIN_SAMPLES:
        raise DataValidationError(
            _("filter input needs at least {0} samples; got {1}").format(MIN_SAMPLES, int(array.size)),
            details={"name": name, "length": int(array.size), "minimum": MIN_SAMPLES},
        )
    if not np.all(np.isfinite(array)):
        raise DataValidationError(
            _("filter input contains non-finite values"),
            details={"name": name, "length": int(array.size)},
        )
    return array


def _positive_float(value: float, name: str) -> float:
    """检查参数是有限正数。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            _("{0} must be a finite positive number").format(name),
            details={"name": name, "value": str(value)},
        ) from exc
    if not np.isfinite(number) or number <= 0.0:
        raise DataValidationError(
            _("{0} must be a finite positive number; got {1}").format(name, number),
            details={"name": name, "value": number},
        )
    return number


def _non_negative_int(value: int, name: str) -> int:
    """检查参数是非负整数。"""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise DataValidationError(
            _("{0} must be a non-negative integer").format(name),
            details={"name": name, "value": str(value)},
        )
    number = int(value)
    if number < 0:
        raise DataValidationError(
            _("{0} must be a non-negative integer; got {1}").format(name, number),
            details={"name": name, "value": number},
        )
    return number


def _nyquist(dt: float) -> float:
    """返回 Nyquist 频率（每 Ma 的周期数）。"""
    return 0.5 / dt


def _validated_time(time: npt.NDArray, size: int) -> npt.NDArray:
    """检查时间轴是一维、有限、严格递增且与记录等长。"""
    try:
        axis = np.asarray(time, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            _("filter time axis must be numeric; got {0}").format(type(time).__name__),
            details={"name": "time"},
        ) from exc
    if axis.ndim != 1:
        raise DataValidationError(
            _("filter time axis must be one-dimensional; got shape {0}").format(axis.shape),
            details={"name": "time", "shape": axis.shape},
        )
    if axis.size != size:
        raise DataValidationError(
            _("filter time axis must have the same length as the record; got {0} and {1}").format(axis.size, size),
            details={"time_length": int(axis.size), "length": int(size)},
        )
    if not np.all(np.isfinite(axis)):
        raise DataValidationError(
            _("filter time axis contains non-finite values"),
            details={"name": "time", "length": int(axis.size)},
        )
    if axis.size > 1 and not np.all(np.diff(axis) > 0.0):
        raise DataValidationError(
            _("filter time axis must be strictly increasing"),
            details={"name": "time", "length": int(axis.size)},
        )
    return axis


def _working_grid(
    values: npt.NDArray,
    time: npt.NDArray | None,
    dt: float,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray | None, float]:
    """
    把记录放到一张均匀工作网格上。

    ``time`` 为 None 时网格就是等间距的（间距 ``dt``）；否则按**实际时间差**的
    中位数建立网格，并用线性插值把记录搬到网格上。返回
    ``(网格上的值, 网格时间, 原始时间轴或 None, 网格间距)``。

    滤波结束后用 :func:`_map_back` 把结果插值回原始时间点。
    """
    if time is None:
        count = int(values.size)
        grid = dt * np.arange(count, dtype=float)
        return values, grid, None, dt

    axis = _validated_time(time, values.size)
    spacing = float(np.median(np.diff(axis)))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise DataValidationError(
            _("filter time axis has a non-positive median spacing"),
            details={"name": "time", "length": int(axis.size)},
        )
    span = float(axis[-1] - axis[0])
    count = int(np.floor(span / spacing)) + 1
    grid = axis[0] + spacing * np.arange(count, dtype=float)
    grid_values = np.interp(grid, axis, values)
    return grid_values, grid, axis, spacing


def _map_back(
    filtered: npt.NDArray,
    grid: npt.NDArray,
    axis: npt.NDArray | None,
) -> npt.NDArray:
    """把网格上的滤波结果插值回原始时间点（等间距时原样返回）。"""
    if axis is None:
        return filtered
    return np.interp(axis, grid, filtered)


# =============================================================================
# Frequency-domain helpers
# =============================================================================


def _frequency_grid(size: int, dt: float) -> npt.NDArray:
    """返回 FFT 频率网格（每 Ma 的周期数，含负频率）。"""
    return np.fft.fftfreq(size, d=dt)


def _gaussian_band_response(frequencies: npt.NDArray, centre: float, sigma: float) -> npt.NDArray:
    """
    偶对称的高斯带通响应。

    高斯核在时间轴上的标准差为 ``sigma``（时间单位），中心频率为 ``centre``
    （每 Ma 的周期数）。取绝对频率保证响应关于 0 偶对称，滤波结果因此是实数
    且零相位。
    """
    offset = np.abs(frequencies) - centre
    return np.exp(-2.0 * np.pi**2 * sigma**2 * offset**2)


def _acycle_band_response(frequencies: npt.NDArray, centre: float, half_width: float) -> npt.NDArray:
    """Acycle 式高斯带通响应，两条带边是 -3 dB（半功率）点。"""
    offset = (np.abs(frequencies) - centre) / half_width
    return np.exp(-HALF_POWER_COEFFICIENT * offset**2)


def _apply_response(values: npt.NDArray, response: npt.NDArray, demean: bool) -> npt.NDArray:
    """把频域响应作用到序列上，返回零相位滤波结果。"""
    working = values - float(values.mean()) if demean else values
    spectrum = np.fft.fft(working)
    filtered = np.fft.ifft(spectrum * response).real
    return np.asarray(filtered, dtype=float)


# =============================================================================
# Time-domain helpers (Taner weighted band-pass)
# =============================================================================


def _sinc_bandpass_kernel(
    low_frequency: float,
    high_frequency: float,
    size: int,
) -> tuple[npt.NDArray, int]:
    """
    构造 sinc 加权的理想带通核。

    Parameters:
        low_frequency: 带边低频（每样本的周期数），大于 0。
        high_frequency: 带边高频（每样本的周期数），大于 ``low_frequency``。
        size: 记录长度（样本数），用来限制核的长度。

    Returns:
        核数组与核的半宽（样本数）。
    """
    desired = int(
        max(
            np.ceil(TANER_KERNEL_OCTAVES / high_frequency),
            np.ceil(TANER_KERNEL_LOBES / low_frequency),
        )
    )
    # 核比记录还长的话，卷积就退化成“用反射数据滤波记录”，没有意义。
    half_width = max(1, min(desired, max(1, (int(size) - 1) // 2)))
    lags = np.arange(-half_width, half_width + 1, dtype=float)
    kernel = 2.0 * high_frequency * np.sinc(2.0 * high_frequency * lags) - 2.0 * low_frequency * np.sinc(
        2.0 * low_frequency * lags
    )
    # Blackman-Harris 窗压低 Gibbs 振铃；代价是有效带边略微变宽。窗**不**做
    # 归一化：窗是平滑的锥形，它的频谱主瓣很窄，所以带内的增益仍接近理想
    # 带通的 1（而直流增益被压到 0，起到去趋势的作用）。
    kernel = kernel * blackmanharris(kernel.size)
    return kernel, half_width


def _convolve_with_padding(values: npt.NDArray, kernel: npt.NDArray, half_width: int, extension: int) -> npt.NDArray:
    """
    镜像延拓后做线性卷积，返回与原序列等长的中心部分。

    延拓长度至少要等于核的半宽，否则卷积结果比原记录还短，取不出等长的结果。
    """
    size = int(values.size)
    extension = max(1, min(int(extension), max(1, size - 1)))
    if extension < half_width:
        logger.info(
            "filter: extending the reflection padding from %d to %d samples to match the kernel half-width",
            extension,
            half_width,
        )
        extension = half_width
    padded = np.pad(values, extension, mode="reflect")
    convolved = fftconvolve(padded, kernel, mode="valid")
    start = extension - half_width
    return np.asarray(convolved[start : start + size], dtype=float)


# =============================================================================
# Public API
# =============================================================================


def gaussian_filter(
    values: npt.NDArray,
    sigma: float | None = None,
    cutoff_period_kyr: float | None = None,
    dt: float = 1.0,
    time: npt.NDArray | None = None,
) -> FilterResult:
    """
    高斯滤波（Berger & Streck 1994；Acycle 的 Gaussian filter）。

    两种模式：

    1. 给了 ``cutoff_period_kyr``：**带通模式**。中心频率取目标周期的倒数
       ``f0 = 1 / T``，高斯宽度 ``sigma`` 缺省为

           sigma = cutoff_period_kyr / (2 * pi * 1000)     （单位 Ma）

       目标周期处增益为 1，直流增益为 ``exp(-1/2) ~ 0.607``，因此线性趋势
       会被明显压低而目标周期被完整保留。显式给出 ``sigma`` 时以 ``sigma``
       为准（覆盖上面由目标周期推出的宽度）。
    2. 只给了 ``sigma``：**低通模式**。响应 ``exp(-2*pi**2*sigma**2*f**2)``，
       即经典的高斯平滑核，直流增益为 1，序列的整体水平被保留。

    两个参数都不给时抛出 :class:`~utils.exceptions.DataValidationError`。

    Parameters:
        values: 一维有限序列，长度至少 ``MIN_SAMPLES``。
        sigma: 高斯核在时间轴上的标准差，单位 **Ma**（与 ``dt`` 同单位）。
        cutoff_period_kyr: 目标周期，单位 **kyr**。
        dt: 采样间隔，单位 **Ma**，必须为正。给出 ``time`` 时被忽略。
        time: 可选的时间轴，单位 **Ma**，严格递增且与 ``values`` 等长。给出时
            滤波按实际时间差进行（见模块文档的“不等间距时间轴”）。

    Returns:
        FilterResult: 滤波结果。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.filtering import gaussian_filter
        >>> series = np.sin(2 * np.pi * np.arange(200) / 20.0)
        >>> out = gaussian_filter(series, cutoff_period_kyr=20.0, dt=0.001)
        >>> out.method
        'gaussian'
    """
    series = _as_series(values)
    dt = _positive_float(dt, "dt")
    if sigma is None and cutoff_period_kyr is None:
        raise DataValidationError(
            _("gaussian_filter needs sigma or cutoff_period_kyr"),
            details={"dt": dt},
        )

    grid_values, grid, axis, spacing = _working_grid(series, time, dt)
    frequencies = _frequency_grid(grid_values.size, spacing)

    if cutoff_period_kyr is not None:
        period_kyr = _positive_float(cutoff_period_kyr, "cutoff_period_kyr")
        period_ma = period_kyr / KYR_PER_MA
        centre = 1.0 / period_ma
        if centre >= _nyquist(spacing):
            raise DataValidationError(
                _("target period {0} kyr is at or below the Nyquist period {1} kyr").format(
                    period_kyr, 2.0 * spacing * KYR_PER_MA
                ),
                details={
                    "cutoff_period_kyr": period_kyr,
                    "nyquist_period_kyr": 2.0 * spacing * KYR_PER_MA,
                },
            )
        if sigma is None:
            sigma_ma = period_kyr / (2.0 * np.pi * KYR_PER_MA)
        else:
            sigma_ma = _positive_float(sigma, "sigma")
        response = _gaussian_band_response(frequencies, centre, sigma_ma)
        filtered = _apply_response(grid_values, response, demean=True)
        parameters: dict[str, Any] = {
            "filter": "gaussian_bandpass",
            "cutoff_period_kyr": period_kyr,
            "cutoff_period_ma": period_ma,
            "sigma_ma": sigma_ma,
            "sigma_samples": sigma_ma / spacing,
            "centre_frequency_per_ma": centre,
            "dc_gain": float(np.exp(-2.0 * np.pi**2 * sigma_ma**2 * centre**2)),
            "dt_ma": dt,
            "sample_spacing_ma": spacing,
            "time_axis": "uniform" if axis is None else "resampled",
        }
    else:
        sigma_ma = _positive_float(float(sigma), "sigma")
        response = np.exp(-2.0 * np.pi**2 * sigma_ma**2 * frequencies**2)
        filtered = _apply_response(grid_values, response, demean=False)
        parameters = {
            "filter": "gaussian_lowpass",
            "sigma_ma": sigma_ma,
            "sigma_samples": sigma_ma / spacing,
            "half_power_period_ma": 2.0 * np.pi * sigma_ma / np.sqrt(HALF_POWER_COEFFICIENT),
            "dt_ma": dt,
            "sample_spacing_ma": spacing,
            "time_axis": "uniform" if axis is None else "resampled",
        }

    return FilterResult(
        filtered=_map_back(filtered, grid, axis),
        original=series.copy(),
        method="gaussian",
        parameters=parameters,
    )


def taner_filter(
    values: npt.NDArray,
    cutoff_period_kyr: float,
    dt: float = 1.0,
    pad: int = 200,
    time: npt.NDArray | None = None,
) -> FilterResult:
    """
    Taner 加权带通滤波（Kodama & Hinnov 2015）。

    滤波核的权重是带边频率的 sinc 函数，也就是理想带通的时间域冲激响应：

        h(k) = 2*fh*sinc(2*fh*k) - 2*fl*sinc(2*fl*k)

    其中 ``fl = 1 / (2T)``、``fh = 2 / T`` 是目标周期 ``T`` 两侧各一个倍频程
    的带边频率（每 Ma 的周期数）。核上再乘 Blackman-Harris 窗以压低 Gibbs
    振铃，代价是有效带边比标称值略宽、过渡带变缓；窗不做归一化，所以带内
    增益仍接近 1（目标周期不被削弱），而直流被压掉。

    核的半宽由带边频率决定（低边约 4 个周期、高边约 8 个周期），记录两端用
    镜像延拓压低边缘效应；延拓长度至少与核半宽相同，否则卷积结果取不出与
    记录等长的序列。

    与 Acycle 的 Taner-Hilbert 实现不同：那边在频域用广义高斯（超高斯）滚降
    并附带希尔伯特包络与瞬时频率；这里只做加权带通本身。

    Parameters:
        values: 一维有限序列，长度至少 ``MIN_SAMPLES``。
        cutoff_period_kyr: 目标周期，单位 **kyr**。
        dt: 采样间隔，单位 **Ma**，必须为正。给出 ``time`` 时被忽略。
        pad: 记录两端镜像延拓的样本数，用于压低边缘效应；必须为非负整数。实际
            延拓长度不小于核的半宽。
        time: 可选的时间轴，单位 **Ma**，严格递增且与 ``values`` 等长。给出时
            滤波按实际时间差进行（见模块文档的“不等间距时间轴”）。

    Returns:
        FilterResult: 滤波结果。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.filtering import taner_filter
        >>> series = np.sin(2 * np.pi * np.arange(500) / 50.0)
        >>> out = taner_filter(series, cutoff_period_kyr=50.0, dt=0.001)
        >>> out.method
        'taner'
    """
    series = _as_series(values)
    dt = _positive_float(dt, "dt")
    period_kyr = _positive_float(cutoff_period_kyr, "cutoff_period_kyr")
    extension = _non_negative_int(pad, "pad")

    grid_values, grid, axis, spacing = _working_grid(series, time, dt)

    period_ma = period_kyr / KYR_PER_MA
    low_frequency = 1.0 / (2.0 * period_ma)
    high_frequency = 2.0 / period_ma
    if high_frequency >= _nyquist(spacing):
        raise DataValidationError(
            _(
                "target period {0} kyr is too short for the sampling interval; "
                "the band-pass upper edge would exceed Nyquist"
            ).format(period_kyr),
            details={
                "cutoff_period_kyr": period_kyr,
                "minimum_period_kyr": 4.0 * spacing * KYR_PER_MA,
            },
        )

    kernel, half_width = _sinc_bandpass_kernel(low_frequency * spacing, high_frequency * spacing, grid_values.size)
    filtered = _convolve_with_padding(grid_values, kernel, half_width, extension)

    return FilterResult(
        filtered=_map_back(filtered, grid, axis),
        original=series.copy(),
        method="taner",
        parameters={
            "filter": "taner_weighted_bandpass",
            "cutoff_period_kyr": period_kyr,
            "band_low_period_kyr": period_kyr * 2.0,
            "band_high_period_kyr": period_kyr / 2.0,
            "dt_ma": dt,
            "sample_spacing_ma": spacing,
            "pad": extension,
            "kernel_half_width": int(half_width),
            "time_axis": "uniform" if axis is None else "resampled",
        },
    )


def bandpass_filter(
    values: npt.NDArray,
    low_period_kyr: float,
    high_period_kyr: float,
    dt: float = 1.0,
    time: npt.NDArray | None = None,
) -> FilterResult:
    """
    带通滤波（Acycle 式高斯响应，两条带边为 -3 dB 半功率点）。

    带边频率由周期换算：低频边 ``fl = 1 / high_period``、高频边
    ``fh = 1 / low_period``（每 Ma 的周期数），中心频率取两者中点，半带宽取
    两者之差的一半，响应为

        H(f) = exp(-ln(sqrt(2)) * ((|f| - fc) / half_width) ** 2)

    ``low_period_kyr`` 是带内**最短**的周期（对应高频边），
    ``high_period_kyr`` 是带内**最长**的周期（对应低频边），所以必须满足
    ``low_period_kyr < high_period_kyr``。滤波结果是去均值的异常量（与 Acycle
    一致：Acycle 滤波前先减去均值）。

    Parameters:
        values: 一维有限序列，长度至少 ``MIN_SAMPLES``。
        low_period_kyr: 带内最短周期，单位 **kyr**。
        high_period_kyr: 带内最长周期，单位 **kyr**。
        dt: 采样间隔，单位 **Ma**，必须为正。给出 ``time`` 时被忽略。
        time: 可选的时间轴，单位 **Ma**，严格递增且与 ``values`` 等长。给出时
            滤波按实际时间差进行（见模块文档的“不等间距时间轴”）。

    Returns:
        FilterResult: 滤波结果。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.filtering import bandpass_filter
        >>> index = np.arange(1000)
        >>> series = np.sin(2 * np.pi * index / 20.0) + np.sin(2 * np.pi * index / 100.0)
        >>> out = bandpass_filter(series, low_period_kyr=15.0, high_period_kyr=30.0, dt=0.001)
        >>> out.method
        'gaussian'
    """
    series = _as_series(values)
    dt = _positive_float(dt, "dt")
    low_period = _positive_float(low_period_kyr, "low_period_kyr")
    high_period = _positive_float(high_period_kyr, "high_period_kyr")
    if low_period >= high_period:
        raise DataValidationError(
            _("low_period_kyr must be shorter than high_period_kyr; got {0} and {1}").format(low_period, high_period),
            details={"low_period_kyr": low_period, "high_period_kyr": high_period},
        )

    grid_values, grid, axis, spacing = _working_grid(series, time, dt)

    low_frequency = 1.0 / (high_period / KYR_PER_MA)
    high_frequency = 1.0 / (low_period / KYR_PER_MA)
    if high_frequency >= _nyquist(spacing):
        raise DataValidationError(
            _("low_period_kyr ({0} kyr) is at or below the Nyquist period ({1} kyr)").format(
                low_period, 2.0 * spacing * KYR_PER_MA
            ),
            details={
                "low_period_kyr": low_period,
                "nyquist_period_kyr": 2.0 * spacing * KYR_PER_MA,
            },
        )

    centre = 0.5 * (low_frequency + high_frequency)
    half_width = 0.5 * (high_frequency - low_frequency)
    frequencies = _frequency_grid(grid_values.size, spacing)
    response = _acycle_band_response(frequencies, centre, half_width)
    filtered = _apply_response(grid_values, response, demean=True)

    return FilterResult(
        filtered=_map_back(filtered, grid, axis),
        original=series.copy(),
        method="gaussian",
        parameters={
            "filter": "acycle_bandpass",
            "low_period_kyr": low_period,
            "high_period_kyr": high_period,
            "centre_frequency_per_ma": centre,
            "half_bandwidth_per_ma": half_width,
            "dt_ma": dt,
            "sample_spacing_ma": spacing,
            "time_axis": "uniform" if axis is None else "resampled",
        },
    )
