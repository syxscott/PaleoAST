# =============================================================================
# FILE: stratigraphy/signal_generator.py
# =============================================================================
"""
信号与噪声生成器：为频谱分析造测试数据。

本模块实现 Acycle（https://github.com/mingsongli/acycle）Math 菜单
"Signal Noise Generator"（``code/guicode/linegenerator.m``）中与数据生成
有关的部分：正弦（谐波）信号、白噪声、红噪声，并额外提供频谱分析常用的
粉噪声（1/f）。Acycle 以 GPL-3.0 发布，因此本文件是**独立实现**，只参考
其公开的行为约定，未翻译或复制其 MATLAB 源码。

行为依据（Acycle 公开行为，非源码翻译）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Acycle 的 harmonic 模式：``y = A * sin(2*pi/T * x + Ph) + B``，其中
  ``T`` 为周期、``Ph`` 为弧度相位。
* Acycle 的 white 模式：``y = sigma * randn(n, 1) + mu``。
* Acycle 的 red 模式：``y = sigma * zscore(AR(1)) + mu``，AR(1) 的递推为
  ``x[t] = alpha * x[t-1] + eps[t]``（MATLAB 的
  ``filter(1, [1 -alpha], randn(n, 1))``），系数 ``alpha`` 即本模块的
  ``rho``。

随机过程约定
~~~~~~~~~~~~
* 所有随机函数都接受 ``random_seed``。给定种子时结果完全可复现；不给
  种子时调用 ``utils.statistics_core.make_rng(None, context=...)``，它会发出
  ``RuntimeWarning`` 提示结果不可复现——这是仓库内其它随机分析（如
  ``stratigraphy.cycles`` 的 REDFIT 蒙特卡洛）沿用的同一约定。
* 生成器都是**独立的** ``numpy.random.Generator``，既不读取也不推进全局
  ``numpy.random`` 流，因此不会扰动调用方自己的随机数据。

噪声模型
~~~~~~~~
* **白噪声**：独立同分布的正态随机变量，标准差为 ``sigma``。
* **红噪声**：一阶自回归过程 AR(1)，``x[t] = rho * x[t-1] + eps[t]``，
  创新项 ``eps`` 的标准差为 ``sigma * sqrt(1 - rho^2)``，因此序列的
  边际标准差为 ``sigma``、滞后 1 阶自相关为 ``rho``。要求
  ``-1 < rho < 1``（平稳性条件）。
* **粉噪声**：功率谱密度正比于 1/f 的噪声。在频域合成——对白噪声的每个
  频点乘以 ``f^(-1/2)`` 的振幅因子（功率即正比于 1/f），再变换回时域，
  最后标准化到零均值、单位方差。DC 分量置零，避免 1/f 在零频处的发散。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.signal import lfilter

from config.i18n import _
from utils.exceptions import ValidationError
from utils.statistics_core import make_rng

logger = logging.getLogger(__name__)

#: 支持的噪声类型。
NOISE_TYPES = ("white", "red", "pink")

#: 白噪声/红噪声/粉噪声生成时使用的上下文名（进入警告信息）。
_NOISE_CONTEXTS = {
    "white": "White noise generation",
    "red": "Red noise generation",
    "pink": "Pink noise generation",
    "signal": "Signal noise generation",
}


# =============================================================================
# 内部工具
# =============================================================================


def _as_float_1d(array: object, name: str, *, require_finite: bool = True) -> npt.NDArray:
    """把输入转换为一维 float 数组，并检查形状与（可选的）有限性。"""
    try:
        arr = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("{0} must be numeric").format(name),
            details={"parameter": name},
            original_exception=exc,
        ) from exc
    if arr.ndim != 1:
        raise ValidationError(
            _("{0} must be a one-dimensional array").format(name),
            details={"parameter": name, "ndim": int(arr.ndim)},
        )
    if arr.size == 0:
        raise ValidationError(
            _("{0} must not be empty").format(name),
            details={"parameter": name},
        )
    if require_finite and not bool(np.all(np.isfinite(arr))):
        raise ValidationError(
            _("{0} must contain only finite values").format(name),
            details={"parameter": name},
        )
    return arr


def _check_length(n: int, name: str = "n") -> int:
    """检查样本数是正整数。"""
    try:
        count = int(n)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("{0} must be an integer").format(name),
            details={"parameter": name},
            original_exception=exc,
        ) from exc
    if count <= 0:
        raise ValidationError(
            _("{0} must be greater than 0").format(name),
            details={"parameter": name, "value": count},
        )
    return count


def _check_sigma(sigma: float) -> float:
    """检查噪声标准差非负且有限。"""
    value = float(sigma)
    if not np.isfinite(value):
        raise ValidationError(_("sigma must be finite"), details={"sigma": value})
    if value < 0.0:
        raise ValidationError(_("sigma must not be negative"), details={"sigma": value})
    return value


# =============================================================================
# 噪声
# =============================================================================


def generate_white_noise(
    n: int,
    sigma: float = 1.0,
    random_seed: int | None = None,
) -> npt.NDArray:
    """
    生成白噪声（独立同分布的正态随机变量）。

    Parameters:
        n: 样本数，必须为正整数。
        sigma: 标准差，必须非负。
        random_seed: 随机种子。``None`` 时结果不可复现，并会发出
            ``RuntimeWarning``。

    Returns:
        长度为 ``n`` 的零均值白噪声序列。

    Raises:
        ValidationError: ``n <= 0`` 或 ``sigma`` 非法。
    """
    count = _check_length(n)
    spread = _check_sigma(sigma)
    rng = make_rng(random_seed, context=_NOISE_CONTEXTS["white"])
    return rng.normal(0.0, spread, size=count)


def generate_red_noise(
    n: int,
    rho: float = 0.7,
    sigma: float = 1.0,
    random_seed: int | None = None,
) -> npt.NDArray:
    """
    生成红噪声（AR(1) 过程）。

    递推为 ``x[t] = rho * x[t-1] + eps[t]``，创新项 ``eps`` 的标准差为
    ``sigma * sqrt(1 - rho^2)``，因此序列的边际标准差为 ``sigma``、滞后
    1 阶自相关系数为 ``rho``。序列前段有一段 burn-in 被丢弃，使起始样本
    也来自平稳分布。

    Parameters:
        n: 样本数，必须为正整数。
        rho: 自回归系数，必须满足 ``-1 < rho < 1``。
        sigma: 边际标准差，必须非负。
        random_seed: 随机种子。``None`` 时结果不可复现，并会发出
            ``RuntimeWarning``。

    Returns:
        长度为 ``n`` 的红噪声序列。

    Raises:
        ValidationError: ``n <= 0``、``rho`` 越界或 ``sigma`` 非法。
    """
    count = _check_length(n)
    coefficient = float(rho)
    if not np.isfinite(coefficient) or not (-1.0 < coefficient < 1.0):
        raise ValidationError(
            _("rho must lie strictly between -1 and 1"),
            details={"rho": coefficient},
        )
    spread = _check_sigma(sigma)

    rng = make_rng(random_seed, context=_NOISE_CONTEXTS["red"])
    # lfilter 实现的就是递推 x[t] = rho * x[t-1] + eps[t]（零初值）。
    # 先跑一段 burn-in 再丢弃，使起始样本也来自平稳分布：
    # burn-in 长度随 rho 接近 +-1 而变长。
    burn_in = int(np.ceil(10.0 / (1.0 - abs(coefficient))))
    full = rng.normal(0.0, spread * np.sqrt(1.0 - coefficient**2), size=count + burn_in)
    values = lfilter([1.0], [1.0, -coefficient], full)
    return np.asarray(values[burn_in:], dtype=float)


def generate_pink_noise(
    n: int,
    sigma: float = 1.0,
    random_seed: int | None = None,
) -> npt.NDArray:
    """
    生成粉噪声（1/f 功率谱）。

    在频域合成：先取白噪声的频谱，对第 ``k`` 个频点乘以 ``k^(-1/2)``
    的振幅因子（于是功率正比于 ``1/f``），再变换回时域；DC 分量置零，
    最后标准化为零均值、标准差 ``sigma``。

    Parameters:
        n: 样本数，必须为正整数。
        sigma: 标准差，必须非负。
        random_seed: 随机种子。``None`` 时结果不可复现，并会发出
            ``RuntimeWarning``。

    Returns:
        长度为 ``n`` 的粉噪声序列。

    Raises:
        ValidationError: ``n <= 0`` 或 ``sigma`` 非法。
    """
    count = _check_length(n)
    spread = _check_sigma(sigma)

    rng = make_rng(random_seed, context=_NOISE_CONTEXTS["pink"])
    n_bins = count // 2 + 1
    # 频率以 bin 序号表示（与 rfftfreq 相差一个常数因子，不影响谱斜率）。
    scale = np.ones(n_bins, dtype=float)
    scale[1:] = 1.0 / np.sqrt(np.arange(1, n_bins, dtype=float))
    scale[0] = 0.0
    spectrum = (rng.normal(size=n_bins) + 1j * rng.normal(size=n_bins)) * scale
    spectrum[0] = 0.0 + 0.0j
    values = np.fft.irfft(spectrum, n=count)
    values -= values.mean()
    deviation = float(values.std())
    if deviation > 0.0:
        values = values / deviation
    return values * spread


# =============================================================================
# 信号
# =============================================================================


def generate_sine(
    time: npt.NDArray,
    amplitude: float = 1.0,
    period: float = 1.0,
    phase: float = 0.0,
) -> npt.NDArray:
    """
    生成一条正弦信号。

    公式与 Acycle 的 harmonic 模式一致：
    ``y = amplitude * sin(2*pi * time / period + phase) + 0``。

    Parameters:
        time: 时间轴。
        amplitude: 振幅。
        period: 周期，必须大于 0。
        phase: 相位，单位为**弧度**。

    Returns:
        与 ``time`` 等长的信号数组。

    Raises:
        ValidationError: 时间轴为空或非有限，``period <= 0``。
    """
    t = _as_float_1d(time, "time")
    cycle = float(period)
    if not np.isfinite(cycle) or cycle <= 0.0:
        raise ValidationError(
            _("period must be greater than 0"),
            details={"period": cycle},
        )
    return float(amplitude) * np.sin(2.0 * np.pi * t / cycle + float(phase))


def generate_signal(
    time: npt.NDArray,
    components: Sequence[tuple[float, float, float]],
    noise: str | None = None,
    noise_sigma: float = 0.0,
    random_seed: int | None = None,
    *,
    noise_rho: float = 0.7,
) -> npt.NDArray:
    """
    合成一条多分量信号，并可叠加噪声。

    每个分量是 ``(振幅, 周期)`` 或 ``(振幅, 周期, 相位)``（相位为弧度），
    所有分量先求和，再叠加一层噪声。

    Parameters:
        time: 时间轴。
        components: 分量列表，每项为 ``(振幅, 周期)`` 或
            ``(振幅, 周期, 相位)``。
        noise: 噪声类型：``None`` / ``"none"``（不加噪声）、``"white"``、
            ``"red"`` 或 ``"pink"``。
        noise_sigma: 噪声的标准差。为 0 时不叠加噪声（也不消耗随机数）。
        random_seed: 随机种子。``None`` 时结果不可复现，并会发出
            ``RuntimeWarning``。
        noise_rho: 红噪声的 AR(1) 系数，仅在 ``noise="red"`` 时使用。

    Returns:
        与 ``time`` 等长的合成信号。

    Raises:
        ValidationError: 时间轴为空或非有限、分量格式非法、噪声类型未知，
            或噪声参数非法。
    """
    t = _as_float_1d(time, "time")

    signal = np.zeros(t.size, dtype=float)
    for position, component in enumerate(components):
        try:
            parts = tuple(component)
        except TypeError as exc:
            raise ValidationError(
                _("Component {0} must be an (amplitude, period[, phase]) tuple").format(position),
                details={"position": position},
                original_exception=exc,
            ) from exc
        if len(parts) == 2:
            amplitude, period = float(parts[0]), float(parts[1])
            phase = 0.0
        elif len(parts) == 3:
            amplitude, period, phase = (float(parts[0]), float(parts[1]), float(parts[2]))
        else:
            raise ValidationError(
                _("Component {0} must be an (amplitude, period[, phase]) tuple").format(position),
                details={"position": position},
            )
        signal += generate_sine(t, amplitude=amplitude, period=period, phase=phase)

    kind = "none" if noise is None else str(noise).lower()
    if kind not in ("none", *NOISE_TYPES):
        raise ValidationError(
            _("noise must be one of {0}; got '{1}'").format(", ".join(("none", *NOISE_TYPES)), noise),
            details={"noise": str(noise)},
        )
    spread = _check_sigma(noise_sigma)
    if kind == "none" or spread == 0.0:
        return signal

    if kind == "white":
        noise_values = generate_white_noise(t.size, spread, random_seed)
    elif kind == "red":
        noise_values = generate_red_noise(t.size, rho=noise_rho, sigma=spread, random_seed=random_seed)
    else:
        noise_values = generate_pink_noise(t.size, spread, random_seed)
    return signal + noise_values
