# =============================================================================
# FILE: stratigraphy/mtm_tuning.py
# =============================================================================
"""
MTM（多锥谱估计）的自适应带宽 / taper 数选择，以及谱的 F 检验。

要回答的两个问题
----------------
1. **该用多少个 taper（K）、多大带宽（NW）？** 仓库已有的
   :func:`stratigraphy.cycles.multitaper_spectrum` 把这两件事**写死**了：
   ``NW = 3.0``、``K = floor(2·NW − 1) = 5``。写死的做法对"随手看一眼"
   是够的，但对"把记录长度、目标频带、显著性水平一起权衡"就不够了：
   短记录用 5 个 taper 会浪费自由度，长记录用 5 个 taper 又白白扔掉了
   估计量本可以提供的分辨率。本模块按 **Thomson (1982)** 的规则把
   K 与 NW、记录长度联系起来，给出**自适应**的选择。

2. **谱上那个峰是信号还是背景涨落？** 这是 F 检验回答的问题：原假设是
   "该频率上的谱功率与背景无异"。

与仓库已有实现的关系
---------------------
* :func:`stratigraphy.cycles.multitaper_spectrum` 的签名与行为**未改动**。
  本模块是并列的新入口，复用它的 DPSS 构造
  (:func:`stratigraphy.cycles.dpss_tapers`)、去趋势
  (:func:`stratigraphy.spectral_analysis._linear_detrend`) 与 AR(1) 背景
  (:func:`stratigraphy.spectral_analysis.ar1_theoretical_spectrum`)。
* 本模块的谱**归一化口径与仓库完全一致**：白噪声下每条谱的平均值为 1，
  因此可以直接和 :func:`stratigraphy.spectral_analysis.ar1_theoretical_spectrum`
  给出的 AR(1) 红噪声曲线逐点比较。

taper 数的选择规则（Thomson）
------------------------------
* **时间-带宽积** ``NW`` 决定 taper 数：``K = 2·NW − 1``（Thomson 1982）。
  本模块强制 ``K`` 为**奇数**，因此把 ``NW`` 量化到满足 ``K = 2·NW − 1``
  的最大取值（``NW`` 取半整数时向下量化，例如 ``NW = 2.5 → K = 3``）。
* **记录长度**给出上限：估计量用的 taper 数不应超过记录能提供的独立谱
  估计数，即 ``K ≤ n // 2``（与 :func:`stratigraphy.cycles.multitaper_spectrum`
  的取用上限一致）。记录太短时带宽被**自适应地收窄**，实际生效的带宽
  ``(K+1)/2`` 随 ``K`` 一起返回。
* 调用方直接给了 ``n_tapers`` 时就用它（并据此反推带宽 ``(K+1)/2``）。

谱的 F 检验怎么构造
--------------------
对频率 ``f``，记多锥谱功率为 ``S(f)``、背景（期望功率）为 ``b(f) > 0``，
归一化功率 ``r(f) = S(f) / b(f)``。在原假设"该频率的功率与背景无异"下，
多锥谱的估计量近似服从 ``ν·r ~ χ²(ν)``，其中 ``ν`` 是估计量的有效自由度
（``ν = 2(Σλ)²/Σλ²``，``λ`` 是 DPSS 锥的聚集特征值）。

把"该频率"与"整条谱的同一背景水平"作比，就得到一个 F 型统计量：

    F(f) = r(f) / r̄                  r̄ = 全谱平均的 r

分子是"该频率上的功率"，分母是"整条谱的平均功率"。零假设下
``ν·r ~ χ²(ν)``，所以这个比值近似服从

    F(ν, M_ind·ν)

其中 ``M_ind = max(2, round(M / K))`` 是**独立**频率单元的个数（M 是参与
检验的频率点数）——多锥谱相邻频率点被同一个锥窗糊在一起，相关长度约
K 个频率格，所以独立单元不是 M 而是 M/K。于是

    p(f) = P( F(ν, M_ind·ν) > F(f) )

在原假设下 p 值应当近似均匀分布。**这个统计量与 Thomson (1982) 式
13.10 的谐波 F 检验不是同一个东西**：式 13.10 需要每个 taper 的复数
傅里叶系数（本模块的公共接口只拿到一条谱，拿不到），它检验的是"该频率上
各锥之间是否同相"；本模块检验的是"该频率的功率是否高于背景"。两者互补，
本模块实现的是后者，并在下面说明它的适用范围。

蒙特卡洛标定
------------
解析 p 值依赖"多锥谱在原假设下近似 χ²(ν)"这个近似。传
``n_simulations > 0`` 时，本模块改用**参数自助法**给出 p 值：按背景的
*形状*（而不是幅度）生成替代序列（随机相位整形），逐条算同一条多锥谱与
同一个 F 统计量，再用 ``(1 + #{零假设统计量 >= 实测统计量}) / (n + 1)``
的 Phipson & Smyth (2010) 形式给出经验 p 值。这样标定出来的零假设把
锥窗造成的频率点相关性也算了进去，比解析 p 值更可靠。

随机过程约定与仓库一致：``random_seed=None`` 时用
:func:`utils.statistics_core.make_rng` 抽取熵并发出 ``RuntimeWarning``，
提示结果不可复现。

频率轴
------
频率一律是**每单位时间**（cycles/Ma 之类），不是每样本：用 ``dt`` 把
样本间隔换算成时间单位。

参考文献
--------
1. Thomson, D.J. 1982. Spectrum estimation and harmonic analysis. Proceedings
   of the IEEE 70: 1055-1096.（多锥方法、taper 数 ``K = 2NW − 1``、
   自适应权重的自由度、谐波 F 检验。）
2. Percival, D.B. & Walden, A.T. 1993. Spectral Analysis and its
   Applications. Cambridge University Press.（DPSS 构造、聚集特征值、
   有效自由度。）
3. Hannan, E.J. 1970. Spectral Analysis. Cambridge University Press.（频域
   检验与泄漏。）
4. Phipson, S.M. & Smyth, G.K. 2010. Permutation P-values should never be
   zero: calculating exact P-values when permutations are randomly drawn.
   Statistical Applications in Genetics and Molecular Biology 9: 39.（经验
   p 值的 ``(1+k)/(n+1)`` 形式。）
5. Acycle（Li, M.）的 ``acycleMtmTaperCount`` / ``mtmdofs`` / ``ftestmtm``
   系列函数提供了同一问题的工程口径；本模块按上面的算法规格与文献**独立
   实现**，未翻译其 MATLAB 源码（Acycle 为 GPL 软件）。

作者: PaleoAST 开发组
版本: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats as sp_stats

from config.i18n import _
from stratigraphy.cycles import dpss_tapers
from stratigraphy.spectral_analysis import _linear_detrend
from utils.exceptions import ValidationError
from utils.statistics_core import make_rng

logger = logging.getLogger(__name__)

#: 缺省的时间-带宽积。与 :func:`stratigraphy.cycles.multitaper_spectrum`
#: 的缺省值一致，便于两个入口的结果互相印证。
DEFAULT_TIME_BANDWIDTH = 3.0

#: 构造一条多锥谱至少需要的样本数。
MIN_SAMPLES = 16

#: F 检验至少需要的 taper 数：一个 taper 没有"锥与锥之间"的对比可言。
MIN_TAPERS = 2


# =============================================================================
# 1. 结果容器
# =============================================================================


@dataclass(frozen=True)
class MTMTuningResult:
    """自适应多锥谱估计的结果。

    Attributes:
        bandwidth: 实际生效的**时间-带宽积** ``NW``（无量纲），满足
            ``NW = (K + 1) / 2``。频域带宽是 ``NW / n``（每单位时间）。
        n_tapers: 实际使用的 taper 数 ``K``（奇数）。
        degrees_of_freedom: 估计量的有效自由度
            ``ν = 2(Σλ)²/Σλ²``（标量）。
        spectrum: 多锥谱功率，白噪声下平均为 1，因此可直接与 AR(1) 红噪声
            曲线逐点比较。
        frequencies: 频率点，单位是**每单位时间**。
        p_values: 逐频率的 F 检验 p 值（None 表示没有做检验）。
        n_samples: 参与估计的样本数。
    """

    bandwidth: float
    n_tapers: int
    degrees_of_freedom: float
    spectrum: npt.NDArray
    frequencies: npt.NDArray
    p_values: npt.NDArray | None = None
    n_samples: int = 0

    def summary(self) -> str:
        """生成摘要文本。"""
        lines = [
            _("Adaptive Multitaper Spectrum"),
            "=" * 50,
            _("Samples: {0}").format(int(self.n_samples)),
            _("Time-bandwidth product (NW): {0}").format(f"{self.bandwidth:.3f}"),
            _("Tapers (K): {0}").format(int(self.n_tapers)),
            _("Effective degrees of freedom: {0}").format(f"{self.degrees_of_freedom:.2f}"),
            _("Frequencies: {0} (per unit time)").format(int(self.frequencies.size)),
        ]
        if self.p_values is not None:
            lines.append(_("F-test p-value < 0.05 at {0} frequencies").format(int(np.sum(self.p_values < 0.05))))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON 友好的视图（不含整条谱）。"""
        result: dict[str, object] = {
            "bandwidth": float(self.bandwidth),
            "n_tapers": int(self.n_tapers),
            "degrees_of_freedom": float(self.degrees_of_freedom),
            "n_samples": int(self.n_samples),
            "n_frequencies": int(self.frequencies.size),
        }
        if self.p_values is not None:
            result["n_significant"] = int(np.sum(self.p_values < 0.05))
        return result


# =============================================================================
# 2. 输入清洗
# =============================================================================


def _clean_series(values: npt.NDArray, name: str) -> npt.NDArray:
    """把输入整理成一维有限 float 数组，并检查样本数与方差。"""
    array = np.asarray(values, dtype=float).flatten()
    if array.size < MIN_SAMPLES:
        raise ValidationError(
            _("'{0}' needs at least {1} samples; got {2}").format(name, MIN_SAMPLES, int(array.size)),
            details={"n": int(array.size), "minimum_required": MIN_SAMPLES},
        )
    if not np.all(np.isfinite(array)):
        raise ValidationError(
            _("'{0}' must contain only finite values").format(name),
            details={"n_invalid": int(np.sum(~np.isfinite(array)))},
        )
    if float(np.ptp(array)) == 0.0:
        raise ValidationError(_("'{0}' is constant, so it has no spectrum").format(name))
    return array


def _clean_background(background: npt.NDArray | float | None, size: int) -> npt.NDArray:
    """把背景整理成与谱等长的、严格为正的数组。"""
    if background is None:
        return np.ones(size, dtype=float)
    array = np.asarray(background, dtype=float).flatten()
    if array.size == 1:
        array = np.full(size, float(array[0]))
    if array.size != size:
        raise ValidationError(
            _("background must have {0} values to match the spectrum; got {1}").format(size, int(array.size)),
            details={"expected": size, "got": int(array.size)},
        )
    if not np.all(np.isfinite(array)):
        raise ValidationError(
            _("background must contain only finite values"),
            details={"n_invalid": int(np.sum(~np.isfinite(array)))},
        )
    if np.any(array <= 0.0):
        raise ValidationError(
            _("background must be strictly positive to normalise the spectrum"),
            details={"n_non_positive": int(np.sum(array <= 0.0))},
        )
    return array


# =============================================================================
# 3. taper 数与带宽的选择
# =============================================================================


def select_taper_count(
    n_samples: int,
    time_bandwidth: float | None = None,
    n_tapers: int | None = None,
) -> tuple[int, float]:
    """按 Thomson 的规则选 taper 数与时间-带宽积。

    规则（详见模块 docstring）：

    1. ``K = 2·NW − 1``（Thomson 1982），并强制 ``K`` 为奇数；
    2. ``K ≤ n // 2``：记录提供的独立谱估计数有限，taper 数不能超过它；
       记录太短时带宽被自适应地收窄；
    3. 调用方给了 ``n_tapers`` 时直接用它。

    Parameters
    ----------
    n_samples:
        记录长度（样本数），至少 4。
    time_bandwidth:
        期望的时间-带宽积 ``NW``。None 表示用
        :data:`DEFAULT_TIME_BANDWIDTH`。
    n_tapers:
        期望的 taper 数。给了就用它（不再受带宽约束，但仍要满足记录
        长度上限）。

    Returns
    -------
    (n_tapers, time_bandwidth):
        实际生效的 taper 数 ``K`` 与时间-带宽积 ``NW``，两者满足
        ``NW = (K + 1) / 2``。

    Raises
    ------
    ValidationError
        样本数太小、带宽不是有限正数、taper 数不是正整数，或 taper 数超过
        记录长度允许的上限。
    """
    try:
        count = int(n_samples)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("n_samples must be a whole number; got {0}").format(n_samples),
            original_exception=exc,
        ) from exc
    if count != n_samples:
        raise ValidationError(
            _("n_samples must be a whole number; got {0}").format(n_samples),
            details={"n_samples": n_samples},
        )
    if count < 4:
        raise ValidationError(
            _("a multitaper estimate needs at least 4 samples; got {0}").format(count),
            details={"n_samples": count, "minimum_required": 4},
        )

    cap = count // 2
    if n_tapers is not None:
        try:
            requested = int(n_tapers)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                _("n_tapers must be a whole number; got {0}").format(n_tapers),
                original_exception=exc,
            ) from exc
        if requested != n_tapers or requested < 1:
            raise ValidationError(
                _("n_tapers must be a positive whole number; got {0}").format(n_tapers),
                details={"n_tapers": n_tapers},
            )
        if requested > cap:
            raise ValidationError(
                _("n_tapers = {0} needs at least {1} samples; the record has {2}").format(
                    requested, 2 * requested, count
                ),
                details={"n_tapers": requested, "n_samples": count, "maximum": cap},
            )
        return requested, (requested + 1) / 2.0

    bandwidth = DEFAULT_TIME_BANDWIDTH if time_bandwidth is None else float(time_bandwidth)
    if not np.isfinite(bandwidth) or bandwidth < 1.0:
        raise ValidationError(
            _("time_bandwidth must be a finite number >= 1; got {0}").format(bandwidth),
            details={"time_bandwidth": bandwidth},
        )

    # K = 2*NW - 1，并且 K 取奇数（等价于"不超过 2*NW-1 的最大奇数"）。
    thomson = 2 * int(np.floor(bandwidth)) - 1
    chosen = min(thomson, cap)
    if chosen % 2 == 0:
        chosen -= 1
    chosen = max(1, chosen)
    return chosen, (chosen + 1) / 2.0


# =============================================================================
# 4. 多锥谱与自由度
# =============================================================================


def _multitaper_power(
    tapers: npt.NDArray,
    eigenvalues: npt.NDArray,
    values: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Thomson 加权的多锥谱功率与各锥的单独功率。

    归一化口径与 :func:`stratigraphy.cycles.multitaper_spectrum` 完全一致
    （除以记录方差），因此白噪声下谱的平均值是 1。

    Returns
    -------
    (power, single_taper_power):
        去掉直流分量后的多锥谱，以及每个锥单独的谱（形状 ``(K, F)``）。
    """
    count = values.size
    centred = values - float(np.mean(values))
    sigma2 = float(np.sum(centred**2) / count)
    if sigma2 <= 0.0:
        raise ValidationError(_("a series with zero variance has no spectrum"))
    bins = np.fft.rfftfreq(count).size
    single = np.empty((tapers.shape[0], bins), dtype=float)
    for index in range(tapers.shape[0]):
        single[index] = np.abs(np.fft.rfft(tapers[index] * centred)) ** 2 / sigma2
    weight_sum = float(np.sum(eigenvalues))
    power = (eigenvalues[:, None] * single).sum(axis=0) / weight_sum
    return power[1:], single[:, 1:]


def _effective_degrees_of_freedom(eigenvalues: npt.NDArray) -> float:
    """多锥估计量的有效自由度 ``ν = 2(Σλ)²/Σλ²``。"""
    sum_sq = float(np.sum(eigenvalues**2))
    if sum_sq <= 0.0:
        raise ValidationError(_("the taper eigenvalues are degenerate"))
    return float(2.0 * float(np.sum(eigenvalues)) ** 2 / sum_sq)


def _adaptive_weights(
    single_taper_power: npt.NDArray,
    eigenvalues: npt.NDArray,
    max_iterations: int = 100,
    tolerance: float = 1.0e-4,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Thomson (1982) 式 5.3–5.5 的自适应权重与逐频率自由度。

    每个频率上迭代求谱水平的自洽估计 ``ax``，再由它给出各锥的自适应权重
    ``wt_k = sqrt(λ_k)·ax / (λ_k·ax + 1 − λ_k)``，最后

        dof(f) = 2·Σ_k ( λ_k·wt_k² ) / ( λ_1·wt_1² )

    Parameters
    ----------
    single_taper_power:
        各锥单独的谱，形状 ``(F, K)``（F 个频率、K 个锥）。

    Returns
    -------
    (weights, degrees_of_freedom):
        权重形状 ``(F, K)``，自由度形状 ``(F,)``。
    """
    lam = eigenvalues[None, :]
    bias = (1.0 - eigenvalues)[None, :]
    level = (single_taper_power[:, 0] + single_taper_power[:, 1]) / 2.0
    for iteration in range(max_iterations):
        weights = np.sqrt(lam) * level[:, None] / np.maximum(lam * level[:, None] + bias, 1.0e-300)
        squared = weights**2
        updated = (squared * single_taper_power).sum(axis=1) / np.maximum(squared.sum(axis=1), 1.0e-300)
        change = np.abs(updated - level)
        level = updated
        if bool(np.all(change < tolerance)):
            break
    weights = np.sqrt(lam) * level[:, None] / np.maximum(lam * level[:, None] + bias, 1.0e-300)
    dof = 2.0 * ((lam * weights**2) / (lam[:, 0:1] * weights[:, 0:1] ** 2)).sum(axis=1)
    return weights, dof


def adaptive_degrees_of_freedom(
    values: npt.NDArray,
    time_bandwidth: float | None = None,
    n_tapers: int | None = None,
) -> npt.NDArray:
    """逐频率的自适应自由度（Thomson 1982 式 5.5）。

    全局有效自由度 ``2(Σλ)²/Σλ²`` 把所有频率一视同仁，但高频处真正
    起作用的锥少，自由度也低；自适应权重按每个频率上的谱水平重新加权，
    给出的自由度更贴近当地的有效信息量。该数组可以直接传给
    :func:`multitaper_f_test` 取代标量自由度。

    Parameters
    ----------
    values:
        记录，样本顺序。
    time_bandwidth, n_tapers:
        与 :func:`select_taper_count` 相同的含义。

    Returns
    -------
    ndarray
        与谱的频率点一一对应的自由度（顺序与
        :func:`tuned_multitaper_spectrum` 的 ``spectrum`` 一致）。
    """
    series = _clean_series(values, "MTM input")
    count, bandwidth = select_taper_count(series.size, time_bandwidth, n_tapers)
    eigenvalues, tapers = dpss_tapers(series.size, bandwidth, count)
    _power, single = _multitaper_power(tapers, eigenvalues, series)
    if count < MIN_TAPERS:
        return np.full(single.shape[1], _effective_degrees_of_freedom(eigenvalues), dtype=float)
    _weights, dof = _adaptive_weights(single.T, eigenvalues)
    return dof


# =============================================================================
# 5. 谱的 F 检验
# =============================================================================


def _f_statistic(spectrum: npt.NDArray, background: npt.NDArray) -> npt.NDArray:
    """F 统计量 ``r(f) / r̄``；对 r 的整体倍数不变。

    ``r(f) = S(f) / b(f)`` 是背景归一化后的功率，``r̄`` 是全谱平均。原假设下
    ``r(f)`` 近似服从 ``χ²(ν)/ν``（均值 1），所以这个比值在零假设下近似服从
    ``F(ν, M_ind·ν)``——F 分布的均值在自由度 2 很大时接近 1，与统计量的
    均值 1 相符。
    """
    ratio = np.asarray(spectrum, dtype=float) / np.asarray(background, dtype=float)
    return ratio / float(np.mean(ratio))


def multitaper_f_test(
    spectrum: npt.NDArray,
    frequencies: npt.NDArray,
    background: npt.NDArray,
    n_tapers: int,
    degrees_of_freedom: float,
    n_frequencies: int | None = None,
) -> npt.NDArray:
    """多锥谱的逐频率 F 检验。

    原假设：**该频率上的谱功率与背景无异**。

    构造（详见模块 docstring）：归一化功率 ``r(f) = S(f) / b(f)`` 在原假设下
    近似服从 ``ν·r ~ χ²(ν)``；把它与整条谱的平均归一化功率作比，得到

        F(f) = r(f) / r̄   ~  F(ν, M_ind·ν)

    其中 ``M_ind = max(2, round(M / K))`` 是参与检验的 ``M`` 个频率点里
    **独立**的频率单元数（多锥谱相邻频率点被同一个锥窗糊在一起，相关长度
    约 K 个频率格）。p 值是 F 分布的上尾概率。

    Parameters
    ----------
    spectrum:
        多锥谱功率（白噪声下平均为 1）。
    frequencies:
        与 ``spectrum`` 等长的频率点，单位是每单位时间。用来做长度与
        合法性检查。
    background:
        背景（期望功率）。标量会被展开成等长的常数数组；必须严格为正。
    n_tapers:
        使用的 taper 数 ``K``。用于估计独立频率单元数，并作为下限检查
        （F 检验至少需要两个锥）。
    degrees_of_freedom:
        有效自由度 ``ν``。标量，或与 ``spectrum`` 等长的逐频率数组
        （例如 :func:`adaptive_degrees_of_freedom` 的输出）。
    n_frequencies:
        参与检验的频率点数（取谱的前若干个点）。None 表示全部。

    Returns
    -------
    ndarray
        逐频率的 p 值，取值 ``(0, 1]``，与被检验的频率点一一对应。

    Raises
    ------
    ValidationError
        数组长度不一致、含非有限值、背景非正、taper 数不足、自由度非正。
    """
    power = np.asarray(spectrum, dtype=float).flatten()
    freqs = np.asarray(frequencies, dtype=float).flatten()
    if power.size < 2:
        raise ValidationError(
            _("the spectrum needs at least two frequency points; got {0}").format(int(power.size)),
            details={"n": int(power.size)},
        )
    if freqs.size != power.size:
        raise ValidationError(
            _("frequencies and spectrum must have the same length; got {0} and {1}").format(
                int(freqs.size), int(power.size)
            ),
            details={"n_frequencies": int(freqs.size), "n_spectrum": int(power.size)},
        )
    if not np.all(np.isfinite(power)):
        raise ValidationError(_("the spectrum must contain only finite values"))
    if not np.all(np.isfinite(freqs)) or np.any(freqs < 0.0):
        raise ValidationError(_("frequencies must be finite and non-negative"))

    # 背景与谱逐点对应，因此先按整条谱检查，再与谱一起截断。
    level = _clean_background(background, power.size)
    full_size = int(power.size)

    if n_frequencies is not None:
        wanted = int(n_frequencies)
        if wanted < 2 or wanted > power.size:
            raise ValidationError(
                _("n_frequencies must lie in [2, {0}]; got {1}").format(int(power.size), wanted),
                details={"n_frequencies": wanted, "n_spectrum": int(power.size)},
            )
        power = power[:wanted]
        freqs = freqs[:wanted]
        level = level[:wanted]

    taps = int(n_tapers)
    if taps < MIN_TAPERS:
        raise ValidationError(
            _("an F test needs at least {0} tapers; got {1}").format(MIN_TAPERS, taps),
            details={"n_tapers": taps, "minimum_required": MIN_TAPERS},
        )

    dof = np.asarray(degrees_of_freedom, dtype=float).flatten()
    if dof.size == 1:
        dof = np.full(full_size, float(dof[0]))
    elif dof.size != full_size:
        raise ValidationError(
            _("degrees_of_freedom must be a scalar or have one value per frequency; got {0}").format(int(dof.size)),
            details={"n_dof": int(dof.size), "n_spectrum": full_size},
        )
    if not np.all(np.isfinite(dof)) or np.any(dof <= 0.0):
        raise ValidationError(_("degrees_of_freedom must be finite and positive"))
    dof = dof[: power.size]

    statistic = _f_statistic(power, level)
    independent = max(2, round(power.size / taps))
    p_values = sp_stats.f.sf(statistic, dof, independent * dof)
    return np.asarray(p_values, dtype=float)


# =============================================================================
# 6. 蒙特卡洛标定
# =============================================================================


def _shaped_noise(count: int, shape: npt.NDArray, rng: np.random.Generator) -> npt.NDArray:
    """按给定功率谱形状生成的零均值随机序列（随机相位整形法）。

    幅度取 ``sqrt(shape)``，相位随机，因此序列的功率谱就是 ``shape``。
    """
    bins = count // 2 + 1
    amplitude = np.zeros(bins, dtype=float)
    usable = int(min(shape.size, bins - 1))
    amplitude[1 : usable + 1] = np.sqrt(np.asarray(shape, dtype=float)[:usable])
    phases = rng.uniform(0.0, 2.0 * np.pi, size=bins)
    spectrum = amplitude * np.exp(1j * phases)
    spectrum[0] = 0.0
    if count % 2 == 0:
        spectrum[-1] = spectrum[-1].real
    series = np.fft.irfft(spectrum, n=count)
    return series - float(np.mean(series))


def _bootstrap_f_test(
    tapers: npt.NDArray,
    eigenvalues: npt.NDArray,
    background: npt.NDArray,
    observed: npt.NDArray,
    n_simulations: int,
    rng: np.random.Generator,
) -> npt.NDArray:
    """参数自助法给出的经验 p 值。

    每条替代序列都走同一条流水线（同样的锥、同样的多锥谱、同样的 F
    统计量），因此零假设分布里包含了锥窗造成的频率点相关性。p 值用
    Phipson & Smyth (2010) 的 ``(1 + k) / (n + 1)`` 形式，一个都没超过的
    频率报 ``1 / (n + 1)`` 而不是 0。
    """
    count = tapers.shape[1]
    shape = background / float(np.mean(background))
    null = np.empty((n_simulations, observed.size), dtype=float)
    for replicate in range(n_simulations):
        simulated = _shaped_noise(count, shape, rng)
        power, _single = _multitaper_power(tapers, eigenvalues, simulated)
        null[replicate] = _f_statistic(power, background)
    exceeded = np.sum(null >= observed[None, :], axis=0)
    return (1.0 + exceeded) / (n_simulations + 1.0)


# =============================================================================
# 7. 自适应多锥谱
# =============================================================================


def tuned_multitaper_spectrum(
    values: npt.NDArray,
    time_bandwidth: float | None = None,
    n_tapers: int | None = None,
    dt: float = 1.0,
    background: npt.NDArray | None = None,
    detrend: bool = True,
    n_simulations: int = 0,
    random_seed: int | None = None,
) -> MTMTuningResult:
    """自适应选参的多锥谱估计，并给出逐频率的 F 检验 p 值。

    与 :func:`stratigraphy.cycles.multitaper_spectrum` 的差别只有两处：
    taper 数与带宽是**自适应**选出来的（见 :func:`select_taper_count`），
    并且附带逐频率的 F 检验 p 值。两者的归一化口径完全相同。

    Parameters
    ----------
    values:
        记录，样本顺序。
    time_bandwidth, n_tapers:
        与 :func:`select_taper_count` 相同的含义；都不给时用
        :data:`DEFAULT_TIME_BANDWIDTH`。
    dt:
        采样间隔（时间单位）。频率轴是**每单位时间**。
    background:
        背景（期望功率）。None 表示白噪声背景（全 1）。标量会展开成等长
        的常数数组。
    detrend:
        是否先去掉线性趋势（长趋势会泄漏到最低频段并抬高红噪声尾部）。
    n_simulations:
        蒙特卡洛标定的替代序列条数。0 表示用解析 F 分布的 p 值。
    random_seed:
        蒙特卡洛的随机种子。None 时抽取熵并发出 ``RuntimeWarning``
        （与仓库其余随机分析一致）。

    Returns
    -------
    MTMTuningResult

    Raises
    ------
    ValidationError
        样本太少、含非有限值、是常数序列、``dt`` 非正、背景长度不符或非正、
        taper 数与带宽非法。
    """
    series = _clean_series(values, "MTM input")
    step = float(dt)
    if not np.isfinite(step) or step <= 0.0:
        raise ValidationError(
            _("dt must be a finite positive sample spacing; got {0}").format(dt),
            details={"dt": dt},
        )

    count, bandwidth = select_taper_count(series.size, time_bandwidth, n_tapers)
    working = _linear_detrend(series) if detrend else series
    eigenvalues, tapers = dpss_tapers(series.size, bandwidth, count)
    power, _single = _multitaper_power(tapers, eigenvalues, working)
    frequencies = np.fft.rfftfreq(series.size, d=step)[1:]
    dof = _effective_degrees_of_freedom(eigenvalues)
    level = _clean_background(background, power.size)

    p_values = multitaper_f_test(power, frequencies, level, count, dof)

    simulations = int(n_simulations)
    if simulations > 0:
        rng = make_rng(random_seed, context="MTM F-test")
        observed = _f_statistic(power, level)
        p_values = _bootstrap_f_test(tapers, eigenvalues, level, observed, simulations, rng)

    logger.info(
        "tuned multitaper: n=%d, K=%d, NW=%.2f, dof=%.2f, n_frequencies=%d",
        int(series.size),
        count,
        bandwidth,
        dof,
        int(power.size),
    )
    return MTMTuningResult(
        bandwidth=float(bandwidth),
        n_tapers=int(count),
        degrees_of_freedom=float(dof),
        spectrum=power,
        frequencies=frequencies,
        p_values=p_values,
        n_samples=int(series.size),
    )
