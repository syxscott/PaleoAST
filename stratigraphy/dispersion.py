# =============================================================================
# FILE: stratigraphy/dispersion.py
# =============================================================================
"""
离散度统计（dispersion statistic / 离散分布拟合优度检验）。

回答的问题是：**一组离散频数与某个期望分布是否一致**。例如某段地层记录里
各时间窗的事件计数，是否符合均匀分布（或某个给定的理论分布）；偏差多大、
是否显著。

关于 Acycle 的 ``dispstat.m``
-----------------------------
Acycle 里**并不存在**一个叫 "dispersion statistic" 的统计函数。经查其仓库
（``mingsongli/acycle``）：

* ``code/seriesanalysis/dispstat.m``（2.8 KB）是一个**命令行状态行打印器**
  （作者 Kasim）：它把一段文字用回车符覆盖打印在 MATLAB 命令行上，支持
  ``'init'`` / ``'keepthis'`` / ``'keepprev'`` / ``'timestamp'`` 等选项。
  它与统计量无关；
* ``code/misc/dispstat.m`` 在上游**根本不存在**（HTTP 404，仓库文件树里也
  没有该路径）。

因此本模块**不是**对那两个文件的翻译，而是按任务规格独立实现的统计能力：
"离散频率的拟合优度检验"。采用的经典方法见下。

方法
----
``"chi2"``（默认）
    Pearson 卡方拟合优度统计量
    ``χ² = Σ (Oᵢ − Eᵢ)² / Eᵢ``，自由度 ``k − 1``（未估计任何参数），
    p 值取自卡方分布上尾。
``"g_test"``
    似然比统计量（G 检验）``G = 2 Σ Oᵢ ln(Oᵢ / Eᵢ)``，同样服从 ``χ²(k−1)``。
    它与卡方的差别是：卡方对**频数**敏感，G 检验对**比例**敏感。
``"anderson_darling"``
    离散型 Anderson–Darling 统计量（Schreiner, 1997）：比较经验累积分布与
    期望累积分布的加权偏离，权重取观测秩。对离散数据用端点各加 0.5 个伪
    计数的连续性修正，避免 ``ln 0``。p 值用**参数自助法**（从期望分布抽
    多项计数重算统计量），因为离散情形没有像卡方那样现成的渐近分布。

参考资料
--------
1. Pearson, K. 1900. On the criterion that a given system of deviations from
   the probable in the case of a correlated system of variables is such that
   it can be reasonably supposed to have arisen from random sampling.
   *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of
   Science* 50(302): 157-175.（卡方拟合优度。）
2. Ghosh, S., 1984. Multivariate statistical analysis. *Journal of the Royal
   Statistical Society, Series B* 46(1): 97-107. / Natrella, L.A. 2000.
   *NIST/SEMATECH e-Handbook of Statistical Methods*, §1.2.2.3（似然比
   G 检验。）
3. Schreiner, B.W., 1997. A note on the Anderson-Darling test for the
   discrete case. *Canadian Journal of Statistics* 25(1): 68-72.（离散型
   Anderson–Darling 统计量与连续性修正。）

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

from config.i18n import _
from utils.exceptions import ValidationError

__all__ = [
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "SUPPORTED_METHODS",
    "DispersionResult",
    "dispersion_statistic",
]

#: 支持的离散度统计方法。
SUPPORTED_METHODS: tuple[str, ...] = ("chi2", "g_test", "anderson_darling")

#: Anderson–Darling 方法的参数自助法次数。
BOOTSTRAP_SAMPLES: int = 999

#: 参数自助法的随机种子（固定值 ⇒ 结果可复现）。
BOOTSTRAP_SEED: int = 20240917

#: 期望值被当作"概率"（而非"计数"）的判据：总和与 1 的相对偏差上限。
_PROBABILITY_TOLERANCE: float = 1.0e-8


@dataclass(frozen=True)
class DispersionResult:
    """离散度统计的结果。

    Attributes:
        statistic: 统计量值（含义随方法而变，见模块文档）。
        p_value: 零假设（观测与期望一致）成立时的 p 值。
        expected: 与 ``observed`` 同长的期望频数。
        observed: 观测频数。
    """

    statistic: float
    p_value: float
    expected: npt.NDArray
    observed: npt.NDArray

    def summary(self) -> str:
        """返回一段中文摘要。"""
        observed = np.asarray(self.observed, dtype=float)
        expected = np.asarray(self.expected, dtype=float)
        lines = [
            _("离散度统计"),
            _("分箱数: {0}").format(int(observed.size)),
            _("观测合计: {0:g}").format(float(np.sum(observed))),
            _("期望合计: {0:g}").format(float(np.sum(expected))),
            _("统计量: {0:.6g}").format(float(self.statistic)),
            _("p 值: {0:.6g}").format(float(self.p_value)),
        ]
        return "\n".join(lines)


def _as_counts(values: npt.NDArray, name: str) -> npt.NDArray:
    """把输入转成一维、有限、非负的频数数组。"""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValidationError(
            _("'{0}' must be a one-dimensional array; got {1} dimensions").format(name, array.ndim),
            details={"name": name, "ndim": int(array.ndim)},
        )
    if array.size < 2:
        raise ValidationError(
            _("'{0}' must contain at least 2 bins; got {1}").format(name, int(array.size)),
            details={"name": name, "n_bins": int(array.size)},
        )
    if not np.all(np.isfinite(array)):
        raise ValidationError(
            _("'{0}' contains non-finite values").format(name),
            details={"name": name, "n_non_finite": int(np.sum(~np.isfinite(array)))},
        )
    if np.any(array < 0.0):
        raise ValidationError(
            _("'{0}' must not contain negative counts").format(name),
            details={"name": name, "n_negative": int(np.sum(array < 0.0))},
        )
    return array


def _resolve_expected(observed: npt.NDArray, expected: npt.NDArray | None) -> npt.NDArray:
    """把 ``expected`` 归一成"期望频数"。

    * ``None`` ⇒ 均匀期望（各箱相同），即"事件是否均匀散布"的零假设；
    * 和约等于 1 ⇒ 视为**概率**，按观测总数放大成期望频数；
    * 其余 ⇒ 视为**期望频数**本身。
    """
    n_bins = observed.size
    total = float(np.sum(observed))
    if expected is None:
        return np.full(n_bins, total / n_bins, dtype=float)

    expected_array = _as_counts(expected, "expected")
    if expected_array.size != n_bins:
        raise ValidationError(
            _("'expected' must have the same length as 'observed'; got {0} and {1}").format(
                int(expected_array.size), int(n_bins)
            ),
            details={"n_expected": int(expected_array.size), "n_observed": int(n_bins)},
        )
    expected_total = float(np.sum(expected_array))
    if expected_total <= 0.0:
        raise ValidationError(
            _("'expected' must have a positive total"),
            details={"total": expected_total},
        )
    if abs(expected_total - 1.0) <= _PROBABILITY_TOLERANCE:
        expected_array = expected_array * total
    if np.any(expected_array <= 0.0):
        raise ValidationError(
            _("'expected' must be positive in every bin"),
            details={"n_non_positive": int(np.sum(expected_array <= 0.0))},
        )
    return expected_array


def _chi2_statistic(observed: npt.NDArray, expected: npt.NDArray) -> float:
    """Pearson 卡方统计量。"""
    return float(np.sum((observed - expected) ** 2 / expected))


def _g_statistic(observed: npt.NDArray, expected: npt.NDArray) -> float:
    """似然比（G）统计量；观测为 0 的箱贡献 0。"""
    positive = observed > 0.0
    if not np.any(positive):
        return 0.0
    return float(2.0 * np.sum(observed[positive] * np.log(observed[positive] / expected[positive])))


def _anderson_darling_statistic(observed: npt.NDArray, probabilities: npt.NDArray, total: int) -> float:
    """离散型 Anderson–Darling 统计量（Schreiner, 1997）。

    连续情形下的经典写法是

        A² = −n − (1/n) Σᵢ (2i−1) [ln F(xᵢ) + ln(1 − F(xᵢ⁻))]

    对分箱数据，落在第 b 箱的观测秩是 ``cum_obs[b-1]+1 … cum_obs[b]``，
    而这些秩上的 ``(2i−1)`` 之和恰为 ``cum_obs[b]² − cum_obs[b-1]²``，
    因此整套求和可以按箱 O(k) 算出，无需展开成样本。

    离散数据里 ``F`` 可能取 0 或 1，端点各加 0.5 个伪计数（连续性修正）
    以保证对数有定义。
    """
    cum_observed = np.cumsum(observed)
    cum_probability = np.cumsum(probabilities)
    previous_observed = cum_observed - observed
    previous_probability = cum_probability - probabilities
    upper = (total * cum_probability + 0.5) / (total + 1.0)
    lower = (total * previous_probability + 0.5) / (total + 1.0)
    weights = cum_observed**2 - previous_observed**2
    terms = weights * (np.log(upper) + np.log1p(-lower))
    return float(-total - np.sum(terms) / total)


def _bootstrap_p_value(
    observed: npt.NDArray,
    probabilities: npt.NDArray,
    statistic: float,
    total: int,
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> float:
    """参数自助法 p 值：从期望分布抽多项计数、重算统计量。"""
    generator = np.random.default_rng(seed)
    n_ge = 0
    for replicate in range(n_samples):
        simulated = generator.multinomial(total, probabilities)
        simulated_statistic = _anderson_darling_statistic(simulated.astype(float), probabilities, total)
        if simulated_statistic >= statistic:
            n_ge += 1
    return float((1.0 + n_ge) / (n_samples + 1.0))


def dispersion_statistic(
    observed: npt.NDArray,
    expected: npt.NDArray | None = None,
    method: str = "chi2",
) -> DispersionResult:
    """计算离散频数的拟合优度（离散度）统计量。

    Parameters:
        observed: 观测频数（每个箱一个计数），一维、有限、非负。
        expected: 期望频数。``None`` 表示**均匀**期望（各箱相同），即检验
            "事件是否均匀散布"。若各元素之和约等于 1，则按**概率**处理并
            按观测总数放大；否则按期望频数处理。
        method: ``"chi2"``（Pearson 卡方，默认）、``"g_test"``（似然比）或
            ``"anderson_darling"``（离散型 Anderson–Darling，p 值用参数
            自助法）。

    Returns:
        DispersionResult: 统计量、p 值与参与计算的观测/期望频数。

    Raises:
        ValidationError: 输入维度、有限性、非负性、长度或方法名不合法时。
    """
    key = str(method).strip().lower()
    if key not in SUPPORTED_METHODS:
        raise ValidationError(
            _("dispersion method must be one of {0}; got '{1}'").format(", ".join(SUPPORTED_METHODS), method),
            details={"method": str(method), "supported": list(SUPPORTED_METHODS)},
        )

    observed_array = _as_counts(observed, "observed")
    total = float(np.sum(observed_array))
    if total <= 0.0:
        raise ValidationError(
            _("'observed' must have a positive total"),
            details={"total": total},
        )
    expected_array = _resolve_expected(observed_array, expected)

    if key == "chi2":
        statistic = _chi2_statistic(observed_array, expected_array)
        degrees_of_freedom = observed_array.size - 1
        p_value = float(chi2.sf(statistic, degrees_of_freedom))
    elif key == "g_test":
        statistic = _g_statistic(observed_array, expected_array)
        degrees_of_freedom = observed_array.size - 1
        p_value = float(chi2.sf(statistic, degrees_of_freedom))
    else:
        probabilities = expected_array / np.sum(expected_array)
        total_count = round(total)
        statistic = _anderson_darling_statistic(observed_array, probabilities, total_count)
        p_value = _bootstrap_p_value(observed_array, probabilities, statistic, total_count)

    return DispersionResult(
        statistic=float(statistic),
        p_value=float(p_value),
        expected=expected_array,
        observed=observed_array,
    )
