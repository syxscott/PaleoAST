# =============================================================================
# FILE: stratigraphy/astronomy.py
# =============================================================================
"""
天文解（astronomical solution）与日照量（insolation）数据层。

本模块提供轨道调谐（COCO / TimeOpt / DYNOT）所需的**物理前提**: 一条真正的
轨道解 —— 偏心率、倾角、近日点黄经、升交点黄经随时间的变化, 以及由这些轨道
要素算出的**每日日照量**。

提供的六个能力
--------------
* :class:`AstronomicalSolution` —— 不可变的轨道解数据容器, 含线性插值
  :meth:`~AstronomicalSolution.at` 与谐波周期查询
  :meth:`~AstronomicalSolution.period_kyr`。
* :func:`load_astronomical_solution` —— 读取内置的 La04 天文解
  (``data/astronomy/la04.npz``)。
* :func:`insolation` —— 指定纬度与日序时的每日平均日照量 (W/m²)。
* :func:`daily_mean_insolation` —— 上述日日照量在一年 365 天上的平均
  (即年平均日日照量)。
* :func:`milankovitch_periods` —— 米兰科维奇周期表 (kyr)。
* :func:`astronomical_target_curve` —— 生成给定周期的理论目标曲线, 供
  TimeOpt 一类方法做相位/包络相关。

数据来源与转换
--------------
内置数据来自 Acycle (https://github.com/mingsongli/acycle) 分发的
``data/La04.mat`` (Laskar et al. 2004 的轨道解, Laskar et al. 2011 的
``La10a.mat`` 结构相同)。该 MAT 文件的 ``data`` 矩阵含四列:

===== ==========================================================
列    含义
===== ==========================================================
1     时间轴, 单位 kyr, 自现在 (0 kyr) 向过去递增
2     偏心率 e
3     黄赤交角 epsilon, **单位为弧度**
4     近日点相对春分点的角距 w, 满足 ``col4 = e * sin(w)``
===== ==========================================================

本模块将其转换为紧凑的 ``.npz`` (``times_ma`` / ``eccentricity`` /
``obliquity`` / ``longitude_perihelion`` / ``longitude_ascending_node``),
其中黄赤交角换算为**度**, 近日点黄经按

    varpi = 180 - arcsin(col4 / e)

给出 (即以春分点为基准的近日点黄经; 该式与 Acycle 自身引用的
Berger & Loutre (1992) 表一致: ``PREC / ECC = sin(OMEGA)``)。
升交点黄经 ``Omega`` 由平均岁差常数 (Laskar et al. 2004 的 5028.796195
角秒/世纪) 线性给出, 取今日 (0 Ma) 为 0°。

**关于这个反演的一个已知局限 (尚未解决)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``arcsin`` 取的是**主值分支**, 因此存下来的这条黄经被限制在
``[90, 270]`` 度内, 而近日点黄经本身应当随春分点进动而转遍整圈。
这意味着**记录里 ``cos(varpi)`` 的符号在约一半的年龄上是反的**,
而 Berger 的日照量公式用到的正是 ``e * cos(varpi)`` 这一项。
实测: 把这条黄经整体转 180 度, 65°N 的年平均日日照量最多相差约 **11%**
(赤道不受影响, 因为那里的积分把这个项消掉了)。

**本模块没有改这条公式**, 原因是:
(1) 在 0 Ma 处它给出的 102.918° 与已知的 102.94° 相符, 日照量也与
Berger (1978) 的经典数值相符;
(2) 换一个分支需要拿 Berger & Loutre (1992) 表里**非 0 Ma** 的日照量
来做外部对照, 而那份表在本机不可得——在**没有外部基准**的情况下改一个
已发布的数据文件, 只会把一个"已知有疑问"换成"看起来没问题"。

因此这里把问题**写在明处**: 使用本模块的日照量时, 若关心高纬度的
**季节性非对称性**, 请自行核对 ``cos(varpi)`` 的符号是否与所用分支一致。

时间覆盖
--------
Acycle 分发的 ``La04.mat`` 覆盖 **0–249 Myr**, 步长 **1 kyr** (共 249001
个采样点), 比经典 La04 的 5 Myr 区间长得多。注意 1 kyr 的步长对最短的
19 kyr 岁差谐波只有约 19 个采样点/周期 —— 做轨道调谐时若需要更细的
岁差采样, 应先对轨道要素本身做上采样, 而不是指望采样密度提高周期分辨
(与 ``stratigraphy.spectral_analysis`` 的口径一致)。

日照量算法
----------
按 Berger (1978) 的每日平均日照量公式独立实现 (未翻译 Acycle 的 MATLAB
源码; Acycle 为 GPL 许可)。对纬度 phi、黄赤交角 epsilon、太阳黄经 lambda:

* 赤纬: ``sin(delta) = sin(epsilon) * sin(lambda)``,
  ``cos(delta) = sqrt(1 - sin^2(delta))``;
* 真近点角: ``nu = lambda + (180 - varpi)`` (春分点取作 lambda = 0,
  春分点位于黄经 80 日);
* 日地距离 (AU): ``r = (1 - e^2) / (1 + e * cos(nu))``;
* 日照量: ``S = S0 / r^2 * (A * H0 + sqrt(B^2 - A^2)) / pi``,
  其中 ``A = sin(phi) sin(delta)``、``B = cos(phi) cos(delta)``、
  ``H0 = arccos(-A / B)`` 为日落时角 (弧度); 极昼 (A >= B) 时取 ``A``,
  极夜 (A <= -B) 时取 0。

太阳常数取 ``S0 = 1365 W/m^2``。算法参照 Laskar et al. (2004) 与
Berger (1978)。

单位约定
--------
* 内部一律使用 **Ma**; 与周期有关的量一律使用 **kyr**。
* 角度: 黄赤交角、近日点黄经、升交点黄经均为 **度**。
* 日照量: W/m² (大气顶)。

参考文献
--------
* Laskar, J., Gastineau, M., Jounder, E. & Bourgeois, F., 2004. Solution for
  the eccentricity and obliquity of the Earth's orbits over 5 Myr.
  *Astronomy and Astrophysics*, 413: 755-767.
* Laskar, J., Gastineau, M., Jounder, E. & Bourgeois, F., 2004. A 250-Myr
  solution to the Pliocene-Pleistocene ice ages. *Earth and Planetary Science
  Letters*, 224: 299-313.
* Berger, A., 1978. Long-term climatic variation: astronomical theory.
  In Understanding the Climate. Academic Press.
* Berger, A. & Loutre, M. F., 1992. Astronomical solutions for paleoclimate
  studies over the last 3 million years. *Earth and Planetary Science Letters*,
  111: 369-382.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.resources import as_file, files

import numpy as np
import numpy.typing as npt

from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "AstronomicalSolution",
    "astronomical_target_curve",
    "daily_mean_insolation",
    "insolation",
    "load_astronomical_solution",
    "milankovitch_periods",
]

# =============================================================================
# 常量
# =============================================================================

#: 太阳常数 (W/m^2)。
SOLAR_CONSTANT = 1365.0

#: 回归年长度 (日); 日照量按非闰年的 365 日计算。
DAYS_PER_YEAR = 365.0

#: 春分点所在的日序 (3 月 20 日为年内第 80 日)。
VERNAL_EQUINOX_DAY = 80.0

#: 平均岁差速率 (度/kyr)。取自 Laskar et al. (2004) 的 5028.796195 角秒/世纪。
PRECESSION_DEG_PER_KYR = 5028.796195 / 100.0 / 3600.0 * 1000.0

#: 米兰科维奇周期表 (kyr)。键名即周期族, 数值取经典的平均轨道周期。
#:
#: * ``eccentricity_100`` —— 偏心率主谐波 ~100 kyr;
#: * ``obliquity_41`` —— 倾角主谐波 ~41 kyr;
#: * ``precession_21`` / ``precession_19`` / ``precession_23`` —— 岁差族:
#:   气候岁差 (climatic precession) 的长/中/短三个分支, 其真实周期随偏心率
#:   在 ~19 kyr 与 ~23 kyr 之间变化, 中心值 ~21 kyr。
_MILANKOVITCH_PERIODS: dict[str, float] = {
    "eccentricity_100": 100.0,
    "obliquity_41": 41.0,
    "precession_21": 21.0,
    "precession_19": 19.0,
    "precession_23": 23.0,
}

#: 内置天文解文件名 (位于 ``data/astronomy/`` 下)。
_SOLUTION_FILES: dict[str, str] = {
    "la04": "la04.npz",
}


# =============================================================================
# 轨道解容器
# =============================================================================


@dataclass(frozen=True)
class AstronomicalSolution:
    """
    一条轨道调谐所需的天文解。

    Attributes:
        name: 解的名称 (如 ``"La04"``)。
        times_ma: 年龄轴 (Ma), **由老到新** (最老在前, 故数值递减)。
        eccentricity: 偏心率 e。
        obliquity: 黄赤交角 epsilon (**度**)。
        longitude_perihelion: 近日点黄经 varpi (**度**, 以春分点为基准)。
        longitude_ascending_node: 升交点黄经 Omega (**度**)。
    """

    name: str
    times_ma: npt.NDArray
    eccentricity: npt.NDArray
    obliquity: npt.NDArray
    longitude_perihelion: npt.NDArray
    longitude_ascending_node: npt.NDArray

    def at(self, times_ma: npt.NDArray) -> AstronomicalSolution:
        """
        把轨道解插值到新的年龄轴上。

        线性插值; 黄经经 ``numpy.unwrap`` 解缠后再插值, 以免跨越 360° 处的
        跳变被抹平成 180° 的错误值。

        Parameters:
            times_ma: 目标年龄 (Ma), 须落在本解的时间范围内。

        Returns:
            新的 :class:`AstronomicalSolution`, 其各数组与 ``times_ma`` 等长。

        Raises:
            ValidationError: 若 ``times_ma`` 含非有限值或落在解的时间范围之外。
        """
        target = np.atleast_1d(np.asarray(times_ma, dtype=float))
        if target.size == 0:
            raise ValidationError("times_ma must contain at least one age")
        if not np.all(np.isfinite(target)):
            raise ValidationError("times_ma must contain finite ages only")

        lo = float(np.min(self.times_ma))
        hi = float(np.max(self.times_ma))
        if float(target.min()) < lo or float(target.max()) > hi:
            raise ValidationError(
                "Requested ages fall outside the astronomical solution's time range",
                details={
                    "solution": self.name,
                    "range_ma": [lo, hi],
                    "requested_ma": [float(target.min()), float(target.max())],
                },
            )

        # 本解的年龄轴由老到新 (数值递减), 插值前先排成升序。
        order = np.argsort(self.times_ma)
        times_sorted = self.times_ma[order]

        return AstronomicalSolution(
            name=self.name,
            times_ma=target,
            eccentricity=_interpolate(self.eccentricity, target, times_sorted, order),
            obliquity=_interpolate(self.obliquity, target, times_sorted, order),
            longitude_perihelion=_interpolate(self.longitude_perihelion, target, times_sorted, order),
            longitude_ascending_node=_interpolate_longitude(self.longitude_ascending_node, target, times_sorted, order),
        )

    def period_kyr(self, harmonic: str) -> float:
        """
        查询某个谐波的周期。

        Parameters:
            harmonic: 谐波名, 如 ``"precession_21"``、``"obliquity_41"``、
                ``"eccentricity_100"``。

        Returns:
            周期 (kyr)。

        Raises:
            ValidationError: 若谐波名未知。
        """
        try:
            return _MILANKOVITCH_PERIODS[harmonic]
        except KeyError as exc:
            raise ValidationError(
                f"Unknown Milankovitch harmonic: {harmonic!r}",
                details={"known": sorted(_MILANKOVITCH_PERIODS)},
            ) from exc


def _interpolate(
    values: npt.NDArray, target: npt.NDArray, times_sorted: npt.NDArray, order: npt.NDArray
) -> npt.NDArray:
    """对连续量 (e、epsilon、varpi) 做线性插值; ``times_sorted`` 必须升序。"""
    return np.interp(target, times_sorted, np.asarray(values)[order])


def _interpolate_longitude(
    values: npt.NDArray, target: npt.NDArray, times_sorted: npt.NDArray, order: npt.NDArray
) -> npt.NDArray:
    """对可能跨越 360° 的黄经先解缠再插值, 避免 0°/360° 处被抹平。"""
    unwrapped = np.unwrap(np.radians(np.asarray(values)[order]))
    interpolated = np.interp(target, times_sorted, unwrapped)
    return np.mod(np.degrees(interpolated), 360.0)


# =============================================================================
# 数据加载
# =============================================================================


def load_astronomical_solution(name: str = "la04") -> AstronomicalSolution:
    """
    载入内置的天文解。

    数据文件随包分发 (``data/astronomy/<name>.npz``), 通过
    ``importlib.resources`` 定位, 因此在源码运行、wheel 安装与打包环境下
    均可找到。

    Parameters:
        name: 天文解名称, 目前支持 ``"la04"``。

    Returns:
        :class:`AstronomicalSolution`。

    Raises:
        ValidationError: 若名称未知或数据文件缺失/损坏。
    """
    key = name.strip().lower()
    filename = _SOLUTION_FILES.get(key)
    if filename is None:
        raise ValidationError(
            f"Unknown astronomical solution: {name!r}",
            details={"known": sorted(_SOLUTION_FILES)},
        )

    resource = files("data") / "astronomy" / filename
    try:
        with as_file(resource) as path, np.load(path) as payload:
            arrays = {key: np.asarray(payload[key], dtype=float) for key in payload.files}
    except FileNotFoundError as exc:
        raise ValidationError(
            f"Astronomical solution file not found: {filename}",
            details={"solution": name},
        ) from exc
    except (OSError, ValueError) as exc:
        raise ValidationError(
            f"Astronomical solution file could not be read: {filename}",
            details={"solution": name, "reason": str(exc)},
        ) from exc

    required = (
        "times_ma",
        "eccentricity",
        "obliquity",
        "longitude_perihelion",
        "longitude_ascending_node",
    )
    missing = [item for item in required if item not in arrays]
    if missing:
        raise ValidationError(
            "Astronomical solution file is missing required arrays",
            details={"solution": name, "missing": missing},
        )

    lengths = {len(arrays[item]) for item in required}
    if len(lengths) != 1:
        raise ValidationError(
            "Astronomical solution arrays have inconsistent lengths",
            details={"solution": name, "lengths": sorted(lengths)},
        )

    return AstronomicalSolution(
        name="La04",
        times_ma=arrays["times_ma"],
        eccentricity=arrays["eccentricity"],
        obliquity=arrays["obliquity"],
        longitude_perihelion=arrays["longitude_perihelion"],
        longitude_ascending_node=arrays["longitude_ascending_node"],
    )


# =============================================================================
# 日照量
# =============================================================================


def _daily_mean_flux(
    lat_deg: float,
    obliquity_deg: npt.NDArray,
    eccentricity: npt.NDArray,
    perihelion_deg: npt.NDArray,
    day_of_year: int,
) -> npt.NDArray:
    """
    Berger (1978) 每日平均日照量 (W/m²), 对一组轨道要素求值。

    Parameters:
        lat_deg: 纬度 (度, 北正南负)。
        obliquity_deg: 黄赤交角 (度)。
        eccentricity: 偏心率。
        perihelion_deg: 近日点黄经 (度, 以春分点为基准)。
        day_of_year: 日序 (1..365)。

    Returns:
        与输入要素等长的日照量数组。
    """
    lam = 2.0 * np.pi * (day_of_year - VERNAL_EQUINOX_DAY) / DAYS_PER_YEAR

    # 赤纬: sin(delta) = sin(epsilon) sin(lambda)
    sin_dec = np.sin(np.radians(obliquity_deg)) * np.sin(lam)
    cos_dec = np.sqrt(np.maximum(1.0 - sin_dec**2, 0.0))

    # 真近点角: 春分点处 nu = 180 - varpi
    nu = lam + np.radians(180.0 - perihelion_deg)

    # 日地距离 (以半长轴为单位)
    distance = (1.0 - eccentricity**2) / (1.0 + eccentricity * np.cos(nu))

    # sin(高度角) = A + B cos(H)
    amp = np.sin(np.radians(lat_deg)) * sin_dec
    cos_h = np.cos(np.radians(lat_deg)) * cos_dec

    polar_day = amp >= cos_h
    polar_night = amp <= -cos_h
    ordinary = ~(polar_day | polar_night)

    flux = np.zeros(np.shape(amp), dtype=float)
    flux[polar_day] = amp[polar_day]

    with np.errstate(invalid="ignore", divide="ignore"):
        hour_angle = np.arccos(np.clip(-amp / cos_h, -1.0, 1.0))
        diurnal = (amp * hour_angle + np.sqrt(np.maximum(cos_h**2 - amp**2, 0.0))) / np.pi
    flux[ordinary] = diurnal[ordinary]

    return SOLAR_CONSTANT / distance**2 * flux


def insolation(
    lat_deg: float,
    day_of_year: int,
    solution: AstronomicalSolution,
    times_ma: npt.NDArray,
) -> npt.NDArray:
    """
    指定纬度与日序的每日平均日照量 (W/m²)。

    "每日平均" 指沿一日之内的昼夜循环取平均 (即 Acycle 的 ``qinso = -2``)。

    Parameters:
        lat_deg: 纬度 (度, -90..90)。
        day_of_year: 日序 (1..365; 1 月 1 日为 1, 春分点为 80)。
        solution: 轨道解。
        times_ma: 目标年龄 (Ma)。

    Returns:
        与 ``times_ma`` 等长的日照量数组。

    Raises:
        ValidationError: 若纬度或日序越界, 或年龄超出解的范围。
    """
    lat = float(lat_deg)
    if not np.isfinite(lat) or not -90.0 <= lat <= 90.0:
        raise ValidationError(f"lat_deg must lie in [-90, 90], got {lat_deg!r}")

    day = int(day_of_year)
    if not 1 <= day <= int(DAYS_PER_YEAR):
        raise ValidationError(f"day_of_year must lie in [1, {int(DAYS_PER_YEAR)}], got {day_of_year!r}")

    target = np.atleast_1d(np.asarray(times_ma, dtype=float))
    sampled = solution.at(target)

    return _daily_mean_flux(
        lat,
        sampled.obliquity,
        sampled.eccentricity,
        sampled.longitude_perihelion,
        day,
    )


def daily_mean_insolation(
    lat_deg: float,
    solution: AstronomicalSolution,
    times_ma: npt.NDArray,
) -> npt.NDArray:
    """
    年平均日日照量 (W/m²): 一年 365 天每日平均日照量的平均。

    Parameters:
        lat_deg: 纬度 (度, -90..90)。
        solution: 轨道解。
        times_ma: 目标年龄 (Ma)。

    Returns:
        与 ``times_ma`` 等长的年平均日日照量数组。

    Raises:
        ValidationError: 若纬度越界或年龄超出解的范围。
    """
    lat = float(lat_deg)
    if not np.isfinite(lat) or not -90.0 <= lat <= 90.0:
        raise ValidationError(f"lat_deg must lie in [-90, 90], got {lat_deg!r}")

    target = np.atleast_1d(np.asarray(times_ma, dtype=float))
    sampled = solution.at(target)

    total = np.zeros(sampled.times_ma.shape, dtype=float)
    for day in range(1, int(DAYS_PER_YEAR) + 1):
        total += _daily_mean_flux(
            lat,
            sampled.obliquity,
            sampled.eccentricity,
            sampled.longitude_perihelion,
            day,
        )
    return total / DAYS_PER_YEAR


# =============================================================================
# 周期与目标曲线
# =============================================================================


def milankovitch_periods() -> dict[str, float]:
    """
    米兰科维奇周期表 (kyr)。

    Returns:
        谐波名到周期 (kyr) 的映射。键名沿用 ``stratigraphy.tuning`` 的约定:
        ``eccentricity_100``、``obliquity_41``、``precession_21``、
        ``precession_19``、``precession_23``。

    Example:
        >>> periods = milankovitch_periods()
        >>> periods["obliquity_41"]
        41.0
    """
    return dict(_MILANKOVITCH_PERIODS)


def astronomical_target_curve(
    period_kyr: float,
    times_ma: npt.NDArray,
    amplitude: float = 1.0,
) -> npt.NDArray:
    """
    生成给定周期的理论目标曲线 (供 TimeOpt 一类方法做相位相关)。

    曲线为余弦, 周期以 kyr 给出, 时间轴以 Ma 给出。

    Parameters:
        period_kyr: 目标周期 (kyr), 必须为正。
        times_ma: 时间轴 (Ma)。
        amplitude: 振幅。

    Returns:
        与 ``times_ma`` 等长的曲线。

    Raises:
        ValidationError: 若周期非正或时间轴含非有限值。
    """
    period = float(period_kyr)
    if not np.isfinite(period) or period <= 0.0:
        raise ValidationError(f"period_kyr must be a finite positive number, got {period_kyr!r}")

    times = np.atleast_1d(np.asarray(times_ma, dtype=float))
    if not np.all(np.isfinite(times)):
        raise ValidationError("times_ma must contain finite ages only")

    phase = 2.0 * np.pi * (times * 1000.0) / period
    return float(amplitude) * np.cos(phase)


# =============================================================================
# 供界面调用的门面
# =============================================================================
@dataclass(frozen=True)
class InsolationResult:
    """一个纬度上的日照量。"""

    times_ma: npt.NDArray
    insolation: npt.NDArray
    latitude_deg: float
    solution_name: str

    def summary(self) -> str:
        values = np.asarray(self.insolation, dtype=float)
        if values.size == 0:
            return "No insolation values."
        return (
            f"Insolation at {self.latitude_deg:g} deg from {self.solution_name}: "
            f"mean {values.mean():.1f} W/m2, min {values.min():.1f}, max {values.max():.1f} "
            f"over {values.size} ages."
        )


class InsolationAnalyzer:
    """日照量分析, 供生成的参数表单驱动。

    轨道解本身是一个对象, 表单给不了; 门面按名字自己加载, 只让用户填纬度
    与时间轴。
    """

    def analyze(
        self,
        times_ma: npt.NDArray,
        latitude_deg: float = 0.0,
        solution_name: str = "la04",
        daily_mean: bool = True,
    ) -> InsolationResult:
        """算一个纬度上的日照量。

        Parameters:
            times_ma: 目标年龄 (Ma), 自表格的一列。
            latitude_deg: 纬度 (度)。
            solution_name: 轨道解的名字。
            daily_mean: True 取年平均日日照量, False 取指定日序的日均值。

        Returns:
            :class:`InsolationResult`
        """
        solution = load_astronomical_solution(solution_name)
        times = np.atleast_1d(np.asarray(times_ma, dtype=float))
        if daily_mean:
            values = daily_mean_insolation(latitude_deg, solution, times)
        else:
            values = insolation(latitude_deg, 172, solution, times)
        return InsolationResult(
            times_ma=times,
            insolation=values,
            latitude_deg=float(latitude_deg),
            solution_name=str(solution_name),
        )
