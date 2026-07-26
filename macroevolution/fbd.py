"""
================================================================================
PaleoAST Macroevolution - Fossilized Birth-Death Process
================================================================================

本模块实现化石生灭过程 (FBD) 的随机模拟。

数学理论:
================================================================================

1. 生灭过程定义
--------------------------------------------------------------------------------
化石生灭过程是系统发育树的先验模型:

    - 出生率 (Speciation): λ
    - 灭绝率 (Extinction): μ
    - 化石保存率 (Fossilization): ψ

2. 连续时间马尔可夫链
--------------------------------------------------------------------------------
状态空间: {N(t), 当前节点活跃}

转移率:
    (n, active) → (n+1, active): nλ    (出生事件)
    (n, active) → (n-1, active): nμ   (灭绝事件)
    (n, active) → (n, fossilizing): nψ (化石保存)

3. Gillespie算法 (随机模拟)
--------------------------------------------------------------------------------
用于精确模拟FBD过程:

步骤:
    1. 确定当前状态 (n 物种活跃)
    2. 计算总速率: R = n(λ + μ + ψ)
    3. 生成事件时间: τ = -ln(U₁)/R, U₁~Uniform(0,1)
    4. 选择事件类型:
       - 出生: 概率 λ/R
       - 灭绝: 概率 μ/R
       - 化石保存: 概率 ψ/R
    5. 更新状态和时间
    6. 重复直到达到时间终点

4. 似然函数
--------------------------------------------------------------------------------
给定完整系统发育树 T 和化石记录 F:

    L(T, λ, μ, ψ | F) = Pr(F | T, ψ) × Pr(T | λ, μ)

其中 Pr(F | T, ψ) 是化石保存的完整性概率。

5. 存活概率
--------------------------------------------------------------------------------
一个物种在时间 t 内存活的概率:

    S(t) = exp(-(λ + μ)t) × [条件项]

6. 化石保存数量分布
--------------------------------------------------------------------------------
保存的化石数量服从复合分布:

    P(k fossils | age, ψ) = Poisson(k; ψ × age)

7. 采样比例 (Sample Proportion)
--------------------------------------------------------------------------------
    ρ = ψ / (λ + μ)

8. 系统发育树的先验
--------------------------------------------------------------------------------
对于生灭过程，树高度和分支长度的分布:

    f(T | λ, μ) = ∏_{branches} λ × exp(-(λ+μ)L_i)

其中 L_i 是第 i 个分支的长度。

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class FBDEventType(Enum):
    """FBD事件类型"""

    BIRTH = auto()  # 物种形成
    DEATH = auto()  # 灭绝
    FOSSILIZATION = auto()  # 化石保存


@dataclass
class FBDEvent:
    """FBD事件"""

    event_type: FBDEventType
    time: float
    parent_id: int
    child_id: int | None = None
    lineage_id: int | None = None


@dataclass
class Lineage:
    """谱系"""

    lineage_id: int
    birth_time: float
    death_time: float | None = None
    fossil_ages: list[float] = field(default_factory=list)
    parent_id: int | None = None
    children_ids: list[int] = field(default_factory=list)
    is_alive: bool = True


@dataclass
class FBDSimulationResult:
    """
    FBD模拟结果

    属性:
        lineages: 谱系列表
        events: 事件序列
        extant_species: 现生物种数
        fossil_count: 化石总数
        speciation_times: 物种形成时间
        extinction_times: 灭绝时间
        diversity_curve: 多样性随时间变化
    """

    lineages: list[Lineage]
    events: list[FBDEvent]
    extant_species: int
    fossil_count: int
    speciation_times: np.ndarray
    extinction_times: np.ndarray
    diversity_curve: np.ndarray

    @property
    def tree_height(self) -> float:
        """树高度 (最老物种年龄)"""
        if not self.lineages:
            return 0.0
        return max(l.birth_time for l in self.lineages)

    @property
    def survival_fraction(self) -> float:
        """存活比例"""
        total = len(self.lineages)
        if total == 0:
            return 0.0
        extinct = sum(1 for l in self.lineages if not l.is_alive)
        return 1 - extinct / total


class GillespieSimulator:
    """
    Gillespie随机模拟器

    实现Gillespie算法进行FBD过程模拟。

    使用示例:
        >>> sim = GillespieSimulator(
        ...     speciation_rate=0.5,
        ...     extinction_rate=0.2,
        ...     fossilization_rate=0.1
        ... )
        >>> sim.initialize(n_lineages=1)
        >>> sim.run(duration=10.0)
        >>> result = sim.get_result()
        >>> print(f"Diversity curve: {result.diversity_curve}")
    """

    def __init__(
        self, speciation_rate: float, extinction_rate: float, fossilization_rate: float, random_seed: int | None = None
    ):
        """
        初始化模拟器

        参数:
            speciation_rate: 物种形成率 λ
            extinction_rate: 灭绝率 μ
            fossilization_rate: 化石保存率 ψ
            random_seed: 随机种子
        """
        if speciation_rate <= 0:
            raise ValueError("Speciation rate must be positive")
        if extinction_rate < 0:
            raise ValueError("Extinction rate must be non-negative")
        if fossilization_rate < 0:
            raise ValueError("Fossilization rate must be non-negative")

        self._lambda = speciation_rate
        self._mu = extinction_rate
        self._psi = fossilization_rate

        if random_seed is not None:
            np.random.seed(random_seed)

        self._logger = logging.getLogger(f"{__name__}.GillespieSimulator")

        self._lineages: list[Lineage] = []
        self._events: list[FBDEvent] = []
        self._next_lineage_id = 0
        self._current_time = 0.0
        self._is_initialized = False

    def initialize(self, n_lineages: int = 1, start_time: float = 0.0) -> None:
        """
        初始化模拟

        参数:
            n_lineages: 初始谱系数
            start_time: 起始时间
        """
        self._lineages = []
        self._events = []
        self._next_lineage_id = 0
        self._current_time = start_time

        for _ in range(n_lineages):
            lineage = Lineage(lineage_id=self._next_lineage_id, birth_time=start_time, parent_id=None, is_alive=True)
            self._lineages.append(lineage)
            self._next_lineage_id += 1

        self._is_initialized = True

        self._logger.info(f"Initialized with {n_lineages} lineages at time {start_time}")

    def run(self, duration: float | None = None, end_time: float | None = None, max_events: int = 100000) -> None:
        """
        运行模拟

        参数:
            duration: 模拟时长
            end_time: 结束时间 (与duration二选一)
            max_events: 最大事件数
        """
        if not self._is_initialized:
            raise RuntimeError("Must call initialize() first")

        if duration is not None:
            end_time = self._current_time + duration
        elif end_time is None:
            raise ValueError("Must specify either duration or end_time")

        self._logger.info(
            f"Running simulation until time {end_time}, "
            f"speciation={self._lambda}, extinction={self._mu}, "
            f"fossilization={self._psi}"
        )

        event_count = 0

        while self._current_time < end_time and event_count < max_events:
            # 获取活跃谱系
            alive_lineages = [l for l in self._lineages if l.is_alive]
            n_alive = len(alive_lineages)

            if n_alive == 0:
                self._logger.info("All lineages extinct, stopping simulation")
                break

            # 计算总速率
            total_rate = n_alive * (self._lambda + self._mu + self._psi)

            if total_rate <= 0:
                break

            # 生成事件时间
            tau = np.random.exponential(1.0 / total_rate)

            # 检查是否超过结束时间
            if self._current_time + tau > end_time:
                self._current_time = end_time
                break

            # 更新时间
            self._current_time += tau

            # 选择事件类型和谱系
            rates = np.array([self._lambda, self._mu, self._psi])
            probs = rates / rates.sum()

            event_type_idx = np.random.choice(3, p=probs)
            event_types = [FBDEventType.BIRTH, FBDEventType.DEATH, FBDEventType.FOSSILIZATION]
            event_type = event_types[event_type_idx]

            # 选择活跃谱系
            lineage_idx = np.random.randint(0, n_alive)
            parent_lineage = alive_lineages[lineage_idx]

            # 创建事件
            event = FBDEvent(event_type=event_type, time=self._current_time, parent_id=parent_lineage.lineage_id)
            self._events.append(event)

            # 处理事件
            if event_type == FBDEventType.BIRTH:
                # 创建新谱系
                new_lineage = Lineage(
                    lineage_id=self._next_lineage_id,
                    birth_time=self._current_time,
                    parent_id=parent_lineage.lineage_id,
                    is_alive=True,
                )
                parent_lineage.children_ids.append(new_lineage.lineage_id)

                self._lineages.append(new_lineage)
                self._next_lineage_id += 1

                event.child_id = new_lineage.lineage_id

            elif event_type == FBDEventType.DEATH:
                # 谱系灭绝
                parent_lineage.is_alive = False
                parent_lineage.death_time = self._current_time

            elif event_type == FBDEventType.FOSSILIZATION:
                # 保存化石年龄
                parent_lineage.fossil_ages.append(self._current_time)

            event_count += 1

        self._logger.info(f"Simulation complete: {event_count} events, time = {self._current_time}")

    def get_result(self) -> FBDSimulationResult:
        """
        获取模拟结果

        返回:
            FBDSimulationResult
        """
        # 统计
        extant = sum(1 for l in self._lineages if l.is_alive)
        fossil_count = sum(len(l.fossil_ages) for l in self._lineages)

        # 物种形成时间
        speciation_times = np.array([l.birth_time for l in self._lineages if l.parent_id is not None])

        # 灭绝时间
        extinction_times = np.array([l.death_time for l in self._lineages if l.death_time is not None])

        # 多样性曲线
        diversity_curve = self._compute_diversity_curve()

        return FBDSimulationResult(
            lineages=self._lineages,
            events=self._events,
            extant_species=extant,
            fossil_count=fossil_count,
            speciation_times=speciation_times,
            extinction_times=extinction_times,
            diversity_curve=diversity_curve,
        )

    def _compute_diversity_curve(self) -> np.ndarray:
        """计算多样性随时间变化。

        返回一维数组，每个元素为对应时刻的多样性计数。
        起始时刻（初始化时刻）的初始多样性也包含在曲线中，
        确保曲线从模拟开始时就有记录。
        """
        if not self._lineages:
            return np.array([])

        from collections import Counter

        # Build event lookup: time -> (births, deaths)
        births: Counter = Counter()
        deaths: Counter = Counter()
        for e in self._events:
            if hasattr(e, "event_type"):
                if e.event_type == FBDEventType.BIRTH:
                    births[e.time] += 1
                elif e.event_type == FBDEventType.DEATH:
                    deaths[e.time] += 1

        # Initial lineages: those created during initialize() (parent_id is None).
        # Their birth_time equals the start_time passed to initialize().
        start_time = self._lineages[0].birth_time
        initial_lineages = sum(
            1 for l in self._lineages if l.parent_id is None and l.birth_time == start_time
        )

        # Build time axis: include start_time so the curve begins at the
        # simulation start, not at the first event.
        all_times = sorted(set(
            [start_time] + [e.time for e in self._events] + [self._current_time]
        ))

        diversity = []
        current_n = initial_lineages
        for t in all_times:
            current_n += births.get(t, 0) - deaths.get(t, 0)
            diversity.append(max(0, current_n))

        return np.array(diversity)


class FossilizedBirthDeathProcess:
    """
    化石生灭过程

    提供FBD分布的解析计算和MCMC采样。

    使用示例:
        >>> fbd = FossilizedBirthDeathProcess(
        ...     lambda_=0.5,
        ...     mu=0.2,
        ...     psi=0.1
        ... )
        >>>
        >>> # 计算存活概率
        >>> S = fbd.survival_probability(age=5.0)
        >>> print(f"Survival prob: {S:.4f}")
        >>>
        >>> # 计算似然
        >>> tree = read_phylogeny(...)
        >>> fossils = [(4.0,), (2.5,), (1.0,)]
        >>> log_lik = fbd.log_likelihood(tree, fossils)
    """

    def __init__(self, lambda_: float, mu: float, psi: float, sampling_probability: float | None = None):
        """
        初始化FBD过程

        参数:
            lambda_: 物种形成率
            mu: 灭绝率
            psi: 化石保存率
            sampling_probability: 显式采样概率 (替代psi)
        """
        self._lambda = lambda_
        self._mu = mu
        self._psi = psi
        self._rho = sampling_probability  # 采样比例

        self._logger = logging.getLogger(f"{__name__}.FBDProcess")

    def survival_probability(self, age: float) -> float:
        """
        计算物种存活概率（至少有一个后裔存活到时间t的概率）

        参数:
            age: 年龄

        返回:
            存活概率 S(t)

        数学公式:
            令 r = λ - μ

            当 r ≠ 0 时:
                S(t) = r / (r + μ × (1 - exp(-r × t)))

            当 r = 0 (λ = μ) 时:
                S(t) = 1 / (1 + μ × t)
        """
        if age < 0:
            raise ValueError("Age must be non-negative")

        if age == 0:
            return 1.0

        r = self._lambda - self._mu

        if abs(r) < 1e-10:
            # λ ≈ μ
            return 1.0 / (1.0 + self._mu * age)
        else:
            # λ ≠ μ
            # S(t) = r / (r + μ * (1 - exp(-r*t)))
            exp_rt = np.exp(-r * age)
            return r / (r + self._mu * (1.0 - exp_rt))

    def expected_diversity(self, time: float) -> float:
        """
        计算期望多样性

        对于纯生灭过程，从1个物种开始，t时刻的期望物种数为：

            E[N(t)] = exp((λ - μ) × t)

        当 λ > μ 时指数增长，当 λ < μ 时指数衰减。

        参数:
            time: 时间

        返回:
            期望物种数
        """
        r = self._lambda - self._mu
        return np.exp(r * time)

    def fossil_count_distribution(self, age: float, max_k: int = 20) -> np.ndarray:
        """
        计算化石数量的概率分布

        参数:
            age: 物种年龄
            max_k: 最大化石数

        返回:
            P(K = k) 概率分布
        """
        mean_count = self._psi * age

        # Poisson分布
        k = np.arange(max_k + 1)
        probs = stats.poisson.pmf(k, mean_count)

        return probs

    def _E(self, t: float) -> float:
        """Probability that a lineage alive at time ``t`` (measured
        backwards from the present, ``t = 0``) leaves *no* sampled
        descendants — neither an extant sampled tip nor any fossil —
        under the time-homogeneous FBD process with rates
        ``(λ, μ, ψ)`` and extant sampling fraction ``ρ``.

        ``E(t)`` satisfies the Riccati ODE
        ``dE/dt = λ E² − (λ + μ + ψ) E + (μ + ψ)`` with boundary
        condition ``E(0) = 1 − ρ``. For constant rates the closed-form
        solution is

            γ = sqrt((λ − μ − ψ)² + 4 λ ψ)
            α = (λ + μ + ψ + γ) / (2 λ)
            β = (λ + μ + ψ − γ) / (2 λ)
            E(t) = (β (r − α) e^{γ t} − α (r − β)) /
                   ((r − α) e^{γ t} − (r − β)),   r = 1 − ρ

        derived by partial fractions of ``dE / ((E − α)(E − β)) = λ dt``.
        This is the standard ``p₀(t)`` of Stadler (2010) / Heath et al.
        (2014); it is the building block of the FBD likelihood because
        each branching event, fossil find, and extinct terminal branch
        must be weighted by the probability that the *unobserved* side
        of the event left no sampled descendants.
        """
        rho = self._rho if self._rho is not None else 1.0
        # Degenerate cases.
        # BUG FIX: Handle λ → 0 limit using Taylor expansion for numerical stability.
        # When λ is very small but positive, the closed-form formula can produce
        # numerical instability due to division by λ. The Taylor expansion
        # E(t) ≈ exp(-(μ+ψ)*t) provides a stable approximation.
        # In the pure-death limit (λ → 0, ψ → 0), this correctly gives exp(-μ*t).
        if self._lambda < 1e-10:
            # Use Taylor expansion: E(t) ≈ exp(-(μ+ψ)*t) for small λ
            # This is the leading-order term when λ → 0
            return max(0.0, min(1.0, np.exp(-(self._mu + self._psi) * t)))
        if self._lambda <= 0:
            # No speciation: lineage either dies (μ) or is sampled (ψ).
            # With λ = 0 the lineage cannot branch, so E(t) is governed by
            # the simpler death-plus-sampling process. Fall back to the
            # extant-sampling-only case E(t) = 1 − ρ e^{−(μ+ψ) t}.
            return max(0.0, min(1.0, 1.0 - rho * np.exp(-(self._mu + self._psi) * t)))
        if t <= 0:
            return max(0.0, min(1.0, 1.0 - rho))

        gamma = np.sqrt((self._lambda - self._mu - self._psi) ** 2 + 4.0 * self._lambda * self._psi)
        alpha = (self._lambda + self._mu + self._psi + gamma) / (2.0 * self._lambda)
        beta = (self._lambda + self._mu + self._psi - gamma) / (2.0 * self._lambda)
        r = 1.0 - rho
        # E(t) = (β(r-α) e^{γt} - α(r-β)) / ((r-α) e^{γt} - (r-β))
        # Guard against overflow: np.exp(>709) overflows to inf for float64
        if gamma * t > 700:
            # For large t, E(t) approaches the smaller root (beta)
            return float(max(0.0, min(1.0, beta)))
        e_gt = np.exp(gamma * t)
        num = beta * (r - alpha) * e_gt - alpha * (r - beta)
        den = (r - alpha) * e_gt - (r - beta)
        if abs(den) < 1e-300:
            return 1.0
        val = num / den
        # Numerical guard: E(t) is a probability.
        return float(max(0.0, min(1.0, val)))

    def log_likelihood(self, tree, fossils: list[tuple[float, ...]], complete_tree: bool = True) -> float:
        """
        计算FBD过程的对数似然 (Stadler 2010, Heath et al. 2014)

        似然函数按事件分解为：

            log L = Σ_branches [−(λ+μ+ψ) × Δt]
                  + Σ_internal_nodes  [log λ + log E(t_node)]
                  + Σ_fossils         [log ψ + log E(t_fossil)]
                  + Σ_extant_leaves   [log ρ]
                  + Σ_extinct_leaves  [log μ + log E(t_leaf)]

        其中 ``E(t)`` 是一条存在于时刻 ``t`` 的谱系不留任何采样
        后代的概率（见 :meth:`_E`）。

        与旧实现的差异：

        - 旧代码用 ``log(λ·ρ)`` 给现存叶节点，这会把分支事件中的
          ``λ`` 重复计入；标准公式只对现存叶节点计 ``ρ``。
        - 旧代码用 ``log(ψ) − ψ·age`` 给化石项，正确形式应为
          ``log ψ + log E(age)``：一个化石意味着该时刻被采样，而
          该谱系自此不再留下其他采样后代（否则会出现在树/化石记
          录里），因此需要 ``E(age)`` 因子。
        - 旧代码完全缺失内部节点与灭绝叶节点的 ``E(t)`` 因子，
          导致似然对采样比例 ρ 与化石采样率 ψ 的依赖被忽略。
        """
        # 解析树
        from ..phylogenetics.tree import PhyloTree

        if isinstance(tree, str):
            tree_obj = PhyloTree.from_newick(tree)
        else:
            tree_obj = tree

        rho = self._rho if self._rho is not None else 1.0
        log_lik = 0.0

        NEG_INF = float("-inf")

        def safe_log(x: float) -> float:
            return np.log(x) if x > 0 else NEG_INF

        # 1. 树拓扑似然：遍历所有分支
        # BUG FIX: Node age direction was reversed. The original code computed
        # node_age as cumulative branch length from node to root (which gives
        # smaller values for older nodes), but we need actual node age (time
        # from present to node), which should be larger for older nodes.
        # For a properly calibrated tree: actual_node_age = tree_height - node_age_to_root.
        if tree_obj.root is not None:
            # First pass: compute tree height (age of root) by finding maximum node_age_to_root
            # node_age_to_root is the cumulative branch length from node to root
            tree_height = 0.0
            for node in tree_obj.root.preorder_traverse():
                node_age_to_root = 0.0
                cursor = node
                while cursor is not None and cursor.parent is not None:
                    node_age_to_root += cursor.branch_length or 0.0
                    cursor = cursor.parent
                if node_age_to_root > tree_height:
                    tree_height = node_age_to_root

            # Second pass: compute likelihood using correct node ages
            for node in tree_obj.root.preorder_traverse():
                if node.parent is None:
                    continue  # 跳过根节点
                branch_length = node.branch_length if node.branch_length is not None else 0.0
                if branch_length > 0:
                    # 分支存活项: exp(-(λ + μ + ψ) × Δt)
                    log_lik += -(self._lambda + self._mu + self._psi) * branch_length

                    # BUG FIX: Compute node_age_to_root (cumulative from node to root),
                    # then convert to actual node age from present: tree_height - node_age_to_root
                    # This ensures parent.age > child.age (parent is older)
                    node_age_to_root = 0.0
                    cursor = node
                    while cursor is not None and cursor.parent is not None:
                        node_age_to_root += cursor.branch_length or 0.0
                        cursor = cursor.parent
                    # node_age is actual age from present (larger for older nodes)
                    node_age = tree_height - node_age_to_root

                    if node.is_leaf:
                        # 叶节点：现存采样或灭绝终止
                        if node.metadata.get("is_extant", True):
                            # 现存采样叶：贡献 ρ（λ 已在父节点分支事件计入）
                            log_lik += safe_log(rho)
                        else:
                            # 灭绝叶：死亡事件 + 此后无采样后代
                            log_lik += safe_log(self._mu) + safe_log(self._E(node_age))
                    else:
                        # 内部分支节点：物种形成事件 + 侧支无采样后代
                        log_lik += safe_log(self._lambda) + safe_log(self._E(node_age))

        # 2. 化石保存似然
        for fossil_group in fossils:
            for age in fossil_group:
                if age < 0:
                    continue
                # 化石项: ψ × E(age)（该时刻被采样 + 此后无其他采样后代）
                log_lik += safe_log(self._psi) + safe_log(self._E(age))

        return log_lik


def simulate_fbd_process(
    lambda_: float, mu: float, psi: float, duration: float, n_replicates: int = 1, random_seed: int | None = None
) -> list[FBDSimulationResult]:
    """
    模拟FBD过程的便捷函数

    参数:
        lambda_: 物种形成率
        mu: 灭绝率
        psi: 化石保存率
        duration: 模拟时长
        n_replicates: 重复次数
        random_seed: 随机种子

    返回:
        FBDSimulationResult列表
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    results = []

    for i in range(n_replicates):
        sim = GillespieSimulator(
            speciation_rate=lambda_,
            extinction_rate=mu,
            fossilization_rate=psi,
            random_seed=random_seed + i if random_seed is not None else None,
        )
        sim.initialize(n_lineages=1)
        sim.run(duration=duration)
        results.append(sim.get_result())

    return results
