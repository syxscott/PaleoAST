"""
================================================================================
PaleoAST Phylogenetics - Phylogenetic Signal (Blomberg's K & Pagel's λ)
================================================================================

本模块实现系统发育信号 (phylogenetic signal) 的量化方法:
- Blomberg's K (Blomberg, Garland & Ives 2003)
- Pagel's λ (Pagel 1999)

这些方法用于量化性状与系统发育的关联强度，是比较系统发育研究的核心工具。

方法原理
================================================================================

1. Blomberg's K (Blomberg et al. 2003, Evolution):
   ----------------------------------------------------
   K 衡量观察到的性状方差与在 Brown 运动 (BM) 期望方差之比:

       K = (Σ_observed / Σ_expected) / n

   其中:
       Σ_observed = Σ_i (x_i - x̄)²  (观察到的性状方差)
       Σ_expected = 在 BM 假设下基于树拓扑和枝长的期望方差

   K 的解释:
       - K = 1: 性状符合 BM 进化
       - K < 1: 性状独立性高于 BM 期望 (弱系统发育信号)
       - K > 1: 性状保守性高于 BM 期望 (强系统发育信号)

   统计检验: 通过 permutation test (置换检验) 获取 p 值

2. Pagel's λ (Pagel 1999, Nature):
   ----------------------------------------------------
   λ 是一个通过最大似然估计的进化参数:

       λ 变换树的枝长: D(λ) = (1-λ)*D_original + λ*D_internal

   其中 D_original 是原始距离矩阵, D_internal 是内部节点距离矩阵。

   λ 的解释:
       - λ = 1: 完全符合 BM 进化
       - λ = 0: 性状完全独立于系统发育 (star tree)
       - λ 接近 0 但不为 0: 弱系统发育依赖

   优化方法: 使用 scipy.optimize 最大化似然函数

参考文献:
----------
- Blomberg, S. P., Garland, T., & Ives, A. R. (2003). Testing for phylogenetic
  signal in comparative data: behavioral traits are more labile. Evolution,
  57(4), 717-745.
- Pagel, M. (1999). Inferring the historical patterns of biological evolution.
  Nature, 401(6756), 877-884.
- Pagel, M., & Meade, A. (2006). Bayesian analysis of correlated evolution
  of discrete characters by reversible-jump Markov chain Monte Carlo.
  American Naturalist, 167(6), 808-825.

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


@dataclass
class PhylogeneticSignalResult:
    """
    系统发育信号计算结果容器

    属性:
        K: Blomberg's K 值
        lambda_: Pagel's λ 值
        K_pvalue: Blomberg's K 的置换检验 p 值
        lambda_pvalue: Pagel's λ 的似然比检验 p 值 (λ=0 vs λ fitted)
        log_likelihood: 最大对数似然值
        AIC: 赤池信息准则
        n_taxa: 分类单元数量
        trait_name: 性状名称 (如果有)
    """
    K: float | None = None
    lambda_: float | None = None
    K_pvalue: float | None = None
    lambda_pvalue: float | None = None
    log_likelihood: float | None = None
    AIC: float | None = None
    n_taxa: int = 0
    trait_name: str | None = None


def _compute_variance_covariance_matrix(tree, lambda_: float = 1.0) -> np.ndarray:
    """
    [DEPRECATED - 使用 _compute_vcv_matrix 替代]
    计算树对应的方差-协方差矩阵 (使用错误的距离形式)

    在 Brown 运动进化模型下，节点 i 和 j 之间的协方差等于
    它们到最近公共祖先 (LCA) 的枝长之和。

    参数:
        tree: PhyloTree 或 PhyloNode 对象
        lambda_: Pagel's λ 变换参数 (0 到 1 之间)

    返回:
        V: n × n 方差-协方差矩阵
    """
    if hasattr(tree, 'root'):
        root = tree.root
    else:
        root = tree

    leaves = root.get_leaves()
    n = len(leaves)
    V = np.zeros((n, n))

    # 创建叶子名称到索引的映射
    leaf_names = [leaf.name for leaf in leaves]
    name_to_idx = {name: i for i, name in enumerate(leaf_names)}

    # 计算每对叶子之间的距离 (协方差)
    for i, leaf1 in enumerate(leaves):
        for j, leaf2 in enumerate(leaves):
            if i <= j:
                # 获取路径
                path1 = set(leaf1.get_path_to_root())
                path2 = leaf2.get_path_to_root()

                # 找 LCA
                lca = None
                for node in path2:
                    if node in path1:
                        lca = node
                        break

                if lca is None:
                    lca = root

                # 计算到 LCA 的枝长
                dist1 = leaf1._distance_to_ancestor(lca)
                dist2 = leaf2._distance_to_ancestor(lca)

                # 原始协方差 = dist_to_lca(leaf1) + dist_to_lca(leaf2)
                cov_original = dist1 + dist2

                # 计算内部距离 (从每个叶子到 root 的距离中减去到 LCA 的部分)
                # 内部距离矩阵 D_internal: 共享祖先越多，协方差越大
                root_dist1 = leaf1._distance_to_ancestor(root)
                root_dist2 = leaf2._distance_to_ancestor(root)

                # 内部距离 = 共享的祖先枝长
                internal_dist = root_dist1 + root_dist2 - cov_original

                # λ 变换: D(λ) = (1-λ)*D_original + λ*D_internal
                V[i, j] = (1 - lambda_) * cov_original + lambda_ * internal_dist
                V[j, i] = V[i, j]

    return V


def _compute_vcv_matrix(tree) -> np.ndarray:
    """
    计算 Brown 运动模型下的方差-协方差 (VCV) 矩阵

    Under BM, VCV[i,j] = shared path length from root to LCA(i,j)
    - V[i,i] = total variance = dist(root, tip i)
    - V[i,j] = covariance = dist(root, LCA(i,j))

    参数:
        tree: PhyloTree 或 PhyloNode 对象

    返回:
        V: n × n VCV 矩阵
    """
    if hasattr(tree, 'root'):
        root = tree.root
    else:
        root = tree

    leaves = root.get_leaves()
    n = len(leaves)
    V = np.zeros((n, n))

    for i, leaf1 in enumerate(leaves):
        for j, leaf2 in enumerate(leaves):
            if i <= j:
                if i == j:
                    # V[i,i] = total variance from root to this tip
                    dist = leaf1._distance_to_ancestor(root)
                    V[i, j] = dist
                else:
                    # V[i,j] = dist from root to LCA(leaf1, leaf2)
                    lca = leaf1.compute_lca(leaf1, leaf2)
                    if lca is None:
                        lca = root
                    # To compute dist(root, lca) where lca may be descendant of root:
                    # dist(root, lca) = dist(leaf, root) - dist(leaf, lca)
                    dist_leaf_to_root = leaf1._distance_to_ancestor(root)
                    dist_leaf_to_lca = leaf1._distance_to_ancestor(lca)
                    dist_root_to_lca = dist_leaf_to_root - dist_leaf_to_lca
                    V[i, j] = dist_root_to_lca
                    V[j, i] = dist_root_to_lca

    return V


def _blomberg_k_from_vcv(y: np.ndarray, V: np.ndarray) -> float:
    """
    Canonical Blomberg et al. (2003) K statistic from a trait vector and
    an ape-convention VCV matrix (diag = root-to-tip distance,
    off-diag = shared path length).

        K = s²_ord / (σ̂²_GLS · tr(V) / n)

    where s²_ord = Σ(yᵢ−ȳ)²/(n−1) is the ordinary mean square and
    σ̂²_GLS = (y−1â)ᵀV⁻¹(y−1â)/(n−1) is the Brownian-rate estimate with
    the GLS intercept â = (1ᵀV⁻¹1)⁻¹ 1ᵀV⁻¹y. Under BM, K ≈ 1; the
    statistic is invariant to linear rescaling of the trait.

    Returns 0.0 when the computation is degenerate (singular VCV,
    zero variance).
    """
    n = len(y)
    if n < 3:
        return 0.0
    V = np.asarray(V, dtype=float) + np.eye(n) * 1e-10
    try:
        V_inv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        return 0.0

    ones = np.ones(n)
    one_vi_one = float(ones @ V_inv @ ones)
    if one_vi_one <= 0:
        return 0.0

    a_hat = float(ones @ V_inv @ y) / one_vi_one
    resid = y - a_hat
    sigma2_gls = float(resid @ V_inv @ resid) / (n - 1)

    y_mean = float(np.mean(y))
    s2_ord = float(np.sum((y - y_mean) ** 2)) / (n - 1)

    denom = sigma2_gls * float(np.trace(V)) / n
    if denom <= 0:
        return 0.0
    return float(s2_ord / denom)


def blomberg_k(
    tree,
    traits: dict[str, float],
    n_permutations: int = 999,
    trait_name: str | None = None,
) -> PhylogeneticSignalResult:
    """
    计算 Blomberg's K 统计量

    参数:
        tree: PhyloTree 或 PhyloNode 对象
        traits: {tip_name: trait_value} 字典
        n_permutations: 置换检验的置换次数
        trait_name: 性状名称 (用于结果记录)

    返回:
        PhylogeneticSignalResult 对象，包含 K 值和 p 值

    算法 (Blomberg, Garland & Ives 2003, 规范定义):
        K = [Σ(yᵢ−ȳ)²/(n−1)] / [σ̂²_GLS · tr(V)/n]
        其中 σ̂²_GLS 为广义最小二乘 Brownian 速率估计。BM 下 K ≈ 1,
        K > 1 表示性状更保守 (系统发育信号强), K < 1 表示信号弱。
        K 对性状量纲不变 (此前实现随单位缩放, 已修正)。
        置换 p 值采用 add-one 修正: p = (1 + #{K_perm >= K}) / (n_perm + 1)。

    示例:
        >>> from phylogenetics import PhyloTree
        >>> tree = PhyloTree.from_newick("(A:1,B:1,C:1)D:1;")
        >>> traits = {"A": 2.0, "B": 4.0, "C": 3.0}
        >>> result = blomberg_k(tree, traits)
        >>> print(f"K = {result.K:.4f}, p-value = {result.K_pvalue:.4f}")
    """
    if hasattr(tree, 'root'):
        root = tree.root
    else:
        root = tree

    leaves = root.get_leaves()
    n = len(leaves)

    if n < 3:
        raise ValueError(f"Need at least 3 taxa for Blomberg's K, got {n}")

    # 构建有序的性状向量
    trait_values = np.array([traits.get(leaf.name, 0.0) for leaf in leaves])

    # 计算 VCV 矩阵
    V = _compute_vcv_matrix(tree)

    # 规范 K 统计量 (量纲不变)
    K = _blomberg_k_from_vcv(trait_values, V)

    # 置换检验: 打乱性状在端元间的分配, 重算 K
    permuted_Ks = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm_traits = trait_values.copy()
        np.random.shuffle(perm_traits)
        permuted_Ks[i] = _blomberg_k_from_vcv(perm_traits, V)

    # add-one 修正的 p 值
    p_value = float((np.sum(permuted_Ks >= K) + 1.0) / (n_permutations + 1.0))

    result = PhylogeneticSignalResult(
        K=K,
        K_pvalue=p_value,
        n_taxa=n,
        trait_name=trait_name,
    )

    logger.info(
        f"Blomberg's K = {K:.4f} (p = {p_value:.4f}) based on {n_permutations} permutations"
    )

    return result


def _compute_log_likelihood(tree, traits: dict[str, float], lambda_: float) -> float:
    """
    计算给定 λ 值下的对数似然

    参数:
        tree: PhyloTree 或 PhyloNode 对象
        traits: 性状字典
        lambda_: λ 参数值

    返回:
        log_likelihood: 对数似然值

    注意:
        使用标准 Pagel λ 变换: V_ij(λ) = λ × V_ij (i ≠ j), V_ii(λ) = V_ii
        参考: Pagel (1999) Nature
    """
    if hasattr(tree, 'root'):
        root = tree.root
    else:
        root = tree

    leaves = root.get_leaves()
    n = len(leaves)

    # 使用标准 BM VCV 矩阵 (Felsenstein 1985)
    V = _compute_vcv_matrix(tree)

    # 应用 Pagel λ 变换: V_ij(λ) = λ × V_ij for i ≠ j
    # 对角线保持不变
    if lambda_ != 1.0:
        diag = np.diag(V).copy()  # 保存原始对角线
        V = V * lambda_
        np.fill_diagonal(V, diag)

    # 添加小的正则化项确保矩阵正定
    V = V + np.eye(n) * 1e-10

    # 检查矩阵是否正定
    try:
        np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        # 如果不正定，添加更大的正则化
        V = V + np.eye(n) * 1e-6

    # 构建性向向量, 用 GLS/ML 均值中心化 (算术均值会使 λ̂ 系统性偏移)
    trait_values = np.array([traits.get(leaf.name, 0.0) for leaf in leaves])

    try:
        sign, logdet = np.linalg.slogdet(V)
        if sign <= 0:
            return -np.inf

        V_inv = np.linalg.inv(V)
        ones = np.ones(n)
        one_vi_one = float(ones @ V_inv @ ones)
        if one_vi_one > 0:
            a_hat = float(ones @ V_inv @ trait_values) / one_vi_one
        else:
            a_hat = float(np.mean(trait_values))
        y = trait_values - a_hat

        quad_form = float(y @ V_inv @ y)

        # BM 多元正态对数似然, 剖出速率参数 σ² (关键: 此前实现缺失
        # -(n/2)·log σ̂² 项, log|V| 随 λ 增大的效应主导似然, 使 λ̂ 恒
        # 塌缩到 0)。σ̂² = yᵀV⁻¹y / n, 剖面似然为
        #   logL = -n/2 · [log(2π σ̂²) + 1 + log|V| / n]
        if quad_form <= 0:
            return -np.inf
        sigma2 = quad_form / n
        log_lik = -0.5 * (n * np.log(2.0 * np.pi * sigma2) + logdet + n)
    except np.linalg.LinAlgError:
        log_lik = -np.inf

    return log_lik


def pagel_lambda(
    tree,
    traits: dict[str, float],
    trait_name: str | None = None,
    return_AIC: bool = True,
) -> PhylogeneticSignalResult:
    """
    计算 Pagel's λ 统计量

    参数:
        tree: PhyloTree 或 PhyloNode 对象
        traits: {tip_name: trait_value} 字典
        trait_name: 性状名称 (用于结果记录)
        return_AIC: 是否计算 AIC

    返回:
        PhylogeneticSignalResult 对象，包含 λ 值和似然比检验 p 值

    算法 (Pagel 1999):
        1. 计算 λ=0 和 λ=1 时的对数似然
        2. 优化 λ 在 (0, 1) 区间内的似然，找到最大似然估计 λ̂
        3. 似然比检验: λ=0 vs λ=λ̂
        4. λ̂ 的 p 值: 根据 0.5*χ²(1) 分布计算

    示例:
        >>> from phylogenetics import PhyloTree
        >>> tree = PhyloTree.from_newick("(A:1,B:1,C:1)D:1;")
        >>> traits = {"A": 2.0, "B": 4.0, "C": 3.0}
        >>> result = pagel_lambda(tree, traits)
        >>> print(f"λ = {result.lambda_:.4f}, p-value = {result.lambda_pvalue:.4f}")
    """
    if hasattr(tree, 'root'):
        root = tree.root
    else:
        root = tree

    leaves = root.get_leaves()
    n = len(leaves)

    if n < 3:
        raise ValueError(f"Need at least 3 taxa for Pagel's λ, got {n}")

    # 计算 λ=0 和 λ=1 时的似然
    log_lik_0 = _compute_log_likelihood(tree, traits, 0.0)
    log_lik_1 = _compute_log_likelihood(tree, traits, 1.0)
    log_lik_fitted = log_lik_0  # 初始化

    # 优化 λ 在 (0, 1) 区间
    def neg_log_lik(lambda_: float) -> float:
        """负对数似然 (用于最小化)"""
        return -_compute_log_likelihood(tree, traits, lambda_)

    # 使用bounded优化
    result = minimize_scalar(
        neg_log_lik,
        bounds=(0.0, 1.0),
        method='bounded',
        options={'xatol': 1e-8}
    )

    lambda_fitted = result.x
    log_lik_fitted = -result.fun

    # 似然比检验: H0: λ=0。λ=0 位于参数空间边界, 零分布不是 χ²(1)
    # 而是 0.5·χ²₀ + 0.5·χ²₁ 混合 (Self & Liang 1987), 即
    # p = 0.5·P(χ²₁ > LR)。
    if log_lik_fitted > log_lik_0:
        LR = 2 * (log_lik_fitted - log_lik_0)
        # p 值 (边界校正)
        p_value = float(0.5 * stats.chi2.sf(max(LR, 0.0), df=1))
    else:
        # 优化似然不超过 λ=0: λ̂ 取 0, LR=0, p=1
        LR = 0.0
        p_value = 1.0
        lambda_fitted = 0.0

    # 计算 AIC (可选)
    AIC = None
    if return_AIC:
        AIC = 2 * 1 - 2 * log_lik_fitted  # k=1 参数

    signal_result = PhylogeneticSignalResult(
        lambda_=lambda_fitted,
        lambda_pvalue=p_value,
        log_likelihood=log_lik_fitted,
        AIC=AIC,
        n_taxa=n,
        trait_name=trait_name,
    )

    logger.info(
        f"Pagel's λ = {lambda_fitted:.4f} (p = {p_value:.4f}), "
        f"logL = {log_lik_fitted:.4f}"
    )

    return signal_result


def phylogenetic_signal(
    tree,
    traits: dict[str, float],
    n_permutations: int = 999,
    trait_name: str | None = None,
) -> PhylogeneticSignalResult:
    """
    综合计算系统发育信号 (Blomberg's K 和 Pagel's λ)

    参数:
        tree: PhyloTree 或 PhyloNode 对象
        traits: {tip_name: trait_value} 字典
        n_permutations: Blomberg's K 置换检验次数
        trait_name: 性状名称

    返回:
        PhylogeneticSignalResult 对象，包含 K, λ 及其 p 值

    示例:
        >>> from phylogenetics import PhyloTree
        >>> tree = PhyloTree.from_newick("(A:1,B:1,C:1)D:1;")
        >>> traits = {"A": 2.0, "B": 4.0, "C": 3.0}
        >>> result = phylogenetic_signal(tree, traits)
        >>> print(f"K = {result.K:.4f}, λ = {result.lambda_:.4f}")
    """
    k_result = blomberg_k(tree, traits, n_permutations, trait_name)
    lambda_result = pagel_lambda(tree, traits, trait_name)

    # 合并结果
    combined = PhylogeneticSignalResult(
        K=k_result.K,
        lambda_=lambda_result.lambda_,
        K_pvalue=k_result.K_pvalue,
        lambda_pvalue=lambda_result.lambda_pvalue,
        log_likelihood=lambda_result.log_likelihood,
        AIC=lambda_result.AIC,
        n_taxa=k_result.n_taxa,
        trait_name=trait_name,
    )

    return combined


def simulate_brownian_motion(
    tree,
    root_value: float = 0.0,
    sigma: float = 1.0,
) -> dict[str, float]:
    """
    在树上模拟 Brown 运动进化的性状值

    参数:
        tree: PhyloTree 或 PhyloNode 对象
        root_value: 根节点的性状值
        sigma: Brown 运动的扩散率 (标准差)

    返回:
        {tip_name: simulated_trait_value} 字典

    算法:
        从根开始，每个子节点的性状值 = 父节点值 + N(0, σ² * branch_length)
    """
    if hasattr(tree, 'root'):
        root = tree.root
    else:
        root = tree

    traits = {}

    def simulate(node: PhyloNode, parent_value: float) -> None:
        """递归模拟"""
        branch_len = node.branch_length if node.branch_length is not None else 0.0
        node_value = parent_value + np.random.normal(0, sigma * np.sqrt(branch_len))

        if node.is_leaf:
            traits[node.name] = node_value
        else:
            for child in node.children:
                simulate(child, node_value)

    # 根节点状态恰为 root_value, 不叠加根自身枝长的噪声
    # (标准 BM 模拟约定; 旧实现从根开始加噪)
    for child in root.children:
        simulate(child, root_value)
    if not root.children and root.name:
        traits[root.name] = root_value
    return traits


def lambda_interpretation(lambda_value: float) -> str:
    """
    解释 Pagel's λ 值的含义

    参数:
        lambda_value: λ 估计值

    返回:
        解释字符串
    """
    if lambda_value < 0:
        return "Invalid (λ < 0)，可能由于数值问题导致"
    elif lambda_value < 0.1:
        return "几乎无系统发育信号 (λ ≈ 0)，性状独立于系统发育"
    elif lambda_value < 0.5:
        return "弱系统发育信号 (0 < λ < 0.5)"
    elif lambda_value < 0.9:
        return "中等系统发育信号 (0.5 < λ < 1)"
    elif lambda_value <= 1.0:
        return "强系统发育信号 (λ ≈ 1)，符合 Brown 运动进化"
    else:
        return f"λ > 1 ({lambda_value:.4f})，可能存在强烈的系统发育依赖或模型误设"
