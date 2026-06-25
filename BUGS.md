# PaleoAST 项目代码审查最终报告

> 范围：D:\GIthub\PaleoAST 全部 Python 源文件
> 方法：grep 定位 + 读取上下文 + 与文献公式/标准做交叉对比
> 已审模块：utils/, models/, statistics/, ecology/, morphometrics/, morpho3d/, phylogenetics/, macroevolution/, stratigraphy/, parsers/, controllers/, views/, main.py, state_machine/, visualization/, reporting/, app_infrastructure/, hpc/, plugins/, config/
> 状态：所有源文件已保持 git HEAD 内容（仅行尾 CRLF 差异），本次未做任何代码修改

---

## 0. 总览

| 等级 | 数量 | 含义 |
|------|------|------|
| 严重 | 17 | 算法/数值错误，直接影响论文结果 |
| 中等 | 24 | API 不一致、性能、边界条件、并发隐患 |
| 小   | 18 | 文档/命名/小 bug/小性能 |

总计 59 条。

---

## 1. 严重 bug

### 1.1 statistics/pcoa.py:174-180 - PCoA 比例解释分母错
- 现象：proportion = eigenvalues_positive[:n_components] / np.sum(eigenvalues_positive[:n_components])
- 后果：当 n_components 远小于全部正值个数时，cumulative 不到 100%；剩余方差被吸收到前 n 个主坐标里，scree plot 误判。
- 修复：total = np.sum(eigenvalues_positive)，先归一化再切片。

### 1.2 statistics/nmds.py:_smacof - 缺 isotonic regression
- 现象：stress 直接对 D - D_hat 计算，没有任何 disparities 步骤。
- 后果：这是 metric MDS，不是真正的 NMDS。所有依赖 rank-order preservation 的论文方法学都不成立。
- 修复：每次迭代加 d_hat = isotonic_regress(d_hat_sorted, D_sorted) 步骤得到 disparities，再算 stress。

### 1.3 statistics/cca.py:330-337 - chi-square 标准化多一个 grand_total 因子
- 现象：Y_std = (p - expected) / np.sqrt(expected)，其中 p = Y/grand_total、expected = (r*c)/grand_total^2。
- 后果：与 ter Braak 1986 差 sqrt(grand_total) 因子。
- 修复：补乘 np.sqrt(grand_total)，或直接 (Y - r*c/T) / sqrt(r*c/T)。

### 1.4 ecology/beta_diversity.py:_hypergeometric_prob:525-543 - 公式抵消退化
- 现象：分子 4 项与分母 4 项完全相同（正负抵消），仅剩 -lgamma(N+1) -> 恒为 1/N!。
- 后果：Hurlbert 稀薄化、覆盖度估计全部失效。
- 修复：标准 C(K,k)*C(N-K, n-k) / C(N, n)，lgamma 数值稳定实现。

### 1.5 ecology/null_models.py:_compute_c_score:366-384 - 完全忽略 S_ij
- 现象：c_ij = (R_i-1)(R_j-1)，没有用到 S_ij（共占位数）。
- 后果：C-score 只反映物种总多度，与共现模式完全无关。
- 修复：(R_i - S_ij)(R_j - S_ij)，其中 S_ij = sum_min(p_i, p_j) over sites。

### 1.6 ecology/beta_diversity.py:285, 291 - Baselga 嵌套度公式简化
- 现象：缺 a/(a+2*min(b,c)) 因子。
- 后果：turnover + nestedness 不等于 total_beta —— Baselga 分解核心恒等式不成立。
- 修复：补乘 a/(a+2*min(b,c))（Jaccard）或 a/(2a+min_bc)（Sorensen）。

### 1.7 statistics/pcm.py:_compute_contrasts_recursive - v_A 少加一条 branch_length
- 现象：二分叉处 v1 = var1, v2 = var2（用子节点到 tip 的累积，不加子节点 branch_length）。
- 后果：IC 分母 sqrt(v_A + v_B) 系统性偏小；t 检验、CI、Blomberg K、Phylo-ANOVA 全部受影响。
- 修复：v1 = var1 + child1.branch_length; v2 = var2 + child2.branch_length。

### 1.8 statistics/pcm.py:compute_phylogenetic_signal - Blomberg K 公式不完整
- 现象：K = mean((IC/sqrt(v))^2) 不等于 Blomberg 2003 标准。
- 修复：标准 K = (sum IC^2/(n-1)) / (sum v/(n-1))，并结合 MSE_0 计算。

### 1.9 statistics/pcm.py:phylogenetic_anova - contrast->group 分配非标准
- 现象：按 subtree dominant group 而非 Garland 1993 的子节点 group pair。
- 修复：按两直接子节点 dominant group 标 contrast，或用跨越 group pair 加权。

### 1.10 morphometrics/efa.py:analyze - EFA 系数用 x(t) 代替 delta_x/delta_t
- 现象：Kuhl & Giardina 公式应使用差分 delta_x/delta_t，当前直接用 x(t)*cos(omega*t)*dt。
- 修复：使用 delta_x*cos(omega*t_k)。

### 1.11 macroevolution/fbd.py:log_likelihood - 与 Stadler 2010 不匹配
- 现象：现存谱系用 log(lambda*rho) 简化、化石项 log psi - psi*age、未区分 extant 谱系衰减。
- 修复：完整 Stadler 公式 E(t) = 1 - 4 exp(-gamma*t)/(...)^2、化石项 psi*E(t_s)*exp(-mu*(t-t_s))。

### 1.12 macroevolution/cohort.py:242-248, 301-308, 333-334 - origination/extinction 颠倒
- 现象：注释与赋值都错，origination_rates = -ln(S)/dt 实际是 mu。
- 修复：mu = -ln(S)/dt, lambda = -ln(1-S)/dt。三处都要改：analyze, foote_analysis, per_capita_rates。

### 1.13 macroevolution/diversity.py:84 - 物种存在性条件方向反
- 现象：count = sum(... if o >= t_end and t_start >= L) 几乎不可能满足。
- 修复：if o <= t_start and L >= t_end。

### 1.14 stratigraphy/extinction.py:_marshall_ci - 置信区间方向
- 现象：ci_upper = lad - ln(q)/n_eff，-ln(q) 为正。
- 后果：方向取决于层号约定（更大=更老 vs 0=顶）。需要核对并加单元测试。

### 1.15 stratigraphy/extinction.py:_strauss_sadler_ci - ad-hoc 公式
- 现象：docstring 声称基于 beta 分布，但实现 delta_lower = k*(log(1/q)/n_taxa)^0.5。
- 修复：scipy.stats.beta.ppf(q/2, k, n-k+1) - k。

### 1.16 parsers/binary_cache.py:304, 311 - 压缩下 metadata_offset 错位
- 现象：metadata_offset 在压缩前用未压缩 matrix_size 计算；实际写入时是压缩后字节数。
- 修复：先压缩再算 header.matrix_size = len(matrix_bytes) 和 metadata_offset。

### 1.17 controllers/data_controller.py:371 - transform_sqrt 静默改号
- 现象：np.sqrt(np.abs(data)) 对负值静默取绝对值。
- 后果：负丰度被改成正值，用户无感。
- 修复：if np.any(data < 0): raise ValidationError(...)。

---

## 2. 中等 bug

| 模块 | 位置 | 问题 | 建议 |
|------|------|------|------|
| models/state_manager.py | __new__ + __init__ | 双重检查 + _initialized 守重入。基本安全但 __init__ 不在锁内 | 把初始化移入锁内 |
| plugins/registry.py | __new__ + __init__ | 同上 | 同上 |
| statistics/univariate.py | one_way_anova | 单样本组时 scipy 抛 ValueError | 显式校验 len(group_data) < 2 |
| statistics/anosim.py | summary vs compute | R 符号约定可能不一致 | 核对 docstring |
| statistics/pca.py | get_scree_data | n_eigenvalues 切片与归一化分母不一致 | 统一用 eigenvalues[:n_eigenvalues].sum() |
| morphometrics/relative_warps.py | _warp_points | coeffs.T @ combined 形状恰好但语义非典型 | 改 np.einsum 或补注释 |
| ecology/rarefaction.py | compute_sample_based_rarefaction | 累积语义反了 | 固定 n_total=n，k 从 1 到 n |
| ecology/dtw.py | distance_matrix | if i==0 分支 warped 列表追加错乱；i>0 不 push | 显式 push warped_seq1/seq2 |
| visualization/*, views/* | 多处 plt.style.use("seaborn-v0_8-paper") | matplotlib >= 3.6 才支持 | try/except 兼容 |
| morphometrics/allometry.py | divide_configuration_into_blocks size_matched | 先按 size 切，再丢弃换成按列切 | 删第二段 |
| utils/matrix_ops.bray_curtis vs statistics/distance_metrics.bray_curtis | 不一致 | 一个 X+X、另一个 abs(X)+abs(X) | 统一两处 |
| morphometrics/allometry.py | merge_configuration_into_blocks | np.random.seed(42) 硬编码 | 接收外部 seed |
| parsers/lexer.py, state_machine/tokenizer.py | regex | 缺空字符串/边界处理 | re.fullmatch 或显式校验 |
| views/* 多处对话框 | from models.state_manager import StateManager | 强耦合 | 构造注入或 get_instance() 延迟 |
| controllers/statistics_controller.py | list_available_analyses | 末尾三元表达式优先级易错 | 简化为 or 链 |
| utils/transformations.py:sqrt_transform vs data_controller.transform_sqrt | 不一致 | 一个 np.maximum(0)，另一个 np.abs | 统一 |
| utils/transformations.py:boxcox_transform | 边界 | valid_data 全空/单元素时 sp_stats.boxcox 报错 | 显式 if len(valid_data) < 2 |
| state_machine/automaton.py | transition() | 理论 race | 加 RLock |
| hpc/process_pool.py | _worker_execute | except Exception 吞掉异常 | 包装 Future + return error |
| hpc/task_scheduler.py | _worker_loop | except Exception + except queue.Empty | 区分异常类型 |
| reporting/figure_handler.py | 路径 | + 而非 os.path.join | 改 os.path.join |
| views/file_drop_handler.py | get_file_drop_handler | hasattr 单例非线程安全 | threading.Lock + 双重检查 |
| app_infrastructure/exception_handler.py | 旧路径 | 多处仍出现 phase5/ 字符串 | 清理 docstring |
| morphometrics/efa.py | _resample_contour | from numpy import interp 触发 DeprecationWarning | 用 np.interp |

---

## 3. 小问题

- controllers/data_controller.py:transform_log 不检查 0/负值（log(0)=-inf，log(neg)=nan）
- phylogenetics/fitch.py:retention_index：g = len(site_scores) 当最大步数；m = sum(s>0) 过度简化
- phylogenetics/fitch.py:_fitch_up：根节点处理时把节点自身当作父节点
- morphometrics/efa.py:reconstruct_from_coefficients 用 T = 2pi，analyze 用真实周长 T = t[-1]，两者不一致
- ecology/diversity.py 与 macroevolution/diversity.py 模块名冲突
- parsers/dat_parser.py、tps_parser.py、nexus_lexer.py：错误处理只 warn 不抛错
- views/ui_main_window.py：硬编码窗口大小未适配高 DPI
- reporting/report_builder.py:add_figure 后图被多次引用可能重复保存
- morpho3d/quaternion.py:from_rotation_matrix Shepperd 算法 4 case 需核对符号
- views/ui_main_window.py 全文件 ~70 处 except Exception（已记录在 BUG_FIX_SUMMARY.md）
- visualization/*_plot.py 多处 except OSError/except Exception 兜底
- views/diagnostic_console.py:ConsoleLogHandler.emit、StatusBarLogHandler.emit except Exception: 静默
- state_machine/automaton.py:to_dfa 的 state_queue.pop(0) 是 O(n)，应 deque.popleft()
- views/floating_toolbar.py:connect_signals except Exception: 静默
- config/i18n/_Translator.set_language Qt 与纯 Python 后端切换不统一
- app_infrastructure/theme/styles.py:Styles 单例可能与 dark/light 模式状态不同步
- parsers/lexer.py:Token.__post_init__ 缺字段校验
- config/imputation.py:impute_knn 当 candidates <= k 时直接加权全 candidates，是合理 fallback

---

## 4. 关键模块统计

| 模块 | 文件数 | 严重 | 中等 | 小 |
|------|--------|------|------|-----|
| statistics/ | 17 | 7 | 4 | 2 |
| ecology/ | 8 | 3 | 2 | 0 |
| morphometrics/ | 7 | 1 | 3 | 1 |
| macroevolution/ | 4 | 2 | 0 | 0 |
| stratigraphy/ | 9 | 2 | 0 | 0 |
| parsers/ | 6 | 1 | 0 | 2 |
| controllers/ | 2 | 1 | 1 | 1 |
| models/ | 5 | 0 | 1 | 0 |
| phylogenetics/ | 5 | 0 | 0 | 1 |
| morpho3d/ | 5 | 0 | 0 | 1 |
| views/ | 12 | 0 | 6 | 4 |
| app_infrastructure/ | 7 | 0 | 1 | 1 |
| utils/ | 6 | 0 | 3 | 1 |
| reporting/, visualization/, hpc/, plugins/, config/, state_machine/, main.py | 14+ | 0 | 3 | 4 |

---

## 5. 验证方法

每条严重 bug 均经过以下流程：
1. 用 Select-String 定位文件与行号
2. 读取该处上下文 5-15 行
3. 与对应文献/标准公式做对比
4. 标记为已逐项核验确认存在 或 边界情形需进一步测试

## 6. 修复优先级建议

1. 立刻修（影响发表/审稿）：#1.1（PCoA 比例）、#1.2（NMDS）、#1.3（CCA chi-square）、#1.4（hypergeometric）、#1.5（C-score）、#1.6（Baselga 嵌套度）、#1.7-#1.9（PCM 三件套）、#1.11（FBD 似然）、#1.12（cohort 颠倒）、#1.13（diversity 条件）、#1.16（binary cache 压缩）
2. 次优先：#1.10（EFA）、#1.14（Marshall CI）、#1.15（Strauss CI）、#1.17（sqrt 静默改号）
3. 中优先级：所有中等项
4. 低优先级：文档与命名清理

## 7. 与已有审计的关系

- BUG_FIX_SUMMARY.md（2024-12）已修：6 项偏 bare except / 依赖 / Qt 异常处理
- CODE_AUDIT_REPORT.md（2026-05）记录：8 项（psutil、5 个 bare except、内存泄漏、隐藏异常）
- 本文补：以上未覆盖的算法/数值正确性层面 bug，影响数值结果

---

## 8. 二次审计修复（2026-06-25）

> 审计范围：基于 commit `7adc61e` 之后的代码，重点检查已修 bug 是否完整同步，
> 以及是否引入新 bug。共发现 10 条遗留/新引入问题，已全部修复。

### 8.1 严重

| # | 模块 | 行 | 问题 | 修复 |
|---|------|----|------|------|
| S1 | ecology/null_models.py | 500-511 | `_compute_c_score_worker` 仍使用旧公式 `(r_i-1)(r_j-1)`，主类 `_compute_c_score` 已修复为 `(r_i-S_ij)(r_j-S_ij)`。并行模式观测值用新公式、模拟值用旧公式，SES/p 值完全错误 | 重写 worker 函数为 `(r_i-S_ij)(r_j-S_ij)` |
| S2 | ecology/null_models.py | 262-277, 440-468 | `_run_sequential` / `_worker_permute` 始终调用 `_compute_c_score`，忽略用户选择的 metric（"checkerboard"/"combo"）。观测值用正确 metric，模拟值全用 c_score | 新增 `_compute_score` 调度函数，metric 参数透传到 worker |

### 8.2 中等

| # | 模块 | 行 | 问题 | 修复 |
|---|------|----|------|------|
| M1 | macroevolution/survival.py | 473 | `np.random.seed(42)` 静默重置全局 RNG（allometry.py 已修，此处遗漏） | 改用 `np.random.default_rng(42)` |
| M2 | controllers/data_controller.py | 323-328 | `transform_log` 不校验零/负值（`log(0)=-inf`, `log(neg)=nan`），`transform_sqrt` 已修但此处遗漏 | 新增 `data_arr <= 0` 校验，抛 `ValidationError` |
| M3 | phylogenetics/fitch.py | 121-127 | `retention_index` 用 `g=len(site_scores)`（site 数）当最大步数，正确应为 `(n_taxa-1) × informative_sites`；`FitchResult` 缺 `n_taxa` 字段 | 新增 `n_taxa` 字段，`compute` 传入，`RI` 使用正确公式 |
| M4 | hpc/process_pool.py | 383-385 | `_worker_map` 无逐项异常处理，单 item 失败导致整块结果丢失 | 改为逐项 try/except，失败返回 None，调用方过滤 |

### 8.3 小

| # | 模块 | 行 | 问题 | 修复 |
|---|------|----|------|------|
| S3 | stratigraphy/extinction.py | 454 | 死代码：`len(lad_sorted)` 独占一行不赋值 | 改为 `n_taxa = len(lad_sorted)` |
| S4 | statistics/distance_metrics.py | 170 | Bray-Curtis 分母 `np.abs(X)+np.abs(Y)` 与 `matrix_ops.py` 的 `X+Y` 不一致（非负数据等价，负值数据不同） | 去掉多余 `np.abs()`，与 `matrix_ops.py` 统一 |
| S5 | macroevolution/fbd.py | 362-391 | `_compute_diversity_curve` 不含起始时刻，曲线从第一个事件开始而非模拟起始 | 加入 `start_time` 到时间轴 |
| S6 | ecology/null_models.py | 全文 | 类方法 `_compute_c_score`（已修）与模块级 `_compute_c_score_worker`（未修）代码重复且分歧 | 新增 `_compute_score_worker` / `_compute_checkerboard_worker` 与类方法一致 |

### 8.4 修复文件清单

| 文件 | 修改行数 |
|------|---------|
| ecology/null_models.py | ~60 行 |
| macroevolution/survival.py | 3 行 |
| controllers/data_controller.py | ~20 行 |
| phylogenetics/fitch.py | ~25 行 |
| hpc/process_pool.py | ~15 行 |
| stratigraphy/extinction.py | 1 行 |
| statistics/distance_metrics.py | 3 行 |
| macroevolution/fbd.py | ~15 行 |

---

## 9. 第二轮审计（2026-06-25，前端 UI + 遗留检查）

> 审计范围：views/、visualization/、utils/transformations.py、
> controllers/、config/i18n/，重点检查前端 UI 显示效果和用户交互。

### 9.1 发现并修复的问题

| # | 等级 | 模块 | 行 | 问题 | 修复 |
|---|------|------|----|------|------|
| U1 | 🟡 | utils/transformations.py | 24-46 | `log_transform` 不校验 `x+offset ≤ 0`：`x=-1` → `-inf`，`x<-1` → `nan`，静默传播到下游 | 新增 `shifted ≤ 0` 校验，抛 `ValueError` |
| U2 | 🟡 | views/ui_main_window.py | 2500-2501 | `_apply_transformation` 用 `str(e)` 显示原始 Python 异常，用户体验差 | 改为 `format_user_error(e, name)` 提供中文友好提示 |
| U3 | 🟢 | views/ui_main_window.py | 2784-2786 | PCA 状态栏 `explained_variance[0]+[1]` 当 `n_components=1`（单列数据）时 `IndexError` | 新增 `len(ev) >= 2` 安全检查 |

### 9.2 已确认无问题的区域

- **主题切换**：`setDarkTheme` 链完整，`RibbonTab`/`RibbonGroup`/`RibbonButton`/`StatusBarWidget`/`WorkspaceArea` 均正确传播。`WorkspaceArea.setDarkTheme` 已处理无 `setDarkTheme` 属性的 widget。
- **i18n**：所有 `QMessageBox.critical/warning` 的标题均使用 `_()` 包裹。
- **错误格式化**：所有分析处理器（PCA/PCoA/NMDS/Diversity/Rarefaction/ANOSIM/SIMPER/EFA/GPA 等）均使用 `format_user_error()`。
- **可视化**：`matplotlib` 样式有 `try/except OSError` 回退，兼容新旧版本。
- **导航树**：`action_map` 路由完整，所有叶子项均有对应处理器。
- **对话框**：`PCADialog`/`PCoADialog`/`NMDSOptionsDialog` 等 `n_components` 范围 `min=2`，但控制器会 clamp 到实际最大值。
- **NullModelDialog**：已正确传递 `metric` 和 `n_workers` 参数到 `NullModelAnalyzer`。
- **拖放**：`FileDropHandler` 信号连接正确，错误处理完整。
- **语言切换**：`_switch_language` 有确认对话框，回滚机制正确。

### 9.3 修复文件清单（第二轮）

| 文件 | 修改 |
|------|------|
| utils/transformations.py | `log_transform` 新增 `x+offset ≤ 0` 校验 |
| views/ui_main_window.py | `_apply_transformation` 改用 `format_user_error`；PCA 状态栏越界保护 |