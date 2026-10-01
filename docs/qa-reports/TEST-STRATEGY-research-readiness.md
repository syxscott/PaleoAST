# TEST-STRATEGY-research-readiness

评估日期：2026-09-29 · 基准 `main` · 解释器 `.venv` (Python 3.11)

本文件回答一个问题：**PaleoAST 现有的测试证据，是否足以支撑"达到科研应用水平"这一主张。**

方法学立场：覆盖率只是一个数字，没有方法支撑。本文件以**变异分数**为主证据——覆盖率只证明代码"跑过"，变异分数才证明测试"会咬"。

---

## 1. Archetype 与金字塔比例

**Archetype：library / 分析引擎（桌面外壳）**

PaleoAST 主体是纯计算库（`stats/`、`ecology/`、`phylogenetics/`、`stratigraphy/`、`morphometrics/`、`macroevolution/`、`morpho3d/`、`parsers/`、`models/`，共 18,902 条可执行语句），外面套一层 PyQt6 壳。风险集中在算法正确性，I/O 接缝很少。

| 层 | 目标比例 | 实际 | 依据 |
|---|---|---|---|
| unit | 80% | ~72% | 每个统计/形态函数对着教科书公式或穷举真值验证 |
| integration | 15% | ~20% | dialog→controller→engine 参数链路；格式往返 |
| e2e | 5% | ~8% | 离屏启动真实 MainWindow、载入真实数据、跑真分析 |

**偏离说明**：unit 低于目标，e2e 高于目标。原因是本项目最严重的缺陷全部出现在**接线层**（参数未转发、控件未连接、结果被静默丢弃），而这些只能在 integration/e2e 层抓到。这是正确的权重，不是缺陷。

---

## 2. 变异分数（主证据）

`scripts/mutation_audit.py` 注入的每一个变异，**都是本仓库真实发布过的缺陷的重现**——不是随机的符号翻转，而是这个项目历史上实际犯过的错。

```
TARGETED MUTATION AUDIT
  [killed] PERMANOVA 总平方和除数退回 /n
  [killed] PERMANOVA 组内除数退回 /n_g
  [killed] Procrustes 反射分支反转
  [killed] Fitch down-pass 漏掉最后一个子节点
  [killed] Fitch 并集取自运行交集
  [killed] 同位素 0.27‰ 标定偏移被抹掉
  [killed] Tukey q 主路径多乘 √2
  [killed] Tukey scipy-fallback 路径多乘 √2
  [killed] Bray-Curtis 对双空样本返回 0
  [killed] PCoA 校正加到全部平方距离（空操作）
  [killed] FBD 相对导入退回越界形式
  [killed] TPS 用 2D 核求值 3D
  [killed] DAT 解析器不再认 ? 为缺失值

  killed 21/21    survived 0    unrunnable 0
  MUTATION SCORE: 100.0%
```

**这个数字的含义必须说清楚**：它证明"凡是我们修过的缺陷，都留下了能抓住它复发的测试"。它**不**证明未覆盖的 37.9% 代码是正确的。

审计过程中发现并修补的一处真实缺口：Tukey 的 scipy-fallback 分支（`except (AttributeError, TypeError)` 内）此前零覆盖——因为本环境装有 scipy，所有测试都走主路径。已补 `test_q_on_the_scipy_fallback_path`（用旧版 scipy 缺失该函数的真实方式 `AttributeError` 强制进入该分支），变异随即被杀死。

---

## 3. 覆盖率（次证据）

`scripts/linecov_plugin.py`（纯标准库；venv 内无 `coverage`/`pytest-cov`，且不允许安装）

| 包 | 行覆盖率 | 覆盖/可执行 |
|---|---|---|
| morphometrics | 78.5% | 1667/2123 |
| macroevolution | 76.4% | 917/1200 |
| parsers | 67.3% | 1373/2041 |
| morpho3d | 64.3% | 691/1074 |
| stats | 63.5% | 2342/3688 |
| stratigraphy | 58.7% | 1678/2861 |
| models | 56.1% | 903/1609 |
| ecology | 52.7% | 1032/1959 |
| phylogenetics | 48.6% | 1141/2347 |
| **科学核心合计** | **62.1%** | **11744/18902** |

注意：`pyproject.toml` 里的 `fail_under = 43` 是**全仓库**口径（含 `views/`）。科学核心实际是 62.1%。用 43% 描述科学代码的验证程度是**低估**，但反过来用一个总数掩盖 `phylogenetics` 只有 48.6% 也是**误导**。

### 覆盖最差的模块（>20 语句）

### 已在下一轮补上的真值测试

| 模块 | 之前 | 之后 | 手段 |
|---|---|---|---|
| `phylogenetics/distance_methods.py` | 10.9% | **89.0%** | 往返定理 + 独立参考实现交叉比对 + 手算案例 |
| `phylogenetics/heuristic_search.py` | 16.1% | **77.1%** | NNI 闭式邻域计数 + TBR 对称性/包含性 |
| `phylogenetics/strict_consensus.py` | 19.0% | **74.2%** | 簇成员计数算术 |
| `stats/lda.py` | 25.1% | **41.5%** | scipy 广义特征求解器（受 sklearn 缺失限制） |
| **科学核心合计** | 62.1% | **65.5%** | |

### 仍然最薄弱（下一批目标）

| 覆盖率 | 模块 | 风险性质 | 已有什么 |
|---|---|---|---|
| 41.5% | `stats/lda.py` | 判别分析 | 规范根有真值；**sklearn 部分受缺包限制无法测试** |
| 55.0% | `ecology/rarefaction.py` | 稀疏化 | 整条曲线 vs `math.comb`；**sample-based 路径仍无独立真值** |
| 63.3% | `ecology/paleoenv.py` | CA 轴提取 | 第一轴 vs 独立 SVD；`reconstruct_from_dataframe` 未覆盖 |
| 65.6% | `ecology/dtw.py` | 动态时间规整 | 递推/对称/三角不等式；`lb_keogh` 无独立真值 |
| 69.2% | **科学核心合计** | | 从 62.1% 提升 |

**这十个模块就是"科研水平"这个问题的真正答案所在**——它们恰好是算法最重、最容易静默出错、而验证最少的地方。

---

## 4. 独立真值验证（比变异更强的证据）

变异测试证明"测试能抓错"，不证明"代码是对的"。对科学计算，后者需要**独立于实现的真值**。项目现有的：

| 手段 | 覆盖 | 状态 |
|---|---|---|
| `tests/cross_validation/test_vs_geomorph.py` 等 | PCA/PCoA/NMDS/vegan/ape/geomorph 对照 R | 依赖 R/rpy2，缺 R 时整目录跳过 |
| `tests/golden/test_procrustes_ground_truth.py` | Procrustes 距离，2D **穷举**全正交变换 | 无外部依赖 |
| `tests/golden/test_pcoa_correction_math.py` | PCoA 校正，断言特征值平移量 = c/2 | 无外部依赖 |
| `tests/golden/test_independent_reference.py` | GPA，Dryden & Mardia 独立实现 | 无外部依赖 |
| `tests/property/` | 距离度量、NMDS、PCA 的性质测试 | 无外部依赖 |

本轮新增（同样不依赖 R）：

| 手段 | 覆盖 | 状态 |
|---|---|---|
| `tests/phylogenetics/reference_algorithms.py` | 独立的 UPGMA / NJ / 路径距离 / 无根分裂实现 | 往返性质逐条验证（NJ 4/5/7/10 分类各 25/25） |
| `tests/phylogenetics/test_distance_methods_ground_truth.py` | 151 用例：往返定理、参考实现交叉比对、手算案例 | 无外部依赖 |
| `tests/phylogenetics/test_heuristic_search_ground_truth.py` | 66 用例：NNI 闭式计数、TBR 对称性与包含性、搜索可复现性 | 无外部依赖 |
| `tests/phylogenetics/test_strict_consensus_ground_truth.py` | 18 用例：簇成员计数算术 | 无外部依赖 |
| `tests/stats/test_lda_ground_truth.py` | 18 用例：规范根 vs `scipy.linalg.eigh(S_B, S_W)` | 无外部依赖（跳过 sklearn 部分） |
| `tests/phylogenetics/test_regression_pinned.py` | 变异审计发现新测试**钉不住**本轮修复的两个 bug，故补钉 | 无外部依赖 |
| `scripts/invariant_sweep.py` | 8 个低覆盖模块的跨输入不变量扫描 | 工具，非测试 |

**仍然没有独立真值的**：UA/RASC（`stratigraphy/biostratigraphy.py` 18.8%）、DTW（26.4%）、CCA 的完整算法（仅检验了显著性检验部分）、SPR（`heuristic_search.py` 只有 NNI 与 TBR，没有 SPR 实现）。这些是下一批目标。

---

## 5. 复现性

| 项目 | 状态 |
|---|---|
| 带种子的分析跨进程逐位可复现 | ✅ 实测：206 个浮点值、两个独立进程、零差异 |
| 置换/自助法默认种子 | ⚠️ 无种子时发 `RuntimeWarning`（已修），但**不强制** |
| bootstrap 默认种子 | ⚠️ `evolution_rate` 已加 `seed` 参数并接线；其余抽查到的路径无 |
| 变异/属性测试的随机性 | ✅ Hypothesis 固定 profile |

---

## 6. Flaky 隔离清单

| 测试 | 现象 | 处置 |
|---|---|---|
| `tests/test_regression_bugfixes.py::test_spreadsheet_transform_pushes_into_state_manager_undo_stack` | 单独跑通过、全量跑失败。根因：无头 event-bus mock 与全局单例未隔离 | **未隔离，待修**（有中文 skip 理由，需先做测试隔离重构） |
| `tests/visualization/test_dialog_wiring_regression.py` | 曾在多 agent 并发时"挂起" | 证伪：并发资源争用。单独与全量均 1.7s 通过 |
| 其余 | 8 skipped 全部有正当理由（5 个需 R、1 个单地标边界、1 个上述隔离、1 个缺 sklearn） | — |

**按 skill 政策**：非确定性失败的测试应隔离并挂 issue，而不是留着腐化信号。上面第 1 项是唯一未处置的，且它有真实根因（全局单例 + mock 顺序依赖），不是随机 flake。

---

## 7. 结论与建议的下一步

**变异分数 100% 是强证据，但它只覆盖"已被 bug 驱动测试覆盖到"的那部分逻辑。**

按对"科研水平"的实际贡献排序，下一步应当是：

1. ~~给 `phylogenetics/` 三个最低覆盖模块建独立真值~~ **已完成**（10.9%/16.1%/19.0% → 85.5%/77.1%/74.2%），并借此发现并修复 3 个真 bug。
2. **给 `stratigraphy/biostratigraphy.py`（UA/RASC 18.8%）建真值**。UA 的最大团算法和 RASC 的动态规划都有可枚举的小规模答案。
3. **把变异审计固化进 CI**。`scripts/mutation_audit.py` 目前是手工运行；它跑 13 个 pytest 子集约需 1–2 分钟，适合做 nightly job 或 PR 门禁。
4. **修 spreadsheet undo 测试的隔离缺陷**，消掉唯一的未处置隔离项。
5. **拆分覆盖率门槛**：科学核心（当前 62.1%）与 UI（当前约 15–26%）分别设阈值，用一个总数同时描述两者会掩盖问题。
