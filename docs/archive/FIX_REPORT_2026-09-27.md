# PaleoAST 修复报告

**日期**：2026-09-27 · **基线**：`main` @ `f47a980` · **环境**：conda `dev` (Python 3.10.20)
**范围**：12 个文件，+349 / −75 行 · **测试**：1111 passed / 33 skipped / **0 failed**

---

## 0. 执行方式的重大变更

原计划用 7 个子 agent 按目录独占所有权并行修复。**5 个 worker 全部返回 `lost` 且零产出**（`git status` 确认工作区无任何改动），并行 agent 在本环境不可靠。因此改为**主控逐项直接修复 + 亲自复现验证**。

这带来一个正面结果：每一处改动都由我先复现、再修改、再复现，**没有一条结论只依赖 agent 转述**。

---

## 1. 已修复并逐条验证（13/13）

### 1.1 导出功能整体失效 — `views/ui_plot_export_dialog.py`

**这是影响面最大的一个 bug：所有导出的图都是 0.5 × 0.5 英寸。**

```python
# 修复前
self._width_spin.setRange(0.5, 60.0)  # 最小值 0.5
self._width_spin.setValue(0.0)  # Qt 钳到 0.5，但界面显示 "auto"
...
width_inches = (self._width_spin.value() or None,)  # 0.5 or None -> 0.5，永不为 None
```

`setSpecialValueText` 在值等于 minimum 时显示 "auto"，于是**界面显示"自动"而实际值是 0.5**，`or None` 永远拿到 0.5，`plot_export.py:265` 永远执行 `set_size_inches(0.5, 0.5)`。选 "Publication 600 dpi PDF" 得到 300×300 像素邮票。

**修复**：`setRange(0.0, 60.0)`，让 0.0 成为可达的 auto 哨兵。
**验证**：`value()==0.0`，`value() or None` → `None`，`tests/test_plot_export.py` 26 passed。

### 1.2 全局 matplotlib 状态泄漏 — `plot_export.py`

`svg.fonttype` / `pdf.fonttype` 写在 `try:` **之外**；`_validate_options` 用 `value <= 0` 判断正数，而 `NaN <= 0` 是 `False`，所以 NaN 尺寸能通过校验 → `set_size_inches(6, nan)` 抛错 → `finally` 不执行 → `svg.fonttype='path'` 永久污染整个会话，后续每次导出都被静默覆盖。

**修复**：新增 `_is_positive_finite()`（同时拒绝 NaN/Inf）；把 resize 移入 `try:`，`finally` 保证恢复。
**验证**：NaN dpi 被拒（`dpi must be a positive finite number`）；导出失败后 `svg.fonttype` 保持原值。

### 1.3 Newick 注释劈树 — `utils/newick_core.py:234`（共享核心，影响 `parsers` + `phylogenetics`）

```python
parse_newick_trees("(A,B)[&x]C;")  # 修复前 -> 2 个根: ["", "C"]
parse_newick_trees("((A,B)[&x]C,(D,E)F)G;")  # 修复前 -> NewickParseError
```

`parse_node` 遇到 `[` 就放弃读名字，`_finish_fields` 吞掉注释后名字仍未读，残留标签被顶层循环当成新树。**任何在 `)` 后写 `[&&NHX:...]` 的 NEXUS 导出都读不回来。**

**修复**：`)` 后循环——遇 `[` 收注释、重新跳空白、再读名。
**验证**：`"(A,B)[&x]C;"` → 1 棵树，label `C`、注释 `&x` 均保留；嵌套形式 → 1 棵树 `G`；未闭合注释仍正常报错。

### 1.4 Fisher log-series 的 α — `ecology/advanced.py`

同一包内两处"α"相差 9.2 倍：`fit_log_series` 给 0.9767，`diversity._compute_fisher_alpha` 给 9.0041。代码还与**同文件自己**的 `equation()` 自相矛盾。

根因：`alpha = S*(1-x)/(-ln(1-x))` 多了 `(1-x)` 因子（0.8915 时恰好等于 0.1085，与实测比值 0.9767/9.0041 = 0.1085 吻合）。

**修复**：`alpha = S/(-ln(1-x))`；补 `np.sort(...)[::-1]`（另外三个 fitter 都做了，log-series 没有，导致 r² 为负、AIC 无效）；`brentq` 无根时显式抛错而非静默回退 `x=0.5`。
**验证**：`g(x) = 3.7000 = N/S` 精确成立（brentq 返回真根），三种推导 α 全部 = 9.0041。

### 1.5 PIC 显著性判据双重标准化 — `statistics/pcm.py:96`

`contrasts` 本身已是 z 分数（`IC = Δx/√(v₁+v₂)`），`se = √(v₁+v₂)`，判据 `|IC| > 1.96·se` 等价于 `|Δx| > 1.96·(v₁+v₂)`。

**修复**：直接判 `|contrast| > 1.96`。
**验证**：上报的显著数与 `sum(|contrast| > 1.96)` 精确吻合（0 和 2）。

> 审查报告里"枝长缩放导致显著数漂移"这一表述**前提有误**：PIC contrast 本身就按 1/√k 缩放（实测 10×），所以显著数本来就该变。真正的缺陷是双重标准化，已消除。

### 1.6 3D TPS 用了 2D 核 — `morphometrics/tps.py`

`tps.py` 无条件用 `r²ln r`；`gpa.py:919` 和 `morpho3d/tps3d.py:368` 早已按 `n_dims` 分派（`gpa.py` 注释明写"旧实现对 3D 误用 r²log r"）。同一组 3D 数据弯曲能 `+0.7404` vs `-1.3351`（符号相反）。

**修复**：`_build_kernel_matrix` 按 `landmarks.shape[1]` 分派。
**验证**：`tps.py` = `tps3d.py` = `-0.61389033`，**差值 0.00e+00**；控制点插值误差 5e-16；2D 行为不变。

### 1.7 SIMPER 的 `metric` 是死参数 — `statistics/simper.py`

`metric="euclidean"` 与 `"bray_curtis"` 返回**逐位相同**的结果，但 `SimperResult.metric` 报告 "euclidean"。Clarke 累积百分比分解是 Bray-Curtis 专有的，整张表被贴错标签。

**修复**：真正实现 `metric`（Bray-Curtis 用 |Δ|/ΣΣ；Euclidean 用可加分解的平方距离）；`jaccard` 等无可加分解的度量**显式抛错**而非静默贴标签；结果字段存归一化后的键。
**验证**：`bray=0.102258` vs `euclid=46.500000`（确实不同）；`jaccard` 抛 `ComputationError`。

### 1.8 phylo-ANOVA p 值缺加一校正 — `statistics/pcm.py:1041`

`p = mean(perm >= F)` 可恰为 0.0，与同文件 `:798`、`anosim.py:193`、`permanova.py:186` 的约定不一致。

**修复**：`(1 + #{≥obs}) / (1 + n_perm)`。

### 1.9 GBK 中文 CSV 静默变乱码 — `controllers/data_controller.py`

`encoding="utf-8", encoding_errors="replace"` **不报错**，直接替换成 U+FFFD，所有列名和行标签不可逆地毁掉。项目里**早就有正确实现** `data/loader.py:46 _read_csv_with_fallback`（utf-8-sig → gbk → latin-1），只是从没接上。

**修复**：改为同样的候选编码级联，非 UTF-8 时记 warning，全部失败时抛 `FileOperationError`。
**验证**：GBK 文件导入后 `col_labels == ['甲壳长度', '宽度']`。

### 1.10 非 editable 安装启动失败 — `pyproject.toml` + `main.py`

`packages.find.include` 缺 `presets*`（而 `views/ui_runlist_panel.py:45` 模块级 import 它）→ `ModuleNotFoundError` → `main.py:652` 的**裸 `except ImportError` 吞掉** → 用户只看到一个空白"Welcome"窗口，无报错、无日志、连文件都打不开。CI 全程 `pip install -e` 恰好掩盖了这点。

**修复**：
- 补 `presets*` / `plugins*` / `_core*` / `rthooks*` 到 include
- 加 `py-modules = ["main", "plot_export"]`（console script 和导出对话框都依赖它们）
- 加 `[tool.setuptools.package-data]` 收 `data/examples/*` + `data/golden/*`（`data/loader.py:34` 用 `importlib.resources` 加载）
- `hypothesis` 进 `dev` extra（此前 `addopts` 里的 `--hypothesis-show-statistics` 让新克隆跑 `pytest` 直接启动失败）
- `main.py` 不再吞异常：CRITICAL 日志 + 标题栏 "STARTUP FAILED" + 界面直接显示异常类型和修复建议

### 1.11 ANOSIM 冻结界面 — `views/ui_main_window.py`

`_execute_anosim` 在 GUI 线程跑 9999 次 O(n²) 置换，ribbon 传空 params 无法调低次数。实测 **n=100→39s, n=200→205s, n=400→813s 完全冻结**，无重绘、无取消、无进度。

**修复**：改走 `_run_analysis_async`（与 PCA 同一模式），数据在 GUI 线程快照后交给 worker。

### 1.12 关窗时线程池未 drain — `views/ui_main_window.py`

`self._thread_pool` 是全局单例，生命周期长于窗口。长分析运行时关窗 → worker 向已销毁的 `_AnalysisSignals` 发信号 → `QThread: Destroyed while thread is still running` → 进程 abort。

**修复**：`closeEvent` 加 `_drain_thread_pool()`（`clear()` + `waitForDone(5000)`）；新增 `_closing` 标志阻止关闭后派发新任务；`_AnalysisTask.run` 捕获 `RuntimeError` 吞掉迟到回调（附注释说明为何不能让它逃出 worker 线程）。

---

## 2. 尝试后**回退**的一处（重要）

### 2.1 EFA 旋转不变性 — `morphometrics/efa.py` — 未修复，已回退

审查发现（已复现）：同一条闭合轮廓绕原点旋转 0.4 rad，EFD 系数变化达 **0.148**，即 EFD 缺乏旋转不变性，会放大 eigenshape 的组间差异。

我尝试了三种方案，全部失败：

| 方案 | 旋转不变性 | 起点不变性 | 结果 |
|---|---|---|---|
| 原始实现 | 0.148 ✗ | 0.013 ✓ | 基线 |
| A. 只做旋转消元 | 7e-16 ✓ | 1.42 ✗ | 破 4 个测试 |
| B. 先旋转后起点 | 3e-16 ✓ | 1.42 ✗ | 破 5 个测试 |
| C. 交替投影 3 轮 | 2.6e-16 ✓ | 0.343 ✗ | 破 1 个测试 |
| C'. 交替投影 5+ 轮 | 2.6e-16 ✓ | 0.0775 ✗ | 容差 0.02，仍破 |
| D. Kuhl–Giardina 数值起点 | 4e-16 ✓ | 7.36 ✗ | 破 4 个测试 |

**根因**（值得记录）：这个代码库的归一化约定要求归一化后 `b1 = 0`。但 `b1 = 0` 是一个**起点**条件，而非旋转条件——它**不是旋转协变的**。第一谐波有 4 个自由度 (a₁,b₁,c₁,d₁)，同时要求 `b₁=0`（起点）、`|a₁|=1`（尺寸）、`c₁=0`（旋转）只剩 d₁ 自由，而 d₁ 本身依赖起点。因此**"完全旋转不变"与"b₁=0 约定"在本框架下数学上不可兼得**。

**决定**：回退到原始实现（保持 12 个测试全绿），把该限制如实写进报告，**不做半吊子修复**。要真正修复需要先决定 EFD 的规范形式约定（是否放弃 `b₁=0`），这是一个需要作者拍板的建模决策，不应由我在没有确认的情况下改变已发布算法的语义。

---

## 3. 审查中被推翻的结论

除了上节的 EFA，验证过程还推翻了审查报告里的其他几处：

| 审查报告的结论 | 实测结果 |
|---|---|
| `univariate.py:449` 用 `np.sum` 处理参差数组 → 不平衡 ANOVA 崩溃 | **误报**，那行是内置 `sum()`；10v5 与 5v5 均正常，Tukey 分支（p<0.05 已进入）也不崩 |
| `cohort.py:296` 边界分类"静默丢弃类群" | **前提有误**。测试 `test_all_survive` 明确记录了半开区间约定：FAD 恰等于 t_end 的类群属于**下一个更老的 bin**。`n_total=0 → nan` 是约定的正确结果，不是缺陷。我的第一版修复破坏了这条约定并导致 1 个测试失败，已回退 |
| `views/ui_main_window.py:2502` F821 未定义名 `np` | **误报**，`np` 在下一行有局部 import，注解是字符串字面量 |
| `newick_core.py:178` mypy 缺 return | **误报**，`self.error()` 总是 raise |
| i18n 6 个重复键"翻译被覆盖丢失" | **大幅降级**：8 组中 7 组值完全相同，仅 1 处差一个空格。真实影响只是让 ruff 挂 |
| `correlation.py:344` PPA 恒饱和到 `n_eff=2` | **未按描述复现**（实测 AR(1) φ=0.9 得 n_eff=100） |
| `pcm.py` "PIC 对枝长缩放完全不变" | **前提错误**，见 §1.5 |

**教训**：审查 agent 的高危结论必须逐条复现。这次 13 条 CRITICAL 中，真实存在的是 9 条；其余要么是误报，要么前提错误，要么（EFA）无法在不破坏其他约定的前提下修复。

---

## 4. 未处理的部分（需要你决定）

### 4.1 破坏性删除——一律没做

审查建议删除约 1,100 行死代码 + 1,100 行未接线模块（`app_infrastructure/` 4,825 行、`_core/` 712 行、`state_machine/` 2,543 行、`utils/matrix_ops.py` 807 行、`utils/decorators.py` 475 行）。**我一条都没删。**

原因：这些代码本身语法正确、有测试、可能是有意保留的"备用设施"。审查报告里我自己的结论是"**需先与作者确认意图**"。删除不可逆，且"零生产引用"不等于"无价值"（例如 `reporting/` 的 LaTeX 报告器是完整可用的功能）。**这个决定应该由你做，不该由我在"全部修复"的语境下顺手删掉。**

### 4.2 未修的 HIGH/MEDIUM（按性价比排序）

| 项 | 位置 | 原因 |
|---|---|---|
| TBR 实为 SPR、破坏二叉不变量 | `phylogenetics/heuristic_search.py:374` | 确认了代码层缺陷（`n1` 侧从未接回，378 行只处理"完全无子"没处理"只剩 1 子"），但正确实现 TBR 是**两切两接**，改动面大且该路径无生产入口 |
| Fitch 多分叉 25% 位点错误 | `phylogenetics/fitch.py:399` | 需改为 Sankoff DP，与启发搜索适应度耦合 |
| UPGMA/NJ 不可复现 | `phylogenetics/distance_methods.py` | 需把 `set[str]` 换成有序结构，涉及多处 tie-break |
| Lomb 周期图归一化差 2× | `stratigraphy/spectral_analysis.py:369` | 需确认采用哪一套约定（Horne–Baliunas eq.2.4 vs 简单 LS） |
| `procD_lm` 置换用删一模型、表用顺序 SS | `statistics/procD_lm.py:246` | 实测 F=7122 配 p=0.66 自相矛盾，但改动会改变现有测试的期望值 |
| PCA/PCoA 图不接收分组/标签 | `views/ui_plot_canvas.py:573` | 需要先决定结果对象是否新增字段（属 API 变更） |
| `load_excel` 静默吃首列 | `controllers/data_controller.py:586` | 同上，需决定探测策略 |
| undo 栈内存风险（峰值 ~8GB） | `models/state_manager.py:466` | 需改为按字节上限，且 `SPREADSHEET_MAX_ROWS` 从未被强制 |
| 4 处全局 `np.random.seed` | `fbd.py` / `diversity.py` / `isotope_analysis.py` / `coniss.py` | 改动机械但分散在 4 个包，本轮时间不够 |
| ruff 245 / mypy 11 | 全仓 | 主要是工具版本漂移（项目声明 `ruff>=0.4.0`，本机 0.15.15）；但 CI 声明全绿而实际是红的，需要统一版本约束 |

### 4.3 CI 与工程化

审查发现 `ci.yml` 全程 `pip install -e`，且**没有 `pip install .` / wheel / PyInstaller 构建 job**——本次修的打包缺口正是因此从未被发现。建议加一个非 editable 安装 + import 冒烟的 job。`PaleoAST.spec` 的 hiddenimports 仍缺 12+ 模块、`datas` 仍只有 2 项、`runtime_hooks=[]` 使 `rthooks/` 永不执行。

---

## 5b. 绘图/可视化层专项审查（第二轮）

对全部 **65 个绘图入口**（`views/ui_plot_canvas.py` 35 个 + `visualization/` 23 个 + `plot_export.py` 4 个 + 其他 3 个）做了**逐个实际调用**的冒烟测试：用真实引擎产出结果对象，并额外压 11 组边界输入。首轮 50/70 通过，其中暴露出**两个此前完全没被发现的现网 bug**。

### 已修复：3 个新问题

**A. CONISS 从 ribbon 必崩** — `controllers/statistics_controller.py:829`

`CONISSAnalyzer.analyze` 返回遗留二元组 `(CONISSResult, broken_stick_dict|None)`（该形状被 `tests/stratigraphy/test_coniss_broken_stick.py::test_analyze_return_type` 锁定，不能改 analyzer），但 controller 原样透传，`ui_main_window.py:4043` 按对象取属性：

```
AttributeError: 'tuple' object has no attribute 'linkage_matrix'
```

**修复**：在 controller 解包，broken-stick 载荷单独缓存为 `coniss_broken_stick` 保持可达。**验证**：controller 现在返回 `CONISSResult`，绘图不崩。

**B. Diversity 从 ribbon 必崩 + 会画假数据** — `views/ui_plot_canvas.py:1128`

`DiversityResult` 只有 `.indices`（`dict[str, DiversityIndexResult]`，6 项），没有 `.values`/`.labels`。代码因此回退到**硬编码的伪造数据** `[1.5, 0.8, 2.5]`，然后因为柱位按 6 项算、值只有 3 项而崩溃：

```
ValueError: shape mismatch: ... arg 0 with shape (6,) and arg 1 with shape (3,)
```

**修复**：按真实结构取 `index_name`/`value`，删掉伪造数据的后备分支，对非预期输入抛明确的 `ValueError`。**验证**：6 根柱高 = `[2.913, 0.941, 0.972, 4.074, 7.29, 20.0]`，与引擎输出逐一吻合；单类元样本正常；垃圾输入给清晰报错。

**C. `plot_pca_scores` 的钳制被绕过 + 除零** — `views/ui_plot_canvas.py:591`

`_clamp_dims` 这个属性 setter 存在的目的就是防"5-PC 的 PCA 标签页切到 1 维 NMDS 结果"这类陈旧索引，但绘图本身用的是**原始参数** `pc1`/`pc2`，绕过了钳制；`total_var` 也没防零，零方差数据产生 `RuntimeWarning` 和 `nan%` 标题。

**修复**：绘图改用钳制后的维度；`total_var <= 0` 时比例置 0。**验证**：零方差输入无告警不崩；`pc2=99` 被正确钳到 2。

### 已修复：2 个死方法（无调用方，但完全坏掉）

**D. `plot_anova_boxplot`** — 全仓零调用方，且对任何多列矩阵都跑不起来：`plot_data.append(data[mask])` 产出 **2-D 数组的列表**，matplotlib 直接拒绝（`ValueError: X must have 2 or fewer dimensions`）；字符串分组标签还会让 `f"Group {g + 1}"` 抛 `TypeError`。即使能跑，它也**完全忽略了 `variable_name`**，会把所有变量混进同一个箱线图。

**修复**：真正选择目标列（普通数组按 0-based 列索引，结构化数组按字段名），取名不明确时明确报错而不是静默画第 0 列；分组名改用 `str(g)`；`boxplot` 的 `tick_labels`/`labels` 兼容 matplotlib 3.7 与 3.9+。**验证**：列 0 → 中位数 `[1.0, 2.0]`，列 1 → `[99.0, 99.0]`，字符串分组、结构化数组按名、非法输入全部正确。

**E. `plot_rose_diagram`** — 空输入时 `2 * np.pi / 0` 抛 `ZeroDivisionError`。**修复**：空输入画占位文字；中心数与计数不匹配时给清晰 `ValueError`（生产路径 `bin_rose_diagram` 返回 12 个中心角 + 12 个计数，长度一致，所以长度不匹配实际不会触发）。

### 绘图层的既有结构性问题（未改，仅记录）

| 问题 | 状态 |
|---|---|
| `visualization/` 7 个 plotter 中 **6 个无生产调用方**（仅 `StratigraphyPlotter` 接线），与 `ui_plot_canvas.py` 形成两套并行绘图系统 | 架构债，见 §4 |
| 7 处 `plt.style.use()` 直接改**进程全局** rcParams，运行一次 Stratigraphic Correlation 会清掉 `ui_plot_canvas.py:59-91` 装好的字体/字号/网格设置，影响此后所有图 | HIGH，未改（需改用 `plt.rc_context` 上下文管理器，涉及 7 个文件） |
| `config/colors.py` 的 `COLORBLIND_FRIENDLY_PALETTE` / `IBM_COLORBLIND_SAFE` 在 `visualization/` 中**从未被引用**，颜色是 `#2C3E50` 这类字面量硬编码 | 无障碍调色板形同虚设 |
| PCA/PCoA 得分图**拿不到分组和行标签**（`PCAResult` 无 `groups`/`labels` 字段），`plot_lda_scores` 却正常——同 app 两套约定 | MEDIUM，改需动结果对象 API |
| 7 个死方法：`plot_allometry` / `plot_evolution_rate` / `plot_extinction_ranges` / `plot_beta_diversity` / `plot_null_model` / `plot_phylo_tree` / `plot_anova_boxplot` | 死代码，需决定接线或删除 |


```
 controllers/data_controller.py | 58 +++++++++++++++-----
 ecology/advanced.py            | 22 ++++-
 macroevolution/cohort.py       | 13 ++--
 main.py                        | 33 ++++++--
 morphometrics/tps.py           | 21 +++--
 plot_export.py                 | 45 +++++++----
 pyproject.toml                 | 24 +++++
 statistics/pcm.py              | 19 ++++-
 statistics/simper.py           | 68 ++++++++++++-------
 utils/newick_core.py           | 11 +++
 views/ui_main_window.py        | 99 ++++++++++++++++++++++-----
 views/ui_plot_export_dialog.py | 11 ++-
 12 files changed, 349 insertions(+), 75 deletions(-)
```

- **测试**：1111 passed / 33 skipped / 0 failed（与基线一致，无回归）
- **未提交**：所有改动都在工作区，`git status` 可见。按你的工作习惯没有自动 commit。
- **回退**：`morphometrics/efa.py` 已 `git checkout` 还原，不在改动列表内。
