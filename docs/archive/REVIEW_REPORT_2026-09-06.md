# PaleoAST 复审报告（第二轮）

**审阅日期**: 2026-09-06
**基线**: commit a66b15d + 工作区未提交改动（12 个修改文件 + 2 个新测试）
**审阅方法**: 主审逐行复核全部未提交 diff 并实测；4 个并行专家 agent（统计+生态 / 系统发育+形态 / 宏观演化+地层 / 解析器+数据+UI）深审，关键结论均经数值探针或测试运行交叉验证
**审阅视角**: 资深古生物学家 + 软件工程
**上一轮报告**: `REVIEW_REPORT_2026.md`（2026-08-04）

---

## 〇、一句话结论

batch2–4 的修复**部分真实有效**（Kabsch 反射、iNEXT bootstrap、Pagel λ 变换、NEXUS 引号、dat 千分位等已修好），但**至少 6 条"修复"只是换了一种错误**（Blomberg K、PIC、Marshall CI、broken-stick、CCA、SIMPER），且**提交前未运行测试**：当前代码库自带测试中至少 **20 个失败**（含 1 个可致内存耗尽的死循环）。工作区未提交的解析器增强（NHX/引号/注释）**核心功能实际不可达**，9 个新测试中 5 个失败。在修复 P0 清单之前，富度曲线、SIMPER、Blomberg K、古温度换算、Marshall CI、小波谱的输出**不可用于任何定量结论**。

### 当前测试实况（本人运行）

| 测试文件 | 结果 |
|---|---|
| tests/phylogenetics（PIC polytomy、Blomberg K golden、Pagel λ golden） | **10 failed** / 64 passed |
| tests/parsers/test_newick_nhx_bom.py（未提交新文件） | **5 failed** / 4 passed（其一为 MemoryError，耗时 4 分钟） |
| tests/ecology/test_simper_golden.py | **2 failed** |
| tests/stratigraphy/test_isotope_paleotemperature.py | **3 failed**（Kim & O'Neil 输出 310 °C） |
| 未提交改动涉及的其余 6 个测试文件 | 74 passed（但部分断言把错误公式固化为 golden，见 §二） |
| tests/cross_validation（R 对照） | 31 skipped（rpy2 缺失；C22 实质未修） |

---

## 一、工作区未提交改动逐项裁定

### 1. `parsers/newick_parser.py` — ❌ 不能按现状提交（本区块最严重）

意图是修复 H14（BOM）/H15（NHX）/H16（引号）+ 注释支持。BOM 与引号解析**正确**（对应测试通过），但注释与 NHX 存在致命缺陷：

| # | 严重度 | 位置 | 问题（均已实测复现） |
|---|--------|------|---------------------|
| N-1 | **CRITICAL** | `_parse_name`:803 + `parse`:571 | **规范冒号式 NHX 触发死循环 + OOM**。名字正则 `[^():,\s;]+` 不排除 `[`，`(...)[&&NHX:B=95:D=N]` 中 `[&&NHX` 被当作内部节点名吞掉，`:` 被当作枝长分隔符，`D=N]` 变成幽灵叶节点；最终游离的 `)` 在 `parse()` 顶层循环永不消耗 → 无限追加空节点直到 `MemoryError`（实测：单棵 5 分类单元树耗尽内存）。**即便提交者自己的测试也因 OOM 失败**。附带修复建议：a) 名字排除集加入 `[`、`]`、`'`、`"`；b) `parse()` 顶层遇到无法消耗的 `)` 立即抛 `ValueError`，消除整类死循环 |
| N-2 | **HIGH** | `parse`:546-555 | **同行注释吞树**。`[&R] (A:0.1,B:0.2);`（BEAST/PAUP 极常见）→ 注释扫描"到行尾为止"，把后面的树一并吞掉 → `ValueError: no valid tree found` → `parse_multi` 仅记 warning **静默丢弃整棵树**。注释应扫描至配对 `]`（支持嵌套），而不是 `\n` |
| N-3 | **HIGH** | `_parse_simple_node`:745 | **叶节点 NHX 完全未处理**（这是 NHX 最常见位置，如 ETE3 输出 `(A[&&NHX:S=human]:0.1,B)`）：叶名被污染成 `[&&NHX...`，枝长错位，树结构被打乱 |
| N-4 | MEDIUM | `_parse_nhx_metadata`:877-889 | **属性分隔符与规范相反**。NHX 规范（Zmasek & Eddy 2001）属性间用 `:`，代码先按 `,` 切分 → 规范文件解析成 `{":B": "95:D=N"}`，属性全部丢失。应优先按 `:` 切分（兼容 FigTree/BEAST 的逗号式 `[&...]` 另行处理） |
| N-5 | MEDIUM | `_parse_subtree_with_children_impl`:733 | NHX 处理代码在"`]` 后跟枝长之前"的位置**不可达**（名字解析先吃掉 `[`）；只有"枝长之后"的 NHX 能进到这里 |
| N-6 | MEDIUM | 测试文件 | `test_nhx_metadata_basic` / `test_nhx_metadata_multiple_attrs` 只断言 `leaf_count==3`，**没有验证任何 metadata 被存下来**（死代码迷宫后必然落到这一句）——这就是错误能溜过去的原因。`test_bom_stripping` 自造双 BOM（`encoding="utf-8-sig"` 又写字面 BOM 字符）；`test_bom_stripping_via_parse` 注释声称"strip() 会去掉 BOM"是错的（`str.strip()` 不删 U+FEFF）；`test_unmatched_paren_raises` 期望抛异常但解析器静默接受（C15 仍未修） |

### 2. `macroevolution/cohort.py` — ⚠️ 方向修复正确，但 Foote 率公式仍错

✅ **已修对**：时间方向判定（深时 Ma 从新到老，`started_before = o > t_end` 等六项全部反转正确）；`dt = t_end - t_start` 由负改正（原实现所有 per-capita 率符号颠倒——这才是旧 C9 的要害）。

❌ **仍存在的问题**：

| # | 严重度 | 位置 | 问题 |
|---|--------|------|------|
| C-9a | **HIGH（科学性）** | cohort.py:316-323 | `origination_rates[i] = -np.log(1-p)/dt` 用**存活率**推起源率——概念错误。`-ln(1-p)` 是灭绝概率的变换，与起源无关；p=1（全部存活）反而得到 λ=0、p=0 得到 λ=∞。起源率必须来自 FAD/边界穿越计数（本函数稍后已算出正确的 `foote97_origination`，两者应统一） |
| C-9b | **HIGH** | cohort.py:340-346 | `foote97_extinction = -ln(n_bl/n_t)/dt` 用**向后**计数（区间内起源者比例），与向后起源率互为补数，本质是同一个量的两种写法。Foote 的灭绝率必须用**向前**存活：`q = -ln(n_ft/n_t)/Δt`。agent 以 birth–death 模拟（p=0.05, q=0.20, Δt=5）实证：现实现灭绝中位估计 **0.58（真值 0.20，+191%）**、起源 0.011（真值 0.05，−78%）；改正后估计值贴近真值。另：`test_foote97_vs_per_capita_rates` 断言"孤立 through-timer 灭绝率=+inf"——`f=1 ⇒ q=0` 才对，**测试把错误固化了** |
| C-9c | MEDIUM | cohort.py:352 | `foote00_origination = n_ft/n_t` 实为**前向存活分数**（=1−前向灭绝概率），标签错误；若作为"起源概率"应为 `n_bl/n_t` |
| C-9d | MEDIUM | cohort.py:250-253 | `started_in`/`ended_in` 双端闭合 `<=`：恰在边界年龄的分类单元会被**相邻两个 bin 重复计入**。建议半开区间 `[t_start, t_end)` |

### 3. `stratigraphy/coniss.py` — ❌ broken-stick 公式不是 broken-stick

新实现把期望值写成 `E[i] = 1/(n_levels - i)`（调和级数项、**升序** 1/(n−1)…1/1），再归一化后与**降序** BD 配对——即最大观测 BD 对上最小期望。**规范公式**（MacArthur 1957；Bennett 1996；rioja::bstick 一致）为第 k 大份额 `E[k] = (1/n)·Σ_{i=k..n}(1/i)`。n=5 时：规范 `[0.457, 0.257, 0.157, 0.090, 0.040]`，代码 `[0.12, 0.16, 0.24, 0.48]`，最大偏差 0.39；同一探针数据 Bennett 判定 2 个显著带、代码判定 0 个。归一化到和为 1 只是掩盖了原先"p 全为 1"的病态——**两个错误叠在一起**。且 `tests/stratigraphy/test_coniss_broken_stick.py` 把调和项当作 golden 断言，需要随公式一起重写。附带：`bd_values` 直接取 merge 顺序的联结距离，与 Bennett 按候选带数比较带内离散度的做法不符（MEDIUM，修公式时一并处理）。

### 4. `statistics/nmds.py` — ⚠️ 可提交，但有一处不一致

双公式参数化方向正确，`raw_stress`（分母 Σd_target²）为默认以保兼容、`stress_1`（分母 Σd_hat²，Kruskal 1964 / vegan::monoMDS）为 opt-in——测试通过。两点提醒：
- **默认值不一致**（LOW-MEDIUM）：`analyze()` 默认 `method="raw_stress"`，`_smacof()` 默认 `method="stress_1"`。任何直接调 `_smacof` 的内部/插件路径会悄悄换公式。建议统一为同一常量。
- 要与 R vegan 对照必须显式传 `method="stress_1"`（GUI 控制器目前未暴露该参数）。

### 5. `macroevolution/survival.py`、`morphometrics/gpa.py`、`views/ui_plot_canvas.py` — ✅ 无问题

`preserve_dimensions=True` 参数真实有效且语义正确（1D 数据不再被重塑成 (n,1)，修复了 lifelines/shape 检查路径）；`GPAResult.centroid_sizes` 字段及 `_compute_sizes`（gpa.py:405）实现正确；ui_plot_canvas 仅变量重命名，行为不变。

### 6. 仓库卫生

`macroevolution/cohort.py.bak` 备份文件遗留未跟踪，应删除（差异已在 git 历史中）。

---

## 二、四个深审域的新发现（精选）

> 以下每条都经 agent 代码精读 + 数值探针验证；标注 ★ 的经主审本人复跑确认。完整清单按域分列。

### CRITICAL（主用例输出错误数字 / 崩溃 / 数据损坏）

| # | 位置 | 问题 | 证据 |
|---|------|------|------|
| K1 ★ | `macroevolution/diversity.py:91` | **富度区间条件反转**：要求 `o <= t_start 且 L >= t_end`——在 Ma 约定下几何上几乎不可能满足（o>L）。真实数据 richness≈0。实测：两个明确出现在 (5,10) Ma 的分类单元 → richness=0。注释声称"修复了方向"但方向仍是反的 | 主审复现 |
| K2 ★ | `statistics/simper.py:227-307` | **SIMPER 回归**（batch2 引入）：贡献项 c_k=2·min/ΣΣ 分解的是相似度（Σc_k=1−BC）却除以 overall_dissimilarity=BC → 累计可达 150%；Av/SD 的 SD 取自**组对间**（2 组时恒为 0）→ ratio=∞。自家 golden 测试 2 失败。正确式：c_k(ij)=\|x_ik−x_jk\|/Σ_l(x_il+x_jl)，SD 跨个体对 | 主审复现 |
| K3 ★ | `statistics/pcm.py:664-670` + `phylogenetics/signal.py:264-279` | **Blomberg K 两处实现都不是 Blomberg K**（量纲依赖：性状 ×10 → K ×10/×100）。pcm.py 版实为 BM 速率估计 σ̂²；signal.py 版是 trace 二次型。规范：K = Σ(y−ȳ_GLS)²/[(n−1)·σ̂²_GLS·tr(V)/n]。自家 golden 测试 10 失败（BM 数据 K=3.09"应≈1"、随机数据 K=6.95"应<0.5"） | 主审复现 |
| K4 | `phylogenetics/pic.py:139-183` | **PIC 仍错（C1 换了一种错法）**：把标准化对比值当作祖先"性状值"向上传递（应传逆方差加权重建值 x̂），方差簿记不一致（叶含枝长、内部节点不含）。实测 `((A:1,B:1):1,C:1)` 根对比 −3.70 vs 规范 −2.0。**pcm.py 的 `_compute_contrasts_recursive` 已实现正确**——应删 pic.py 实现改为委托 | agent 探针 |
| K5 | `statistics/pcm.py:351,385,410` | PIC 方差向上合并用 `v1+v2`，规范为 `v1·v2/(v1+v2)`（逆方差加权重建的方差）→ 深层对比 SE 系统性膨胀（探针：SE 3.00 vs 规范 2.20） | agent 探针 |
| K6 | `statistics/cca.py:423-449` | 特征问题不是 ter Braak (1986) CCA：特征值 0.010 vs 参考 0.435（~40 倍）；总惯量定义也非卡方惯量。H3 只修了字面权重、未触及方法本身 | agent 探针 |
| K7 ★ | `stratigraphy/isotope_analysis.py:522-549` | **Kim & O'Neil 混用 VPDB/VSMOW 标尺**（δc 未做 `1.03091·δc+30.91` 转换即进入分馏方程）→ 输出 ~310 °C。自家测试 3 失败。Erez & Luz 常数正确；Bemis 用了非其 Table 3 的种级系数 | 主审复现 |
| K8 | `stratigraphy/spectral_analysis.py:50-116` | **CWT 无尺度膨胀**："scales"只改变窗长、载波频率不变（无 `ψ((t−b)/a)`）→ 变换在原理上无法检测周期。探针：周期 10 正弦在 scale=2 处取峰（暗示波长 2.46）；正确膨胀 Morlet 在 scale=8（波长 9.86）。`frequencies` 与 COI 因此全部无效（H8 只修了不等号方向） | agent 探针 |
| K9 | `stratigraphy/extinction.py:313-346` | **Marshall 1990 CI 三重错误**：① 分位数用 `chi2.ppf(1−2q,2)` = 90% 区间标 95%（2.3026/r vs 正确 2.9957/r）；② **方向反了**——CI 向更老端延伸，Signor-Lipps/Marshall 语义是真灭绝时间在 LAD **更年轻端**；③ r 取自其它分类单元 LAD 排名，非该分类单元自身恢复率；k=0（最顶部 LAD，恰是最需要 CI 的）反而无 CI。测试同样固化错误方向。`_strauss_sadler_ci` 也非 S&S 1989 间距枢轴量 | agent 探针 |
| K10 | `parsers/nexus_lexer.py:371-389` | **回归（9228a7c 引入）**：`_scan_comment` 只在 `]]` 序列才减 depth → 普通 `[...]` 注释永不终止，吞掉文件剩余全部内容（`[&R]`、PAUP 注释即触发）。C16 的旧问题换成了更致命的新问题 | agent 探针 |
| K11 | `parsers/binary_cache.py:290-297` | dtype 白名单外（如 numpy 2.x 默认 **int64**）静默按 FLOAT64 头写原始字节 → 读回为乱码浮点；float16 直接 load 失败。保存时无任何告警——**静默数据损坏** | agent 探针 |
| K12 | `morphometrics/tps.py:147-153,249-285` | **TPS 系数顺序错**：存的是 `[w; a]`（非仿射在前），求值向量却按 `[1,x,y,U…]`（仿射在前）拼 → 所有形变网格/`warp_grid` 可视化错误（在源地标处求值误差 13.4，应为 ~1e-12） | agent 探针 |

### HIGH（常见场景错误 / 崩溃）

| # | 位置 | 问题 |
|---|------|------|
| H-A | `ecology/beta_diversity.py:308` | Baselga β_sne 误用 Jaccard 分子（应为 `2a·max(b,c)/[(2a+b+c)(2a+min(b,c))]`）→ 分解不可加（0.583 vs 总 0.778） |
| H-B | `ecology/beta_diversity.py:686-749` | 稀疏化 Shannon/Simpson 曲线对多数 n **恒等于观测值**（f1 线性缩放的覆盖度近似在 f1·n/N≥1 时退化）；应按 Chao & Jost 2012 Eq.4 隐式求解 |
| H-C | `ecology/beta_diversity.py:589-601` | iNEXT 式点估计取 **bootstrap 曲线中位数**（随种子变、偏低），Chao et al. 2014 的点估计来自观测数据，bootstrap 只用于 CI |
| H-D | `ecology/paleoenv.py:278-279` | CA 得分缺 D^(-1/2) 质量标准化（=主坐标×√质量），χ² 距离不保持（H5 半修） |
| H-E | `ecology/advanced.py:322-325` | SHE 分析仍按丰度排序累积（H24 原样）——必须按采样/地层顺序，否则 S 峰位置无意义 |
| H-F | `statistics/pcm.py:854-968` | phylo-ANOVA 观测 F 与置换 F 是**两个不同统计量**（H13 原样）→ p 值无效 |
| H-G | `morphometrics/gpa.py:635-732` | `partial_gpa` 滑动目标含"偏离共识点²"项 → 半标志点被拉向共识、真实形状变异被抹掉（实测方差 0.160→0.001）。不是 Bookstein/Gunz 滑动 |
| H-H | `morpho3d/sliding.py:318-424` | 3D 滑动常态配置下 `IndexError` 崩溃（用全树索引访问固定点子集数组）；弯曲能分支的位移方向取反 |
| H-I | `phylogenetics/strict_consensus.py:381-426` | 共识树构造器会**编造 clade**：`((A,B),C,D)` 与自身的一致 consensus 输出 `((A,B),(C,D))`（连恒等往返都失败） |
| H-J | `phylogenetics/heuristic_search.py:235-257` | NNI 节点按 (name, 子数) 匹配 → 未命名内部节点全部映射到根；`((A,B),(C,D))` 根边 NNI 输出 `((C,D),(C,D))`（树损坏后继续搜索） |
| H-K ★ | `macroevolution/survival.py:191-241` | Kaplan–Meier：并列时刻若删失值排在事件前（排序不稳定），**事件被静默丢弃**（[5,5,5]/[0,1,1] → S(t)≡1，正确 1/3）；log-log CI 除以负数 → **lower > upper 全线互换** |
| H-L | `stratigraphy/arma.py:232-276` | 手写 Yule–Walker 崩溃链：validate 返回 (n,1) → `np.correlate` "object too deep"；1D 时 RHS 滞后错位（AR(1) 恒得 φ=1.0）；statsmodels 路径 `include_intercept=False` 切片错位把常数当 φ₁。本机无 statsmodels → `fit()` 每次必抛（H10 恶化） |
| H-M | `stratigraphy/correlation.py:222` | 年龄模型仍 cubic interp1d（H9 原样）——单调稀疏节点场景必须 PCHIP；CI 为 ±1.96·平均误差均匀带，无单调性守卫 |
| H-N | `statistics/pcm.py:568-576` | ASR 权重仍 1/枝长（H12 原样，应为 1/(cum_var+branch)）；`or` 把真 0 枝长变成 0.001 |
| H-O | `parsers/dat_parser.py:185-202` | 表头存在但无行标签的合法 PAST .dat → `expected_field_count` 错 → 硬性 `DATParseError`（batch4 修了千分位但留了此坑） |
| H-P | `parsers/lexer.py:321-393` | 基类 `_try_match`（无命名组）与 `_create_token`（强制取命名组）契约不匹配 → 任何用默认路径的子类**第一个 token 即 IndexError**（仅 NexusLexer 因覆写而幸存） |
| H-Q | `plot_export.py:181-235` | 导出尺寸快照取在 `set_size_inches` **之后** → "还原"的是导出尺寸，画布被永久改小；灰度导出直接改画布 artist 颜色且不还原（永久变灰）。应快照前置 + 在副本 figure 上做灰度 |
| H-R | `state_machine/automaton.py:690-697` | 正则解析器顶层交替只解析一个 `|`：`a|b|c` 静默拒绝 `c`（batch4 修了 `\D\W\S` 但留此坑） |

### MEDIUM（择要；完整清单见 agent 明细）

`pcoa.py:192-217` 负特征值回归（√\|λ\|+按\|λ\|排序，违背自述的 R 约定）· `statistics/pca.py:358` 零方差列静默 σ→1（H21 原样）· `diversity.py:95-109` 起源/灭绝率实为净变化率 dR 拆分（H7 原样）· `macroevolution/fbd.py:583-684` complete_tree 参数被忽略、根 λ 遗漏、小 λ 分支公式错（Stadler 2010 语义不符）· `morpho3d/gpa3d.py:208-252` centroid_sizes 恒 ≈1.0（首轮缩放后覆盖）· `morpho3d/tps3d.py:105-174` Result 用 r³ 核而 fit 用 r → 对象不能复现自身映射 · `gpa.py:900-923` 3D 弯曲能用 2D 核 r²log r · `evolution_rate.py:483-487` OU σ² 除以 Σdt² · `allometry.py:330` 同距检验 df 用相关变量数（反保守）· `markov.py:137-148` 引 P&E 1982 却用独立期望（应除以 N−row_i）· `signal.py:367,453` λ 用算术均值非 GLS 均值（iid 数据 λ̂≈0.6）且 LRT 忽略 0 边界混合 χ² · `cca.py:398` 零行/列静默造边际 · `binary_cache.py:505-560` mmap/fd 异常路径泄漏 · `load():477` 一把 `except Exception→None` 使 C18 的硬失败失效 · `nexus_writer.py:280,297` 多字符 cell 无分隔拼接（H20 残留）· `report_builder.py:403-464` 标题/图注未过 LaTeX 转义 · `report_builder.py:517-549` pdflatex 成功判定路径错（cwd 与 output_dir 不一致时误报失败）· `tokenizer.py:294-314` 同类型双规则 redefinition 崩溃 · `ui_plot_export_dialog.py:228` `.lstrip("")` 无操作 → 默认格式恒 PNG · `tps_parser.py:303-317` `_in_curve` 跨标本不重置 → TypeError · `biostratigraphy.py:806-911` "RASC" 实为贪心 TSP 启发式，非 RASC · `diversity.py:300` 等处 z 值硬编码 95/99 二选一（90% CI 静默返回 99%）· `null_models.py:359` swap 链混合不足且每复制从观测矩阵重启

### 旧报告 25 条 CRITICAL 的当前状态速览

**已真正修好**: C4（Kabsch 反射——上轮批评有误，单侧翻转 Vt 即可且已重算 R）、C8（iNEXT bootstrap 真重采样）、C2（Pagel λ 协方差）、H14（BOM，parse_file 路径）、H16（引号）、H17（dat 千分位）、H18（NEXUS 引号）、H4（swap 边际保持）、H22（DTW 记录）
**换了一种错**: C1/C24（pic.py 仍错、pcm.py 对）、C3（两处 K 都不是 K）、C6（NMDS 默认仍非 Kruskal 规范，opt-in 才对）、C10（Marshall CI 换成 α 拆分错误+方向反）、C13（broken-stick 换成调和项）、C7（SIMPER 回归）
**未修**: C5 残留（局部弯曲能）、C11（Pyper-Peterman 仍为 Chelton 式）、C12 半修、C14、C15、C16 恶化、C17、C19（真实失效方式与上轮描述不同）、C21、C22、C23（行为已统一，代码仍三份）、C25（pcm.py VCV=零对角距离阵，GLS 不可用）、H1 半修、H5 半修、H7、H9、H10 恶化、H11、H12、H13、H20、H21、H23~H26

---

## 三、流程层面的问题（与公式同等重要）

1. **提交前未跑测试**。batch1-4 各自新增的 golden 测试中，仅系统发育域就有 10 个失败、SIMPER 2 个、古温度 3 个、解析器（未提交）5 个——合计 20 个。这意味着部分"修复"是在未验证状态下合入的。建议在 CI 中强制 `pytest -q` 全绿才允许 merge。
2. **golden 测试反向固化错误**。至少 4 处测试把错误公式断言为正确（broken-stick 调和项、Foote through-timer 灭绝率=inf、Marshall CI 方向、SIMPER 累计和）。修公式时必须同步重写这些断言并在注释中给出文献页码/公式号。
3. **R 对照形同虚设**（C22）：31 个 cross-validation 测试全部 skip。哪怕不引入 rpy2，也至少应把 R 的输出（ape::pic、phytools::phylosignal、vegan::simper/monoMDS、betapart）手工固化为 golden 数值写进测试。
4. **双实现并存**仍在制造 bug：pic.py（错）vs pcm.py（对）；signal.py VCV（对）vs pcm.py VCV（零对角距离阵）；Blomberg K 两处都错。收敛为单一参考实现是消除此类问题的根本手段。

---

## 四、修复优先级

**P0 — 数据可靠性地雷（直接产出错误数字/崩溃/损坏，建议立即修）**
1. `diversity.py:91` 富度条件反转（K1）
2. `newick_parser.py` NHX 死循环 + 注释吞树 + 游离 `)` 守卫（N-1/N-2/N-3）——未提交改动需返工
3. `nexus_lexer.py` 注释吞噬回归（K10）
4. `binary_cache.py` dtype 静默损坏（K11）
5. `simper.py` 回归（K2）
6. `isotope_analysis.py` Kim & O'Neil 标尺转换（K7）
7. `spectral_analysis.py` CWT 尺度膨胀（K8）
8. `survival.py` KM 事件丢失 + CI 互换（H-K）
9. `cohort.py` Foote 灭绝率改向前存活 + 起源率概念修正（C-9a/b）

**P1 — 核心统计正确性（影响发表级结论）**
Blomberg K 单一规范实现 · PIC 统一到 pcm.py 并修 v1·v2/(v1+v2) · CCA 特征问题重写 · Marshall/S&S CI（方向+分位数+恢复率）· phylo-ANOVA 统一统计量 · TPS 系数顺序 · 2D/3D 滑动半标志点目标函数 · broken-stick 规范公式（连同测试重写）

**P2 — 一致性与鲁棒性**
β_sne · iNEXT 点估计 · CA 得分标准化 · SHE 排序 · ARMA 手写分支 · 严格共识/NNI · ASR 权重 · PCHIP 年龄模型 · dat 表头推断 · lexer 契约 · plot_export 状态还原 · automaton 交替解析 · VCV 单一约定（C25）

**流程**: CI 全量测试门禁 · golden 数值必须附文献出处 · 删除 `cohort.py.bak` · cross-validation 用固化 R 数值落地

---

## 五、仍值得肯定的部分

- 已验证正确的模块面相当广：ANOSIM、PERMANOVA、Bray-Curtis（两条路径）、Shannon/Simpson/Pielou/Margalef/Fisher-α/Chao1 及其 CI、UPGMA、NJ、Fitch 下行打分、四元数全套（Shepperd/SLERP/双覆盖安全）、efa.py（Kuhl & Giardina 闭式系数与独立推导吻合）、tps3d 拟合/雅可比（1e-16 量级）、Kabsch/反射、mesh 采样与体积、state_machine 基类转移表。
- 工程骨架（MVC、i18n、异常体系、验证器、文件大小守卫、递归深度守卫）持续在变好；本轮未提交的 BOM/引号修复本身是正确的。
- 上一轮指出的"测试先行"意识在增强（新增了 golden 测试文件）——只是还差"提交前跑一遍"这最后一步。

---

**审阅人**: Claude（主审 + 4 个并行领域专家 agent，全部关键结论经实测复核）
**方法与可复现性**: 所有"实测"标注项均以 pytest 或独立数值探针复现；探针脚本为一次性产物，未留在仓库中。
