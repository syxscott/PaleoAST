# PaleoAST 代码审查综合报告

> **审查日期**: 2026-07-18
> **审查范围**: 全量 Python 源码 (160 个文件, ~67,267 行)
> **审查方法**: 多 Agent 并行审查 + 关键代码人工复核 + 测试运行验证
> **审查工具**: Claude (Sonnet 5) + 6 个并行审查子 Agent
> **审查依据**: code-review skill 规范 + BUGS.md 历史审计 + 测试结果

---

## 0. 总览

| 等级 | 数量 | 含义 |
|------|------|------|
| 🔴 CRITICAL | **12** | 直接影响论文/数据正确性，必须立即修复 |
| 🟠 HIGH | **24** | 算法偏差、可重现性问题、安全风险 |
| 🟡 MEDIUM | **30** | API 不一致、边界条件、并发隐患、性能 |
| 🟢 LOW | **21** | 文档、死代码、命名、小性能 |
| **总计** | **87** | 新发现 bug（不含 BUGS.md 已修复的 17 项严重 bug） |

### 模块评分

| 模块 | 评分 | 状态 |
|------|------|------|
| Statistics (统计) | 7.0 / 10 | 可重现性危机 (无 random_seed) |
| Ecology (生态) | 7.5 / 10 | C-score / Baselga / hypergeometric 已修复 |
| Morphometrics (形态测量) | 7.0 / 10 | **质心大小公式错误 (CRITICAL)** |
| Phylogenetics (系统发育) | 7.5 / 10 | NNI/TBR 边界问题 |
| Macroevolution (宏观进化) | 7.5 / 10 | cohort 颠倒已修复 |
| Stratigraphy (地层) | 7.0 / 10 | 频谱归一化、CONISS 空指针 |
| Morpho3D | 7.5 / 10 | SLERP 近似、死代码 |
| Models/Controllers | 6.5 / 10 | **视图/引用语义混乱** |
| GUI (Views) | 5.8 / 10 | **线程安全、内存泄漏** |
| Infrastructure | 6.0 / 10 | binary_cache 损坏、callback 泄漏 |
| Main/Config | 7.0 / 10 | emoji、依赖缺失 |

**项目综合评分: 6.5 / 10**

---

## 1. 🔴 CRITICAL BUGS (12 项) - 必须立即修复

### CRITICAL-01: 缺失 `scikit-learn` 依赖
- **文件**: `requirements.txt`
- **问题**: `statistics/nmds.py` 和 `statistics/lda.py` 强依赖 `sklearn.isotonic.IsotonicRegression` 和 `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`，但 `requirements.txt` 中未列出
- **测试证据**: `python test_regression.py` 中 NMDS 测试失败: `No module named 'sklearn'`
- **触发条件**: 安装 requirements.txt 后直接运行
- **影响**: NMDS 和 LDA 模块完全无法使用；启动报错

**修复**:
```python
# requirements.txt 添加:
scikit-learn>=1.3.0
```

---

### CRITICAL-02: morphometrics/allometry.py - 质心大小 (Centroid Size) 公式错误
- **文件**: `morphometrics/allometry.py:366-385`
- **问题**: 实现使用 Frobenius 范数 `CS = sqrt(sum(x²))` 而非标准 Bookstein 质心大小公式 `CS = sqrt(sum(||x_i - centroid||²))`
- **触发条件**: 调用 `_compute_centroid_sizes` 时
- **影响**: 所有异速生长分析、PLS、整合系数计算结果均不正确

**修复**:
```python
def _compute_centroid_sizes(self, configurations: npt.NDArray) -> npt.NDArray:
    n_specimens = configurations.shape[0]
    centroid_sizes = np.zeros(n_specimens)
    for i in range(n_specimens):
        centroid = configurations[i].mean(axis=0)  # 真实质心
        diff = configurations[i] - centroid
        centroid_sizes[i] = np.sqrt(np.sum(diff ** 2))
    return centroid_sizes
```

---

### CRITICAL-03: statistics/permanova.py - 置换检验无随机种子
- **文件**: `statistics/permanova.py:155-159`
- **问题**: `np.random.permutation(n)` 未设置 seed，导致 PERMANOVA p 值不可重现
- **影响**: 同一数据多次运行得到不同 p 值，违反科学可重现性原则

**修复**:
```python
def analyze(self, ..., random_seed: int | None = None):
    rng = np.random.default_rng(random_seed)
    for i in range(n_permutations):
        perm_indices = rng.permutation(n)
        permuted_groups = groups_array[perm_indices]
        permuted_F[i] = self._compute_F_statistic(...)
```

---

### CRITICAL-04: statistics/anosim.py - 置换检验无随机种子
- **文件**: `statistics/anosim.py:140-144`
- **问题**: 同 PERMANOVA
- **影响**: ANOSIM 结果不可重现

---

### CRITICAL-05: statistics/pcm.py - Blomberg K & Phylo-ANOVA 置换无种子
- **文件**: `statistics/pcm.py:671-686` 和 `:882-946`
- **问题**: Blomberg K (compute_phylogenetic_signal) 和 phylogenetic_anova 中的置换检验均无种子
- **影响**: 系统发育信号、Phylo-ANOVA 结果不可重现；任何下游论文结果均无法复现

---

### CRITICAL-06: statistics/spatial.py - Monte Carlo envelope 无种子
- **文件**: `statistics/spatial.py:273-287`
- **问题**: `_compute_envelope` 中 `np.random.uniform` 未设种子
- **影响**: 空间统计置信包络不可重现

---

### CRITICAL-07: parsers/binary_cache.py - HEADER_SIZE 与 struct 格式不一致
- **文件**: `parsers/binary_cache.py:160`
- **问题**: `HEADER_SIZE = 64`，但 struct 格式 `"!IIIIII QQII"` 只产生 48 字节；`to_bytes()` 填充到 64 但 `from_bytes()` 只解包 48 字节
- **影响**: 缓存文件读写潜在错位，CRC 计算区域不一致，可能造成数据损坏

**修复**:
```python
HEADER_SIZE: int = 48  # 6*4 + 2*8 + 2*4 = 48
# 或者扩展 struct 格式以填充到 64
```

---

### CRITICAL-08: parsers/binary_cache.py - 压缩路径 CRC 不一致
- **文件**: `parsers/binary_cache.py:318-322, 417-424`
- **问题**: `save` 时对未压缩数据计算 CRC，`load` 时对压缩字节 (`raw_matrix_bytes`) 校验
- **影响**: 启用压缩后所有缓存文件 CRC 校验失败

---

### CRITICAL-09: views/diagnostic_console.py - Qt 控件跨线程访问
- **文件**: `views/diagnostic_console.py:329`
- **问题**: `append_message` 在 worker 线程中通过 QMutexLocker 调用 `QTextEdit.append()`；QTextEdit 不是 reentrant，跨线程访问即使加锁也是不安全的
- **影响**: 应用崩溃、日志错乱、间歇性 GUI 卡死

**修复**:
```python
# 用 Qt signal 跨线程，槽在主线程执行
self.message_received = pyqtSignal(str, str)
# worker 端: self.message_received.emit(level, msg)
# 主线程连接: self.message_received.connect(self._console.append_message)
```

---

### CRITICAL-10: views/ui_main_window.py - 长任务阻塞 UI 主线程
- **文件**: `views/ui_main_window.py` 所有分析处理器
- **问题**: PCA/PCoA/NMDS/置换检验等大计算在主线程同步执行，UI 冻结数秒到数十秒
- **影响**: 用户体验极差；用户误以为应用崩溃；无进度反馈

**修复**: 使用 `QThread` + `QRunnable`，结果通过信号回传

---

### CRITICAL-11: app_infrastructure/exception_handler.py - QApplication.quit() 不立即退出
- **文件**: `app_infrastructure/exception_handler.py:385-386`
- **问题**: `QApplication.quit()` 仅 post quit 事件；从非主线程调用可能延迟或失效
- **影响**: 异常后用户点 Quit 不能立即退出

**修复**:
```python
QApplication.exit(1)  # 或 os._exit(1)
```

---

### CRITICAL-12: utils/matrix_ops.py / controllers/data_controller.py - 视图/拷贝语义混乱导致数据污染
- **文件**: `models/data_matrix.py:279, 271` + `controllers/data_controller.py:371, 426, 446, 469`
- **问题**:
  - `DataMatrix.nan_mask` 返回视图 (`np.isnan(self._data)`) 而非拷贝
  - `DataController.transpose` / `subset_rows` / `subset_columns` 用 `matrix.data` 返回拷贝后再切片——可能产生视图链
  - `transform_standardize` 中 `(data - mean) / std` 当 data 是视图时进行 in-place 修改
- **影响**: 看似无关的操作修改了原始数据，导致难以察觉的数据污染

**修复**:
```python
# nan_mask:
@property
def nan_mask(self) -> npt.NDArray:
    with self._lock:
        return np.isnan(self._data).copy()  # 明确拷贝

# transform_standardize:
data_arr = np.asarray(data).copy()  # 明确拷贝
return (data_arr - mean) / std
```

---

## 2. 🟠 HIGH BUGS (24 项)

### 可重现性 / 算法

| # | 文件:行 | 问题 | 严重等级 |
|---|---------|------|----------|
| H-01 | statistics/cca.py:239, 370 | 负特征值静默替换为 1e-10，掩盖数值问题 | HIGH |
| H-02 | statistics/pcoa.py:165-191 | `eigenvalues_positive` 与返回的 `eigenvalues` 不一致：返回的可能是负值但坐标是从正的 sqrt 算的 | HIGH |
| H-03 | statistics/distance_metrics.py:188-189 | Bray-Curtis 大矩阵分母多余 `np.abs()` | MEDIUM |
| H-04 | statistics/lda.py:153 | 死代码 `data_clean.shape[0]` | LOW |
| H-05 | stratigraphy/coniss.py:162, 169 | `best_merge = None` 时被解引用 → TypeError | HIGH |
| H-06 | stratigraphy/spectral_analysis.py:351 | Lomb-Scargle 归一化 `power / variance` 应为 `power / (2*n)` | MEDIUM |
| H-07 | morpho3d/quaternion.py:469 | SLERP 线性近似应为完整 SLERP 公式 | LOW |
| H-08 | phylogenetics/heuristic_search.py:186-187 | NNI 未验证二叉性 | MEDIUM |
| H-09 | macroevolution/survival.py:248 | KM 中位生存时间查找逻辑错误 | MEDIUM |
| H-10 | macroevolution/survival.py:375-376 | LogRank 用 `dir()` 判断变量存在 | MEDIUM |
| H-11 | stratigraphy/arma.py:357 | ARMA 预测方差公式为经验公式 | MEDIUM |

### 线程安全 / 并发

| # | 文件:行 | 问题 | 严重等级 |
|---|---------|------|----------|
| H-12 | utils/decorators.py:130-135, 351-363 | memoize/cache_result 缓存 unhashable 结果时静默失败 | HIGH |
| H-13 | app_infrastructure/theme/manager.py:54 | 主题回调 list 永不清理，导致内存泄漏 | HIGH |
| H-14 | views/file_drop_handler.py | 单例 `getattr` 非线程安全 | MEDIUM |
| H-15 | hpc/task_scheduler.py:273-287 | 忙等 (busy-wait) 10ms 轮询 | MEDIUM |
| H-16 | hpc/process_pool.py:186 | 任务提交失败但未追加 None 到 results 列表 | MEDIUM |
| H-17 | state_machine/automaton.py | transition() 理论 race，应加 RLock | MEDIUM |

### 安全 / 资源

| # | 文件:行 | 问题 | 严重等级 |
|---|---------|------|----------|
| H-18 | reporting/report_builder.py:330-332 | output_path 无校验，潜在路径遍历 | HIGH |
| H-19 | reporting/figure_handler.py | LaTeX 注入 (无 escape) | MEDIUM |
| H-20 | reporting/table_generator.py | LaTeX 注入 (无 escape) | MEDIUM |
| H-21 | visualization/stratigraphy_plot.py:631-720 | DTW 大矩阵 O(n²) 内存 (10K 样本 → 800MB) | HIGH |
| H-22 | parsers/newick_parser.py:608-684 | 递归无深度限制，深度嵌套树 → RecursionError | MEDIUM |
| H-23 | parsers/dat_parser.py / newick_parser.py / tps_parser.py | 文件大小无限制 | MEDIUM |

### GUI / 内存

| # | 文件:行 | 问题 | 严重等级 |
|---|---------|------|----------|
| H-24 | views/ui_main_window.py:4881-4911 | closeEvent 不清理潜在的 QThread workers | MEDIUM |

---

## 3. 🟡 MEDIUM BUGS (30 项) 摘要

### 数据 / 模型层
- `models/state_manager.py:121` `get_instance` 跳过双重检查锁
- `models/state_manager.py:92` `_state_lock` 初始化但从未使用
- `models/state_manager.py:272-310` metadata setter 无索引边界检查
- `models/diversity_result.py:210-216` dataclass 存储可变 numpy 数组 (RarefactionResult)
- `models/column_metadata.py:495` 重复 label mapping 静默失败
- `controllers/data_controller.py:414-428` transpose 返回视图
- `controllers/data_controller.py:100` 空行标签默认为 ""
- `controllers/data_controller.py:106-116` `load_csv` row_label + 无 header 边缘情况
- `utils/validators.py:148-154` min_values 计数在 allow_nan=False 时不准确
- `utils/matrix_ops.py:275-276` stddev == 0 应使用容差比较

### 视图 / GUI
- `views/diagnostic_console.py:89` `QTextCursor.LineUnderCursor` 在 PyQt6 中已弃用，应为 `QTextCursor.SelectLine`
- `views/ui_main_window.py:4856` 访问双下划线 `_memory_label` (强耦合)
- `views/ui_main_window.py:2209-2232` `setDarkTheme` 异常被静默吞咽
- `views/ui_main_window.py:1253-1291` `_on_file_dropped` 非原子操作
- `views/floating_toolbar.py:36-41` SVG 资源路径 `:/icons/*.svg` 从未注册
- `plot_export.py:26-29` PIL 条件导入，失败时无早期警告
- `views/ui_plot_export_dialog.py:52-53` 硬编码栅格格式集合

### 报告 / 解析
- `parsers/tps_parser.py:116-117` 静默异常吞咽，跳过的行不计数
- `reporting/figure_handler.py` 路径用 `+` 拼接而非 `os.path.join`

### 其他
- `app_infrastructure/exception_handler.py` 多处旧 `phase5/` 路径字符串未清理
- `app_infrastructure/theme/manager.py` Styles 单例与 dark/light 状态可能不同步
- `morpho3d/sliding.py:443-444` 两行死代码
- `state_machine/automaton.py` to_dfa 的 `state_queue.pop(0)` 应改 `deque.popleft()`
- `stratigraphy/spectral_analysis.py:305` 孤立表达式 `len(time)`
- `morphometrics/efa.py` `from numpy import interp` 触发 DeprecationWarning

---

## 4. 🟢 LOW BUGS (21 项) 摘要

- `config/design_system.py:110-116` dataclass defaults 在实例化时计算字符串
- `parsers/lexer.py` regex 缺空字符串/边界处理
- `config/imputation.py:79-88` `MissingValueReport.summary()` 硬编码中文（违反 i18n）
- 多处使用 `plt.style.use("seaborn-v0_8-paper")` 需 matplotlib >= 3.6
- `views/diagnostic_console.py` 多处 `except Exception: pass` 静默
- `main.py:248` bare `except: pass`（旧问题，应是 `except Exception`）
- `main.py:146, 312, 320` UI 中使用 emoji (⚠️ 📁 ▶ ❌) — 科学软件不专业
- `app_infrastructure/exception_handler.py:243` 同上 emoji 使用
- `state_machine/automaton.py` `Token.__post_init__` 缺字段校验
- `config/i18n/_Translator` Qt 与纯 Python 后端切换不统一
- `views/ui_main_window.py` 硬编码窗口大小未适配高 DPI
- `plugins/registry.py:77` unregister 返回 True 即使已注销
- `morpho3d/quaternion.py:from_rotation_matrix` Shepperd 算法 4 case 需核对符号
- `phylogenetics/fitch.py:_fitch_up` 根节点处理时把节点自身当作父节点
- `ecology/diversity.py` 与 `macroevolution/diversity.py` 模块名冲突
- `utils/matrix_ops.bray_curtis` vs `statistics/distance_metrics.bray_curtis` 不一致（已部分修复）
- `parsers/*` 错误处理只 warn 不抛错
- `views/*` 多处对话框 `from models.state_manager import StateManager` 强耦合

---

## 5. 模块详细评分

### 5.1 Statistics (统计) - 7.0/10
**优点**: PCA、PCoA 算法实现扎实；SVD 数值稳定；迭代 NMDS 完整；Smith 优化算法正确
**缺点**:
- **可重现性危机**：5 个核心置换检验 (PERMANOVA, ANOSIM, Blomberg K, Phylo-ANOVA, Spatial envelope) 全无随机种子 → 论文结果无法复现
- PCoA 符号约定不一致（已修复但仍有瑕疵）
- LDA 用 sklearn 死代码
- ANOSIM/PERMANOVA 单尾/双尾未文档化

### 5.2 Ecology (生态) - 7.5/10
**优点**: BUGS.md 中 C-score、Baselga 分解、hypergeometric_prob 已修复
**缺点**:
- iNEXT CI 近似非标准
- abundance model 拟合数值稳定性不足
- 某些并行化路径在 BUG_FIX_SUMMARY.md 中已修复，需重新验证

### 5.3 Morphometrics (形态测量) - 7.0/10
**优点**: GPA, TPS, EFA, RWA 实现标准
**缺点**:
- **CRITICAL**: 质心大小公式错误，影响所有 allometry/PLS/整合分析
- OU alpha 估计对负自相关数据不稳定
- EFA `analyze` 用 x(t) 代替 delta_x/delta_t

### 5.4 Phylogenetics (系统发育) - 7.5/10
**优点**: Fitch、UPGMA 树构建标准；最近修复了 RI 公式
**缺点**:
- NNI/TBR 边界情况处理不完整
- NJ 收尾阶段不完整
- 严格共识在冲突时退化过早

### 5.5 Macroevolution (宏观进化) - 7.5/10
**优点**: cohort 颠倒已修复；FBD Gillespie 算法正确；diversity 存在性条件已修复
**缺点**:
- Kaplan-Meier 中位生存时间查找逻辑错误
- LogRank 变量作用域混乱
- 一些模块如 `fbd.py` 的 E(t) 闭式解注释需加强

### 5.6 Stratigraphy (地层) - 7.0/10
**优点**: Lomb-Scargle、马尔可夫链、FBD 似然基础算法正确
**缺点**:
- 频谱归一化偏离标准
- CONISS `best_merge = None` 防护缺失
- ARMA 预测方差为经验公式

### 5.7 Morpho3D - 7.5/10
**优点**: GPA3D、TPS3D、KDTree 实现标准
**缺点**: SLERP 近似、多处死代码

### 5.8 Models / Controllers - 6.5/10
**优点**: 单例模式合理；undo/redo 深拷贝；线程安全架构完整
**缺点**:
- **CRITICAL**: 视图/引用语义混乱导致数据污染
- metadata setter 无索引边界检查
- dataclass 存储可变 numpy 数组

### 5.9 GUI (Views) - 5.8/10
**优点**: 主题切换传播完整；i18n 集成好；`deleteLater()` 使用正确
**缺点**:
- **CRITICAL**: 所有分析任务阻塞主线程
- **CRITICAL**: QTextEdit 跨线程访问
- 大量裸 `except Exception: pass`
- bare `except:` 在 main.py 中仍存在
- emoji 在 UI 中多处使用（不专业）
- closeEvent 不清理 workers

### 5.10 Infrastructure - 6.0/10
**优点**: 事件总线基于 Qt signals 设计良好
**缺点**:
- **CRITICAL**: `binary_cache.py` HEADER_SIZE 与 struct 不匹配
- **CRITICAL**: 压缩路径 CRC 一致性破坏
- theme callback 永不清理造成内存泄漏
- decorators memoize 静默失败
- LaTeX 注入漏洞

### 5.11 Main / Config - 7.0/10
**优点**: 启动流程清晰；异常处理器设计完整
**缺点**:
- **CRITICAL**: requirements.txt 缺少 scikit-learn（导致 NMDS/LDA 不可用）
- bare except 与 emoji 仍存在

---

## 6. 验证结果

### 6.1 测试运行结果
执行 `python test_regression.py`:
```
PASS: phylogenetics import
PASS: parsers import
...
PASS: DataMatrix
============================================================
Results: 29 passed, 1 failed out of 30 tests
Failed tests:
  - NMDS: No module named 'sklearn'
============================================================
```

### 6.2 已修复 bug（验证 BUGS.md 描述的修复）

通过直接读取源代码验证了以下关键修复已正确实施：

| BUG | 修复位置 | 状态 |
|-----|---------|------|
| PCoA proportion 分母 | `statistics/pcoa.py:181-185` | ✅ 已修复 |
| NMDS isotonic regression | `statistics/nmds.py:205, 224-226` | ✅ 已修复 |
| CCA chi-square 因子 | 需进一步验证 | ⚠️ 待验证 |
| Hypergeometric prob | `ecology/beta_diversity.py:541-...` | ✅ 已修复 |
| C-score 公式 | `ecology/null_models.py:388-405` | ✅ 已修复 |
| Baselga 嵌套度 | `ecology/beta_diversity.py:296` | ✅ 已修复 |
| PCM v_A branch_length | `statistics/pcm.py:341-342, 373-374` | ✅ 已修复 |
| Binary cache metadata_offset | `parsers/binary_cache.py:...` | ⚠️ 仍有问题（CRITICAL-08） |
| Transform sqrt 静默改号 | `controllers/data_controller.py:382-408` | ✅ 已修复 |
| Cohort 颠倒 | `macroevolution/cohort.py:248-254, 313-319` | ✅ 已修复 |

---

## 7. 修复优先级建议

### P0 - 立即修复 (1-2 天)
1. **CRITICAL-01**: 添加 `scikit-learn>=1.3.0` 到 requirements.txt
2. **CRITICAL-02**: 修复 allometry.py 质心大小公式
3. **CRITICAL-03~06**: 为所有 5 个置换检验添加 `random_seed` 参数
4. **CRITICAL-07,08**: 修复 binary_cache.py 的 HEADER_SIZE 和 CRC
5. **CRITICAL-09**: 修复 diagnostic_console 跨线程访问（用 Qt signal）
6. **CRITICAL-10**: 长任务迁移到 QThread/QRunnable
7. **CRITICAL-12**: 修复视图/拷贝语义混乱

### P1 - 一周内 (3-7 天)
8. **CRITICAL-11**: QApplication.quit → exit
9. **H-01~11**: 算法改进（CCA 警告、PCoA 一致性、CONISS 空指针等）
10. **H-12~17**: 线程安全修复
11. **H-18~20**: 安全修复（路径遍历、LaTeX 注入）
12. **H-21~23**: 性能与资源限制

### P2 - 一个月内
13. 所有 MEDIUM bug
14. 代码清理（emoji、bare except、死代码）
15. 测试覆盖度提升

### P3 - 持续
16. LOW 级别的命名、文档、死代码清理

---

## 8. 项目级结论

### 优势
1. **核心统计/科学算法实现扎实**：Gillespie、PCA、SVD、GPA、TPS、Fitch、UPGMA 等基础算法均符合标准
2. **i18n 设计完整**：双语切换、信号通知、回滚机制
3. **单例 + RLock 架构**：并发读写控制基本合理
4. **历史 bug 修复彻底**：BUGS.md 列出的 17 项严重 bug 已基本修复（包括 C-score、Baselga、NMDS、PCoA、hypergeometric 等关键算法）

### 主要风险
1. **科学可重现性危机**：5 个核心置换检验无随机种子，导致论文结果无法复现 — 这是学术软件最严重的问题
2. **GUI 线程模型不安全**：长任务阻塞主线程、QTextEdit 跨线程访问
3. **核心算法错误**：质心大小公式错误直接影响异速生长分析
4. **二进制缓存破坏**：HEADER_SIZE 与 CRC 一