<div align="center">

# PaleoAST

### Paleontological Advanced Statistical Toolkit / 古生物学高级统计分析工具包

![Version](https://img.shields.io/badge/version-1.0.1-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
![GUI](https://img.shields.io/badge/GUI-PyQt6-orange)
![License](https://img.shields.io/badge/license-MIT-yellow)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

**Professional desktop platform for paleontological, paleoecological, and biosratigraphic research.**

**专业古生物学桌面数据分析平台，集成多元统计、形态测量、生态分析与系统发育功能。**

### User-Friendly Error Handling / 用户友好的错误处理

- **Chinese Error Messages** — All error messages displayed in Chinese with actionable guidance
  中文错误提示 — 所有错误信息以中文显示，提供可操作的指导
- **Foolproof Design** — Validates input data before analysis, preventing crashes with helpful suggestions
  防呆设计 — 在分析前验证输入数据，通过友好建议防止崩溃
- **Error Classification** — Automatic error type detection (invalid characters, numeric errors, dimension mismatch, empty data)
  错误分类 — 自动检测错误类型（无效字符、数值错误、维度不匹配、空数据）

[English](#-quick-start) | [中文](#-快速开始)

</div>

---

## Table of Contents / 目录

- [Features / 功能特性](#-features--功能特性)
- [Quick Start / 快速开始](#-quick-start--快速开始)
- [Operation Guide / 操作指南](#-operation-guide--操作指南)
- [Module Overview / 模块概览](#-module-overview--模块概览)
- [Troubleshooting / 常见问题](#-troubleshooting--常见问题)
- [Development / 开发指南](#-development--开发指南)
- [License / 许可证](#-license--许可证)

---

## Features / 功能特性

### Multivariate Statistics / 多元统计分析

| Feature | Description |
|---------|-------------|
| **PCA** (Principal Component Analysis) | 主成分分析 — 降维、方差解释、双标图 |
| **PCoA** (Principal Coordinate Analysis) | 主坐标分析 — 基于距离矩阵的排序 |
| **NMDS** (Non-metric MDS) | 非度量多维标度 — SMACOF 优化，自动 stress 检验 |
| **ANOSIM** | 组间相似性分析 — 基于秩的置换检验 |
| **PERMANOVA** | 多元方差分析 — 基于距离的置换检验 |

### Ecology / 生态分析

| Feature | Description |
|---------|-------------|
| **Diversity Indices** | 多样性指数 — Shannon, Simpson, Chao1, Fisher's Alpha |
| **Rarefaction** | 稀疏化分析 — 标准化采样努力量 |
| **Spectral Analysis** | 频谱分析 — Lomb-Scargle 周期图 |

### Morphometrics / 形态测量

| Feature | Description |
|---------|-------------|
| **2D/3D GPA** | 普氏分析 — 形状对齐与叠加 |
| **TPS** (Thin-Plate Spline) | 薄板样条 — 形变可视化与弯曲能量 |
| **RWA** (Relative Warps Analysis) | 相对扭曲分析 — 局部与整体形状变化 |

### Phylogenetics / 系统发育

| Feature | Description |
|---------|-------------|
| **UPGMA / NJ** | 距离法建树 — 基于距离矩阵 |
| **Fitch Parsimony** | 最简约法 — 祖先状态重建 |
| **Consensus Trees** | 共识树 — 严格共识拓扑 |
| **PCM** | 系统发育比较方法 — PIC、ASR、Blomberg's K、Phylo-ANOVA |

### Macroevolution / 宏观进化

| Feature | Description |
|---------|-------------|
| **FBD Process** | 化石生灭过程 — Gillespie 随机模拟 |
| **Cohort Survivorship** | 群组存活曲线 — 灭绝动态建模 |
| **Diversity Dynamics** | 多样性动态 — 指数/逻辑斯蒂增长模型 |
| **Extinction Intervals** | 灭绝置信区间 — Marshall & Strauss-Sadler 方法 |
| **Evolution Rate** | 演化速率分析 — BM/Directional/OU 模型与 AIC 选择 |

### Ecology / 生态分析 (Extended)

| Feature | Description |
|---------|-------------|
| **Diversity Indices** | 多样性指数 — Shannon, Simpson, Chao1, Fisher's Alpha |
| **Rarefaction** | 稀疏化分析 — 标准化采样努力量 |
| **Spectral Analysis** | 频谱分析 — Lomb-Scargle 周期图 |
| **Coverage Rarefaction** | 覆盖度稀疏化 — iNEXT 样本覆盖度估计 |
| **Beta Diversity** | Beta 多样性分解 — Jaccard/Sørensen 拆分为 turnover 与 nestedness |
| **Null Models** | 零模型共现分析 — C-score 与 Swap 算法 |

### Morphometrics / 形态测量 (Extended)

| Feature | Description |
|---------|-------------|
| **2D/3D GPA** | 普氏分析 — 形状对齐与叠加 |
| **TPS** (Thin-Plate Spline) | 薄板样条 — 形变可视化与弯曲能量 |
| **RWA** (Relative Warps Analysis) | 相对扭曲分析 — 局部与整体形状变化 |
| **Allometry** | 异速生长分析 — 尺寸-形状多元回归 |
| **2B-PLS** | 形态整合分析 — 两块偏最小二乘法 |

---

## Quick Start / 快速开始

### Prerequisites / 环境要求

- **Python** 3.10 或更高版本
- **操作系统** Windows 10+, macOS 12+, Ubuntu 20.04+

### Step 1: Clone the repository / 克隆仓库

```bash
git clone https://github.com/syxscott/PaleoAST.git
cd PaleoAST
```

### Step 2: Create virtual environment (recommended) / 创建虚拟环境（推荐）

```bash
# Using conda / 使用 conda
conda create -n paleoast python=3.13
conda activate paleoast

# Or using venv / 或使用 venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install dependencies / 安装依赖

```bash
pip install -r requirements.txt
```

> **What gets installed / 安装内容:**
> `numpy`, `scipy`, `pandas`, `matplotlib`, `PyQt6`, `psutil`

### Step 4: Launch / 启动应用

```bash
python main.py
```

The application will show a splash screen, then open the main window.

应用将显示启动画面，随后打开主窗口。

---

## Operation Guide / 操作指南

### Interface Overview / 界面总览

The main window has three areas:

```
+-----------------------------------------------------------+
|  [File] [Multivariate] [Ecology] [Morpho] [Phylo] [Evo]  |  <- Ribbon toolbar
+----------+------------------------------------------------+
|          |                                                |
| Nav Tree |          Workspace / 工作区                     |
| 导航树   |     (Spreadsheet / Plot / Results)              |
|          |                                                |
|          |                                                |
+----------+------------------------------------------------+
|  Status Bar / 状态栏                                       |
+-----------------------------------------------------------+
```

- **Left panel** — Navigation tree with all modules
- **Top ribbon** — Quick-access buttons grouped by category
- **Center** — Data spreadsheet or analysis plots
- **Bottom** — Status bar with progress and info

---

### 1. Import Data / 导入数据

There are three ways to load data into PaleoAST:

#### Method A: Open CSV file / 打开 CSV 文件

1. Click **Open** button in the top-left of the ribbon
   点击功能区左上角的 **Open** 按钮
2. Select a `.csv` or `.txt` file
   选择 `.csv` 或 `.txt` 文件
3. Data loads into the spreadsheet automatically
   数据自动加载到电子表格中

#### Method B: Create sample data / 创建示例数据

1. Click **New** button in the ribbon
   点击功能区的 **New** 按钮
2. Set number of samples and variables in the dialog
   在对话框中设置样本数和变量数
3. Random data is generated for testing
   自动生成随机测试数据

#### Method C: Import dialog / 导入对话框

1. In the left nav tree, click **Data Management** -> **Import Data**
   在左侧导航树中，点击 **数据管理** -> **导入数据**
2. Browse for your file, configure options (header, row labels, delimiter)
   浏览文件，配置选项（表头、行标签、分隔符）
3. Preview the data, then click **Import**
   预览数据，然后点击 **导入**

> **Supported formats / 支持格式:**
> - CSV (comma-separated)
> - TSV (tab-separated)
> - TXT (space-separated)
> - Excel (.xlsx) — requires `openpyxl`

> **Data format tip / 数据格式提示:**
> - First row = column headers (variable names)
>   第一行 = 列标题（变量名）
> - First column = row labels (sample names)
>   第一行 = 行标签（样本名）
> - All other cells = numeric values
>   其他所有单元格 = 数值

---

### 2. Run PCA Analysis / 运行主成分分析

PCA reduces high-dimensional data to principal components for visualization.

PCA 将高维数据降维到主成分以便可视化。

**Steps / 步骤:**

1. **Load data first** — Use any method above
   先加载数据 — 使用上述任一方法
2. Click **PCA** in the Multivariate section of the ribbon
   点击功能区 Multivariate 区域的 **PCA**
3. Configure in the dialog:
   在对话框中配置：
   - **Analysis Method**: Correlation (standardized) or Covariance (centered)
     分析方法：相关矩阵（标准化）或协方差矩阵（中心化）
   - **Components**: Number of PCs to compute (default: 3)
     成分数：要计算的主成分数（默认 3）
   - **Display Options**: Check what to show (loadings, scores, scree plot, biplot)
     显示选项：勾选要显示的内容（载荷表、得分表、碎石图、双标图）
4. Click **Run** — Plots appear in the workspace
   点击 **Run** — 图表显示在工作区中

**Reading the results / 解读结果:**

- **Scree plot**: Shows variance explained by each PC. Look for the "elbow" to decide how many PCs to keep.
  碎石图：显示每个主成分解释的方差。寻找"肘部"来决定保留多少个主成分。
- **Scores plot**: Each point is a sample. Samples close together are similar.
  得分图：每个点是一个样本。靠近的样本相似。
- **Biplot**: Shows both samples (points) and variables (arrows). Arrows indicate variable directions.
  双标图：同时显示样本（点）和变量（箭头）。箭头指示变量方向。

---

### 3. Run PCoA / NMDS / 运行主坐标分析 / 非度量多维标度

These ordination methods work on distance/dissimilarity matrices.

这些排序方法基于距离/相异度矩阵。

**Steps / 步骤:**

1. Load data (samples in rows, variables in columns)
   加载数据（行为样本，列为变量）
2. Click **PCoA** or **NMDS** in the ribbon
   点击功能区的 **PCoA** 或 **NMDS**
3. Choose distance metric:
   选择距离度量：
   - **Bray-Curtis** — Best for abundance data / 最适合丰度数据
   - **Jaccard** — Presence/absence data / 有/无数据
   - **Euclidean** — Continuous measurements / 连续测量值
4. Click **Run** to generate the ordination plot
   点击 **Run** 生成排序图

> **NMDS stress / NMDS 压力值:**
> - < 0.05: Excellent / 优秀
> - < 0.10: Good / 良好
> - < 0.15: Acceptable / 可接受
> - > 0.20: Poor, try more dimensions / 差，尝试更多维度

---

### 4. Run ANOSIM / PERMANOVA / 运行组间差异检验

These tests check whether groups of samples are significantly different.

这些检验判断样本组之间是否有显著差异。

**Steps / 步骤:**

1. Load data and define groups (groups can be in a separate column or assigned manually)
   加载数据并定义组（组可以在单独列中或手动分配）
2. Click **ANOSIM** or **PERMANOVA** in the ribbon
   点击功能区的 **ANOSIM** 或 **PERMANOVA**
3. Set the number of permutations (default: 9999, higher = more accurate p-value)
   设置置换次数（默认 9999，越高 p 值越准确）
4. Click **Run** — Results show R statistic (ANOSIM) or F statistic (PERMANOVA) with p-value
   点击 **Run** — 结果显示 R 统计量（ANOSIM）或 F 统计量（PERMANOVA）及 p 值

---

### 5. Biodiversity Analysis / 生物多样性分析

**Steps / 步骤:**

1. Load abundance data (species counts per sample)
   加载丰度数据（每个样本的物种计数）
2. In the left nav tree, click **Ecology** -> **Diversity**
   在左侧导航树中，点击 **生态分析** -> **多样性分析**
3. Select which indices to compute:
   选择要计算的指数：
   - **Shannon H'** — Information entropy / 信息熵
   - **Simpson 1-D** — Dominance measure / 优势度
   - **Chao1** — Richness estimator / 丰富度估计
   - **Fisher's α** — Fitting parameter / 拟合参数
4. Click **Run** — Results displayed as bar charts or radar plots
   点击 **Run** — 结果以柱状图或雷达图显示

---

### 6. Rarefaction Analysis / 稀疏化分析

Rarefaction standardizes samples to compare diversity at equal sampling effort.

稀疏化将样本标准化到相同的采样努力量以比较多样性。

**Steps / 步骤:**

1. Load abundance data
   加载丰度数据
2. In the left nav tree, click **Ecology** -> **Rarefaction**
   在左侧导航树中，点击 **生态分析** -> **稀疏化分析**
3. Select samples to compare
   选择要比较的样本
4. Set maximum individuals and step size
   设置最大个体数和步长
5. Click **Run** — Rarefaction curves appear
   点击 **Run** — 稀疏化曲线显示

> **Interpretation / 解读:**
> Curves that plateau indicate adequate sampling. Curves still rising suggest more sampling is needed.
> 曲线趋于平坦表示采样充分。仍在上升的曲线表示需要更多采样。

---

### 7. Spectral Analysis / 频谱分析

Lomb-Scargle periodogram for unevenly sampled time series (e.g., geological sections).

Lomb-Scargle 周期图用于非均匀采样时间序列（如地质剖面）。

**Steps / 步骤:**

1. Load time-series data (first column = time/depth, second column = values)
   加载时间序列数据（第一列 = 时间/深度，第二列 = 数值）
2. In the left nav tree, click **Stratigraphy** -> **Spectral Analysis**
   在左侧导航树中，点击 **地层学** -> **频谱分析**
3. Click **Run** — Power spectrum with peak frequency highlighted
   点击 **Run** — 显示功率谱并高亮峰值频率

---

### 8. Saving Results / 保存结果

- **File -> Save** — Save current workspace data to CSV
  文件 -> 保存 — 将当前工作区数据保存为 CSV
- **Right-click on plot** — Save plot as PNG/SVG/PDF
  右键点击图表 — 将图表保存为 PNG/SVG/PDF
- **Export Log** — If an error occurs, export the error log for debugging
  导出日志 — 如果出错，导出错误日志用于调试

---

### Keyboard Shortcuts / 快捷键

| Shortcut | Action |
|----------|--------|
| `Ctrl+N` | New file / 新建 |
| `Ctrl+O` | Open file / 打开 |
| `Ctrl+S` | Save / 保存 |
| `Ctrl+I` | Import data / 导入数据 |
| `Ctrl+Shift+E` | Export data / 导出数据 |
| `Ctrl+Shift+P` | Run PCA / 运行主成分分析 |
| `Ctrl+Shift+C` | Run PCoA / 运行主坐标分析 |
| `Ctrl+Shift+M` | Run NMDS / 运行非度量MDS |
| `Ctrl+Shift+D` | Run Diversity / 运行多样性分析 |
| `Ctrl+Shift+R` | Run Rarefaction / 运行稀疏化分析 |
| `Ctrl+Shift+S` | Run Spectral / 运行频谱分析 |

---

## Module Overview / 模块概览

```
PaleoAST/
├── config/                 # Configuration & i18n / 配置与国际化
│   ├── colors.py           # Color palette / 调色板
│   ├── constants.py        # App constants / 应用常量
│   ├── design_system.py    # UI design tokens / 设计系统
│   └── i18n/               # Translations (EN/ZH) / 翻译
├── models/                 # Core data models / 核心数据模型
│   ├── state_manager.py    # Singleton state / 全局状态管理
│   └── data_matrix.py      # Data container / 数据矩阵容器
├── controllers/            # Business logic / 业务逻辑
│   ├── data_controller.py  # Data I/O / 数据读写
│   └── statistics_controller.py  # Analysis dispatch / 分析调度
├── views/                  # UI components / 界面组件
│   ├── ui_main_window.py   # Main window / 主窗口
│   ├── ui_dialogs.py       # Analysis dialogs / 分析对话框
│   ├── ui_plot_canvas.py   # Matplotlib plots / 图表画布
│   ├── ui_navigation.py    # Left nav tree / 左侧导航树
│   └── ui_spreadsheet.py   # Data spreadsheet / 数据表格
├── statistics/             # Statistical engines / 统计引擎
├── ecology/                # Ecology algorithms / 生态算法
├── morphometrics/          # 2D morphometrics / 二维形态测量
├── morpho3d/               # 3D morphometrics / 三维形态测量
├── phylogenetics/          # Phylogenetic trees / 系统发育树
├── macroevolution/         # Macroevolution models / 宏观进化模型
├── stratigraphy/           # Spectral & strat analysis / 频谱与地层分析
├── visualization/          # Plotting utilities / 绘图工具
├── utils/                  # Shared utilities / 通用工具
├── main.py                 # Application entry / 应用入口
├── requirements.txt        # Dependencies / 依赖列表
└── pyproject.toml          # Project config / 项目配置
```

### Technical Stack / 技术栈

| Layer | Technology |
|-------|------------|
| GUI | PyQt6 with custom vector icon engine |
| Computing | NumPy, SciPy, scikit-learn |
| Plotting | Matplotlib integrated via FigureCanvasQTAgg |
| Architecture | MVC pattern with singleton StateManager |
| i18n | Chinese/English bilingual with runtime switching |

---

## Troubleshooting / 常见问题

### DLL load failed (Windows) / DLL 加载失败

```
DLL load failed while importing QtWidgets
```

**Solution / 解决方案:**

1. Install [Visual C++ Redistributable (2015-2022)](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
2. Or reinstall PyQt6: `pip install PyQt6 --force-reinstall`

### Data not loading / 数据无法加载

**Check / 检查:**
- File encoding should be UTF-8
  文件编码应为 UTF-8
- First row should contain column names
  第一行应包含列名
- All data cells must be numeric (no text in data columns)
  所有数据单元格必须是数值（数据列中不能有文本）

### Plots show empty / 图表显示空白

**Check / 检查:**
- Did you load data first?
  是否先加载了数据？
- Check the status bar for error messages
  检查状态栏是否有错误信息
- Try clicking the analysis button again after loading data
  加载数据后尝试再次点击分析按钮

### Chinese characters not showing in plots / 图表中不显示中文

PaleoAST automatically configures Chinese fonts (Microsoft YaHei / SimHei). If Chinese still doesn't show:
PaleoAST 自动配置中文字体（微软雅黑/黑体）。如果中文仍不显示：

1. Ensure Microsoft YaHei or SimHei is installed on your system
   确保系统安装了微软雅黑或黑体字体
2. Restart the application after font installation
   安装字体后重启应用

### Application crashes on startup / 启动时崩溃

1. Check the log file at `~/.paleoast/logs/`
   检查 `~/.paleoast/logs/` 下的日志文件
2. Run from terminal to see error output: `python main.py`
   从终端运行以查看错误输出
3. Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
   重新安装依赖

---

## Development / 开发指南

### Run tests / 运行测试

```bash
python test_regression.py
```

### Run with pytest / 使用 pytest

```bash
pip install pytest pytest-cov
pytest -v
```

### Code linting / 代码检查

```bash
pip install ruff
ruff check .
ruff format .
```

### Type checking / 类型检查

```bash
pip install mypy
mypy .
```

---

## License / 许可证

MIT License. See [LICENSE](LICENSE) for details.

MIT 许可证。详情见 [LICENSE](LICENSE)。

---

<div align="center">

**Made with care for the paleontological community.**

**为古生物学研究社区精心打造。**

</div>
