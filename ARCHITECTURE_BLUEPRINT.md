# PaleoAST 软件架构蓝图与核心数学算法设计白皮书

## 一、项目概述

### 1.1 项目背景与目标

PaleoAST (Paleontological Advanced Statistical Toolkit) 是一款专为古生物学、古生态学及生物地层学研究设计的工业级桌面数据分析平台。本软件整合了多元统计分析、几何形态测量学、生物多样性分析及时序频谱分析等核心功能，为古生物学家提供从原始数据导入到出版级可视化输出的完整工作流。

### 1.2 核心技术栈

| 层次 | 技术选型 | 说明 |
|------|----------|------|
| GUI框架 | PyQt6 / PySide6 | 现代化跨平台桌面应用框架 |
| 核心计算 | NumPy / SciPy | 高性能数值计算与科学算法 |
| 机器学习 | scikit-learn | 降维、聚类等高级分析 |
| 图论分析 | NetworkX | 生物地层学图论算法 |
| 数据可视化 | Matplotlib | 出版级图表渲染 |
| 交互图表 | mplcursors / HoverTool | 数据点悬停交互 |
| 编程语言 | Python 3.10+ | 严格类型提示与数据类 |

---

## 二、系统架构设计

### 2.1 MVC分层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        视图层 (Views)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ MainWindow  │  │ Spreadsheet │  │ PlotWidgets (MplCanvas) │  │
│  │  (Ribbon)   │  │ (QTableView)│  │ (Embedded Matplotlib)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      控制器层 (Controllers)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Data      │  │ Statistics  │  │     Plot                │  │
│  │ Controller  │  │ Controller  │  │     Controller          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                        模型层 (Models)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ DataMatrix  │  │ ColumnMeta  │  │     StateManager        │  │
│  │             │  │             │  │   (Thread-safe)         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                       引擎层 (Engines)                           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐              │
│  │  Statistics  │ │ Morphometrics│ │   Ecology    │              │
│  │    Engine    │ │   Engine    │ │   Engine    │              │
│  └──────────────┘ └──────────────┘ └──────────────┘              │
│  ┌──────────────┐ ┌──────────────┐                                │
│  │ Stratigraphy│ │  Visualization│                                │
│  │   Engine    │ │    Engine    │                                │
│  └──────────────┘ └──────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 线程安全状态管理

采用单例模式+读写锁实现全局状态管理：
- **DataMatrix**: 核心数据矩阵 (numpy.ndarray)
- **ColumnMetadata**: 列属性字典 {列索引: {type, group, color, marker}}
- **RowMetadata**: 行属性字典 {行索引: {label, color, marker}}
- **GlobalColorScheme**: 全局配色映射

---

## 三、核心数学算法设计

### 3.1 主成分分析 (PCA)

**数学原理**:

给定原始数据矩阵 $\mathbf{X} \in \mathbb{R}^{n \times p}$，其中 $n$ 为样本数，$p$ 为变量数。

**步骤1: 数据标准化**

基于方差-协方差矩阵:
$$\mathbf{Z} = \mathbf{X} - \bar{\mathbf{X}}$$

基于相关系数矩阵 (Z-score标准化):
$$z_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j}$$
其中 $\bar{x}_j = \frac{1}{n}\sum_{i=1}^{n}x_{ij}$，$s_j = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_{ij}-\bar{x}_j)^2}$

**步骤2: 协方差矩阵**

$$\mathbf{S} = \frac{1}{n-1}\mathbf{Z}^\top\mathbf{Z}$$

**步骤3: 特征值分解**

$$\mathbf{S}\mathbf{V} = \mathbf{V}\mathbf{\Lambda}$$

其中 $\mathbf{\Lambda} = \text{diag}(\lambda_1, \lambda_2, ..., \lambda_p)$ 为特征值对角矩阵，$\mathbf{V}$ 为特征向量矩阵。

**步骤4: 方差贡献率**

$$r_k = \frac{\lambda_k}{\sum_{i=1}^{p}\lambda_i} \times 100\%$$

累积贡献率:
$$\text{Cumulative}_k = \sum_{i=1}^{k}\frac{\lambda_i}{\sum_{j=1}^{p}\lambda_j} \times 100\%$$

**步骤5: 主成分得分**

$$\mathbf{PC} = \mathbf{Z}\mathbf{V}$$

### 3.2 主坐标分析 (PCoA)

**数学原理**:

给定距离矩阵 $\mathbf{D} = [d_{ij}]$，首先进行平方距离变换：

**步骤1: Gower中心化**

$$B = -\frac{1}{2}J\mathbf{D}^{(2)}J$$

其中 $\mathbf{J} = \mathbf{I} - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$ 为中心化矩阵，$\mathbf{D}^{(2)}$ 为元素平方距离矩阵。

**步骤2: 特征值分解**

$$\mathbf{B} = \mathbf{U}\mathbf{\Lambda}\mathbf{U}^\top$$

**步骤3: 坐标计算**

$$\text{PCoA coordinates} = \mathbf{U}\mathbf{\Lambda}^{1/2}$$

### 3.3 距离度量引擎

**欧氏距离 (Euclidean)**:
$$d_{Euclidean}(x,y) = \sqrt{\sum_{i=1}^{p}(x_i - y_i)^2}$$

**布雷-柯蒂斯距离 (Bray-Curtis)**:
$$d_{Bray-Curtis}(x,y) = \frac{\sum_{i=1}^{p}|x_i - y_i|}{\sum_{i=1}^{p}(x_i + y_i)}$$

**雅卡尔距离 (Jaccard)**:
$$d_{Jaccard}(x,y) = 1 - \frac{|x \cap y|}{|x \cup y|}$$

**曼哈顿距离 (Manhattan)**:
$$d_{Manhattan}(x,y) = \sum_{i=1}^{p}|x_i - y_i|$$

### 3.4 非度量多维尺度分析 (NMDS)

**目标函数 (Stress)**:

$$\text{Stress} = \sqrt{\frac{\sum_{i<j}(d_{ij} - \hat{d}_{ij})^2}{\sum_{i<j}d_{ij}^2}}$$

其中 $d_{ij}$ 为原始距离，$\hat{d}_{ij}$ 为降维后计算的距离。

**迭代优化**: 使用SMACOF算法，结合500+次随机重启避免局部最优。

### 3.5 ANOSIM (相似性分析)

**统计量R**:

$$R = \frac{\bar{r}_B - \bar{r}_W}{\frac{1}{2}n(n-1)}$$

其中 $\bar{r}_B$ 为组间相似性秩平均，$\bar{r}_W$ 为组内相似性秩平均，$n$ 为总样本数。

**显著性检验**: 通过9999次随机置换计算p值。

### 3.6 PERMANOVA

**检验统计量**:

$$F = \frac{SS_B/(g-1)}{SS_W/(n-g)}$$

其中 $SS_B$ 为组间平方和，$SS_W$ 为组内平方和，$g$ 为组数，$n$ 为总样本数。

**置换检验**: 9999次随机置换计算p值。

### 3.7 广义普氏分析 (GPA)

**Procrustes距离**:

$$d_{Procrustes}^2(X, Y) = \frac{1}{N}\sum_{i=1}^{N}\|x_i - y_i\|^2$$

**最优变换参数**:

给定目标构型 $\mathbf{X}$ 和参考构型 $\mathbf{Y}$:

1. **平移至质心**:
$$\mathbf{X}_{centered} = \mathbf{X} - \bar{\mathbf{X}}$$
$$\mathbf{Y}_{centered} = \mathbf{Y} - \bar{\mathbf{Y}}$$

2. **缩放至单位质心大小**:
$$\mathbf{X}_{scaled} = \frac{\mathbf{X}_{centered}}{C_s}$$
其中 $C_s = \sqrt{\text{trace}(\mathbf{X}_{centered}^\top\mathbf{X}_{centered})}$

3. **最优旋转** (通过SVD):
$$\mathbf{X}_{rotated} = \mathbf{U}\mathbf{V}^\top \mathbf{X}_{scaled}$$
其中 $\mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top = \mathbf{X}_{scaled}^\top \mathbf{Y}_{scaled}$ 的SVD分解。

### 3.8 薄板样条分析 (TPS)

**弯曲能量函数**:

$$E = \int\int_{\mathbb{R}^2}\left[\left(\frac{\partial^2 f}{\partial x^2}\right)^2 + 2\left(\frac{\partial^2 f}{\partial x \partial y}\right)^2 + \left(\frac{\partial^2 f}{\partial y^2}\right)^2\right]dxdy$$

**插值函数**:

$$f(x,y) = a_0 + a_1x + a_2y + \sum_{i=1}^{n}w_i U(r_i)$$

其中 $U(r) = r^2\log(r)$ 为径向基函数，$r_i = \sqrt{(x-x_i)^2 + (y-y_i)^2}$。

### 3.9 Alpha多样性指数

**物种丰富度 (S)**:
$$S = \sum_{i=1}^{N_{taxa}}1$$

**Simpson指数 (D)**:
$$D = 1 - \sum_{i=1}^{N_{taxa}}p_i^2$$

**Shannon指数 (H)**:
$$H = -\sum_{i=1}^{N_{taxa}}p_i \ln(p_i)$$
或
$$H' = -\sum_{i=1}^{N_{taxa}}p_i \log_2(p_i)$$

**Fisher Alpha**:
$$\alpha = \frac{N_{taxa} - 1}{\ln(1 - n/N)}$$

其中 $n$ 为个体总数，$N$ 为样本总数。

**Chao-1估计**:
$$\hat{S}_{Chao1} = S_{obs} + \frac{f_1^2}{2f_2}$$

其中 $f_1$ 为只出现1次的物种数，$f_2$ 为只出现2次的物种数。

### 3.10 Lomb-Scargle周期图

**周期图值**:

$$P_n(\omega) = \frac{1}{2}\left[\frac{\left(\sum_j y_j \cos\omega(t_j-\tau)\right)^2}{\sum_j \cos^2\omega(t_j-\tau)} + \frac{\left(\sum_j y_j \sin\omega(t_j-\tau)\right)^2}{\sum_j \sin^2\omega(t_j-\tau)}\right]$$

其中 $\tau$ 为时间偏移量，满足:
$$\tau = \frac{1}{2\omega}\arctan\frac{\sum_j \sin 2\omega t_j}{\sum_j \cos 2\omega t_j}$$

---

## 四、文件结构规划

```
PaleoAST/
├── main.py                          # 应用程序入口点
├── requirements.txt                 # 依赖包清单
├── README.md                        # 项目说明文档
│
├── config/                          # 配置模块
│   ├── __init__.py
│   ├── constants.py                 # 全局常量定义
│   ├── colors.py                    # 配色方案定义
│   └── validators.py                # 配置验证器
│
├── models/                          # 数据模型层
│   ├── __init__.py
│   ├── data_matrix.py               # 核心数据矩阵类
│   ├── column_metadata.py           # 列元数据类
│   ├── row_metadata.py              # 行元数据类
│   ├── diversity_result.py          # 多样性分析结果
│   └── state_manager.py             # 线程安全状态管理器
│
├── views/                            # 视图层 (GUI)
│   ├── __init__.py
│   ├── main_window.py               # 主窗口类
│   ├── ribbon_bar.py                # 功能区菜单栏
│   ├── navigation_tree.py           # 左侧导航树
│   ├── spreadsheet.py               # 科学电子表格
│   ├── spreadsheet_model.py         # 表格数据模型
│   ├── plot_widgets.py              # 图表组件
│   └── dialogs.py                   # 对话框
│
├── controllers/                     # 控制器层
│   ├── __init__.py
│   ├── data_controller.py           # 数据操作控制器
│   ├── statistics_controller.py     # 统计分析控制器
│   ├── morphometrics_controller.py  # 形态测量控制器
│   ├── ecology_controller.py        # 生态分析控制器
│   ├── stratigraphy_controller.py   # 地层分析控制器
│   └── plot_controller.py           # 绘图控制器
│
├── statistics/                       # 统计分析引擎
│   ├── __init__.py
│   ├── pca.py                       # 主成分分析
│   ├── pcoa.py                      # 主坐标分析
│   ├── nmds.py                      # 非度量MDS
│   ├── anosim.py                    # ANOSIM分析
│   ├── permanova.py                 # PERMANOVA分析
│   ├── distance_metrics.py          # 距离度量计算
│   ├── factor_analysis.py           # 因子分析
│   ├── cluster_analysis.py          # 聚类分析
│   └── manova.py                    # 多元方差分析
│
├── morphometrics/                    # 几何形态测量学引擎
│   ├── __init__.py
│   ├── gpa.py                       # 广义普氏分析
│   ├── tps.py                       # 薄板样条分析
│   ├── relative_warps.py            # 相对扭曲分析
│   ├── landmarks.py                 # 标志点数据处理
│   └── visualization.py             # 形态测量可视化
│
├── ecology/                          # 古生态学引擎
│   ├── __init__.py
│   ├── diversity.py                  # Alpha多样性指数
│   ├── rarefaction.py               # 稀疏化曲线
│   ├── similarity.py                # 相似性指数
│   └── beta_diversity.py            # Beta多样性
│
├── stratigraphy/                     # 生物地层学引擎
│   ├── __init__.py
│   ├── unitary_associations.py       # 单一组合方法
│   ├── spectral_analysis.py          # 频谱分析(Lomb-Scargle)
│   ├── time_series.py               # 时间序列处理
│   └── confidence.py                # 置信区间计算
│
├── visualization/                    # 可视化渲染引擎
│   ├── __init__.py
│   ├── base_plot.py                 # 基础绘图类
│   ├── pca_plot.py                  # PCA专用绘图
│   ├── pcoa_plot.py                 # PCoA专用绘图
│   ├── nmds_plot.py                 # NMDS专用绘图
│   ├── diversity_plot.py            # 多样性曲线绘图
│   ├── tps_grid_plot.py             # TPS网格绘图
│   ├── spectral_plot.py             # 频谱图绘图
│   ├── style.py                     # 出版级样式配置
│   └── export.py                    # 导出功能
│
├── utils/                            # 工具模块
│   ├── __init__.py
│   ├── matrix_ops.py                # 矩阵运算工具
│   ├── validators.py                # 数据验证器
│   ├── exceptions.py                # 自定义异常类
│   ├── decorators.py                # 装饰器
│   └── parallel.py                  # 并行计算工具
│
└── tests/                            # 测试模块
    ├── __init__.py
    ├── test_pca.py
    ├── test_distance.py
    ├── test_diversity.py
    └── test_gpa.py
```

---

## 五、关键设计决策

### 5.1 数据结构选择

- **DataMatrix**: 使用 NumPy ndarray 的子类，支持 masked array 处理缺失值
- **Metadata**: 使用 dataclasses.frozen 实现不可变元数据
- **StateManager**: 使用 threading.RLock 实现读写锁

### 5.2 异常处理策略

所有数学运算中的异常（除以零、矩阵不可逆、数值不稳定等）必须：
1. 捕获具体异常类型
2. 记录详细错误堆栈
3. 通过GUI弹出格式化警告窗口
4. 返回NaN或None而非崩溃

### 5.3 性能优化策略

- 电子表格使用虚拟渲染(QAbstractTableModel)
- 大矩阵运算使用NumPy向量化
- 可并行计算使用multiprocessing
- 图表使用缓存机制避免重复渲染

---

## 六、UI/UX设计规范

### 6.1 配色方案

**主色调**:
- Primary: #2C3E50 (深蓝灰)
- Secondary: #3498DB (亮蓝)
- Accent: #E74C3C (珊瑚红)

**图表配色** (色弱友好):
- Category 1: #0077BB
- Category 2: #EE7733
- Category 3: #009988
- Category 4: #CC3311
- Category 5: #33BBEE
- Category 6: #EE3377

### 6.2 字体规范

**正文字体**: Arial, Helvetica, sans-serif
**代码/数字**: Consolas, Monaco, monospace
**图表字体**: Helvetica (Nature/Science标准)

### 6.3 布局规范

- 主窗口: 左侧导航(250px) + 右侧工作区(弹性)
- 功能区: Ribbon风格，图标+文字
- 状态栏: 显示当前数据集信息、内存使用

---

## 七、结论

PaleoAST平台采用严格的MVC架构，通过深度封装NumPy、SciPy、Matplotlib等科学计算库，实现了古生物学研究的全流程支持。核心算法均基于严谨的数学公式实现，确保结果的科学准确性。零省略的代码规范和详尽的LaTeX数学推导注释，使本项目具备极高的学术价值和工程可维护性。
