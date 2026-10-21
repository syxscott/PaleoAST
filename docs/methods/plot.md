# 绘图方法

绘图菜单的二十个入口共用一条原则：**图是数据的另一种编码，不是数据的装饰。**
这一域的方法验证因此不看"像素像不像"，而看**图背后的数据变换**与参考实现是否一致。

## 通用假设

| 编号 | 假设 | 说明 |
|---|---|---|
| **A1** | 轴标签写明单位 | "Size" 与 "Shell length (mm)" 是两张不同的图 |
| **A2** | 类别顺序是有意义的 | 排序改变了柱高顺序，但读者会读出"趋势" |
| **A3** | 变换（对数、标准化）写进标题或轴标签 | 对数轴上的等距 ≠ 等量 |
| **A4** | 缺失值被显示，不是被悄悄丢掉 | 一条缺 30% 的序列和不缺的是两回事 |

## 逐方法的注意事项

- **`percentiles`（百分位图）是箱线图去掉离群点**，
  **"离群"是按 1.5×IQR 定义的**，不是"极端值"。
- **`normal_probability_plot`（Q-Q 图）**的**参考线是斜率为 1、截距 0**，
  点偏离直线的形状说明了偏度还是厚尾。
- **`box_plot` 的箱体是四分位距**，**须线默认伸到 1.5×IQR 内的最远点**，
  而不是 min/max——**离群点是单独画出来的点**。
- **`histogram` 的分箱规则**（等宽/等频/Sturges/Freedman-Diaconis）会改变图的形状。
  **R 的 `hist()` 默认是 Sturges，`numpy` 默认是等宽**——两者对同一份数据给的图不同。
- **`stacked_chart` 的堆叠顺序**决定了每一条带的起点；
  **百分比堆叠**与**绝对值堆叠**的读法完全不同。
- **`mosaic_plot` 的面积是频数**（不是百分比除非另行归一化）。
- **`xy_error_bars` 的误差棒含义**（标准差/标准误/置信区间）必须在图注里写明。
- **`surface_plot` 与 `plot_3d` 的 z 轴**是投影，**不是数据的三维正交视图**；
  读透视关系时容易看错。

## 与 R 的对照方式

绘图本身无法逐像素与 R 对比（渲染器、字体、抗锯齿都不同）。
**能对比的是图背后的数据变换**：

| 图 | 可与 R 比对的东西 | R 的函数 |
|---|---|---|
| `percentiles` | 分位数点 | `quantile(type = 7)` |
| `histogram` | 分箱边界与每箱计数 | `hist(breaks = ...)` |
| `box_plot` | 箱线图五数概括 | `boxplot.stats` |
| `normal_probability_plot` | QQ 分位数 | `qqnorm` |
| `pie_chart` | 各扇区份额 | `pie` |
| `stacked_chart` | 堆叠后的矩阵 | `barplot` / `matrix` |
| `matrix_plot` | 归一化后的矩阵 | `image` / `heatmap` |
| `xy_plot` | 拟合线与坐标变换 | `lm` / `plot` |

## 参考

- Tufte, E. R. (2001). *The Visual Display of Quantitative
  Information.* 2nd ed. Graphics Press.
- Cleveland, W. S. (1993). *Visualizing Data.* Addison-Wesley.
- Wainer, H. (1997). *Visual Revelations: An Introduction to
  Visualizing Categorical, Quantitative Data, and Statistical
  Patterns.* Springer.
- Tukey, J. W. (1977). *Exploratory Data Analysis.* Addison-Wesley.
- R Core Team. (2023). *R: A Language and Environment for Statistical
  Computing.* R Foundation for Statistical Computing.
- Hunter, J. D. (2007). *Matplotlib: A 2D graphics environment.*
  Computing in Science & Engineering 9, 90–95.
- Wilkinson, L., et al. (2016). *ggplot2: Elegant Graphics for Data
  Analysis.* 2nd ed. Springer.
- Chambers, J. M., Cleveland, W. S., Cleveland, W. M., et al. (1983).
  *Graphical Methods for Data Analysis.* Bell Laboratories.
- Anscombe, F. J. (1973). *Graphs in Statistical Analysis.*
  Academic Press.
- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles
  and Practice.* 3rd ed. OTexts.

## 分析清单

<!-- matrix:start -->
| 分析 | 模块 | 验证证据 | 说明 |
|---|---|---|---|
| `bar_chart` | `visualization.plot_menu` | 单元 | Bar chart of a column, or sums per category |
| `box_plot` | `visualization.plot_menu` | 单元 | Box plot of one column per group |
| `bubble_plot` | `visualization.plot_menu` | 单元 | Scatter with a third variable as point area |
| `histogram` | `visualization.plot_menu` | 单元 | Histogram of one column |
| `matrix_plot` | `visualization.plot_menu` | 单元 | Matrix plot of a 2-D array |
| `mosaic_plot` | `visualization.plot_menu` | 单元 | Mosaic plot of a contingency table |
| `network_plot` | `visualization.plot_menu` | 单元 | Network plot from an edge list |
| `normal_probability_plot` | `visualization.plot_menu` | 单元 | Normal probability (Q-Q) plot |
| `percentiles` | `visualization.plot_menu` | 单元 | Percentile plot: a box plot without outliers |
| `pie_chart` | `visualization.plot_menu` | 单元 | Pie chart of a column's values or labels |
| `plot_3d` | `visualization.plot_menu` | 单元 | 3-D scatter, line or bubble plot |
| `polar_plot` | `visualization.plot_menu` | 单元 | Polar plot of one column |
| `radar_chart` | `visualization.plot_menu` | 单元 | Radar chart: one polygon per matrix row |
| `stacked_chart` | `visualization.plot_menu` | 单元 | Stacked bars over the rows of a matrix |
| `surface_plot` | `visualization.plot_menu` | 单元 | 3-D surface over a grid |
| `ternary_plot` | `visualization.plot_menu` | 单元 | Ternary plot of three components |
| `vector_plot` | `visualization.plot_menu` | 单元 | Vector (quiver) field |
| `venn_diagram` | `visualization.plot_menu` | 单元 | Venn diagram for two or three sets |
| `xy_error_bars` | `visualization.plot_menu` | 单元 | XY scatter with error bars |
| `xy_plot` | `visualization.plot_menu` | 单元 | XY scatter with an optional fitted line and log axes |
<!-- matrix:end -->
