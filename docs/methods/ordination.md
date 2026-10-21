# 排序与空间

排序（ordination）把"多变量数据 + 结构"压成一张图；空间统计量描述"点在哪里"。
这一域的共同点是：**它们读的都是距离矩阵或相似度矩阵，而距离矩阵的选择会改变一切结论。**

## 通用假设

| 编号 | 假设 | 说明 |
|---|---|---|
| **A1** | 距离度量与研究问题相符 | 计数用 Jaccard / Bray-Curtis，连续测量用 Euclidean；**混用会得出完全不同的图** |
| **A2** | 行是样本、列是变量，且已标准化 | 未标准化时 PCA 被量纲最大的变量主导 |
| **A3** | 距离矩阵满足三角不等式 | 非度量方法（NMDS）对此敏感；`Metric MDS` 则要求它成立 |
| **A4** | 点之间独立 | 空间数据通常**不**独立，这正是 Moran's I 存在的原因 |

## 距离度量的选择（这一域最重要的一个决定）

| 数据类型 | 推荐度量 | 不该用的 |
|---|---|---|
| 物种丰度（计数） | Bray-Curtis、Jaccard、**Hellinger** | Euclidean（计数当连续量） |
| 连续测量 | Euclidean（标准化后）、Correlation distance | Manhattan（维度灾难） |
| 二元存在/缺失 | Jaccard、Sørensen | Bray-Curtis（对 0-1 数据信息量低） |

**Hellinger（平方根变换后再算欧氏）值得单独提**：它把计数拉向高斯，PCA 的结果
会稳定很多，在生态学里是标准做法。

## 逐方法的注意事项

- **`pca` 的两种算法**：`method="covariance"` 与 `method="correlation"`
  差在是否标准化变量。**尺度差异大的变量（体长 vs 骨骼比例）必须用 correlation**，
  否则 PCA 就是在画体长。
- **`pcoa` 里的负特征值**：当距离矩阵不是"欧氏型"时，PCoA 会产生虚轴（negative
  eigenvalue）。本实现用 Lingoes correction 校正，但**校正后的轴仍然要按比例解释**，
  不能像 PCA 那样直接读"解释了百分之多少的方差"。
- **`nmds` 的 stress**：> 0.2 表示二维图不可信，> 0.1 勉强。**报 NMDS 结果必须报 stress**。
- **`cca` / `rda` 的约束轴**："第一个约束轴"不是"第一个主轴"——它被环境变量
  预先定义了，解释率天然低于 PCA。
- **`dca` 的轴必须按顺序解释**：DCA 的第二轴是"去趋势后"的，**不能像 PCA 那样对称地
  解释前后两个轴**。
- **`mantel` 的 p 值是模拟出来的**（Mantel 检验没有解析分布），
  且**对距离度量的选择高度敏感**；partial Mantel 控制第三张矩阵的方式
  （残差化 vs 直接相关）会给出不同 p 值。
- **`morans_i` 的期望值是 -1/(n-1)**（随机分布下），不是 0。
- **`ripley_k` 的边缘校正**：`edge` 修正对边界点很重要，不修正会把边缘效应
  当成聚集。

## 参考

- Anderson, M. J. (2001). *A new method for non-parametric multivariate
  analysis of variance.* Austral Ecology 26, 32–46.
- Anderson, M. J. (2001). *Nonmetric multivariate analysis of
  distances.* In (*eds*) *Per ordination.* Springer, 1–40.
- Gower, J. C. (1966). *A note on the similarity of continua and all
  its numbers.* Biometrika 53, 325–338. （PCoA）
- Benzécri, J.-P. (1971). *Les plans d'expérience.* Annales de l'Institut
  Statistique. （CA）
- Ter Braak, C. J. F. (1986). *Canonical correspondence analysis: a new
  method for constrained ordination.* Vegetatio 66, 37–47.
- Legendre, P. & Legendre, L. (1998). *Numerical Ecology.* 2nd ed. Elsevier.
- Kruskal, J. B. (1964). *Multidimensional scaling for use with
  dissimilarity data in numerical taxonomy.* Systematic Zoology 13, 117–141.
- Hill, M. O. (1974). *Correspondence analysis in the study of a site
  matrix.* Vegetatio 26, 79–89. （DCA）
- Mantel, N. (1967). *The detection of association in two-way contingency
  tables.* Journal of the American Statistical Association 62, 200–210.
- Moran, P. A. P. (1950). *Notes on the continuous stochastic
  population model.* Biometrika 37, 17–23.
- Getis, A. & Ord, J. K. (1992). *The analysis of spatial association by
  use of distance statistics.* Geographical Analysis 24, 189–206.
- Ripley, B. D. (1976). *The Second-Order Analysis of Spatial Point
  Patterns.* Cambridge University Press.
- Clarke, E. A. & Evans, F. C. (1955). *On the relation between the
  transitive neighbourhood and the spatial distribution of individuals.*
  Ecology 36, 445–453.
- Mardia, K. V. & Jupp, P. E. (2000). *Circular Statistics.* 2nd ed. Wiley.

## 分析清单

<!-- matrix:start -->
| 分析 | 模块 | 验证证据 | 说明 |
|---|---|---|---|
| `ca` | `stats.detriding` | 单元 | Correspondence analysis of a contingency table |
| `cca` | `stats.cca` | 单元 | Canonical correspondence analysis / redundancy analysis |
| `dca` | `stats.detriding` | 单元 | Detrended correspondence analysis |
| `grid_interpolate` | `stats.spatial_stats` | 性质 | Interpolate scattered sites onto a regular grid |
| `lda` | `stats.lda` | 单元 | Linear discriminant analysis |
| `mantel` | `stats.mantel` | 单元 | Mantel test between two distance matrices |
| `minimum_spanning_tree` | `stats.geometry` | 单元 | Minimum spanning tree of a distance matrix |
| `morans_i` | `stats.spatial_stats` | 性质 | Global spatial autocorrelation (Moran's I) |
| `morphospace_disparity` | `stats.geometry` | 单元 | Disparity through time of a morphospace |
| `nearest_neighbour` | `stats.spatial_stats` | 性质 | Nearest-neighbour point pattern statistics |
| `nmds` | `stats.nmds` | 性质 | Non-metric multidimensional scaling |
| `partial_mantel` | `stats.mantel` | 单元 | Partial Mantel test controlling for a third matrix |
| `pca` | `stats.pca` | R + 性质 | Principal component analysis |
| `pcoa` | `stats.pcoa` | 真值 | Principal coordinate analysis |
| `ripley_k` | `stats.spatial` | 真值 + 性质 | Ripley's K point pattern analysis |
| `spherical` | `stats.spatial_stats` | 性质 | Mean direction and concentration on the sphere |
<!-- matrix:end -->
