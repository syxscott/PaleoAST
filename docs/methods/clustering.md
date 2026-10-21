# 聚类

聚类把一组对象按相似度分组。**它回答的是"有没有结构"，不是"有几个组"**——
这是这一域最常被误解的一点。

## 通用假设

| 编号 | 假设 | 说明 |
|---|---|---|
| **A1** | 变量已标准化 | 未标准化时，量纲大的变量主导整个距离结构 |
| **A2** | 距离度量与数据类型相符 | 计数用 Bray-Curtis，连续量用 Euclidean |
| **A3** | 簇是"球状"的 | k-means 只能找到凸簇；双螺旋数据上它一定失败 |
| **A4** | k 是外部给定的，或由明确的准则给出 | "看肘部"是启发式，不是检验 |

## 两类方法

| | 凝聚层次聚类 | k-means |
|---|---|---|
| 需要预先指定 k | **不需要** | 需要 |
| 簇的形状 | 任意 | 凸、大小相近 |
| 结果稳定性 | 稳定（只要距离度量与连接方式定下） | 受初值影响（但本实现用 k-means++ 固定了随机源） |
| 切割成 k 簇的方式 | 看树的高度 | — |

**层次聚类在探索阶段更有用**：它不要求先知道 k，树的形状本身就是信息。
**"有几个簇"应该由树的高度、轮廓系数或 gap statistic 一起支持，而不是只看肘部。**

## 逐方法的注意事项

- **`kmeans` 的初值**：本实现用 k-means++，**随机源是固定的**，
  所以同样的数据每次给同样的结果——这很重要，因为**k-means 的局部最优会让
  "跑了三次"给出不同答案**。
- **`kmeans_elbow` 画的是组内平方和随 k 的曲线**。肘部是**启发式**：
  它没有零假设，**没有 p 值**，曲线也可能根本没有肘部（尤其是维度高时）。
  报告时应写"目视/经验准则"，而不是"检验得到 k = 4"。
- **层次聚类的连接方式**（single / complete / average / Ward）会给出不同的树，
  **必须写明用的是哪一种**。Ward 只适用于 Euclidean 距离。
- **树的高度是距离，不是"差异的统计量"**：切割高度的选择没有 p 值。

## 参考

- Lloyd, S. P. (1982). *Least squares quantization in PCM.* IEEE
  Transactions on Information Theory 28, 129–137.
- Hartigan, J. A. & Wong, M. A. (1979). *A simple algorithm for
  k-means clustering.* Applied Statistics 28, 100–118.
- MacQueen, J. (1967). *Some methods for classification and analysis of
  multivariate observations.* In (eds) *Proceedings of the Fifth Berkeley
  Symposium on Mathematical Statistics and Probability.* University of
  California Press.
- Arthur, D. & Vassilvitskii, S. (2007). *k-means++: the advantages of
  careful seeding.* In (eds) *Proceedings of the 24th Annual ACM-SIAM
  Symposium on Discrete Algorithms.* SIAM.
- Sokal, R. R. & Michener, C. D. (1958). *A statistical method for
  evaluating systematic relationships.* University of Kansas Science
  Bulletin 38, 1409–1438.
- Ward, J. H. (1963). *Hierarchical grouping to optimize an objective
  function.* Journal of the American Statistical Association 58, 236–244.
- Kaufman, L. & Rousseeuw, P. J. (1990). *Finding Groups in Data: an
  Introduction to Cluster Analysis.* Wiley.
- Thorndike, R. L. (1953). *Who belongs in the family?* Psychometrika
  18, 267–276.
- Rousseeuw, P. J. (1987). *Silhouettes: a graphical aid to the
  interpretation and validation of cluster analysis.* Journal of
  Computational and Applied Statistics 20, 53–65.
- Tibshirani, R., Walther, G. & Hastie, T. (2001). *Estimating the
  number of clusters in a data set via the gap statistic.* Journal of the
  Royal Statistical Society B 63, 411–423.

## 分析清单

<!-- matrix:start -->
| 分析 | 模块 | 验证证据 | 说明 |
|---|---|---|---|
| `hierarchical_clustering` | `stats.clustering` | 性质 | Agglomerative hierarchical clustering |
| `kmeans` | `stats.clustering` | 性质 | K-means partitioning |
| `kmeans_elbow` | `stats.clustering` | 性质 | Within-cluster sum of squares against k |
<!-- matrix:end -->
