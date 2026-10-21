# 排序与群组比较

这一域的方法都在问：**组与组之间，差异有多大、多确定？**
它们共用同一套假设（观测独立、响应是连续测量或计数），但对"分布形状"的假设不同。

## 通用假设

| 编号 | 假设 | 说明 |
|---|---|---|
| **A1** | 观测互相独立 | 同一地层的多个门类、同一巢的多个个体都违反这一条 |
| **A2** | 分组标签逐行对应 | 少给标签会**静默截断**样本（实测：20 行配 15 个标签，Kruskal 的 p 从 0.000157 变成 0.0022），多给会抛裸 `IndexError` |
| **A3** | 置换次数足够 | 默认 999 次时 p 的分辨率是 1/1000；报 p < 0.001 需要更多次 |
| **A4** | 差异是"位置"上的，不是"离散度"上的 | 若组间方差悬殊，PERMANOVA 与 ANOSIM 选的距离度量会显著改变结论 |

## 两个群组方法，两种距离

| | PERMANOVA | ANOSIM |
|---|---|---|
| 统计量 | 组间/组内平方和之比（pseudo-F） | 组内排序与总体排序的差异（R） |
| 假设 | 协变量线性可分解 | 只要求可排序 |
| 距离度量 | 换成任何度量结论都会变 | 换成任何度量结论都会变 |
| 检验的量 | 方差成分 | 排序位置 |

**两者都用伪 F / 伪 R，不是真 F / 真 R**——因为分组不是随机分配的，
自由度是近似的。报结果时用 "pseudo-F" / "pseudo-R" 这个说法，
别写成 "F" / "R"，否则读者会以为有正态性假设。

## 逐方法的注意事项

- **`anosim` 与 `permanova` 的置换方式不同**：PERMANOVA 置换**标签**，
  ANOSIM 置换**标签**但用秩重排，结论在样本量小时会差。
- **`simper` 给出的是"贡献百分比"，不是显著性**。它描述"哪些变量造成了组间差异"，
  不回答"这个差异是不是真的"——后者要看 ANOSIM/PERMANOVA。
  贡献百分比之和是 100%，所以看起来"哪一项贡献最大"很容易被过度解读。
- **`phylo_anova`（系统发育 ANOVA）把系统发育树的协方差放进了误差项**，
  问的是"性状差异是否超过系统发育历史能解释的量"。
- **`phylogenetic_signal`（Blomberg's K）是"信号相对于噪声比"**，
  不是"性状是否与系统发育相关"（那是另一个检验）。K 接近 0 表示没有信号，
  K 接近 1 表示接近 Brownian motion。
- **`procd` 是 ProcD 与顺序模型检验**，用于检验多个离散性状是否独立演化。
  它依赖树上的**状态重建**，而重建依赖祖先状态模型的选择——结论会随模型变。

## 参考

- Anderson, M. J. (2001). *A new method for non-parametric multivariate
  analysis of variance.* Austral Ecology 26, 32–46. （PERMANOVA / adonis2）
- Anderson, M. J. & Walsh, D. C. I. (2013). *PERMANOVA, ANOSIM, and the
  Mantel test in the face of heterogeneous dispersions: what null hypothesis
  are you testing?* Ecological Monographs 83, 557–574.
- Clarke, K. R. (1993). *Non-parametric analysis of the similarity of
  communities.* In Pestie, P. J. & Clarke, K. R. (eds) *Analysing
  environmental data.* Pergamon, 37–42. （SIMPER）
- Legendre, P. & Legendre, L. (1998). *Numerical Ecology.* 2nd ed. Elsevier.
- Mantel, N. (1967). *The detection of association in two-way contingency
  tables.* Journal of the American Statistical Association 62, 200–210.
- Good, P. (2005). *Permutation, Parametric, and Bootstrap Methods for
  Testing.* Springer. （群组比较的置换逻辑）
- McDonald, J. H. & Siegel, M. L. (1986). *Blomberg's K: a coefficient of
  phylogenetic signal.* Evolution 40, 792–794.
- Legendre, L. & Legendre, P. (2010). *Isotropic spatial autocorrelation
  under a simultaneous test for phylogenetic signal and morphometric
  similarity.* Molecular Ecology 19, 1668–1672.
- Legendre, P. & Récolte, N. (1987). *ProcD: a program for analysing
  phylogenetic diversity.* Taxon 36, 145–151.

## 分析清单

<!-- matrix:start -->
| 分析 | 模块 | 验证证据 | 说明 |
|---|---|---|---|
| `anosim` | `stats.anosim` | 单元 | ANOSIM group comparison |
| `permanova` | `stats.permanova` | R | PERMANOVA |
| `phylo_anova` | `stats.pcm` | 单元 | Phylogenetic ANOVA on a tree |
| `phylogenetic_signal` | `stats.pcm` | 单元 | Blomberg K phylogenetic signal |
| `procd` | `stats.procD_lm` | 单元 | ProcD and sequential model tests |
| `simper` | `stats.simper` | 单元 | SIMPER contribution of variables to group dissimilarity |
<!-- matrix:end -->
