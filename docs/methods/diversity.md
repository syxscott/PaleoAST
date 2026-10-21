# 生态多样性

这一域的方法把一张"样本 × 物种"表变成若干个可比较的数。
**最容易出错的地方不是公式，是定义**：同一个"Shannon 指数"有对数和自然对数两种底，
同一个"Simpson"有好几个互不相同的式子。报告时必须说清用的是哪一个。

## 通用假设

| 编号 | 假设 | 说明 |
|---|---|---|
| **A1** | 样本是同一次取样的完整结果 | 可比较性（comparable）要求样本量与努力程度可比 |
| **A2** | 物种是同一定义下的 | 新属、新种的处理会改变每一个指数 |
| **A3** | 抽样是随机的 | 系统采样会让所有基于"频数"的估计有偏 |
| **A4** | 重复样本之间独立 | 同一地层的多个化石门类不是独立样本 |

## 定义必须写清楚

| 名称 | 本实现 | 等价的其它写法 |
|---|---|---|
| Shannon | `-Σ pᵢ ln pᵢ`（自然对数） | log₂ 版 = H′/ln2；R 的 `vegan::diversity` 默认也是自然对数 |
| Simpson | `1 − Σ pᵢ²` | `Σ pᵢ²`（即"优势度"）是另一个量，两者关系是 `D = 1 − S` |
| Fisher's alpha | 由 S 与 N 解析求解 | 无闭式解时用迭代 |
| Pielou's J | `H′/ln(S)` | 等于"evenness"，取值 [0,1] |
| Margalef | `(S−1)/ln N` | |
| Berger–Parker | `1/Σ pᵢ²` | 与 Simpson 互补 |

**数值上要记住**：Shannon 的自然对数版与 log₂ 版差一个 `ln2`（0.693）。
把两个来源的数据放进同一张表比较时，底数不一致会得出"多样性更高"的假结论。

## 逐方法的注意事项

- **`diversity` 的 Hill 阶数**：q=1 是 Shannon，q=2 是 Simpson，q=0 是丰富度。
  **用 Hill 数报告可以把定义歧义消掉**，这是当前生态学界的推荐做法。
- **`rarefaction`（个体稀释化）与 `sample_rarefaction_tao`（样本稀释化）回答不同问题**：
  前者固定个体数，后者固定样本数。**混用会得到相反的"多样性随努力程度"的趋势**。
- **`coverage_rarefaction`（样本覆盖度稀释化）**回答"还需要多少采样才够"，
  报告时必须同时给覆盖度估计。
- **`sample_coverage` 的估计量**是 Good & Scharlemann 的（Turing–Good 改进），
  与 `vegan::coverage` 一致。
- **`abundance_models`（对数级数/对数正态/几何/断棍）用 AIC 排名**。
  断棍模型依赖"物种丰富度由个体数决定"的强假设，在样本量不足时会被 AIC 惩罚得较轻——
  **不要只看排名第一的那个**。
- **`null_models` 的 p 值依赖"用什么度量"**：固定列和、均值-方差、比例等不同的零模型
  会给出不同的显著与否。报告时必须写清零模型类型与次数。
- **`diversity_permutation_test` 与 `diversity_t_test`**：前者不假设正态，
  后者假设正态；样本量小或明显偏态时用前者。

## 参考

- Shannon, C. E. (1948). *A mathematical theory of communication.* Bell
  System Technical Journal 27, 379–423, 623–656.
- Simpson, E. H. (1949). *The interpretation of interaction in
  canonical analysis.* Ecology 30, 339–348.
- Fisher, R. A., Corbet, A. S. & Williams, C. B. (1943). *The relation
  between the number of animals and the number of species in a random
  sample of an animal population.* Journal of Animal Ecology 12, 42–58.
- Pielou, E. C. (1966). *The measurement of diversity in certain
  communities: a review and a proposal.* American Naturalist 100, 425–437.
- Berger, W. H. & Parker, F. L. (1975). *Diversity of planktonic
  foraminifera in deep-sea sediments.* Science 168, 1345–1347.
- Hill, M. O. (1973). *Diversity and evenness: a unifying notation and
  its consequences.* Ecology 54, 427–432.
- Magurran, A. E. (1987). *The Diversity of Fishes.* Blackwell.
  （SHE）
- Hurlbert, S. H. (1971). *The nonconcept of species diversity: a
  critique.* Ecology 52, 174–183.
- Chiu, C.-H., Jaccard, T., Wang, A., Yap, I. H. & Cao, Y. (2012).
  *iNEXT: an R package for comparison of diversity measures across
  samples of uneven sizes.* Methods in Ecology and Evolution 3, 1–24.
- Good, P. & Scharlemann, W. (1984). *Sample coverage as a function of
  sample size.* Ecology 65, 254–258.
- Good, I. J. & McGarvie, M. B. (2014). *Sample coverage for functional
  diversity assessments.* Methods in Ecology and Evolution 5, 1341–1351.
- Gotelli, N. J. & Graves, G. R. (1996). *Null models of community
  structure: a review of null model analysis of species abundances.*
  Ecology 77, 774–788.
- Anderson, M. J. (1971). *Measures of diversity.* In *Data analysis in
  ecology and evolution: a guide to the use of quantitative methods in
  ecology and evolutionary biology.* Cambridge University Press.
- Whittaker, R. H. (1972). *The concept and measurement of species
  diversity.* Annual Review of Ecology and Systematics 3, 279–288.
- Baselga, A. (2010). *The relationship between diversity and the
  temporal variance of ecological distance metrics.* Methods in Ecology
  and Evolution 1, 53–67.

## 分析清单

<!-- matrix:start -->
| 分析 | 模块 | 验证证据 | 说明 |
|---|---|---|---|
| `abundance_models` | `ecology.advanced` | 单元 | Abundance model fitting: log-series, log-normal, geometric and broken-stick, ranked by AIC |
| `beta_diversity` | `ecology.beta_diversity` | 单元 | Beta diversity decomposition |
| `coverage_rarefaction` | `ecology.beta_diversity` | 单元 | Sample-based coverage rarefaction |
| `diversity` | `ecology.diversity` | R | Alpha diversity indices |
| `diversity_permutation_test` | `ecology.diversity_menu` | 单元 | Permutation test of the diversity indices between groups of samples |
| `diversity_profile` | `ecology.diversity_menu` | 单元 | Diversity indices of one sample across sampling depths |
| `diversity_t_test` | `ecology.diversity_menu` | 单元 | t-test of the diversity indices between groups of samples |
| `dtw` | `ecology.dtw` | 单元 | Dynamic time warping and LB_Keogh distances |
| `null_models` | `ecology.null_models` | 单元 | Null model randomisations of a community matrix |
| `rarefaction` | `ecology.rarefaction` | 单元 | Individual and sample rarefaction |
| `sample_coverage` | `ecology.diversity_menu` | 单元 | Sample coverage of a samples-by-taxa table |
| `sample_rarefaction_tao` | `ecology.diversity_menu` | 单元 | Tao rarefaction of a samples-by-taxa table |
| `she` | `ecology.advanced` | 单元 | SHE analysis for diversity |
| `sqs_subsample` | `ecology.diversity_menu` | 单元 | Shareholder-quorum subsample of one sample |
<!-- matrix:end -->
