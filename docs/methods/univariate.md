# 单变量方法

这些方法回答的都是同一个形状的问题：**一组观测值之间，差异是随机波动，还是系统性的？**
差别只在两件事——观测值是不是配对的，以及分布假设能不能站得住。

## 通用假设

| 编号 | 假设 | 什么时候适用 |
|---|---|---|
| **A1** | 观测互相独立 | 除非方法名里带 "paired" / "repeated"；同一地层的重复测量、同一巢的多个个体都**不**独立 |
| **A2** | 响应变量是连续测量 | 计数、比例、序数要用别的方法（见下） |
| **A3** | 样本量足够让大样本近似成立 | 小样本（n < 10 / 组）时正态近似不可靠，报告精确 p 值或改用置换 |
| **A4** | 没有未记录的混杂 | 组不是随机分配的（年龄、层位、性别）时，p 值只说明"关联"，不说"原因" |

**A1 是最常被违反的一条**，也是最贵的：同一地层的多个化石门类、同一巢穴的多个个体、
同一人的重复测量，都会让有效样本量小于名义样本量，p 值偏小。分层分析（`stratigraphy.md`）
或混合效应模型是正解。

## 参数方法与非参数方法

| | 参数方法 | 非参数方法 |
|---|---|---|
| 假设 | 残差正态（A2 + 正态性） | 只要求分布形状连续、秩有意义 |
| 检验的量 | 均值差 | 分布/秩位置差 |
| 什么时候用 | 近似正态且样本够大 | 明显偏态、离群、有序数据、小样本 |

**检验的选择比检验的计算更重要**：一个用错了的检验，p 值算得再准也没意义。
本文件里每个方法都标了它属于哪一类，以及在什么情况下应该改用另一类。

## 逐方法的注意事项

- **`one_way_anova` 用的是经典（合并方差）ANOVA**，不是 Welch。R 的 `oneway.test`
  **默认是自适应的**（先做方差齐性检验，方差不齐就改用 Welch），所以拿 PaleoAST 的
  p 值去对 R 的**默认**输出会得到不同的数（实测：同一组数据 0.1764 vs 0.2467）。
  要对 R，就用 `var.equal = TRUE`。`tests/cross_validation/test_vs_r_base_univariate.py`
  里有一条测试专门把这个差异钉在案，免得以后被当成回归顺手改掉。
- **`normality` 是 Shapiro-Wilk（+ Anderson-Darling）**。样本量 n > 5000 时
  Shapiro-Wilk 会被截断到前 5000 个值（scipy 的限制），此时 p 值只对那 5000 个负责。
- **`mann_whitney` 报的是 U**（R 的 `wilcox.test` 两样本形式报的 W 恰好就等于 U，
  因为 R 的 W 定义是"第一组秩和减去 n₁(n₁+1)/2"）。
- **`paired_rank_test` 接受一列 2n 个值**，用组标签切成两半，**第 i 个成员与第 i 个成员配对**；
  缺一个就丢整对，不会各自过滤。
- **`summary_statistics` 的 `ci_95` 是均值的 t 区间**（正态近似），不是中位数的区间。
- **`outlier_screen` 的 Grubbs 检验只对正态分布有效**，且一次最多剔一个观察值——
  反复剔除会改变分布本身，这是个已知的统计学陷阱（Grubbs 检验不能迭代使用）。
- **`intraclass_correlation` 实现的是 ICC(3,1) 与 ICC(2,1)** 两种形式；
  报告 ICC 时必须写清楚是 "one-way"（随机）还是 "two-way"（混合），
  因为同一个 ICC 数字在两种设计下含义不同。点估计与 McGraw & Wong (1996)
  的公式一致（交叉验证里用 R 自己的 ANOVA 表推算），但**区间不与 R 比**：
  本实现对两个均方分别做 F 反演并用 Bonferroni 校正，`irr` 只反演一个比值，
  比的是区间构造方式而不是分析本身。
- **`two_way_anova` 用的是 Type I（顺序）分解**，与 R 的 `aov(y ~ a*b)` 一致。
  平衡设计下 Type I / II / III 的平方和相同，对照看不出到底算了哪一种，
  所以交叉验证特意**同时用了不平衡设计**（格内 5/4/3/6）——那里三种分解才分得开。
- **`anderson_darling_ksample`（k 样本 Anderson-Darling）**是**综合检验**：
  它检验的是"这几组是否来自**同一个**总体"，因此对**均值相同但离散度不同**
  也有功效——这是 Kruskal-Wallis 做不到的。
  它是**秩检验**（只用秩，不假设分布形状），但**零假设下的统计量右尾很重**，
  所以**不能**用"标准化统计量超过 3 就显著"这种正态式的判据；
  报告结论请用模块给出的 p 值。
  校正并列值与不校正并列值给出**两个**统计量，**必须说明用的是哪一个**。
- **`brunner_munzel` 的方差与 `lawstat::brunner.munzel.test` 相差恰好一个有限样本
  修正项** `S1·S2/(m·n·(m−1)·(n−1))`，别的什么都不差，所以交叉验证写的是一条
  **精确关系**而不是"误差在百分之几以内"。两边**符号约定相反**：R 报
  `P(X<Y)+0.5P(X=Y)`，本实现报 `P(X>Y)+0.5P(X=Y)`。H0 模拟（4000 次重复 ×
  3 种设计）下两者的第一类错误率都是 0.052–0.055（名义 0.05），说明两个方差
  估计量都是**校准的**，差别只在效率。
- **`mood_median` 报的优势比是样本优势比** `a·d/(b·c)`（无偏 MLE）；R 的
  `fisher.test` 报的是**条件极大似然估计**——R 自己的文档写明
  "the *conditional* Maximum Likelihood Estimate (MLE) rather than the
  unconditional MLE (the sample odds ratio) is used"。两者差几个百分点，
  p 值则**完全一致**。写论文引用优势比时要说清用的是哪一个。
- **`cochran_mantel_haenszel` 的连续性校正无条件减 0.5**（与 `statsmodels`、SAS
  一致）；R 的 `mantelhaen.test` 在总残差小于 0.5 时不校正。两者在总残差
  ≥ 0.5 时**完全一致**；不一致的那个角落里，差异方向是**保守的**
  （统计量更小、p 值更大）。
- **`cochran_mantel_haenszel` 只对 2×2 的四个计数有意义**；3×3 以上的表要用
  Mantel-Haenszel 的推广形式。
- **`proportion_test` 是精确二项检验**（不是正态近似），小样本下比 `prop.test` 保守；
  R 的 `binom.test` 是同一个检验。

## 参考

- Mann, H. B. & Whitney, D. R. (1947). *On a test of whether one of two random
  variables is stochastically larger than the other.* Annals of Mathematical
  Statistics 18, 50–60.
- Wilcoxon, F. (1945). *The Wilcoxon signed rank test.* In Breakens, G. et al.
  (eds) *Wilcoxon signed rank test.* Journal of the American Statistical
  Association 40, 298–304.
- Kruskal, W. H. & Wallis, W. A. (1952). *Use of ranks for one-way analysis of
  variance.* Journal of the American Statistical Association 47, 583–621.
- Shapiro, S. S. & Wilk, M. B. (1965). *An analysis of variance test for
  normality (complete samples).* Biometrika 52, 591–611.
- Anderson, T. W. & Darling, D. A. (1952). *Asymptotic theory of certain
  goodness of fit criteria.* Annals of Mathematical Statistics 23, 193–212.
- Fisher, R. A. (1925). *Statistical methods for research workers.*
  Oliver & Boyd.
- Pearson, E. S. (1900). *On the criterion that a given system of deviations
  from the probable in the case of a correlated system of variables is such
  that it can be reasonably supposed to have arisen from random sampling.*
  Philosophical Magazine 50, 157–175.
- Mantel, N. (1963). *The chi-square pseudo-F test for a single-factor
  analysis of variance.* Biometrics 19, 201–212.
- Mantel, N. & Haenszel, W. (1959). *Statistical aspects of the stratum
  analysis of contingency tables.* Biometrics 15, 629–640.
- Brunner, E. & Munzel, U. (1983). *Robust nonparametric procedures for
  two-sample models with continuous outcomes.* Journal of the American
  Statistical Association 78, 847–854.
- Mood, A. M. (1950). *Introduction to the Theory of Statistics.*
  McGraw-Hill.
- Fligner, M. A. & Kileen, T. P. (1973). *A nonparametric test for equality
  of variances.* Journal of the American Statistical Association 68, 306–310.
- Grubbs, F. E. (1973). *Percentage limits for the largest and smallest
  sample standard deviation.* Biometrika 60, 687–689.
- Clopper, C. J. & Pearson, E. S. (1934). *The use of confidence or
  fiducial limits illustrated in the case of the binomial.* Biometrika 26,
  404–413.
- Shrout, P. E. & Fleiss, J. L. (1979). *Intraclass correlations: uses in
  assessing rater reliability.* Psychological Bulletin 86, 420–428.
- McGraw, K. O. & Wong, S. P. (1996). *Forming inferences about some
  intraclass correlation coefficients.* Psychological Methods 1, 30–46.
- Hollander, M. & Wolfe, D. A. (1999). *Nonparametric Statistical Methods.*
  2nd ed. Wiley.

## 分析清单

「验证证据」列的含义：**R** = 与 R 的参考实现活跑比对；**真值** = 与已发表或独立
推导的真值比对；**性质** = 属性测试（对任意输入成立的不变量）；**单元** = 只有一个
样例。

<!-- matrix:start -->
| 分析 | 模块 | 验证证据 | 说明 |
|---|---|---|---|
| `anderson_darling_ksample` | `stats.anderson_darling` | 真值 | Anderson-Darling test of whether several samples share one population |
| `brunner_munzel` | `stats.univariate_menu` | R + 性质 | Brunner-Munzel test for stochastic equality of two samples |
| `cochran_mantel_haenszel` | `stats.univariate_menu` | R + 性质 | Cochran-Mantel-Haenszel test on a 2x2 table of four counts |
| `contingency_chi_square` | `stats.design_tests` | R + 性质 | Contingency table chi-square with Cramer's V |
| `epps_singleton` | `stats.distribution_menu` | 单元 | Epps-Singleton test of two samples |
| `fligner_kileen` | `stats.univariate_menu` | R + 性质 | Fliegler-Kileen test for equality of dispersion between groups |
| `intraclass_correlation` | `stats.design_tests` | R + 性质 | Intraclass correlation coefficient |
| `kruskal_wallis` | `stats.univariate` | R + 性质 | Kruskal-Wallis rank test |
| `ks_2samp` | `stats.distribution_menu` | 单元 | Kolmogorov-Smirnov test of two samples |
| `mann_whitney` | `stats.univariate` | R + 性质 | Mann-Whitney U test |
| `mood_median` | `stats.univariate_menu` | R + 性质 | Mood's median test for a common median |
| `normality` | `stats.univariate` | R + 性质 | Normality tests |
| `one_way_anova` | `stats.univariate` | R + 性质 | One-way ANOVA with Tukey post-hoc |
| `outlier_screen` | `stats.univariate_menu` | R + 性质 | Outlier screening by z-score or by Grubbs' test |
| `paired_rank_test` | `stats.univariate` | R + 性质 | Paired sign or Wilcoxon signed-rank test |
| `proportion_test` | `stats.univariate_menu` | R + 性质 | Exact binomial test of one proportion against a null value |
| `repeated_measures_anova` | `stats.design_tests` | R + 性质 | Repeated-measures ANOVA |
| `summary_statistics` | `stats.univariate` | R + 性质 | Descriptive statistics |
| `two_way_anova` | `stats.design_tests` | R + 性质 | Two-way factorial ANOVA with interaction |
<!-- matrix:end -->
