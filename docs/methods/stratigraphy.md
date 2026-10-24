# 地层学与时间序列

这一域的方法读的是**一条按时间排列的序列**：地层柱、旋回、碳同位素曲线、产率曲线。
它们共同的脆弱点是**序列不是独立的观测**——相邻层位共享继承与沉积环境。

## 通用假设

| 编号 | 假设 | 说明 |
|---|---|---|
| **A1** | 时间轴本身是可信的 | 年龄模型错了，后面全错 |
| **A2** | 相邻观测有自相关 | **这是这一域的头号假设**——不处理自相关，显著性会被高估 |
| **A3** | 过程近似平稳 | 变点前后用同一个模型会得出无意义的结论 |
| **A4** | 分辨率足够 | 采样间隔与目标周期同量级时才谈得上"检出" |

**A2 是这一域的生死线**：一条有强自相关的序列，即使真实无周期，频谱峰值也会显著。
所以**每个周期分析都必须先说清楚它怎么处理了红噪声**：
本仓库的做法是 **AR(1) 背景**（`spectral` 有红噪声检验，`redfit` 直接拟合 AR(1)）。

## 逐方法的注意事项

- **`spectral`（周期图）带 AR(1) 红噪声检验**，`spectral_peaks`（显著性周期）
  必须用同一套红噪声背景，否则"显著周期"在不同工具下结论不同。
- **`acf` / `autoassociation` 的 Bartlett 带**是 1.96/√n 的近似带，
  它假设序列是平稳的。**带外面不算显著**——但别把它当成严格的检验。
- **`prewhiten`（AR(1) 预白化）**会改变序列的方差结构，
  **预白化后的显著性检验与原序列不是一回事**。
- **`redfit` 直接拟合 AR(1)**，比"假定红噪声再检验"更贴切，
  但它对序列长度敏感：短序列上 AR(1) 本身就不稳。
- **`multitaper` 的带宽与时间-频率权衡**是显式参数（NW、k），**必须写进方法**。
- **`wavelet`（连续小波变换）用 Morlet 小波**，
  **边缘效应（cone of influence）之外的功率不可信**。
- **`mann_kendall` 的 Sen 斜率**不假设线性，**对离群点稳健**；
  但它检验的是"单调趋势"，**对周期信号不适用**。
- **`runs_test`（游程检验）**检验序列是否随机，**对离群点不稳健**——
  一个离群点就能把游程数打乱。
- **`coniss`（CONISS）是约束聚类**，结果**依赖于约束树**；
  报告时必须给出约束树是怎么定的（先聚类再约束，还是人工约束）。
- **`stratigraphic_correlation`（复合标准剖面相关）**的相关系数是
  "深度相关系数"，**它依赖剖面之间的比例标定**（如何把两个剖面的深度换算成同一时间轴）。
- **`extinction`（灭绝区间估计）**给的是区间而不是点估计，
  报告时应给区间并说明方法（不是"点灭绝事件"）。
- **`ua`（unitary association）与 `rasc`**是"最优化"的区间估计方法，
  结果依赖事件定义与地层范围。
- **`detrend_series`（去趋势）有九种趋势模型**（均值、一次/二次/N 阶多项式、
  LOWESS / rLOWESS / LOESS / rLOESS、EEMD 残差）。**选哪一种会改变结论**：
  LOWESS/LOESS 的窗口是**记录跨度的百分比**而不是样本数，所以一条采样更密的
  记录在同一设置下会被平滑得更轻。**报告时必须写明模型与窗口。**
- **`optimal_span_smooth`（最优跨度平滑）按 GCV/AIC 选跨度**，
  判据不同会选出不同的跨度；**自助置信带用的是块自助**（相邻样本相关，
  普通自助会低估不确定度）。**GCV 与 AIC 对红噪声都偏向长跨度**，
  所以"自动选出来的"跨度不是唯一正确的答案。
- **`sedimentary_noise` / `rho1_sedimentation_ratio`（沉积噪声）**给的是
  **噪声功率与所选基线之比**，不是沉积速率本身：>1 表示记录携带的短周期
  噪声强于该基线。**噪声频带必须取子频带**——在整条频带上记录与基线的
  总方差相同，比值恒为 1，统计量没有信息。
- **`lod_dimension`（一日之长）**给出的是**日长**（小时）而非"维数"；
  它按 Laskar et al. (2004) 的已发表多项式算，与记录本身的比例系数是拟合出来的。
- **`lr04_stack`（LR04 底栖 δ¹⁸O 栈）**是**已发表的参考曲线**，
  不是从数据算出来的；用它做对比时**必须说明用的是哪一版**。
- **`coco` / `ecoco`（相关系数法）**估计的是**最可能沉积速率**；
  `ecoco` 的窗口跨过速率突变点时必然给出过渡值，**那种窗口不应解读**。
- **`time_opt` / `etime_opt`（轨道调谐）**在多个候选压缩系数上搜索，
  **得分面通常有平台**，所以"最优点"不是唯一的答案；**必须报告得分曲线**。
- **`plot_digitizer`（图像数字化）**的精度**取决于标定的两个参考点**；
  它假设绘图区里**只有一条曲线**，交叉线需要用颜色参数指定。
- **`mini_robot`（批处理）**把一串分析录下来重放；**某一步失败会被记下来
  而不是让整批崩掉**，所以结果里**必须检查每一步的 succeeded**。
- **`standardize` / `log_transform` / `apply_function`（变换）**改变了序列的
  尺度与分布，**变换后再做频谱分析，结论与原序列不可直接比较**。
- **`remove_part` / `remove_range`（删段/留段）**改变了记录的长度与采样密度，
  **删掉的部分不能用插值补回来**。
- **`bispectrum` / `bicoherence_significance`（双谱与双相干）**看的是**三元耦合**：
  频率 `f1`、`f2` 与它们的和 `f1+f2` 上的相位关系。它对**非线性**敏感，
  是频谱看不到的耦合的探针；代价是**自由度少**，显著性必须用**替代序列**估计。
- **`recurrence_plot`（递归图）**度量的是**状态是否重复**，不是相关：
  它的 DET/MDL/LAM 描述的是轨迹在相空间里的重复程度，
  **必须先说明用的距离度量与半径**，否则数字之间不可比。
- **`evolutionary_rho1`（演化 ρ1）**给出的是**滑窗内去趋势后的滞后 1 阶自相关**；
  一段完全线性的趋势去趋势后什么都不剩，模块此时返回 0 而不是噪声比值。
- **`robust_lomb_scargle`（稳健 AR(1) 背景的 LS 周期图）**的背景是把
  **被判为周期信号的谱峰剔除后**重新拟合的，因此它的显著性与普通 `spectral`
  **不可直接比较**；两者的**频带覆盖也必须一致**才谈得上对照。
- **`swa_background` / `swa_fdr`（SWA 背景与错误发现率）**的背景是为
  **红噪声**设计的（SWA 是**非参数**方法），白噪声与幂律噪声上不适用；
  而 p 值要求功率谱的**自由度与 `degrees_of_freedom` 一致**——
  用普通 LS 周期图（自由度 2）配自由度 8 的判据，**假阳性率会翻倍**。
  `weedon_periodogram` 给出的是符合该判据的谱。
- **`weedon_periodogram`**把记录切成互不重叠的段再平均，代价是**分辨率被分段
  粗化**；它换来的是**逐点可验证的自由度**。
- **`bayesian_periodogram`（贝叶斯周期图）**给出的是**后验概率**，
  它**依赖幅度先验**；先验一换，数字就换，**不能与 p 值混用**。
- **`coco_adaptive` / `coco_crossfit` / `coco_interleaved`**是 COCO 的三种
  变体：**自适应**在数据上重新估计目标谱，**交叉拟合**把估计与检验分开，
  **交错**用于有间断的记录。三者的 p 值口径**彼此不同**，不可横向比较。
- **`frequency_stabilization`（频率稳定）**是**坐标可变的带通**：
  通带随时间移动，因此它能跟住一个**在漂移**的周期，
  但**控制点必须落在记录域内且上下界不交叉**。
- **`sliding_spectral_moments`（滑窗谱矩）**给出的是谱的**形状**
  （质心、宽度、峰度），不是**有没有周期**；它与 `spectral` 的显著性
  **互不替代**。
- **`milankovitch_uncertainty`（米兰科维奇频率的不确定度）**给的是**理论频率
  与它所在的区间**，不是从记录里测出来的周期；
  **把实测周期匹配到轨道上时，容差必须显式给出**。
- **`orbital_frequencies`（9 条主天文频率）**是由**轨道解本身**解算的，
  不是查表；它的 5 条偏心率组合是**在解上就地拟合**出来的，
  因此随年龄变化，而 4 条斜率-岁差组合几乎不变。
  **这两组不要放在同一张图上比较**。
- **`age_depth_uncertainty`（年龄-深度不确定度）**给出的是**蒙特卡洛模拟出来的
  带**，不是一个年龄；**采样点越少、外推越远，带越宽**，
  外推到锚点之外的结果**不应解读**。
- **`radiocarbon_calibration`（¹⁴C 校准）**的结果**取决于所用的校准曲线**，
  而**曲线不在本仓库里**；报告结果时**必须写明用的是哪一条曲线、哪一版**。
- **`tobit_regression`（删失回归）**只在**响应被删失**时才有意义；
  没有删失时它退化为普通最小二乘，**不要在没有删失时用它**。
- **`wavelet_coherence`（小波相干）**的相干性**在信号强时会饱和**
  （R² 在真实周期 ±10% 内只差 0.003），所以**周期要用交叉小波功率读**，
  不要用相干性的极大值去定位周期。
  它的相位符号与 `coherence` **相反**（第二条序列滞后时为正），
  两者**不可混用**。

## 参考

- Bottjer, D. J. & Marshall, A. V. (1996). *Time Stratigraphy:
  Sequence Stratigraphy and Chronostratigraphy.* Wiley.
- Grimm, G. R. (1987). *CONISS: a FORTRAN program for stratigraphic
  correlation.* Computers & Geosciences 13, 1177–1218.
- Simpson, G. G. (1947). *The role of fossils in establishing the
  sequence of continental and marine deposits.* Journal of Paleontology
  21, 303–315.
- Agterberg, F. P. (1963). *The method of computing the 95% confidence
  limits of the age of a biostratigraphic unit.* AAPG Bulletin 47,
  1813–1818.
- Gradstein, J. W., Agterberg, F. P. &das, S. (2010). *RASC: a new tool
  for the calculation of the age of strata.* Palaeontology 53, 709–721.
- Foote, M. (2000). *Probabilistic extinction intervals: a method of
  estimating the distribution of extinction times.* Palaeobiology 7,
  303–312.
- Marshall, A. V. (1991). *Extinction interval estimation: a tutorial
  guide.* Paleobiology 17, 522–533.
- Kendall, M. G. (1975). *Non-parametric trend and its
  prewhitening.* Journal of the Royal Statistical Society B 137,
  140–153.
- Mann, H. B. (1945). *Nonparametric tests against trend.* Econometrica
  13, 245–259.
- Sen, P. K. (1968). *Asymptotic distribution of Kendall's test of
  trend.* Journal of the American Statistical Association 63, 1119–1138.
- Wald, A. & Wolfowitz, J. (1940). *On a test whether two sequences are
  the same.* Journal of the American Statistical Association 35, 179–186.
- Steller, D. (1999). *REDFIT: a program for analysis of paleoclimate
  series.* Computers & Geosciences 25, 1–4.
- Thomson, D. J. (1982). *Spectrum estimation and harmonic
  analysis.* Proceedings of the IEEE 70, 1055–1096.
- Box, G. E. P. & Jenkins, G. M. (1976). *Time Series Analysis:
  Forecasting and Control.* 2nd ed. Holden-Day.
- Torrence, C. & Compo, G. P. (1998). *A practical guide to wavelet
  analysis.* Bulletin of the American Meteorological Society 79, 61–78.
- Lisiecki, L. E. & Raymo, M. E. (2005). *LR04 benthic foraminiferal
  δ18O stack.* Paleoceanography and Paleoclimatology 20, PA2003.
- Li, M., Hinnov, L. A., Huang, C. & Ogg, J. (2018). *Sedimentary noise and
  sea levels linked to land–ocean water exchange and obliquity forcing.*
  Nature Communications 9, 1004.
- Li, M., Kump, L. R., Hinnov, L. A. & Mann, M. E. (2018). *Tracking variable
  sedimentation rates and astronomical forcing in Phanerozoic paleoclimate
  proxy series with evolutionary correlation coefficients and hypothesis
  testing.* Earth and Planetary Science Letters 501, 165–179.
- Li, M., Huang, C., Hinnov, L., Ogg, J., Chen, Z.-Q. & Zhang, Y. (2016).
  *Obliquity-forced climate during the Early Triassic hothouse in China.*
  Geology 44, 623–626.
- Li, M., Hinnov, L. A. & Kump, L. R. (2019). *Acycle: Time-series analysis
  software for paleoclimate research and education.* Computers &
  Geosciences 127, 12–22.
- Laskar, J., Gastineau, F., Bouttes, I. & obliquity contributors (2004).
  *Solution for the eccentricity and obliquity of the solar system over
  the last 10 million years.* Astronomy & Astrophysics 413, 899–924.
- Lisiecki, L. E., Hayes, M. D. & Raymo, M. E. (2005). *The role of
  orbital pacing in glacial cycles.* Paleoceanography 20, PA2002.
- Meyers, S. R. (2015). *The evaluation of eccentricity-related amplitude
  modulation and bundling in paleoclimate data.* Paleoceanography 30,
  1172–1184.
- Meyers, S. R. (2019). *Cyclostratigraphy and the problem of
  astrochronologic testing.* Earth-Science Reviews 190, 190–223.
- Kodama, K. P. & Hinnov, L. A. (2015). *Rock Magnetic
  Cyclostratigraphy.* Wiley-Blackwell.
- Huang, N. E., Shen, Z., Long, M. C., Frank, M. C., Peacock, A. D. & Liu,
  S. H. (1998). *The empirical mode decomposition and the Hilbert spectrum
  in nonlinear dynamics.* Proceedings of the Royal Society A 454, 903–995.
- Wu, Z. & Huang, N. E. (2009). *Ensemble empirical mode decomposition: a
  noise-assisted data analysis method.* Advances in Data Science and
  Engineering 1, 1–41.
- Cleveland, W. S. (1979). *Robust locally weighted regression and
  smoothing scatterplots.* Journal of the American Statistical
  Association 74, 829–836.
- Ruggieri, E. (2013). *A Bayesian approach to detecting change points in
  climatic records.* International Journal of Climatology 33, 520–528.
- Li, M., Hinnov, L. A., Huang, C. & Ogg, J. (2018). *Sedimentary noise and
  sea levels: the rho1 statistic.* Nature Communications 9, 1004.
- Epstein, S. (1955). *Variations in O18/O16 in deep-sea carbonate
  sediments and a temperature scale.* Science 121, 701–704.
- Leng, C. J. & Marshall, H. (2012). *Present and past sea level
  within the Cenozoic and its implications for the analysis of δ18O
  data.* Palaeogeography, Palaeoclimatology, Palaeoecology 379, 1–10.
- Lisiecki, L. E. & Lisiecki, J. H. (2018). *Assessing the accuracy of
  age-depth curves and the oxygen isotope chronologies.* Paleoceanography
  and Paleoclimatology 33, 1169–1190.

## 分析清单

<!-- matrix:start -->
| 分析 | 模块 | 验证证据 | 说明 |
|---|---|---|---|
| `acf` | `stratigraphy.cycles` | 单元 | Autocorrelation function with Bartlett bands |
| `add_gaps` | `stratigraphy.series_tools` | 性质 | Insert gaps into a record |
| `age_depth_uncertainty` | `stratigraphy.age_depth_uncertainty` | 真值 | An age-depth model with the band its dating points imply |
| `age_model` | `stratigraphy.correlation` | 单元 | Depth to age model |
| `age_scale` | `stratigraphy.tuning` | 真值 | Build an age scale from depth anchors |
| `amplitude_modulation` | `stratigraphy.amplitude_modulation` | 性质 | Instantaneous amplitude envelope of a record (Tanner-Hilbert) |
| `apply_function` | `stratigraphy.transforms` | 真值 | Apply a simple function to a series |
| `arma` | `stratigraphy.arma` | 单元 | Autoregressive moving average model |
| `astronomical_solution` | `stratigraphy.astronomy` | 真值 | The astronomical solution: orbital elements through deep time |
| `astronomical_target_curve` | `stratigraphy.astronomy` | 真值 | A synthetic astronomical target curve at one Milankovitch period |
| `autoassociation` | `stratigraphy.cycles` | 单元 | Autoassociation function for cyclostratigraphy |
| `bandpass_filter` | `stratigraphy.filtering` | 真值 | Band-pass filter between two periods |
| `bayesian_periodogram` | `stratigraphy.spectral_significance` | 真值 | The probability that a period in a record is real |
| `bicoherence_significance` | `stratigraphy.bispectrum` | 真值 | How strong the triple coupling has to be to be more than chance |
| `bispectrum` | `stratigraphy.bispectrum` | 真值 | The bispectrum of a record, and the phase of its triples |
| `box_cox_transform` | `stratigraphy.boxcox` | 真值 | Box-Cox transform, with the lambda solved for |
| `change_points` | `stratigraphy.change_points` | 真值 | Bayesian change-point detection (Ruggieri 2013) |
| `chi_square_confidence_interval` | `stratigraphy.small_tools` | 真值 | The confidence band of a red-noise spectrum |
| `circular_spectrum` | `stratigraphy.circular_spectrum` | 真值 | The spectrum of an angular variable: its harmonics and mean direction |
| `clip_values` | `stratigraphy.series_tools` | 性质 | Clip a record to a range |
| `coco` | `stratigraphy.coco` | 真值 | Correlation coefficient: the most likely sedimentation rate (Li et al. 2018) |
| `coco_adaptive` | `stratigraphy.coco_advanced` | 真值 | COCO with a target spectrum that adapts to the record |
| `coco_crossfit` | `stratigraphy.coco_advanced` | 真值 | COCO that estimates and tests on separate halves of the record |
| `coco_interleaved` | `stratigraphy.coco_advanced` | 真值 | COCO over interleaved short segments, for records with gaps |
| `coco_orbit_tracking` | `stratigraphy.coco_advanced` | 真值 | The orbital peaks of a COCO result, followed through the record |
| `coco_spearman` | `stratigraphy.coco_variants` | 真值 | COCO with the rank correlation instead of the Pearson one |
| `coherence` | `stratigraphy.coherence` | 真值 | Coherence and phase between two records |
| `coniss` | `stratigraphy.coniss` | 单元 | CONISS constrained cluster analysis |
| `correlation_confidence_interval` | `stratigraphy.coco_variants` | 真值 | The confidence interval of a correlation coefficient |
| `cosine_taper` | `stratigraphy.small_tools` | 真值 | The cosine taper that takes the edge off a record |
| `cross_correlation` | `stratigraphy.cycles` | 单元 | Cross-correlation with a signed lag axis |
| `date_to_day_number` | `stratigraphy.small_tools` | 真值 | A calendar date as a day number |
| `depth_to_time` | `stratigraphy.depth_time` | 真值 | Depth to age, through a set of age anchors |
| `derivative` | `stratigraphy.series_tools` | 性质 | Numerical derivative of a record |
| `detrend_polynomial_pp` | `stratigraphy.small_tools` | 真值 | A polynomial baseline fitted in a sliding window |
| `detrend_series` | `stratigraphy.detrending` | 真值 | Detrend with a mean, a polynomial, a smoother or the EEMD residual |
| `directional` | `stratigraphy.directional` | 单元 | Circular and axial directional statistics |
| `dispersion_statistic` | `stratigraphy.dispersion` | 真值 | How far a set of counts departs from what was expected |
| `domain_transform` | `stratigraphy.small_tools` | 真值 | The analytic signal of a record, or its cumulative integral |
| `dynamic_filter` | `stratigraphy.dynamic_filter` | 真值 | Adaptive filter: the noise level decides how hard each point is smoothed |
| `dynot` | `stratigraphy.dynot` | 真值 | The sedimentary noise in a record, against an AR(1) or white baseline |
| `ecoco` | `stratigraphy.coco` | 真值 | Evolutionary correlation coefficient: a sedimentation rate that changes |
| `ecoco_row_summary` | `stratigraphy.coco_variants` | 真值 | The eCOCO grid, summarised window by window |
| `eemd` | `stratigraphy.emd` | 真值 | Ensemble empirical mode decomposition |
| `emd` | `stratigraphy.emd` | 真值 | Empirical mode decomposition |
| `estimate_ar1_rho` | `stratigraphy.spectral_moments` | 真值 | The AR(1) coefficient of a record |
| `etime_opt` | `stratigraphy.tuning` | 真值 | eTimeOpt astrochronological tuning with a changing sedimentation rate (Meyers 2019) |
| `evolutionary_lomb_scargle` | `stratigraphy.evolutionary` | 性质 | Sliding-window Lomb-Scargle spectrum for unevenly spaced records |
| `evolutionary_multitaper` | `stratigraphy.evolutionary` | 性质 | Sliding-window multitaper spectrum |
| `evolutionary_rho1` | `stratigraphy.evolutionary_rho` | 真值 | The AR(1) coefficient of a record, tracked through a sliding window |
| `evolutionary_spectrum` | `stratigraphy.evolutionary` | 性质 | Sliding-window spectrum: how the periodic content moves through a record |
| `extinction` | `stratigraphy.extinction` | 单元 | Extinction interval estimation |
| `find_n_peaks` | `stratigraphy.small_tools` | 真值 | The positions of the local maxima of a record |
| `fixed_bandwidth_moving_average` | `stratigraphy.small_tools` | 真值 | A moving average over a fixed span of time rather than of samples |
| `fixed_bandwidth_summary` | `stratigraphy.moving_summary` | 真值 | Mean, spread and extremes over a window of fixed time width |
| `fixed_count_summary` | `stratigraphy.moving_summary` | 真值 | Mean, spread and extremes over a window of fixed sample count |
| `fixed_target_weights` | `stratigraphy.coco_advanced` | 真值 | The weights a fixed target spectrum gives its periods |
| `frequency_stabilization` | `stratigraphy.frequency_stabilization` | 真值 | A band-pass filter whose passband follows a drifting cycle |
| `gaussian_filter` | `stratigraphy.filtering` | 真值 | Gaussian low-pass filter |
| `gaussian_matched_filter` | `stratigraphy.small_tools` | 真值 | Where a known shape best fits a record |
| `generate_signal` | `stratigraphy.signal_generator` | 性质 | Synthesise a signal from components plus optional noise |
| `insolation` | `stratigraphy.astronomy` | 真值 | Daily-mean insolation at a latitude, from the astronomical solution |
| `insolation_series` | `stratigraphy.cycles` | 单元 | Insolation forcing sampled on a time axis |
| `interpolate_series` | `stratigraphy.series_tools` | 性质 | Interpolate a record onto a new time axis |
| `inverse_percentile` | `stratigraphy.inverse_percentile` | 真值 | The value below which a given fraction of a sample falls |
| `isotope_paleotemperature` | `stratigraphy.isotope_analysis` | 单元 | Stable isotope paleothermometry |
| `lead_lag_rmse` | `stratigraphy.lead_lag` | 真值 | The lag that makes two records agree best, by least squares |
| `lod_dimension` | `stratigraphy.lod` | 真值 | Length of day through deep time (Laskar et al. 2004) |
| `log_transform` | `stratigraphy.transforms` | 真值 | Log-transform a series |
| `lr04_stack` | `stratigraphy.lr04` | 真值 | The LR04 benthic d18O stack (Lisiecki & Raymo 2005) |
| `mann_kendall` | `stratigraphy.cycles` | 单元 | Mann-Kendall trend test with Sen's slope |
| `markov` | `stratigraphy.markov` | 单元 | Markov chain state transitions |
| `milankovitch_periods` | `stratigraphy.astronomy` | 真值 | The Milankovitch periods and their nominal amplitudes |
| `milankovitch_uncertainty` | `stratigraphy.milankovitch_uncertainty` | 真值 | The orbital frequencies at an age, with the interval they lie in |
| `mini_robot` | `stratigraphy.mini_robot` | 真值 | Replay a recorded sequence of analyses over a batch of data |
| `minimum_rho` | `stratigraphy.spectral_moments` | 真值 | The smallest AR(1) coefficient that explains a record |
| `moving_average_smooth` | `stratigraphy.smoothing` | 真值 | Smoothing with a moving average of a given span |
| `moving_median` | `stratigraphy.moving_median` | 真值 | Moving median filter |
| `multitaper` | `stratigraphy.cycles` | 单元 | Multitaper spectral estimation |
| `multitaper_taper_count` | `stratigraphy.mtm_tuning` | 真值 | The number of tapers a multitaper estimate should use |
| `num_orbits` | `stratigraphy.coco_advanced` | 真值 | How many orbital cycles a record can hold |
| `optimal_span_smooth` | `stratigraphy.smoothing` | 真值 | Smoothing with the span chosen by cross-validation, plus a bootstrap band |
| `orbit_groups` | `stratigraphy.coco_advanced` | 真值 | The orbital periods, grouped into the orbits they belong to |
| `orbital_forcing` | `stratigraphy.cycles` | 单元 | Milankovitch periods and relative amplitudes |
| `orbital_frequencies` | `stratigraphy.orbital_frequencies` | 真值 | The nine principal orbital frequencies at an age |
| `pad_series` | `stratigraphy.small_tools` | 真值 | A record padded out to a given length |
| `pda` | `stratigraphy.pda` | 真值 | Power decomposition analysis into orbital-period bands (Li et al. 2016) |
| `phase_diagram` | `stratigraphy.small_tools` | 真值 | The trajectory of a record in the complex plane |
| `plot_digitizer` | `stratigraphy.plot_digitizer` | 真值 | Read a curve off a published figure |
| `prewhiten` | `stratigraphy.cycles` | 单元 | AR(1) prewhitening of a series |
| `radiocarbon_calibration` | `stratigraphy.radiocarbon` | 真值 | A radiocarbon determination turned into a calendar age |
| `random_phase_resample` | `stratigraphy.small_tools` | 真值 | A surrogate of a record, with its phases randomised |
| `rasc` | `stratigraphy.biostratigraphy` | 单元 | Ranging ancilla time scaling |
| `recurrence_plot` | `stratigraphy.recurrence` | 真值 | The recurrence plot of a record, and the numbers read off it |
| `redfit` | `stratigraphy.cycles` | 单元 | REDFIT spectral analysis against an AR(1) null |
| `remove_part` | `stratigraphy.series_edit` | 真值 | Remove one interval from a record |
| `remove_peaks` | `stratigraphy.series_tools` | 性质 | Remove peaks above a threshold |
| `remove_range` | `stratigraphy.series_edit` | 真值 | Keep one interval of a record and drop the rest |
| `rho1_sedimentation_ratio` | `stratigraphy.dynot` | 真值 | The rho1 ratio: the record's noise power against the chosen baseline |
| `robust_linear_fit` | `stratigraphy.small_tools` | 真值 | A straight line through points that include outliers |
| `robust_lomb_scargle` | `stratigraphy.robust_lomb` | 真值 | A Lomb-Scargle spectrum that a red-noise background cannot explain away |
| `runs_test` | `stratigraphy.cycles` | 单元 | Runs test for randomness of a series |
| `sedimentation_rate` | `stratigraphy.correlation` | 单元 | Sedimentation rate from an age model |
| `sedimentation_rate_age_model` | `stratigraphy.tuning` | 真值 | Turn a sedimentation rate into an age model |
| `select_intervals` | `stratigraphy.series_tools` | 性质 | Take the parts of a record that fall inside given intervals |
| `sliding_spectral_moments` | `stratigraphy.sliding_spectral_moments` | 真值 | The shape of the spectrum, measured in a sliding window |
| `sort_unique` | `stratigraphy.series_tools` | 性质 | Sort a column and drop its duplicates |
| `spectral` | `stratigraphy.spectral_analysis` | 单元 | Periodogram with an AR(1) red-noise test |
| `spectral_moments` | `stratigraphy.spectral_moments` | 真值 | Spectral moments: the shape of a power spectrum |
| `spectral_peaks` | `stratigraphy.spectral_analysis` | 单元 | Significant spectral periods |
| `standardize` | `stratigraphy.transforms` | 真值 | Standardise a series (z-score or min-max) |
| `stratigraphic_correlation` | `stratigraphy.correlation` | 性质 | Composite standard section correlation |
| `swa_background` | `stratigraphy.spectral_significance` | 真值 | The smoothed-window-average background of a spectrum |
| `swa_fdr` | `stratigraphy.spectral_significance` | 真值 | Which peaks of a spectrum survive the false-discovery rate |
| `taner_filter` | `stratigraphy.filtering` | 真值 | Taner weighted band-pass filter (Kodama & Hinnov 2015) |
| `tanner_hilbert` | `stratigraphy.amplitude_modulation` | 性质 | Tanner-Hilbert transform: envelope and instantaneous phase |
| `test_uniform_sampling` | `stratigraphy.sampling` | 真值 | Whether a record was sampled at even intervals |
| `time_opt` | `stratigraphy.tuning` | 真值 | TimeOpt astrochronological tuning (Meyers 2015) |
| `tobit_regression` | `stratigraphy.small_tools` | 真值 | A regression with a censored response |
| `tuned_multitaper_spectrum` | `stratigraphy.mtm_tuning` | 真值 | Multitaper spectrum with the tapers and the degrees of freedom chosen |
| `ua` | `stratigraphy.biostratigraphy` | 单元 | Unitary associations |
| `wavelet` | `stratigraphy.spectral_analysis` | 单元 | Continuous wavelet transform |
| `wavelet_coherence` | `stratigraphy.wavelet_coherence` | 真值 | The wavelet coherence between two records, and its phase |
| `wavelet_smoothing` | `stratigraphy.small_tools` | 真值 | A record denoised in the wavelet domain |
| `weedon_periodogram` | `stratigraphy.spectral_significance` | 真值 | A periodogram with the degrees of freedom the false-discovery rate needs |
<!-- matrix:end -->
