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
| `age_model` | `stratigraphy.correlation` | 单元 | Depth to age model |
| `age_scale` | `stratigraphy.tuning` | 真值 | Build an age scale from depth anchors |
| `amplitude_modulation` | `stratigraphy.amplitude_modulation` | 性质 | Instantaneous amplitude envelope of a record (Tanner-Hilbert) |
| `arma` | `stratigraphy.arma` | 单元 | Autoregressive moving average model |
| `astronomical_solution` | `stratigraphy.astronomy` | 真值 | The astronomical solution: orbital elements through deep time |
| `astronomical_target_curve` | `stratigraphy.astronomy` | 真值 | A synthetic astronomical target curve at one Milankovitch period |
| `autoassociation` | `stratigraphy.cycles` | 单元 | Autoassociation function for cyclostratigraphy |
| `bandpass_filter` | `stratigraphy.filtering` | 真值 | Band-pass filter between two periods |
| `change_points` | `stratigraphy.change_points` | 真值 | Bayesian change-point detection (Ruggieri 2013) |
| `clip_values` | `stratigraphy.series_tools` | 性质 | Clip a record to a range |
| `coco` | `stratigraphy.coco` | 真值 | Correlation coefficient: the most likely sedimentation rate (Li et al. 2018) |
| `coniss` | `stratigraphy.coniss` | 单元 | CONISS constrained cluster analysis |
| `cross_correlation` | `stratigraphy.cycles` | 单元 | Cross-correlation with a signed lag axis |
| `derivative` | `stratigraphy.series_tools` | 性质 | Numerical derivative of a record |
| `directional` | `stratigraphy.directional` | 单元 | Circular and axial directional statistics |
| `dynamic_filter` | `stratigraphy.dynamic_filter` | 真值 | Adaptive filter: the noise level decides how hard each point is smoothed |
| `ecoco` | `stratigraphy.coco` | 真值 | Evolutionary correlation coefficient: a sedimentation rate that changes |
| `eemd` | `stratigraphy.emd` | 真值 | Ensemble empirical mode decomposition |
| `emd` | `stratigraphy.emd` | 真值 | Empirical mode decomposition |
| `estimate_ar1_rho` | `stratigraphy.spectral_moments` | 真值 | The AR(1) coefficient of a record |
| `etime_opt` | `stratigraphy.tuning` | 真值 | eTimeOpt astrochronological tuning with a changing sedimentation rate (Meyers 2019) |
| `evolutionary_lomb_scargle` | `stratigraphy.evolutionary` | 性质 | Sliding-window Lomb-Scargle spectrum for unevenly spaced records |
| `evolutionary_multitaper` | `stratigraphy.evolutionary` | 性质 | Sliding-window multitaper spectrum |
| `evolutionary_spectrum` | `stratigraphy.evolutionary` | 性质 | Sliding-window spectrum: how the periodic content moves through a record |
| `extinction` | `stratigraphy.extinction` | 单元 | Extinction interval estimation |
| `fixed_bandwidth_summary` | `stratigraphy.moving_summary` | 真值 | Mean, spread and extremes over a window of fixed time width |
| `fixed_count_summary` | `stratigraphy.moving_summary` | 真值 | Mean, spread and extremes over a window of fixed sample count |
| `gaussian_filter` | `stratigraphy.filtering` | 真值 | Gaussian low-pass filter |
| `generate_signal` | `stratigraphy.signal_generator` | 性质 | Synthesise a signal from components plus optional noise |
| `insolation` | `stratigraphy.astronomy` | 真值 | Daily-mean insolation at a latitude, from the astronomical solution |
| `insolation_series` | `stratigraphy.cycles` | 单元 | Insolation forcing sampled on a time axis |
| `interpolate_series` | `stratigraphy.series_tools` | 性质 | Interpolate a record onto a new time axis |
| `isotope_paleotemperature` | `stratigraphy.isotope_analysis` | 单元 | Stable isotope paleothermometry |
| `mann_kendall` | `stratigraphy.cycles` | 单元 | Mann-Kendall trend test with Sen's slope |
| `markov` | `stratigraphy.markov` | 单元 | Markov chain state transitions |
| `milankovitch_periods` | `stratigraphy.astronomy` | 真值 | The Milankovitch periods and their nominal amplitudes |
| `minimum_rho` | `stratigraphy.spectral_moments` | 真值 | The smallest AR(1) coefficient that explains a record |
| `multitaper` | `stratigraphy.cycles` | 单元 | Multitaper spectral estimation |
| `orbital_forcing` | `stratigraphy.cycles` | 单元 | Milankovitch periods and relative amplitudes |
| `pda` | `stratigraphy.pda` | 真值 | Power decomposition analysis into orbital-period bands (Li et al. 2016) |
| `prewhiten` | `stratigraphy.cycles` | 单元 | AR(1) prewhitening of a series |
| `rasc` | `stratigraphy.biostratigraphy` | 单元 | Ranging ancilla time scaling |
| `redfit` | `stratigraphy.cycles` | 单元 | REDFIT spectral analysis against an AR(1) null |
| `remove_peaks` | `stratigraphy.series_tools` | 性质 | Remove peaks above a threshold |
| `runs_test` | `stratigraphy.cycles` | 单元 | Runs test for randomness of a series |
| `sedimentation_rate` | `stratigraphy.correlation` | 单元 | Sedimentation rate from an age model |
| `sedimentation_rate_age_model` | `stratigraphy.tuning` | 真值 | Turn a sedimentation rate into an age model |
| `select_intervals` | `stratigraphy.series_tools` | 性质 | Take the parts of a record that fall inside given intervals |
| `sort_unique` | `stratigraphy.series_tools` | 性质 | Sort a column and drop its duplicates |
| `spectral` | `stratigraphy.spectral_analysis` | 单元 | Periodogram with an AR(1) red-noise test |
| `spectral_moments` | `stratigraphy.spectral_moments` | 真值 | Spectral moments: the shape of a power spectrum |
| `spectral_peaks` | `stratigraphy.spectral_analysis` | 单元 | Significant spectral periods |
| `stratigraphic_correlation` | `stratigraphy.correlation` | 性质 | Composite standard section correlation |
| `taner_filter` | `stratigraphy.filtering` | 真值 | Taner weighted band-pass filter (Kodama & Hinnov 2015) |
| `tanner_hilbert` | `stratigraphy.amplitude_modulation` | 性质 | Tanner-Hilbert transform: envelope and instantaneous phase |
| `time_opt` | `stratigraphy.tuning` | 真值 | TimeOpt astrochronological tuning (Meyers 2015) |
| `ua` | `stratigraphy.biostratigraphy` | 单元 | Unitary associations |
| `wavelet` | `stratigraphy.spectral_analysis` | 单元 | Continuous wavelet transform |
<!-- matrix:end -->
