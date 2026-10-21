# 演化与生长

这一域的方法描述"一个量怎么随时间变"：存活曲线、风险率、个体的生长曲线。
它们的共同问题是**删失数据**（censoring）与**模型选择**。

## 通用假设

| 编号 | 假设 | 说明 |
|---|---|---|
| **A1** | 事件时间与观测单位定义清楚 | "灭绝"是最后一个化石的年龄，还是一个区间？ |
| **A2** | 删失是独立删失（右删失） | 区间删失（只知道"在这段时间内灭绝"）需要专门方法 |
| **A3** | 风险率模型的形式被检验过 | 比例风险假设被违反时，系数没有统一的解释 |
| **A4** | 生长曲线的渐近线有意义 | 没有上限的 S 形曲线在数学上可能不存在 |

## 逐方法的注意事项

- **`kaplan_meier`（Kaplan-Meier 生存曲线）**在**删失比例高或删失与分组相关**时
  会系统性偏倚。生存曲线的横轴是**时间**，纵轴是**存活比例**，两边都要标单位。
- **`cox_ph`（Cox 比例风险）**的系数是**对数风险比**。
  **比例风险假设（PH）必须检验**：违反时系数只在某段时间内成立，
  曲线的形状也会变。报告应给出 Schoenfeld 残差的检验结果。
- **`growth_model`（非线性生长曲线拟合）**：
  - **参数的标准误与 95% 区间**：本实现修正了 `absolute_sigma` 的用法
    （`utils/statistics_core.py`）。修正前区间**窄了 3.3 倍**——
    报告的区间是错的。
  - **模型比较用 AICc**（小样本下 AIC 的偏差修正版），不是 AIC。
  - **初值会决定收敛到哪个解**：非线性拟合是多解的，**初值必须写进方法**。
- **`growth_model_comparison`** 给出多个生长曲线的 AICc 排名。
  **ΔAICc < 2 的模型在统计上无法区分**，不应只报第一名。

## 参考

- Kaplan, E. L. & Meier, P. (1958). *Nonparametric estimation from
  incomplete observations.* Journal of the American Statistical
  Association 53, 457–481.
- Cox, D. R. (1972). *Regression models and life-tables.* Journal of
  the Royal Statistical Society B 34, 187–220.
- Schoenfeld, D. (1982). *Partial residuals for the proportional hazards
  model.* Journal of the Royal Statistical Society B 44, 265–279.
- Grambsch, P. M. & Therneau, T. M. (1994). *Proportional hazards models
  and diagnostics based on weighted residuals.* Biometrika 81, 515–526.
- von Bertalanffy, L. (1938). *Über die Theorie der plastischen
  Körpergröße.* Archiv für die gesamte experimentelle Medizin und
  Physiologie 115, 423–439.
- Richards, F. J. (1959). *The use of growth curves for analysing
  core data.* Journal of the Paleontological Society 33, 222–224.
- Schnute, J. E. (1979). *A general model for fish growth.*
  Mathematical Biosciences 28, 141–154.
- Hurvich, Z. E. & Ts AIC, C. L. (1978). *A report on the goodness of fit
  of individual models.* Canadian Journal of Statistics 6, 86–89.
- Sugiura, H. (1978). *Comments on the use of AIC for the assessment of
  goodness of fit.* American Statistician 32, 118–121.
- Burnham, K. P. & Anderson, D. R. (2002). *Model Selection and
  Multimodel Inference.* 2nd ed. Springer.
- Royston, P. & Parmar, M. K. B. (2013). *Restricted cubic splines
  for the Cox model.* Statistics in Medicine 32, 4775–4790.

## 分析清单

<!-- matrix:start -->
| 分析 | 模块 | 验证证据 | 说明 |
|---|---|---|---|
| `cox_ph` | `macroevolution.survival` | 单元 | Cox proportional hazards model |
| `growth_model` | `models.growth_models` | 单元 | Nonlinear growth curve fitting |
| `growth_model_comparison` | `models.growth_models` | 单元 | Fit several growth curves and rank them by AICc |
| `kaplan_meier` | `macroevolution.survival` | 单元 | Kaplan-Meier survivorship |
<!-- matrix:end -->
