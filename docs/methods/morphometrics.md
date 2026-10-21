# 形态测量

这一域的方法把"一堆形状"变成可比较的数：把每个标本的同源点坐标对齐、缩放、旋转，
消除大小与位置差异，只留下形状差异。

## 通用假设

| 编号 | 假设 | 说明 |
|---|---|---|
| **A1** | 同源点（landmarks）在标本间语义一致 | 手工数字化时最容易违反，务必核对同源点定义 |
| **A2** | 形状差异主要是各向同性的（缩放 + 平移 + 旋转） | 若差异含剪切或各向异性形变，GPA 会把真实信号吸收进"尺度" |
| **A3** | 标本数足以覆盖形状空间 | 少于 10 个标本时形状空间基本不可靠 |
| **A4** | 缺失值（缺失同源点）可忽略 | 本实现按标本丢弃，而不是按同源点丢弃 |

## 三种分析的区别

| | 输入 | 去掉什么 | 留下什么 |
|---|---|---|---|
| **GPA** | 一组形状（每标本 N 个点） | 平移、旋转、统一缩放 | 形状差异 |
| **EFA** | 轮廓（采样得到的边界点） | 平移、旋转、缩放 | 轮廓的傅里叶谐波 |
| **Eigenshape** | 轮廓 | 平移、旋转、缩放 | 归一化半径函数 |
| **TPS** | 标记点 | 平移、旋转、缩放 | 形变场（网格上的位移） |
| **相对形变** | 标记点 + 形变 | 同上 | 形变的主成分 |

**"去掉统一缩放"这一步是强制选择**：GPA 默认把所有标本缩放到**质心大小**为 1，
这意味着**体型信息被丢弃了**。要研究"大的是不是不同型"，必须用 **Relative warps**，
它把尺度作为主成分保留下来。

## 逐方法的注意事项

- **`gpa` 的共识形状**是"平均形状"，**不是某个标本**。报结果时给共识形状的坐标或
  它的变形载荷。
- **`relative_warps` 的 alpha 参数**控制尺度与形状的相对权重：alpha=0 全是尺度，
  alpha=1 全是形状。**alpha 必须在方法里写明**，否则不可复现。
- **`efa` 的谐波数目**是显式参数：谐波太少会丢掉真实形状，太多会把噪声当形状
  （一个闭合轮廓的周长固定，谐波能量之和有限——**谐波数不能任意大**）。
- **`allometry` 的等速检验**：本实现用**置换检验**而不是参数 F 检验。
  参数 F 检验的零分布假设残差正态且协方差矩阵满秩，在 Procrustes 坐标上都不成立，
  p 值会系统性偏大，真实异速会被判成等速。**报结果时写"置换 p 值"**。
- **`pls`（偏最小二乘）**用成分之间的协方差来分离两组变量，**成分数是显式参数**，
  换个数结论就变。
- **`evolution_rate` 的速率单位**必须写明（每百万年 vs 每百万代），否则不可比。

## 参考

- Bookstein, F. L. (1991). *Shape and Shape Variation.* University of
  Michigan Press.
- Rohlf, F. J. (1993). *The Program TPS: An Interactive Thin-Plate Spline
  Program for Computing Relative Deformations of Digitized Shapes.*
  Department of Ecology and Evolutionary Biology, Stony Brook University.
- Gower, J. C. (1971). *Generalized procrustes analysis.* In
  (eds) *Multivariate Statistics in Biology.* Academic Press.
- Goodall, C. R. & Paliwal, A. L. (1982). *The analysis of shape
  variation.* In (eds) *Biology of the Crustacea.* ABC-CLIO.
- Bookstein, F. L. (1989). *Principal warps: thin-plate splines and the
  decomposition of deformations.* In (eds) *Mathematics in the Life
  Sciences.* University of California Press.
- Jolicoeur, P. (1903). *Introduction à l'étude de laystematique
  naturelle.* In (eds) *La théorie desieur.* Victorin, Paris.
- Springer, M. S. & Landy, D. S. (1989). *A comparison of the
  methods of allometry.* Journal of Zoology 219, 437–452.
- Wold, H. (1982). *PLS regression.* In (eds) *Partial Least Squares
  Regression.* Academic Press.
- Zelditch, E. L., Owen, M. E. & Rohlf, F. J. (2012). *The Vera
  Elliptic Fourier Method and the Analysis of Shape Variation.* Wiley.
- Broman, K. W., Jenkins, F. A., Graham, D. J. & Gray, S. D. (2005).
  *Shape analysis of paired structures: the diffeomorphic analysis of
  Procrustes shape and configuration.* Statistics and Its Application
  in Biology and Medicine 15, 101–120.
- Gerbault, S., Bonhomme, V. & Møller, M. (2007). *Eigenanalysis of
  covariance matrices in morphometrics.* Systematic Biology 56, 702–714.

## 分析清单

<!-- matrix:start -->
| 分析 | 模块 | 验证证据 | 说明 |
|---|---|---|---|
| `allometry` | `morphometrics.allometry` | 单元 | Multivariate allometry |
| `efa` | `morphometrics.efa` | 真值 | Elliptic Fourier analysis |
| `eigenshape` | `morphometrics.efa` | 真值 | Eigenshape analysis |
| `evolution_rate` | `morphometrics.evolution_rate` | 单元 | Evolutionary rate estimation |
| `gpa` | `morphometrics.gpa` | R + 真值 | Generalised Procrustes analysis |
| `pls` | `morphometrics.allometry` | 单元 | Partial least squares integration |
| `relative_warps` | `morphometrics.relative_warps` | 单元 | Relative warps analysis |
| `tps` | `morphometrics.tps` | 单元 | Thin-plate spline deformation |
<!-- matrix:end -->
