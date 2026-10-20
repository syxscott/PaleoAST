# PaleoAST 设计系统

> **本文件已被 `config/design_system.py` 取代。**
>
> 早先这里有一份 "flat UI" 规范（主蓝 `#3498DB`、正文 `#2C3E50`、错误 `#E74C3C`）。
> 那套色板**早已不再使用**：它写在这里的每一张色值表，如今都是仓库里被明令禁止的
> 硬编码值——`tests/test_design_conformance.py` 会在 CI 上把它们拦下来。
> 保留这份文档而不删掉它，是因为它比删掉更能说明问题：同一个仓库里曾经同时活着
> 三套配色，而其中两套没人知道是过时的。
>
> **代码以 `config/design_system.py` 为准。** 它的 docstring 写得很明确：
> "This module, not the docstring of any other file."

## 令牌在哪

| 要找什么 | 去哪 |
|---|---|
| 主色 / 语义色 / 中性色（浅色与暗色两套） | `config/design_system.py` 的 `ColorPalette` / `ColorPaletteDark` |
| 间距（4px 网格） | 同文件的 `Spacing` |
| 圆角 | 同文件的 `BorderRadius` |
| 界面字号 | 同文件的 `Typography` |
| 图表字号 | 同文件的 `FigureTypography` |
| 全局 QSS | 同文件的 `get_modern_stylesheet(palette)` / `get_stylesheet(dark)` |
| **数据系列配色**（与 R 导出共用） | `config/colors.py` 的 `PALETTES` / `current_palette()` |

## 三条规矩

1. **界面里不写死颜色。** 需要一个颜色，就去令牌里取。
   `tests/test_design_conformance.py` 把「已废弃的 flat-UI 色板不得出现在界面层」
   变成了 CI 上可执行的断言。
2. **数据色板只有一份。** `config/colors.py` 的登记表同时喂给交互图和 R 导出，
   所以**屏幕上的图和导出的图是同一张图**。改色板要改那里，不要在绘图模块里另开一份。
3. **方向是减法。** 仓库现状是 0 阴影、1px 边框、2–6px 圆角，
   这已经比 VSCode（0–2px 圆角、5px 输入框上限）更圆润了。
   正确方向是**删硬编码、收敛到令牌、去掉多余的颜色**，
   不是加阴影、渐变、卡片式圆角。

## 还没有做完的

- 数据系列配色此前仍有约 60 处硬编码，已于 2026-10-20 统一到注册色板；
- 焦点环此前只有输入框有，现已按 VSCode 的 `focusBorder` 原则补齐到所有可聚焦控件；
- 导航树此前只能用鼠标点中，现已可以 Tab 到达；
- 4px 网格守卫此前只覆盖 `config/design_system.py`，现已扩展到 `views/` 与 `visualization/`。

其余待办见 `FIX_PLAN_2026-10-19.md`。
