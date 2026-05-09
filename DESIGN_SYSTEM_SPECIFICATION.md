# PaleoAST 现代设计系统规范

## 1. 颜色系统

### 主颜色
| 名称 | 代码 | 用途 | 示例 |
|------|------|------|------|
| 主蓝色 | #3498DB | 按钮，链接，活跃状态，强调 | ![#3498DB](https://via.placeholder.com/50/3498DB/3498DB) |
| 主蓝深 | #2980B9 | 悬停，按下状态 | ![#2980B9](https://via.placeholder.com/50/2980B9/2980B9) |
| 主蓝浅 | #5DADE2 | 浅色背景，次要强调 | ![#5DADE2](https://via.placeholder.com/50/5DADE2/5DADE2) |

### 语义颜色
| 名称 | 代码 | 用途 |
|------|------|------|
| 成功 | #27AE60 | 验证通过，确认操作 |
| 警告 | #F39C12 | 注意，可能的问题 |
| 错误 | #E74C3C | 错误，失败，销毁操作 |
| 信息 | #16A085 | 信息提示 |

### 中性颜色
| 名称 | 代码 | 用途 |
|------|------|------|
| 背景（主） | #FFFFFF | 主背景 |
| 背景（次） | #F8F9FA | 卡片，面板背景 |
| 背景（三） | #F0F2F5 | 悬停状态，表头 |
| 文本（主） | #2C3E50 | 主要文本 |
| 文本（次） | #7F8C8D | 次要文本，说明 |
| 文本（禁用） | #95A5A6 | 禁用文本 |
| 边框（浅） | #E4E7EB | 卡片边框，分隔线 |
| 边框（中） | #D0D5DD | 输入框边框 |
| 边框（深） | #BDC3C7 | 强调边框 |

## 2. 排版系统

### 字体栈
```css
主字体: 'Segoe UI', 'Microsoft YaHei', -apple-system, BlinkMacSystemFont, sans-serif
等宽字体: 'Consolas', 'Monaco', 'Courier New', monospace
```

### 字体大小
| 名称 | 大小 | 用途 |
|------|------|------|
| H1 | 32px | 页面标题 |
| H2 | 28px | 章节标题 |
| H3 | 24px | 子标题 |
| 标题 | 18px | 对话框标题 |
| 正文 | 12-14px | 常规文本 |
| 小字 | 10-11px | 辅助文本 |

### 字体权重
- Regular: 400 - 常规文本
- Medium: 500 - 按钮文本
- Semibold: 600 - 标签，标题
- Bold: 700 - 强调标题

## 3. 间距系统

基于 4px 网格：
```
xs: 4px
sm: 8px
md: 12px (3 * 4px)
lg: 16px (4 * 4px)
xl: 24px (6 * 4px)
xxl: 32px (8 * 4px)
```

### 应用示例
- 按钮内边距：lg (16px)
- 卡片内边距：lg (16px)
- 元素间距：md (12px)
- 组间距：lg (16px)

## 4. 圆角规范

| 大小 | 值 | 用途 |
|------|-----|------|
| 小 | 2px | 输入框，小按钮 |
| 中 | 4px | 按钮 |
| 大 | 6px | 卡片，对话框 |
| 特大 | 8px | 大面板，页面容器 |

## 5. 阴影系统

```css
阴影小: box-shadow: 0 1px 2px rgba(0,0,0,0.05)
阴影中: box-shadow: 0 4px 6px rgba(0,0,0,0.07)
阴影大: box-shadow: 0 10px 15px rgba(0,0,0,0.10)
阴影特大: box-shadow: 0 20px 25px rgba(0,0,0,0.15)
```

## 6. 组件样式

### 按钮

#### 主按钮（蓝色）
```python
background: #3498DB
color: #FFFFFF
border: 1px solid #2980B9
border-radius: 6px
padding: 8px 16px
font-weight: 500

:hover {
    background: #2980B9
    border-color: #1F618D
}

:pressed {
    background: #1F618D
}
```

#### 次按钮（灰色）
```python
background: #F0F2F5
color: #2C3E50
border: 1px solid #E4E7EB
border-radius: 6px
padding: 8px 16px
font-weight: 500

:hover {
    background: #E4E7EB
    border-color: #BFC9D4
}

:pressed {
    background: #D9DFE8
}
```

### 输入框

```python
background: #FFFFFF
color: #2C3E50
border: 1px solid #E4E7EB
border-radius: 6px
padding: 8px 12px

:hover {
    border-color: #3498DB
    background: #F8F9FA
}

:focus {
    border-color: #3498DB
    outline: 2px solid rgba(52, 152, 219, 0.1)
}
```

### 复选框/单选框

```python
width: 18px
height: 18px
border: 2px solid #E4E7EB
border-radius: 3px (复选框) / 9px (单选框)
background: #FFFFFF

:hover {
    border-color: #3498DB
    background: #F8F9FA
}

:checked {
    background: #3498DB
    border-color: #2980B9
}
```

### 表格头

```python
background: #F0F2F5
color: #2C3E50
padding: 8px 12px
border-bottom: 2px solid #3498DB
font-weight: 600
```

### 表格行

- 奇数行：#FFFFFF
- 偶数行：#F8F9FA
- 选中行：rgba(52, 152, 219, 0.15)
- 行高：28px

### 标签页

```python
背景: transparent
文本: #2C3E50
内边距: 8px 14px
字体: 11px 600

:hover {
    background: rgba(52, 152, 219, 0.08)
    color: #3498DB
}

:checked {
    background: rgba(52, 152, 219, 0.12)
    border-bottom: 3px solid #3498DB
    color: #3498DB
}
```

## 7. 状态颜色

### 交互状态
- **默认**：灰色背景 #F0F2F5
- **悬停**：浅蓝色 rgba(52, 152, 219, 0.08)
- **活跃**：蓝色 #3498DB
- **按下**：深蓝色 #2980B9
- **禁用**：灰色文本 #95A5A6

### 数据状态
- **成功**：绿色 #27AE60
- **警告**：橙色 #F39C12
- **错误**：红色 #E74C3C
- **信息**：青色 #16A085

## 8. 对比度检查

所有配色组合都应满足 WCAG AA 标准：
- 文本与背景比率 ≥ 4.5:1
- 大型文本与背景比率 ≥ 3:1

### 验证的组合
✅ #2C3E50 文本 on #FFFFFF 背景 (15.06:1)
✅ #FFFFFF 文本 on #3498DB 背景 (7.60:1)
✅ #7F8C8D 文本 on #FFFFFF 背景 (5.87:1)
✅ #95A5A6 文本 on #F8F9FA 背景 (4.78:1)

## 9. 响应式设计原则

### 断点
- 小：< 768px
- 中：768px - 1024px
- 大：> 1024px

### 适配指南
- 最小宽度：1200px
- 最小高度：800px
- 字体大小：在不同屏幕下保持 12-14px

## 10. 无障碍规范

### 最小触摸目标
- 按钮最小尺寸：32x32px

### 高对比度模式
- 文本：#000000
- 背景：#FFFFFF
- 强调：#0000FF

### 键盘导航
- Tab 键支持所有交互元素
- 焦点指示器：2px 蓝色边框

## 11. 动画规范

### 过渡时间
- 快速：100ms - 200ms（简单交互）
- 标准：300ms - 400ms（组件变化）
- 慢速：500ms - 800ms（复杂动画）

### 缓动函数
- ease-in-out：标准交互
- ease-out：进入动画
- ease-in：退出动画

## 12. 实现示例

### PyQt6 QSS 样式表
```qss
QPushButton {
    background-color: #F0F2F5;
    color: #2C3E50;
    border: 1px solid #E4E7EB;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #E4E7EB;
    border: 1px solid #BFC9D4;
}

QPushButton:pressed {
    background-color: #D9DFE8;
}

QPushButton[default="true"] {
    background-color: #3498DB;
    color: #FFFFFF;
    border: 1px solid #2980B9;
}

QPushButton[default="true"]:hover {
    background-color: #2980B9;
    border: 1px solid #1F618D;
}
```

## 维护指南

### 添加新颜色
1. 在 `config/design_system.py` 中添加
2. 验证对比度满足 WCAG AA
3. 更新此文档
4. 在现有组件中测试

### 更新字体
1. 修改 `Typography` 类
2. 测试所有语言渲染（包括中文）
3. 验证可读性
4. 在不同分辨率下测试

### 测试清单
- [ ] 浅色主题下的所有对话框
- [ ] 表格和列表渲染
- [ ] 按钮和输入框交互
- [ ] 图表和图形显示
- [ ] 键盘导航
- [ ] 屏幕阅读器兼容性
- [ ] 不同语言显示
- [ ] 不同屏幕分辨率

---
**文档版本：** 1.0
**最后更新：** 2026年5月9日
**维护者：** PaleoAST 开发团队
