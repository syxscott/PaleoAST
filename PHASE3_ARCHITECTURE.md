# PaleoAST Phase 3 - Enterprise Architecture

## 一、概述

Phase 3将PaleoAST升级为企业级科研基础设施，具备：
- 自主I/O解析引擎（零依赖外部库）
- 系统发育树推断引擎
- HPC并发计算框架
- LaTeX学术报告生成器

## 二、目录结构

```
PaleoAST/
├── parsers/                      # 自主I/O解析引擎
│   ├── __init__.py
│   ├── lexer.py                  # 词法分析器基类
│   ├── nexus_lexer.py           # NEXUS词法分析
│   ├── nexus_parser.py          # NEXUS语法解析
│   ├── fasta_parser.py          # FASTA格式解析
│   ├── phylip_parser.py         # PHYLIP格式解析
│   ├── newick_parser.py         # Newick树格式解析
│   ├── nexus_writer.py          # NEXUS写入器
│   └── binary_cache.py          # .pastx二进制缓存
│
├── phylogenetics/                # 系统发育推断引擎
│   ├── __init__.py
│   ├── tree.py                  # 树数据结构
│   ├── node.py                  # 节点数据结构
│   ├── fitch.py                 # Fitch算法
│   ├── heuristic_search.py      # 启发式树搜索(NNI/TBR)
│   ├── strict_consensus.py      # 严格一致性树
│   ├── distance_methods.py       # 距离法(UPGMA/NJ)
│   ├── likelihood.py            # 最大似然估计
│   └── bootstrap.py             # Bootstrap分析
│
├── hpc/                         # 高性能计算
│   ├── __init__.py
│   ├── process_pool.py          # 多进程池
│   ├── task_scheduler.py         # 任务调度器
│   ├── shared_memory.py          # 共享内存管理
│   ├── memory_map.py             # mmap内存映射
│   ├── progress_queue.py         # 进度队列
│   └── thread_safe.py            # 线程安全工具
│
├── reporting/                    # 自动化报告
│   ├── __init__.py
│   ├── latex_preamble.py         # LaTeX导言区
│   ├── table_generator.py         # 表格生成器
│   ├── figure_handler.py          # 图表处理
│   ├── matrix_converter.py        # 矩阵转LaTeX
│   ├── report_builder.py         # 报告构建器
│   └── compiler.py               # LaTeX编译器封装
│
└── state_machine/               # 状态机框架
    ├── __init__.py
    ├── base.py                   # 状态机基类
    ├── tokenizer.py              # 分词器
    └── automaton.py              # 自动机
```

## 三、核心算法数学推导

### 3.1 Fitch算法（最大简约）

对于给定位点，树长计算：

$$L = \sum_{i=1}^{sites} l_i$$

其中$l_i$为位点$i$的最小变换次数：

$$
\text{Intersection}(A, B) = 
\begin{cases}
A \cap B & \text{if } A \cap B \neq \emptyset \\
A \cup B & \text{otherwise}
\end{cases}
$$

### 3.2 NNI拓扑变换

邻居连接互换，交换边$(A,B)$两侧的子树：

$$\text{Newick}_{NNI} = (((X, Y), Z), W) \rightarrow (((X, Z), Y), W)$$

### 3.3 似然函数

$$\mathcal{L}(\theta | D) = \prod_{i=1}^{n} \prod_{j=1}^{k} \pi_j^{(j)} \prod_{t} p_{uv}(t, \theta)$$

其中$p_{uv}$为转移概率。
