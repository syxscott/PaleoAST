"""
================================================================================
PaleoAST Phase 5 - Directory Structure Fixer
================================================================================

本模块实现目录结构强制生成器，确保PaleoAST所有合法文件夹存在。

作者: PaleoAST Development Team
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DirectoryFixerResult:
    """目录修复结果"""
    missing_directories: List[str] = field(default_factory=list)
    created_directories: List[str] = field(default_factory=list)
    existing_directories: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def is_complete(self) -> bool:
        """是否所有目录都已存在"""
        return len(self.missing_directories) == 0 and len(self.errors) == 0
    
    def get_summary(self) -> str:
        """获取摘要"""
        return (
            f"Directory Fix Summary:\n"
            f"  - Existing: {len(self.existing_directories)}\n"
            f"  - Created: {len(self.created_directories)}\n"
            f"  - Missing: {len(self.missing_directories)}\n"
            f"  - Errors: {len(self.errors)}"
        )


class DirectoryFixer:
    """
    目录结构强制生成器
    
    确保PaleoAST所有合法文件夹存在，不存在则自动创建。
    
    使用示例:
        >>> fixer = DirectoryFixer()
        >>> fixer.fix_all()
        >>> print(fixer.result.get_summary())
    """
    
    # PaleoAST合法目录结构
    EXPECTED_DIRECTORY_STRUCTURE = {
        # 根目录
        "": [],
        
        # Phase 5目录
        "phase5": ["audit", "startup", "theme"],
        "phase5/audit": [],
        "phase5/startup": [],
        "phase5/theme": [],
        
        # 核心模块目录
        "config": [],
        "models": [],
        "views": [],
        "controllers": [],
        
        # 科学计算引擎目录
        "statistics": [],
        "morphometrics": [],
        "morpho3d": [],
        "ecology": [],
        "stratigraphy": [],
        "visualization": [],
        
        # Phase 3-4目录
        "phylogenetics": [],
        "state_machine": [],
        "parsers": [],
        "hpc": [],
        "reporting": [],
        "macroevolution": [],
        "phase4": ["tests"],
        "phase4/tests": [],
        
        # 工具目录
        "utils": [],
        
        # 测试目录
        "tests": [],
        
        # 文档目录
        "docs": ["api", "tutorials", "examples"],
        "docs/api": [],
        "docs/tutorials": [],
        "docs/examples": [],
        
        # 资源目录
        "resources": ["icons", "styles", "data", "fonts"],
        "resources/icons": [],
        "resources/styles": [],
        "resources/data": [],
        "resources/fonts": [],
        
        # 输出目录
        "output": ["reports", "exports", "cache"],
        "output/reports": [],
        "output/exports": [],
        "output/cache": [],
    }
    
    # 合法Python文件模式
    VALID_PYTHON_FILES = [
        # 根目录
        "main.py",
        "setup.py",
        "requirements.txt",
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        
        # config
        "config/__init__.py",
        "config/constants.py",
        "config/colors.py",
        "config/validators.py",
        
        # models
        "models/__init__.py",
        "models/data_matrix.py",
        "models/column_metadata.py",
        "models/row_metadata.py",
        "models/diversity_result.py",
        "models/state_manager.py",
        
        # views
        "views/__init__.py",
        "views/main_window.py",
        "views/ui_main_window.py",
        "views/ui_navigation.py",
        "views/ui_spreadsheet.py",
        "views/ui_dialogs.py",
        "views/ui_plot_canvas.py",
        "views/ribbon_bar.py",
        
        # controllers
        "controllers/__init__.py",
        "controllers/data_controller.py",
        "controllers/statistics_controller.py",
        "controllers/morphometrics_controller.py",
        "controllers/ecology_controller.py",
        "controllers/stratigraphy_controller.py",
        "controllers/plot_controller.py",
        
        # statistics
        "statistics/__init__.py",
        "statistics/pca.py",
        "statistics/pcoa.py",
        "statistics/nmds.py",
        "statistics/distance_metrics.py",
        "statistics/anosim.py",
        "statistics/permanova.py",
        "statistics/factor_analysis.py",
        "statistics/cluster_analysis.py",
        "statistics/manova.py",
        
        # morphometrics
        "morphometrics/__init__.py",
        "morphometrics/gpa.py",
        "morphometrics/tps.py",
        "morphometrics/relative_warps.py",
        "morphometrics/visualization.py",
        
        # morpho3d
        "morpho3d/__init__.py",
        "morpho3d/quaternion.py",
        "morpho3d/gpa3d.py",
        "morpho3d/tps3d.py",
        "morpho3d/sliding.py",
        "morpho3d/mesh.py",
        
        # ecology
        "ecology/__init__.py",
        "ecology/diversity.py",
        "ecology/rarefaction.py",
        "ecology/similarity.py",
        "ecology/beta_diversity.py",
        
        # stratigraphy
        "stratigraphy/__init__.py",
        "stratigraphy/spectral_analysis.py",
        "stratigraphy/time_series.py",
        "stratigraphy/confidence.py",
        "stratigraphy/unitary_associations.py",
        
        # visualization
        "visualization/__init__.py",
        "visualization/base_plot.py",
        "visualization/pca_plot.py",
        "visualization/pcoa_plot.py",
        "visualization/nmds_plot.py",
        "visualization/diversity_plot.py",
        "visualization/tps_grid_plot.py",
        "visualization/spectral_plot.py",
        "visualization/style.py",
        "visualization/export.py",
        
        # phylogenetics
        "phylogenetics/__init__.py",
        "phylogenetics/tree.py",
        "phylogenetics/fitch.py",
        "phylogenetics/heuristic_search.py",
        "phylogenetics/strict_consensus.py",
        "phylogenetics/distance_methods.py",
        
        # state_machine
        "state_machine/__init__.py",
        "state_machine/base.py",
        "state_machine/tokenizer.py",
        "state_machine/automaton.py",
        
        # parsers
        "parsers/__init__.py",
        "parsers/lexer.py",
        "parsers/nexus_lexer.py",
        "parsers/newick_parser.py",
        "parsers/binary_cache.py",
        
        # hpc
        "hpc/__init__.py",
        "hpc/process_pool.py",
        "hpc/task_scheduler.py",
        
        # reporting
        "reporting/__init__.py",
        "reporting/report_builder.py",
        
        # macroevolution
        "macroevolution/__init__.py",
        "macroevolution/cohort.py",
        "macroevolution/fbd.py",
        "macroevolution/diversity.py",
        
        # utils
        "utils/__init__.py",
        "utils/exceptions.py",
        "utils/matrix_ops.py",
        "utils/validators.py",
        "utils/decorators.py",
        "utils/parallel.py",
        
        # phase4
        "phase4/__init__.py",
        "phase4/tests/__init__.py",
        "phase4/tests/test_quaternion.py",
        "phase4/tests/test_3d_gpa.py",
        "phase4/tests/test_tps3d.py",
        "phase4/tests/test_macroevolution.py",
        "phase4/tests/test_integration.py",
        
        # phase5
        "phase5/__init__.py",
        "phase5/audit/__init__.py",
        "phase5/audit/directory_fixer.py",
        "phase5/audit/ast_auditor.py",
        "phase5/startup/__init__.py",
        "phase5/startup/splash.py",
        "phase5/startup/loader.py",
        "phase5/theme/__init__.py",
        "phase5/theme/styles.py",
        "phase5/theme/manager.py",
        
        # tests
        "tests/__init__.py",
        "tests/test_pca.py",
        "tests/test_distance.py",
        "tests/test_diversity.py",
        "tests/test_gpa.py",
    ]
    
    def __init__(self, project_root: str = None):
        """
        初始化目录修复器
        
        参数:
            project_root: 项目根目录，默认为当前工作目录
        """
        if project_root is None:
            self._project_root = Path.cwd()
        else:
            self._project_root = Path(project_root)
        
        self._result = DirectoryFixerResult()
        self._logger = logging.getLogger(f"{__name__}.DirectoryFixer")
    
    @property
    def result(self) -> DirectoryFixerResult:
        """获取修复结果"""
        return self._result
    
    def fix_all(self, dry_run: bool = False) -> DirectoryFixerResult:
        """
        修复所有缺失目录
        
        参数:
            dry_run: 如果为True，只检查不创建
        
        返回:
            DirectoryFixerResult
        """
        self._logger.info(f"Starting directory fix in {self._project_root}")
        
        # 重建结果
        self._result = DirectoryFixerResult()
        
        # 获取所有需要创建的目录
        all_dirs = set()
        for dirs in self.EXPECTED_DIRECTORY_STRUCTURE.values():
            for d in dirs:
                all_dirs.add(d)
        
        # 检查并创建目录
        for rel_dir in all_dirs:
            abs_dir = self._project_root / rel_dir
            
            if abs_dir.exists():
                if abs_dir.is_dir():
                    self._result.existing_directories.append(str(abs_dir))
                else:
                    self._result.errors.append(f"Path exists but is not directory: {abs_dir}")
            else:
                self._result.missing_directories.append(str(abs_dir))
                
                if not dry_run:
                    try:
                        abs_dir.mkdir(parents=True, exist_ok=True)
                        self._result.created_directories.append(str(abs_dir))
                        self._logger.info(f"Created directory: {abs_dir}")
                    except Exception as e:
                        self._result.errors.append(f"Failed to create {abs_dir}: {str(e)}")
                        self._logger.error(f"Error creating {abs_dir}: {e}")
        
        self._logger.info(self._result.get_summary())
        return self._result
    
    def verify_structure(self) -> Dict[str, List[str]]:
        """
        验证目录结构
        
        返回:
            缺失文件/目录的字典
        """
        missing = {
            'directories': [],
            'files': []
        }
        
        # 检查目录
        all_dirs = set()
        for dirs in self.EXPECTED_DIRECTORY_STRUCTURE.values():
            for d in dirs:
                all_dirs.add(d)
        
        for rel_dir in all_dirs:
            abs_dir = self._project_root / rel_dir
            if not abs_dir.exists():
                missing['directories'].append(str(abs_dir))
        
        # 检查文件 (只检查重要文件)
        important_files = [
            "main.py",
            "requirements.txt",
            "config/__init__.py",
            "models/__init__.py",
            "statistics/__init__.py",
            "morpho3d/__init__.py",
        ]
        
        for rel_file in important_files:
            abs_file = self._project_root / rel_file
            if not abs_file.exists():
                missing['files'].append(str(abs_file))
        
        return missing
    
    def generate_structure_markdown(self) -> str:
        """
        生成目录结构的Markdown文档
        
        返回:
            Markdown格式的目录树
        """
        lines = ["# PaleoAST Directory Structure", "", "```"]
        
        def add_tree(dir_path: Path, prefix: str = "", is_last: bool = True):
            """递归添加目录树"""
            parts = []
            for i, subdir in enumerate(sorted(dir_path.iterdir())):
                if subdir.is_dir() and not subdir.name.startswith('.'):
                    is_last_sub = (i == len(list(dir_path.iterdir())) - 1)
                    connector = "└── " if is_last_sub else "├── "
                    lines.append(f"{prefix}{connector}{subdir.name}/")
                    
                    new_prefix = prefix + ("    " if is_last_sub else "│   ")
                    add_tree(subdir, new_prefix, is_last_sub)
        
        add_tree(self._project_root)
        lines.append("```")
        
        return "\n".join(lines)
    
    def check_write_permissions(self) -> Dict[str, bool]:
        """
        检查写入权限
        
        返回:
            各目录的写入权限
        """
        permissions = {}
        
        for rel_dir in ["output", "cache", "temp"]:
            abs_dir = self._project_root / rel_dir
            if abs_dir.exists():
                permissions[rel_dir] = os.access(abs_dir, os.W_OK)
            else:
                parent = self._project_root
                permissions[rel_dir] = os.access(parent, os.W_OK)
        
        return permissions


def fix_paleoast_directories(project_root: str = None) -> DirectoryFixerResult:
    """
    便捷函数：修复PaleoAST目录结构
    
    参数:
        project_root: 项目根目录
    
    返回:
        DirectoryFixerResult
    """
    fixer = DirectoryFixer(project_root)
    return fixer.fix_all()


if __name__ == "__main__":
    # 直接运行时的测试
    fixer = DirectoryFixer()
    result = fixer.fix_all()
    print(result.get_summary())
