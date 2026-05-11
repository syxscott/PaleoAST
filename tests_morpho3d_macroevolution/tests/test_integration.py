"""
================================================================================
PaleoAST Phase 4 - Integration Tests
================================================================================

端到端集成测试套件。

测试完整工作流:
1. 数据导入 -> 2. PCA分析 -> 3. 距离计算 -> 4. 树搜索 -> 5. 结果验证

作者: PaleoAST Development Team
"""

import unittest
import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from statistics.pca import PCA
from statistics.distance_metrics import DistanceMetrics, DISTANCE_EUCLIDEAN
from morpho3d.gpa3d import GPA3D
from macroevolution.cohort import CohortSurvivorshipAnalysis


class TestPCAIntegration(unittest.TestCase):
    """PCA集成测试"""
    
    def setUp(self):
        """测试准备"""
        np.random.seed(42)
        
        # 生成测试数据
        n_samples = 50
        n_features = 10
        
        # 创建具有相关性的数据
        true_eigenvalues = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.05])
        
        self.data = np.random.randn(n_samples, n_features)
        self.data = self.data @ np.diag(true_eigenvalues) @ np.random.randn(n_features, n_features)
    
    def test_pca_basic(self):
        """测试PCA基本功能"""
        pca = PCA()
        pca.fit(self.data)
        
        # 验证特征值
        self.assertEqual(len(pca.eigenvalues), n_features)
        
        # 验证方差解释率
        total_var = np.sum(pca.eigenvalues)
        explained = pca.explained_variance_ratio
        
        self.assertAlmostEqual(np.sum(explained), 1.0, places=10)
        
        # 前3个主成分应解释大部分方差
        self.assertGreater(np.sum(explained[:3]), 0.5)
    
    def test_pca_projection(self):
        """测试PCA投影"""
        pca = PCA()
        pca.fit(self.data)
        
        scores = pca.transform(self.data)
        
        self.assertEqual(scores.shape, self.data.shape)
    
    def test_pca_eigenvalue_sum(self):
        """测试特征值总和等于总方差"""
        pca = PCA()
        pca.fit(self.data)
        
        pca_eigenvalue_sum = np.sum(pca.eigenvalues)
        
        # 计算原始数据方差
        data_centered = self.data - self.data.mean(axis=0)
        original_var = np.sum(np.var(data_centered, axis=0, ddof=1))
        
        self.assertAlmostEqual(pca_eigenvalue_sum, original_var, places=5)


class TestDistanceMetricsIntegration(unittest.TestCase):
    """距离度量集成测试"""
    
    def setUp(self):
        """测试准备"""
        np.random.seed(123)
        
        self.data = np.random.randn(20, 5)
    
    def test_euclidean_distance(self):
        """测试欧氏距离"""
        dm = DistanceMetrics(self.data)
        D = dm.compute(DISTANCE_EUCLIDEAN)
        
        # 验证距离矩阵性质
        self.assertEqual(D.shape, (20, 20))
        self.assertTrue(np.allclose(np.diag(D), 0))  # 对角线为0
        self.assertTrue(np.allclose(D, D.T))  # 对称性
        self.assertTrue(np.all(D >= 0))  # 非负性
        
        # 三角不等式
        for i in range(20):
            for j in range(20):
                for k in range(20):
                    self.assertLessEqual(
                        D[i, j],
                        D[i, k] + D[k, j]
                    )
    
    def test_distance_with_nan(self):
        """测试含NaN的距离计算"""
        data_nan = self.data.copy()
        data_nan[5, 2] = np.nan
        data_nan[10, 3] = np.nan
        
        dm = DistanceMetrics(data_nan)
        D = dm.compute(DISTANCE_EUCLIDEAN)
        
        # 应该产生NaN
        self.assertTrue(np.any(np.isnan(D)))
    
    def test_distance_consistency(self):
        """测试距离计算一致性"""
        dm1 = DistanceMetrics(self.data)
        D1 = dm1.compute(DISTANCE_EUCLIDEAN)
        
        dm2 = DistanceMetrics(self.data)
        D2 = dm2.compute(DISTANCE_EUCLIDEAN)
        
        self.assertTrue(np.allclose(D1, D2))


class Test3DMorphometricsIntegration(unittest.TestCase):
    """3D形态测量集成测试"""
    
    def setUp(self):
        """测试准备"""
        np.random.seed(456)
        
        # 创建3D标志点数据
        self.configs = [
            np.random.randn(15, 3) + np.array([i, 0, 0])  # 每个样本有偏移
            for i in range(10)
        ]
    
    def test_gpa3d_workflow(self):
        """测试GPA 3D完整工作流"""
        gpa = GPA3D(tolerance=1e-8, max_iterations=100)
        result = gpa.analyze(self.configs)
        
        # 验证结果
        self.assertEqual(result.n_samples, 10)
        self.assertEqual(result.n_landmarks, 15)
        self.assertGreater(result.n_iterations, 0)
        self.assertGreater(result.final_spread, 0)
        
        # 验证旋转矩阵
        for R in result.rotations:
            self.assertTrue(
                np.allclose(R @ R.T, np.eye(3), atol=1e-8)
            )
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=8)
    
    def test_procrustes_distance_matrix(self):
        """测试Procrustes距离矩阵"""
        gpa = GPA3D()
        result = gpa.analyze(self.configs)
        
        D = result.procrustes_distances
        
        # 验证距离矩阵
        self.assertEqual(D.shape, (10, 10))
        self.assertTrue(np.allclose(np.diag(D), 0))
        self.assertTrue(np.allclose(D, D.T))


class TestMacroevolutionIntegration(unittest.TestCase):
    """宏观演化集成测试"""
    
    def setUp(self):
        """测试准备"""
        np.random.seed(789)
        
        # 创建化石记录
        self.records = [
            (10.0 + np.random.rand(), np.random.rand() * 5)
            for _ in range(100)
        ]
        
        self.intervals = [
            (0, 2), (2, 4), (4, 6), (6, 8), (8, 10)
        ]
    
    def test_cohort_analysis_workflow(self):
        """测试存活分析工作流"""
        analysis = CohortSurvivorshipAnalysis(confidence_level=0.95)
        result = analysis.analyze(self.records, self.intervals)
        
        # 验证结果
        self.assertEqual(len(result.survival_rates), 5)
        self.assertEqual(len(result.origination_rates), 5)
        self.assertEqual(len(result.extinction_rates), 5)
        
        # 所有存活率在[0, 1]内
        for rate in result.survival_rates:
            if not np.isnan(rate):
                self.assertGreaterEqual(rate, 0.0)
                self.assertLessEqual(rate, 1.0)
    
    def test_rate_ratio(self):
        """测试速率比率"""
        analysis = CohortSurvivorshipAnalysis()
        result = analysis.analyze(self.records, self.intervals)
        
        ratio = result.get_rate_ratio()
        
        # 不应该有无穷大或NaN
        for r in ratio:
            if not np.isnan(r):
                self.assertFalse(np.isinf(r))


class TestEndToEndWorkflow(unittest.TestCase):
    """端到端工作流测试"""
    
    def setUp(self):
        """测试准备"""
        np.random.seed(999)
    
    def test_complete_workflow(self):
        """
        测试完整工作流:
        1. 生成数据
        2. 计算距离
        3. PCA分析
        4. 3D GPA
        5. 存活分析
        """
        # 步骤1: 生成数据
        n_samples = 30
        n_features = 8
        
        data = np.random.randn(n_samples, n_features)
        
        # 步骤2: 距离计算
        dm = DistanceMetrics(data)
        distance_matrix = dm.compute(DISTANCE_EUCLIDEAN)
        
        # 验证距离矩阵
        self.assertTrue(np.allclose(np.diag(distance_matrix), 0))
        self.assertTrue(np.allclose(distance_matrix, distance_matrix.T))
        
        # 步骤3: PCA分析
        pca = PCA()
        pca.fit(data)
        
        # 验证特征值
        total_var = np.sum(pca.eigenvalues)
        original_var = np.sum(np.var(data, axis=0, ddof=1))
        self.assertAlmostEqual(total_var, original_var, places=5)
        
        # 步骤4: 3D GPA
        configs_3d = [
            np.random.randn(10, 3) for _ in range(5)
        ]
        gpa_3d = GPA3D()
        result_3d = gpa_3d.analyze(configs_3d)
        
        self.assertEqual(result_3d.n_samples, 5)
        self.assertEqual(result_3d.n_landmarks, 10)
        
        # 步骤5: 存活分析
        fossil_records = [(10.0, 0.0)] * 20
        intervals = [(0, 5), (5, 10)]
        cohort = CohortSurvivorshipAnalysis()
        cohort_result = cohort.analyze(fossil_records, intervals)
        
        self.assertEqual(len(cohort_result.survival_rates), 2)
    
    def test_nan_propagation(self):
        """测试NaN传播"""
        data = np.random.randn(10, 5)
        data[3, 2] = np.nan
        data[7, 0] = np.nan
        
        # 距离计算应该产生NaN
        dm = DistanceMetrics(data)
        D = dm.compute(DISTANCE_EUCLIDEAN)
        
        self.assertTrue(np.any(np.isnan(D)))
    
    def test_edge_case_workflow(self):
        """测试边界用例工作流"""
        # 极小数据
        data_tiny = np.random.randn(2, 2)
        
        dm = DistanceMetrics(data_tiny)
        D = dm.compute(DISTANCE_EUCLIDEAN)
        
        self.assertEqual(D.shape, (2, 2))
        
        # 单样本
        pca = PCA()
        pca.fit(data_tiny)
        
        self.assertEqual(len(pca.eigenvalues), 2)


class TestAssertionCheckpoints(unittest.TestCase):
    """断言检查点测试"""
    
    def test_pca_eigenvalue_nonnegative(self):
        """PCA特征值非负"""
        np.random.seed(111)
        data = np.random.randn(20, 5)
        
        pca = PCA()
        pca.fit(data)
        
        self.assertTrue(np.all(pca.eigenvalues >= 0))
    
    def test_distance_matrix_properties(self):
        """距离矩阵性质检查"""
        np.random.seed(222)
        data = np.random.randn(15, 4)
        
        dm = DistanceMetrics(data)
        D = dm.compute(DISTANCE_EUCLIDEAN)
        
        # 对角线为0
        self.assertTrue(np.allclose(np.diag(D), 0))
        
        # 对称
        self.assertTrue(np.allclose(D, D.T))
        
        # 非负
        self.assertTrue(np.all(D >= 0))
    
    def test_gpa_rotations_special(self):
        """GPA旋转矩阵是SO(3)"""
        np.random.seed(333)
        configs = [np.random.randn(12, 3) for _ in range(6)]
        
        gpa = GPA3D()
        result = gpa.analyze(configs)
        
        for R in result.rotations:
            # 正交性
            self.assertTrue(
                np.allclose(R @ R.T, np.eye(3), atol=1e-8)
            )
            # 行列式为1
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=8)


class TestIntegrationSuite(unittest.TestCase):
    """集成测试完整套件"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        np.random.seed(444)
        cls.test_count = 0
    
    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        print(f"\n总共运行 {cls.test_count} 个集成测试")
    
    def test_run_all(self):
        """运行所有测试"""
        TestIntegrationSuite.test_count += 1
        
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestPCAIntegration))
        suite.addTest(unittest.makeSuite(TestDistanceMetricsIntegration))
        suite.addTest(unittest.makeSuite(Test3DMorphometricsIntegration))
        suite.addTest(unittest.makeSuite(TestMacroevolutionIntegration))
        suite.addTest(unittest.makeSuite(TestEndToEndWorkflow))
        suite.addTest(unittest.makeSuite(TestAssertionCheckpoints))
        
        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)
        
        self.assertTrue(result.wasSuccessful())


class TestChaosMonkey(unittest.TestCase):
    """混沌测试 (Chaos Monkey)"""
    
    def test_all_nan_matrix(self):
        """测试全NaN矩阵"""
        data = np.full((10, 5), np.nan)
        
        dm = DistanceMetrics(data)
        D = dm.compute(DISTANCE_EUCLIDEAN)
        
        self.assertTrue(np.all(np.isnan(D)))
    
    def test_all_inf_matrix(self):
        """测试全Inf矩阵"""
        data = np.full((10, 5), np.inf)
        
        dm = DistanceMetrics(data)
        D = dm.compute(DISTANCE_EUCLIDEAN)
        
        self.assertTrue(np.all(np.isinf(D)))
    
    def test_identical_rows(self):
        """测试完全相同的行"""
        row = np.random.randn(5)
        data = np.tile(row, (10, 1))
        
        dm = DistanceMetrics(data)
        D = dm.compute(DISTANCE_EUCLIDEAN)
        
        # 行间距离应该为0
        self.assertTrue(np.allclose(D, 0))
    
    def test_single_value_column(self):
        """测试单值列 (方差为0)"""
        data = np.random.randn(10, 5)
        data[:, 2] = 5.0  # 常数列
        
        pca = PCA()
        pca.fit(data)
        
        # 常数特征对应的特征值应该接近0
        self.assertLessEqual(pca.eigenvalues[-1], 1e-10)


if __name__ == '__main__':
    unittest.main()
