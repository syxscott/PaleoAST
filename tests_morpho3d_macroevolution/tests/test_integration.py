"""
================================================================================
PaleoAST - Integration Tests
================================================================================

端到端集成测试套件。

测试完整工作流:
1. 数据导入 -> 2. PCA分析 -> 3. 距离计算 -> 4. 树搜索 -> 5. 结果验证

作者: PaleoAST Development Team
"""

import unittest

import numpy as np

from macroevolution.cohort import CohortSurvivorshipAnalysis
from morpho3d.gpa3d import GPA3D
from statistics.distance_metrics import DistanceMatrixResult, compute_distance_matrix
from statistics.pca import PCAAnalyzer, PCAResult


class TestPCAIntegration(unittest.TestCase):
    """PCA集成测试"""

    def setUp(self):
        """测试准备"""
        np.random.seed(42)

        n_samples = 50
        n_features = 10

        true_eigenvalues = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.05])
        self.data = np.random.randn(n_samples, n_features)
        self.data = self.data @ np.diag(true_eigenvalues) @ np.random.randn(n_features, n_features)

    def test_pca_basic(self):
        """测试PCA基本功能"""
        analyzer = PCAAnalyzer()
        result = analyzer.analyze(self.data)

        self.assertIsInstance(result, PCAResult)
        self.assertEqual(len(result.eigenvalues), result.n_components)

        total_var = np.sum(result.eigenvalues)
        self.assertGreater(total_var, 0)

        # ``explained_variance`` 在本项目中以百分比形式返回 (sum=100),
        # 与 scikit-learn 的 ``explained_variance_ratio_`` (sum=1)
        # 约定不同。
        self.assertAlmostEqual(np.sum(result.explained_variance), 100.0, places=8)

        self.assertGreater(np.sum(result.explained_variance[:3]), 0.5)

    def test_pca_projection(self):
        """测试PCA投影"""
        analyzer = PCAAnalyzer()
        result = analyzer.analyze(self.data)

        self.assertEqual(result.scores.shape[0], self.data.shape[0])
        self.assertEqual(result.scores.shape[1], result.n_components)

    def test_pca_eigenvalue_sum(self):
        """测试特征值总和等于总方差"""
        analyzer = PCAAnalyzer()
        result = analyzer.analyze(self.data)

        pca_eigenvalue_sum = np.sum(result.eigenvalues)

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
        dm = compute_distance_matrix(self.data, metric="euclidean")

        self.assertIsInstance(dm, DistanceMatrixResult)
        D = dm.matrix
        self.assertEqual(D.shape, (20, 20))
        self.assertTrue(np.allclose(np.diag(D), 0))
        self.assertTrue(np.allclose(D, D.T))
        self.assertTrue(np.all(D >= 0))

    def test_distance_with_nan(self):
        """测试含NaN的距离计算"""
        data_nan = self.data.copy()
        data_nan[5, 2] = np.nan
        data_nan[10, 3] = np.nan

        dm = compute_distance_matrix(data_nan, metric="euclidean")
        self.assertTrue(np.any(np.isnan(dm.matrix)))

    def test_distance_consistency(self):
        """测试距离计算一致性"""
        dm1 = compute_distance_matrix(self.data, metric="euclidean")
        dm2 = compute_distance_matrix(self.data, metric="euclidean")

        self.assertTrue(np.allclose(dm1.matrix, dm2.matrix))

    def test_multiple_metrics(self):
        """测试多种距离度量"""
        for metric in ["euclidean", "manhattan", "canberra"]:
            dm = compute_distance_matrix(self.data, metric=metric)
            self.assertEqual(dm.matrix.shape, (20, 20))
            self.assertTrue(np.allclose(np.diag(dm.matrix), 0))


class Test3DMorphometricsIntegration(unittest.TestCase):
    """3D形态测量集成测试"""

    def setUp(self):
        """测试准备"""
        np.random.seed(456)

        self.configs = [np.random.randn(15, 3) + np.array([i, 0, 0]) for i in range(10)]

    def test_gpa3d_workflow(self):
        """测试GPA 3D完整工作流"""
        gpa = GPA3D(tolerance=1e-8, max_iterations=100)
        result = gpa.analyze(self.configs)

        self.assertEqual(result.n_samples, 10)
        self.assertEqual(result.n_landmarks, 15)
        self.assertGreater(result.n_iterations, 0)
        self.assertGreater(result.final_spread, 0)

        for R in result.rotations:
            self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-8))
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=8)

    def test_procrustes_distance_matrix(self):
        """测试Procrustes距离矩阵"""
        gpa = GPA3D()
        result = gpa.analyze(self.configs)

        D = result.procrustes_distances
        self.assertEqual(D.shape, (10, 10))
        self.assertTrue(np.allclose(np.diag(D), 0))
        self.assertTrue(np.allclose(D, D.T))


class TestMacroevolutionIntegration(unittest.TestCase):
    """宏观演化集成测试"""

    def setUp(self):
        """测试准备"""
        np.random.seed(789)

        self.records = [(10.0 + np.random.rand(), np.random.rand() * 5) for _ in range(100)]
        self.intervals = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]

    def test_cohort_analysis_workflow(self):
        """测试存活分析工作流"""
        analysis = CohortSurvivorshipAnalysis(confidence_level=0.95)
        result = analysis.analyze(self.records, self.intervals)

        self.assertEqual(len(result.survival_rates), 5)
        self.assertEqual(len(result.origination_rates), 5)
        self.assertEqual(len(result.extinction_rates), 5)

        for rate in result.survival_rates:
            if not np.isnan(rate):
                self.assertGreaterEqual(rate, 0.0)
                self.assertLessEqual(rate, 1.0)

    def test_rate_ratio(self):
        """测试速率比率"""
        analysis = CohortSurvivorshipAnalysis()
        result = analysis.analyze(self.records, self.intervals)

        ratio = result.get_rate_ratio()

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
        n_samples = 30
        n_features = 8
        data = np.random.randn(n_samples, n_features)

        dm = compute_distance_matrix(data, metric="euclidean")
        self.assertTrue(np.allclose(np.diag(dm.matrix), 0))
        self.assertTrue(np.allclose(dm.matrix, dm.matrix.T))

        analyzer = PCAAnalyzer()
        result = analyzer.analyze(data)

        total_var = np.sum(result.eigenvalues)
        original_var = np.sum(np.var(data, axis=0, ddof=1))
        self.assertAlmostEqual(total_var, original_var, places=5)

        configs_3d = [np.random.randn(10, 3) for _ in range(5)]
        gpa_3d = GPA3D()
        result_3d = gpa_3d.analyze(configs_3d)

        self.assertEqual(result_3d.n_samples, 5)
        self.assertEqual(result_3d.n_landmarks, 10)

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

        dm = compute_distance_matrix(data, metric="euclidean")
        self.assertTrue(np.any(np.isnan(dm.matrix)))

    def test_edge_case_workflow(self):
        """测试边界用例工作流"""
        data_tiny = np.random.randn(2, 2)

        dm = compute_distance_matrix(data_tiny, metric="euclidean")
        self.assertEqual(dm.matrix.shape, (2, 2))

        analyzer = PCAAnalyzer()
        result = analyzer.analyze(data_tiny)
        self.assertEqual(len(result.eigenvalues), result.n_components)


class TestAssertionCheckpoints(unittest.TestCase):
    """断言检查点测试"""

    def test_pca_eigenvalue_nonnegative(self):
        """PCA特征值非负"""
        np.random.seed(111)
        data = np.random.randn(20, 5)

        analyzer = PCAAnalyzer()
        result = analyzer.analyze(data)
        self.assertTrue(np.all(result.eigenvalues >= 0))

    def test_distance_matrix_properties(self):
        """距离矩阵性质检查"""
        np.random.seed(222)
        data = np.random.randn(15, 4)

        dm = compute_distance_matrix(data, metric="euclidean")
        D = dm.matrix

        self.assertTrue(np.allclose(np.diag(D), 0))
        self.assertTrue(np.allclose(D, D.T))
        self.assertTrue(np.all(D >= 0))

    def test_gpa_rotations_special(self):
        """GPA旋转矩阵是SO(3)"""
        np.random.seed(333)
        configs = [np.random.randn(12, 3) for _ in range(6)]

        gpa = GPA3D()
        result = gpa.analyze(configs)

        for R in result.rotations:
            self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-8))
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
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestPCAIntegration))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestDistanceMetricsIntegration))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(Test3DMorphometricsIntegration))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestMacroevolutionIntegration))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestEndToEndWorkflow))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestAssertionCheckpoints))

        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)

        self.assertTrue(result.wasSuccessful())


class TestChaosMonkey(unittest.TestCase):
    """混沌测试 (Chaos Monkey)"""

    def test_all_nan_matrix(self):
        """测试全NaN矩阵"""
        data = np.full((10, 5), np.nan)

        dm = compute_distance_matrix(data, metric="euclidean")
        # 对角线 d(i, i) = 0,非对角线 (i != j) 在输入全 NaN 时
        # 没有可计算的差异,应为 NaN。``squareform`` 会强制把
        # 对角线写成 0,所以这里分别检查两个区域。
        n = dm.matrix.shape[0]
        self.assertEqual(n, 10)
        # 对角线为 0
        self.assertTrue(np.allclose(np.diag(dm.matrix), 0.0))
        # 非对角线为 NaN
        off_diag = dm.matrix[~np.eye(n, dtype=bool)]
        self.assertTrue(np.all(np.isnan(off_diag)))

    def test_all_inf_matrix(self):
        """测试全Inf矩阵"""
        data = np.full((10, 5), np.inf)

        dm = compute_distance_matrix(data, metric="euclidean")
        # 注意:``inf - inf = nan`` 是 IEEE 754 的规定行为,所以
        # 即便输入全 Inf,pdist 返回的距离也是 NaN。这里检查
        # 实现不会崩溃、矩阵形状正确,并且对角线为 0。
        n = dm.matrix.shape[0]
        self.assertEqual(n, 10)
        self.assertTrue(np.allclose(np.diag(dm.matrix), 0.0))
        off_diag = dm.matrix[~np.eye(n, dtype=bool)]
        # 非对角线不应该是有限的数值 (因为 inf 减去 inf 不可约简)
        self.assertTrue(np.all(np.isnan(off_diag)))

    def test_identical_rows(self):
        """测试完全相同的行"""
        row = np.random.randn(5)
        data = np.tile(row, (10, 1))

        dm = compute_distance_matrix(data, metric="euclidean")
        self.assertTrue(np.allclose(dm.matrix, 0))

    def test_single_value_column(self):
        """测试单值列 (方差为0)"""
        data = np.random.randn(10, 5)
        data[:, 2] = 5.0

        analyzer = PCAAnalyzer()
        result = analyzer.analyze(data)

        self.assertLessEqual(result.eigenvalues[-1], 1e-10)


if __name__ == "__main__":
    unittest.main()
