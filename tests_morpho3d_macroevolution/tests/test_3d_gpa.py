"""
================================================================================
PaleoAST Phase 4 - 3D GPA Tests
================================================================================

三维广义普氏分析测试套件。

作者: PaleoAST Development Team
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from morpho3d.gpa3d import GPA3D, procrustes_distance_3d


class TestGPA3DBasics(unittest.TestCase):
    """3D GPA基础测试"""

    def setUp(self):
        """测试准备"""
        np.random.seed(42)

    def test_gpa_single_config(self):
        """测试单构型"""
        config = np.random.randn(10, 3)

        gpa = GPA3D()
        result = gpa.analyze([config])

        self.assertEqual(result.n_samples, 1)
        self.assertEqual(result.n_landmarks, 10)
        self.assertTrue(result.n_iterations >= 1)

    def test_gpa_identical_configs(self):
        """测试完全相同的构型"""
        config = np.random.randn(15, 3)

        gpa = GPA3D()
        result = gpa.analyze([config, config.copy(), config.copy()])

        # 质心大小应该相同
        self.assertTrue(np.allclose(result.centroid_sizes, result.centroid_sizes[0]))

    def test_gpa_convergence(self):
        """测试收敛性"""
        configs = [np.random.randn(20, 3) for _ in range(10)]

        gpa = GPA3D(tolerance=1e-10, max_iterations=100)
        result = gpa.analyze(configs)

        # 应该收敛
        self.assertTrue(result.n_iterations <= 100)
        self.assertTrue(result.final_spread >= 0)

    def test_gpa_no_scale(self):
        """测试无缩放"""
        configs = [np.random.randn(10, 3) * 10 for _ in range(5)]

        gpa = GPA3D(scale=False)
        result = gpa.analyze(configs)

        # 质心大小应该差异很大
        self.assertFalse(np.allclose(result.centroid_sizes, result.centroid_sizes[0]))


class TestGPA3DResults(unittest.TestCase):
    """3D GPA结果测试"""

    def setUp(self):
        """测试准备"""
        np.random.seed(123)
        configs = [np.random.randn(12, 3) for _ in range(8)]

        gpa = GPA3D()
        self.result = gpa.analyze(configs)

    def test_result_properties(self):
        """测试结果属性"""
        self.assertEqual(self.result.n_samples, 8)
        self.assertEqual(self.result.n_landmarks, 12)
        self.assertIsNotNone(self.result.mean_config)
        self.assertEqual(self.result.mean_config.shape, (12, 3))

    def test_centroid_sizes(self):
        """测试质心大小"""
        self.assertEqual(len(self.result.centroid_sizes), 8)
        self.assertTrue(np.all(self.result.centroid_sizes > 0))

    def test_rotations(self):
        """测试旋转矩阵"""
        self.assertEqual(len(self.result.rotations), 8)

        for R in self.result.rotations:
            # 验证正交性
            self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-8))
            # 验证行列式
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=8)

    def test_procrustes_distances(self):
        """测试Procrustes距离矩阵"""
        D = self.result.procrustes_distances

        self.assertEqual(D.shape, (8, 8))

        # 对角线为0
        self.assertTrue(np.allclose(np.diag(D), 0))

        # 对称性
        self.assertTrue(np.allclose(D, D.T))

        # 非负性
        self.assertTrue(np.all(D >= 0))

    def test_shape_coordinates(self):
        """测试展平形状坐标"""
        coords = self.result.get_shape_coordinates()

        self.assertEqual(coords.shape, (8, 12 * 3))

    def test_covariance_matrix(self):
        """测试协方差矩阵"""
        cov = self.result.compute_covariance()

        expected_shape = (12 * 3, 12 * 3)
        self.assertEqual(cov.shape, expected_shape)

        # 协方差矩阵应是对称的
        self.assertTrue(np.allclose(cov, cov.T))


class TestGPA3DEdgeCases(unittest.TestCase):
    """3D GPA边界用例测试"""

    def test_empty_configs(self):
        """测试空配置列表"""
        gpa = GPA3D()

        with self.assertRaises(ValueError):
            gpa.analyze([])

    def test_single_landmark(self):
        """测试单标志点"""
        config = np.array([[1.0, 2.0, 3.0]])

        gpa = GPA3D()
        result = gpa.analyze([config])

        self.assertEqual(result.n_landmarks, 1)

    def test_invalid_config_shape(self):
        """测试无效配置形状"""
        config_2d = np.random.randn(10)  # 应该失败
        config_wrong_dim = np.random.randn(10, 4)  # 应该失败

        gpa = GPA3D()

        with self.assertRaises(ValueError):
            gpa.analyze([config_2d])

        with self.assertRaises(ValueError):
            gpa.analyze([config_wrong_dim])

    def test_mixed_landmark_counts(self):
        """测试不同标志点数"""
        config1 = np.random.randn(10, 3)
        config2 = np.random.randn(15, 3)

        gpa = GPA3D()

        with self.assertRaises(ValueError):
            gpa.analyze([config1, config2])


class TestProcrustesDistance(unittest.TestCase):
    """Procrustes距离测试"""

    def test_identical_configs(self):
        """测试相同构型"""
        config = np.random.randn(10, 3)

        d, R, s = procrustes_distance_3d(config, config)

        self.assertAlmostEqual(d, 0.0, places=10)
        self.assertTrue(np.allclose(R, np.eye(3)))
        self.assertAlmostEqual(s, 1.0, places=10)

    def test_rotation_only(self):
        """测试纯旋转变换"""
        angle = np.pi / 3

        R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

        config1 = np.random.randn(15, 3)
        config2 = config1 @ R.T

        d, _R_recovered, s = procrustes_distance_3d(config1, config2)

        self.assertAlmostEqual(d, 0.0, places=5)
        self.assertAlmostEqual(s, 1.0, places=5)

    def test_distance_metric_properties(self):
        """测试距离度量性质"""
        np.random.seed(789)

        configs = [np.random.randn(10, 3) for _ in range(5)]

        distances = []
        for i in range(5):
            for j in range(i + 1, 5):
                d, _, _ = procrustes_distance_3d(configs[i], configs[j])
                distances.append(d)

        # 所有距离应非负
        self.assertTrue(all(d >= 0 for d in distances))

        # 同一性
        for i, config in enumerate(configs):
            d, _, _ = procrustes_distance_3d(config, config)
            self.assertAlmostEqual(d, 0.0, places=10)


class TestGPA3DChaos(unittest.TestCase):
    """3D GPA混沌测试"""

    def test_extreme_scale_differences(self):
        """测试极端缩放差异"""
        configs = [np.random.randn(10, 3) * 1e-10, np.random.randn(10, 3) * 1e10]

        gpa = GPA3D()
        result = gpa.analyze(configs)

        self.assertIsNotNone(result)
        self.assertFalse(np.any(np.isnan(result.mean_config)))

    def test_degenerate_config(self):
        """测试退化构型(所有点共线)"""
        config = np.zeros((10, 3))
        config[:, 0] = np.linspace(0, 1, 10)  # 所有点在x轴上

        gpa = GPA3D()
        result = gpa.analyze([config])

        self.assertIsNotNone(result)
        self.assertFalse(np.any(np.isnan(result.centroid_sizes)))

    def test_large_iterations(self):
        """测试大量迭代"""
        configs = [np.random.randn(8, 3) for _ in range(3)]

        gpa = GPA3D(max_iterations=1000, tolerance=1e-15)
        result = gpa.analyze(configs)

        # 无论如何都应该结束
        self.assertTrue(result.n_iterations <= 1000)


class TestGPA3DSuite(unittest.TestCase):
    """3D GPA完整测试套件"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        np.random.seed(888)
        cls.test_count = 0

    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        print(f"\n总共运行 {cls.test_count} 个3D GPA测试")

    def test_run_all(self):
        """运行所有测试"""
        TestGPA3DSuite.test_count += 1

        suite = unittest.TestSuite()
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestGPA3DBasics))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestGPA3DResults))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestGPA3DEdgeCases))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestProcrustesDistance))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestGPA3DChaos))

        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)

        self.assertTrue(result.wasSuccessful())


if __name__ == "__main__":
    unittest.main()
