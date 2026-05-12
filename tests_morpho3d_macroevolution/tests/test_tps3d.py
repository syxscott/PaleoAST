"""
================================================================================
PaleoAST - TPS 3D Tests
================================================================================

三维薄板样条测试套件。

作者: PaleoAST Development Team
"""

import unittest

import numpy as np

from morpho3d.tps3d import TPS3D


class TestTPS3DBasics(unittest.TestCase):
    """TPS 3D基础测试"""

    def setUp(self):
        """测试准备"""
        np.random.seed(42)

    def test_tps_identity_transform(self):
        """测试恒等变换"""
        source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        tps = TPS3D()
        tps.fit(source, source)

        # 在源点处应该精确插值
        result = tps.transform(source)

        self.assertTrue(np.allclose(result, source, atol=1e-10))

    def test_tps_translation(self):
        """测试平移变换"""
        source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        translation = np.array([5.0, -3.0, 2.0])
        target = source + translation

        tps = TPS3D()
        tps.fit(source, target)

        # 测试点
        test_point = np.array([[0.5, 0.5, 0.0]])
        result = tps.transform(test_point)

        expected = test_point + translation
        self.assertTrue(np.allclose(result, expected, atol=1e-6))

    def test_tps_scaling(self):
        """测试缩放变换"""
        source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        scale = 2.0
        target = source * scale

        tps = TPS3D()
        tps.fit(source, target)

        # 测试点
        test_point = np.array([[0.5, 0.5, 0.0]])
        result = tps.transform(test_point)

        expected = test_point * scale
        self.assertTrue(np.allclose(result, expected, atol=1e-6))


class TestTPS3DKernel(unittest.TestCase):
    """TPS 3D核函数测试"""

    def test_cubic_kernel(self):
        """测试立方核"""
        tps = TPS3D(kernel="cubic")
        source = np.random.randn(5, 3)
        target = np.random.randn(5, 3)

        tps.fit(source, target)

        # 应该成功拟合
        self.assertIsNotNone(tps.get_weights())
        self.assertIsNotNone(tps.get_affine_matrix())

    def test_thin_plate_kernel(self):
        """测试薄板核"""
        tps = TPS3D(kernel="thin_plate")
        source = np.random.randn(5, 3)
        target = np.random.randn(5, 3)

        tps.fit(source, target)

        self.assertIsNotNone(tps.get_weights())

    def test_gaussian_kernel(self):
        """测试高斯核"""
        tps = TPS3D(kernel="gaussian")
        source = np.random.randn(5, 3)
        target = np.random.randn(5, 3)

        tps.fit(source, target)

        self.assertIsNotNone(tps.get_weights())


class TestTPS3DEdgeCases(unittest.TestCase):
    """TPS 3D边界用例测试"""

    def test_collinear_points(self):
        """测试共线点"""
        source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        target = source + np.array([0, 1, 0])

        tps = TPS3D()
        tps.fit(source, target)

        self.assertIsNotNone(tps)

    def test_coplanar_points(self):
        """测试共面点"""
        source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

        target = source * 1.5

        tps = TPS3D()
        tps.fit(source, target)

        self.assertIsNotNone(tps)

    def test_shape_mismatch(self):
        """测试形状不匹配"""
        source = np.random.randn(5, 3)
        target = np.random.randn(7, 3)

        tps = TPS3D()

        with self.assertRaises(ValueError):
            tps.fit(source, target)

    def test_wrong_dimension(self):
        """测试错误维度"""
        source = np.random.randn(5, 2)  # 应该是3D
        target = np.random.randn(5, 2)

        tps = TPS3D()

        with self.assertRaises(ValueError):
            tps.fit(source, target)


class TestTPS3DChaos(unittest.TestCase):
    """TPS 3D混沌测试"""

    def test_extreme_values(self):
        """测试极端值"""
        source = np.array([[1e-10, 1e-10, 1e-10], [1e10, 1e10, 1e10], [-1e10, 1e10, -1e10]])

        target = source * (-1)

        tps = TPS3D()
        tps.fit(source, target)

        # 应该不产生NaN
        weights = tps.get_weights()
        self.assertFalse(np.any(np.isnan(weights)))

    def test_identical_points(self):
        """测试重复点"""
        source = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

        target = np.array([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])

        tps = TPS3D()
        tps.fit(source, target)

        self.assertIsNotNone(tps)

    def test_single_point(self):
        """测试单点"""
        source = np.array([[1.0, 2.0, 3.0]])
        target = np.array([[4.0, 5.0, 6.0]])

        tps = TPS3D()
        tps.fit(source, target)

        result = tps.transform(np.array([[1.0, 2.0, 3.0]]))
        self.assertTrue(np.allclose(result, target, atol=1e-6))


class TestTPS3DGrid(unittest.TestCase):
    """TPS 3D网格测试"""

    def test_deformation_grid(self):
        """测试变形网格生成"""
        source = np.random.randn(5, 3)
        target = np.random.randn(5, 3)

        tps = TPS3D()
        tps.fit(source, target)

        grid_points, deformations = tps.create_deformation_grid(grid_range=(-1, 1, -1, 1, -1, 1), resolution=(5, 5, 5))

        self.assertEqual(grid_points.shape[0], 125)  # 5^3
        self.assertEqual(deformations.shape, (125, 3))

    def test_jacobian(self):
        """测试Jacobian矩阵"""
        source = np.random.randn(6, 3)
        target = np.random.randn(6, 3)

        tps = TPS3D()
        tps.fit(source, target)

        points = np.random.randn(10, 3)
        jacobians = tps.compute_jacobian(points)

        self.assertEqual(jacobians.shape, (10, 3, 3))

        # Jacobian应该接近仿射变换
        for J in jacobians:
            self.assertFalse(np.any(np.isnan(J)))


class TestTPS3DSuite(unittest.TestCase):
    """TPS 3D完整测试套件"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        np.random.seed(777)
        cls.test_count = 0

    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        print(f"\n总共运行 {cls.test_count} 个TPS 3D测试")

    def test_run_all(self):
        """运行所有测试"""
        TestTPS3DSuite.test_count += 1

        suite = unittest.TestSuite()
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestTPS3DBasics))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestTPS3DKernel))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestTPS3DEdgeCases))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestTPS3DChaos))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestTPS3DGrid))

        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)

        self.assertTrue(result.wasSuccessful())


if __name__ == "__main__":
    unittest.main()
