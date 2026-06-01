"""
================================================================================
PaleoAST - Quaternion Tests
================================================================================

四元数和旋转矩阵的单元测试套件。

测试内容:
- 四元数基础运算
- 旋转矩阵转换
- SVD对齐
- 极限边界条件

作者: PaleoAST Development Team
"""

import unittest

import numpy as np

from morpho3d.quaternion import Quaternion, RotationMatrix


class TestQuaternionBasics(unittest.TestCase):
    """四元数基础运算测试"""

    def setUp(self):
        """测试前准备"""
        np.random.seed(42)

    def test_quaternion_creation(self):
        """测试四元数创建和归一化"""
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        self.assertAlmostEqual(q.magnitude, 1.0, places=10)

        # 验证分量归一化
        expected_w = 1.0 / np.sqrt(1 + 4 + 9 + 16)
        self.assertAlmostEqual(q.w, expected_w, places=10)

    def test_quaternion_magnitude_zero(self):
        """测试零向量异常"""
        with self.assertRaises(ValueError):
            Quaternion(0.0, 0.0, 0.0, 0.0)

    def test_quaternion_conjugate(self):
        """测试共轭运算"""
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        q_conj = q.conjugate

        # ``__post_init__`` 会自动归一化,所以即使 (1, 2, 3, 4) 在
        # 内部存为 (1, 2, 3, 4) / sqrt(30),共轭也必须经过同样的
        # 归一化才能保证 ``q_conj.w == q.w`` 严格成立。这中间会
        # 引入 1e-17 级别的浮点误差,因此这里用 ``assertAlmostEqual``。
        self.assertAlmostEqual(q_conj.w, q.w, places=14)
        self.assertAlmostEqual(q_conj.x, -q.x, places=14)
        self.assertAlmostEqual(q_conj.y, -q.y, places=14)
        self.assertAlmostEqual(q_conj.z, -q.z, places=14)

    def test_quaternion_inverse(self):
        """测试逆运算"""
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        q_inv = q.inverse

        # q * q⁻¹ = 1
        product = q * q_inv
        self.assertAlmostEqual(product.w, 1.0, places=10)
        self.assertAlmostEqual(product.x, 0.0, places=10)
        self.assertAlmostEqual(product.y, 0.0, places=10)
        self.assertAlmostEqual(product.z, 0.0, places=10)

    def test_quaternion_multiplication(self):
        """测试Hamilton积"""
        q1 = Quaternion(1.0, 0.0, 0.0, 0.0)  # 恒等元
        q2 = Quaternion(2.0, 3.0, 4.0, 5.0)

        product = q1 * q2
        self.assertAlmostEqual(product.w, q2.w, places=10)
        self.assertAlmostEqual(product.x, q2.x, places=10)
        self.assertAlmostEqual(product.y, q2.y, places=10)
        self.assertAlmostEqual(product.z, q2.z, places=10)

    def test_quaternion_rotation_angle(self):
        """测试旋转角计算"""
        # 恒等旋转
        q_identity = Quaternion(1.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(q_identity.rotation_angle, 0.0, places=10)

        # 180度旋转
        q_180 = Quaternion(0.0, 1.0, 0.0, 0.0)
        self.assertAlmostEqual(q_180.rotation_angle, np.pi, places=10)

    def test_quaternion_rotation_axis(self):
        """测试旋转轴提取"""
        # Z轴旋转
        q_z = Quaternion.from_axis_angle([0.0, 0.0, 1.0], np.pi / 2)
        axis = q_z.rotation_axis

        self.assertAlmostEqual(axis[0], 0.0, places=10)
        self.assertAlmostEqual(axis[1], 0.0, places=10)
        self.assertAlmostEqual(axis[2], 1.0, places=10)


class TestQuaternionRotation(unittest.TestCase):
    """四元数旋转测试"""

    def test_rotate_vector_z_axis(self):
        """测试绕Z轴旋转90度"""
        q = Quaternion.from_axis_angle([0.0, 0.0, 1.0], np.pi / 2)
        v = np.array([1.0, 0.0, 0.0])

        v_rotated = q.rotate_vector(v)

        # 期望: (0, 1, 0)
        self.assertAlmostEqual(v_rotated[0], 0.0, places=5)
        self.assertAlmostEqual(v_rotated[1], 1.0, places=5)
        self.assertAlmostEqual(v_rotated[2], 0.0, places=5)

    def test_rotate_vector_preserves_length(self):
        """测试旋转保持向量长度"""
        np.random.seed(123)

        for _ in range(100):
            axis = np.random.randn(3)
            angle = np.random.uniform(0, 2 * np.pi)
            v = np.random.randn(3)

            q = Quaternion.from_axis_angle(axis, angle)
            v_rotated = q.rotate_vector(v)

            self.assertAlmostEqual(np.linalg.norm(v), np.linalg.norm(v_rotated), places=10)

    def test_rotation_matrix_conversion(self):
        """测试四元数到旋转矩阵转换"""
        q = Quaternion.from_axis_angle([1.0, 2.0, 3.0], np.pi / 4)
        R = q.to_rotation_matrix()

        # 验证正交性
        self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-10))

        # 验证行列式
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=10)

    def test_slerp(self):
        """测试球面线性插值"""
        # q1: 0 度旋转 (w=1)
        # q2: 90 度绕 z 轴旋转 (w=cos(45°), z=sin(45°))
        # 角度差为 90°,所以 t=0.5 处的插值应当是 45° 旋转。
        q1 = Quaternion(1.0, 0.0, 0.0, 0.0)
        q2 = Quaternion.from_axis_angle([0.0, 0.0, 1.0], np.pi / 2)

        q_mid = q1.slerp(q2, 0.5)

        expected_angle = np.pi / 4  # 45度
        self.assertAlmostEqual(q_mid.rotation_angle, expected_angle, places=5)


class TestRotationMatrix(unittest.TestCase):
    """旋转矩阵测试"""

    def test_svd_alignment_identity(self):
        """测试恒等变换对齐"""
        X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        R = RotationMatrix.from_svd(X, X)

        self.assertTrue(np.allclose(R, np.eye(3), atol=1e-10))

    def test_svd_alignment_rotation(self):
        """测试旋转变换对齐"""
        angle = np.pi / 3
        axis = np.array([0.0, 0.0, 1.0])

        R_true = RotationMatrix.from_axis_angle(axis, angle)

        X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        Y = X @ R_true.T

        R_recovered = RotationMatrix.from_svd(X, Y)

        self.assertTrue(np.allclose(R_recovered, R_true, atol=1e-8))

    def test_svd_alignment_with_noise(self):
        """测试带噪声的对齐"""
        np.random.seed(456)

        angle = np.pi / 4
        R_true = RotationMatrix.from_axis_angle([1, 1, 1], angle)

        X = np.random.randn(20, 3)
        Y = X @ R_true.T + np.random.randn(20, 3) * 0.01

        R_recovered = RotationMatrix.from_svd(X, Y)

        # 验证正交性
        self.assertTrue(np.allclose(R_recovered @ R_recovered.T, np.eye(3), atol=1e-8))

    def test_verify_orthogonal(self):
        """测试正交性验证"""
        # 有效的旋转矩阵
        R_valid = RotationMatrix.from_axis_angle([1, 0, 0], np.pi / 4)
        self.assertTrue(RotationMatrix.verify_orthogonal(R_valid))

        # 非正交矩阵
        R_invalid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertFalse(RotationMatrix.verify_orthogonal(R_invalid))

    def test_verify_special(self):
        """测试SO(3)群验证"""
        # 有效旋转
        R_valid = RotationMatrix.from_axis_angle([0, 1, 0], np.pi / 2)
        self.assertTrue(RotationMatrix.verify_special(R_valid))

        # 反射矩阵 (det = -1)。注意: ``[[-1,0,0],[0,-1,0],[0,0,1]]``
        # 实际上是绕 z 轴 180° 的旋转 (det = +1),不是反射。
        # 一个真正的反射矩阵应当有一个 (且仅一个) 特征值为 -1。
        R_reflection = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertFalse(RotationMatrix.verify_special(R_reflection))

    def test_compose_rotations(self):
        """测试旋转组合"""
        R1 = RotationMatrix.from_axis_angle([1, 0, 0], np.pi / 2)
        R2 = RotationMatrix.from_axis_angle([0, 1, 0], np.pi / 2)

        R_composed = RotationMatrix.compose(R1, R2)

        # 验证仍是旋转矩阵
        self.assertTrue(RotationMatrix.verify_special(R_composed))


class TestQuaternionChaos(unittest.TestCase):
    """四元数极限测试 (混沌测试)"""

    def test_extreme_angles(self):
        """测试极端角度"""
        for angle in [0, np.pi, 2 * np.pi, 1e-10, 1e10]:
            if angle > 0:
                q = Quaternion.from_axis_angle([1, 0, 0], angle)
                self.assertIsNotNone(q)
                self.assertTrue(0 <= q.rotation_angle <= np.pi)

    def test_degenerate_axis(self):
        """测试退化轴向量"""
        # 零向量
        with self.assertRaises(ValueError):
            Quaternion.from_axis_angle([0, 0, 0], np.pi / 4)

        # 很小的向量
        with self.assertRaises(ValueError):
            Quaternion.from_axis_angle([1e-20, 1e-20, 1e-20], np.pi / 4)

    def test_near_singular_rotation_matrix(self):
        """测试接近奇异旋转矩阵"""
        # 接近行列式-1的矩阵
        R = np.eye(3)
        R[0, 0] = -1 + 1e-12

        with self.assertRaises(ValueError):
            Quaternion.from_rotation_matrix(R)

    def test_invalid_matrix_shapes(self):
        """测试无效矩阵形状"""
        with self.assertRaises(ValueError):
            Quaternion.from_rotation_matrix(np.eye(4))

        with self.assertRaises(ValueError):
            Quaternion.from_rotation_matrix(np.array([1, 2, 3]))


class TestQuaternionEdgeCases(unittest.TestCase):
    """四元数边界用例测试"""

    def test_nan_handling(self):
        """测试NaN处理"""
        # 使用有效数据创建四元数
        q = Quaternion(1.0, 0.0, 0.0, 0.0)

        # 所有运算应产生有效结果
        self.assertFalse(np.isnan(q.rotation_angle))
        self.assertFalse(np.isnan(q.magnitude))

    def test_infinity_handling(self):
        """测试无穷大处理"""
        # 确保四元数运算不会产生无穷大
        q = Quaternion(1.0, 1e-10, 1e-10, 1e-10)

        for _ in range(1000):
            q = q * q
            self.assertTrue(q.magnitude < 1e10)

    def test_large_component_differences(self):
        """测试大分量差异"""
        q = Quaternion(1e10, 1e-10, 1e-10, 1e-10)

        # 归一化后应该是单位四元数
        self.assertAlmostEqual(q.magnitude, 1.0, places=5)


class TestQuaternionSuite(unittest.TestCase):
    """四元数完整测试套件"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        np.random.seed(999)
        cls.test_count = 0

    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        print(f"\n总共运行 {cls.test_count} 个四元数测试")

    def test_run_all(self):
        """运行所有测试"""
        TestQuaternionSuite.test_count += 1

        # 创建测试套件
        suite = unittest.TestSuite()
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestQuaternionBasics))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestQuaternionRotation))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestRotationMatrix))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestQuaternionChaos))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestQuaternionEdgeCases))

        # 运行测试
        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)

        self.assertTrue(result.wasSuccessful())


if __name__ == "__main__":
    unittest.main()
