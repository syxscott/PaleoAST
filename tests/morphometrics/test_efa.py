# tests/morphometrics/test_efa.py
import numpy as np

from morphometrics.efa import EFAAnalyzer


def test_efa_analyzer_creation():
    """测试 EFAAnalyzer 创建"""
    analyzer = EFAAnalyzer()
    assert analyzer is not None


def test_fourier_coefficients():
    """测试傅里叶系数计算"""
    # 单位圆
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    x = np.cos(t)
    y = np.sin(t)
    coords = np.column_stack([x, y])

    analyzer = EFAAnalyzer()
    result = analyzer.analyze(coords, n_harmonics=5, n_points=100)

    assert result.coefficients.shape == (5, 4)  # 5谐波 × 4系数 (a,b,c,d)
    assert result.n_harmonics == 5
    assert result.n_points == 100


def test_contour_reconstruction():
    """测试轮廓重建"""
    # 原始坐标 - 稍微变形的椭圆
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    x = 2 * np.cos(t) + 0.1 * np.sin(3 * t)
    y = np.sin(t)
    coords = np.column_stack([x, y])

    analyzer = EFAAnalyzer()
    result = analyzer.analyze(coords, n_harmonics=10, n_points=100)

    # 重建误差应小于 5%
    error = np.sqrt(np.mean((result.original - result.reconstructed) ** 2))
    assert error < 0.1, f"Reconstruction error {error} too high"


def test_reconstruct_from_coefficients():
    """测试从系数重建轮廓"""
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    x = np.cos(t)
    y = np.sin(t)
    coords = np.column_stack([x, y])

    analyzer = EFAAnalyzer()
    result = analyzer.analyze(coords, n_harmonics=10, n_points=100)

    # 使用新方法从系数重建
    reconstructed = analyzer.reconstruct_from_coefficients(result.a0, result.c0, result.coefficients, n_points=100)

    assert reconstructed.shape == (100, 2)


def test_efaresult_summary():
    """测试 EFAResult.summary() 方法"""
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    x = np.cos(t)
    y = np.sin(t)
    coords = np.column_stack([x, y])

    analyzer = EFAAnalyzer()
    result = analyzer.analyze(coords, n_harmonics=5, n_points=100)

    summary = result.summary()
    assert "Elliptic Fourier Analysis" in summary
    assert "Harmonics" in summary or "谐波" in summary
