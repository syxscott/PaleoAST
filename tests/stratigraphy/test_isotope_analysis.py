# tests/stratigraphy/test_isotope_analysis.py
import numpy as np
from stratigraphy.isotope_analysis import (
    IsotopeData,
    IsotopeResult,
    compute_moving_average,
    detect_excursions_from_values,
    compute_correlation
)

def test_isotope_data_creation():
    """测试 IsotopeData 数据类"""
    depth = np.array([0, 10, 20, 30, 40, 50])
    age = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    d13C = np.array([-2.5, -2.3, -2.8, -2.1, -3.0, -2.4])
    d18O = np.array([-1.0, -0.8, -1.2, -0.7, -1.5, -0.9])

    data = IsotopeData(depth=depth, age=age, d13C=d13C, d18O=d18O)

    assert len(data.depth) == 6
    assert len(data.d13C) == 6
    assert len(data.d18O) == 6

def test_moving_average():
    """测试移动平均"""
    # 模拟有趋势的数据
    t = np.linspace(0, 10, 100)
    signal = np.sin(t) + 0.1 * np.random.randn(100)

    smoothed = compute_moving_average(signal, window=5)

    # 平滑后长度应与原始相同
    assert len(smoothed) == len(signal)

    # 平滑后波动应减小
    original_var = np.var(np.diff(signal))
    smoothed_var = np.var(np.diff(smoothed))
    assert smoothed_var < original_var

    print(f"Original variance: {original_var:.4f}")
    print(f"Smoothed variance: {smoothed_var:.4f}")

def test_excursion_detection():
    """测试 excursion 检测"""
    # 模拟 excursion 数据
    signal = np.array([0.0, 0.1, 0.2, 1.5, 1.8, 0.3, 0.1, -0.1, 0.2])

    excursions = detect_excursions_from_values(signal, threshold=1.0, min_duration=2)

    # 应检测到 1 个 excursion (索引 3-5)
    assert len(excursions) >= 1

    # 验证 excursion 位置
    if len(excursions) > 0:
        excursion = excursions[0]
        assert excursion.start_idx <= 3 <= excursion.end_idx

    print(f"Detected {len(excursions)} excursion(s)")

def test_correlation():
    """测试相关性分析"""
    d13C = np.array([-2.5, -2.3, -2.8, -2.1, -3.0, -2.4])
    d18O = np.array([-1.0, -0.8, -1.2, -0.7, -1.5, -0.9])

    r, p = compute_correlation(d13C, d18O)

    # 验证相关系数在 [-1, 1]
    assert -1 <= r <= 1
    assert 0 <= p <= 1

    print(f"Correlation: r={r:.4f}, p={p:.4f}")
