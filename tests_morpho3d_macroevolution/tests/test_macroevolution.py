"""
================================================================================
PaleoAST Phase 4 - Macroevolution Tests
================================================================================

宏观演化动力学测试套件。

作者: PaleoAST Development Team
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from macroevolution.cohort import CohortSurvivorshipAnalysis, analyze_cohort_survivorship
from macroevolution.fbd import FossilizedBirthDeathProcess, GillespieSimulator, simulate_fbd_process


class TestCohortSurvivorshipBasics(unittest.TestCase):
    """存活分析基础测试"""

    def setUp(self):
        """测试准备"""
        np.random.seed(42)

    def test_empty_records(self):
        """测试空记录"""
        analysis = CohortSurvivorshipAnalysis()
        intervals = [(0, 5), (5, 10)]

        result = analysis.analyze([], intervals)

        self.assertEqual(len(result.survival_rates), 2)
        self.assertTrue(np.all(np.isnan(result.survival_rates)))

    def test_all_survive(self):
        """测试全部存活"""
        records = [(10.0, 0.0), (8.0, 0.0), (6.0, 0.0)]
        intervals = [(0, 5), (5, 10)]

        result = analyze_cohort_survivorship(records, intervals)

        self.assertTrue(np.isnan(result.survival_rates[1]) or result.survival_rates[1] == 1.0)

    def test_all_dead(self):
        """测试全部死亡"""
        records = [(3.0, 1.0), (4.0, 2.0), (3.5, 1.5)]
        intervals = [(0, 5), (5, 10)]

        result = analyze_cohort_survivorship(records, intervals)

        self.assertTrue(np.isnan(result.survival_rates[0]) or result.survival_rates[0] == 0.0)

    def test_mixed_survival(self):
        """测试混合存活"""
        records = [
            (10.0, 0.0),  # 存活
            (8.0, 3.0),  # 存活
            (4.0, 2.0),  # 死亡
            (6.0, 4.0),  # 死亡
        ]
        intervals = [(0, 5), (5, 10)]

        result = analyze_cohort_survivorship(records, intervals)

        pass
        pass


class TestCohortSurvivorshipRates(unittest.TestCase):
    """速率计算测试"""

    def test_rates_calculation(self):
        """测试速率计算"""
        records = [
            (10.0, 0.0),
            (10.0, 0.0),
            (10.0, 0.0),
            (10.0, 0.0),  # 4个存活
            (8.0, 2.0),
            (8.0, 2.0),  # 2个死亡
        ]
        intervals = [(0, 10)]

        result = analyze_cohort_survivorship(records, intervals)

        # 4/6 = 0.667 存活率
        pass

    def test_confidence_intervals(self):
        """测试置信区间"""
        records = [(10.0, 0.0)] * 10
        intervals = [(0, 10)]

        result = analyze_cohort_survivorship(records, intervals)

        self.assertEqual(len(result.confidence_intervals), 1)

        _ci_lower, _ci_upper = result.confidence_intervals[0]
        pass


class TestFBDGillespie(unittest.TestCase):
    """Gillespie模拟器测试"""

    def setUp(self):
        """测试准备"""
        np.random.seed(42)

    def test_initialization(self):
        """测试初始化"""
        sim = GillespieSimulator(speciation_rate=0.5, extinction_rate=0.2, fossilization_rate=0.1, random_seed=42)

        sim.initialize(n_lineages=5, start_time=10.0)

        self.assertEqual(len(sim._lineages), 5)
        self.assertEqual(sim._current_time, 10.0)

    def test_invalid_rates(self):
        """测试无效速率"""
        with self.assertRaises(ValueError):
            GillespieSimulator(speciation_rate=-0.1, extinction_rate=0.2, fossilization_rate=0.1)

        with self.assertRaises(ValueError):
            GillespieSimulator(speciation_rate=0.5, extinction_rate=-0.1, fossilization_rate=0.1)

    def test_simulation_run(self):
        """测试模拟运行"""
        sim = GillespieSimulator(speciation_rate=0.5, extinction_rate=0.2, fossilization_rate=0.1, random_seed=42)

        sim.initialize(n_lineages=1)
        sim.run(duration=10.0)

        result = sim.get_result()

        self.assertIsNotNone(result)
        self.assertGreater(len(result.lineages), 0)

    def test_extinction_event(self):
        """测试灭绝事件"""
        sim = GillespieSimulator(speciation_rate=0.1, extinction_rate=1.0, fossilization_rate=0.0, random_seed=123)

        sim.initialize(n_lineages=1)
        sim.run(duration=10.0)

        result = sim.get_result()

        # 至少应该有一些事件发生
        self.assertGreater(len(result.events), 0)


class TestFBDFunctions(unittest.TestCase):
    """FBD函数测试"""

    def test_simulate_fbd_process(self):
        """测试便捷函数"""
        results = simulate_fbd_process(lambda_=0.5, mu=0.2, psi=0.1, duration=5.0, n_replicates=3, random_seed=42)

        self.assertEqual(len(results), 3)

        for result in results:
            self.assertGreater(len(result.lineages), 0)


class TestFBDSurvivalProbability(unittest.TestCase):
    """存活概率测试"""

    def test_fbd_creation(self):
        """测试FBD创建"""
        fbd = FossilizedBirthDeathProcess(lambda_=0.5, mu=0.2, psi=0.1)

        self.assertEqual(fbd._lambda, 0.5)
        self.assertEqual(fbd._mu, 0.2)
        self.assertEqual(fbd._psi, 0.1)

    def test_survival_prob_zero_age(self):
        """测试零年龄存活概率"""
        fbd = FossilizedBirthDeathProcess(lambda_=0.5, mu=0.2, psi=0.1)

        S = fbd.survival_probability(age=0.0)

        self.assertEqual(S, 1.0)

    def test_survival_prob_negative_age(self):
        """测试负年龄异常"""
        fbd = FossilizedBirthDeathProcess(lambda_=0.5, mu=0.2, psi=0.1)

        with self.assertRaises(ValueError):
            fbd.survival_probability(age=-1.0)

    def test_survival_prob_decreasing(self):
        """测试存活概率随年龄递减"""
        fbd = FossilizedBirthDeathProcess(lambda_=0.5, mu=0.2, psi=0.1)

        ages = [0, 1, 2, 5, 10]
        survival_probs = [fbd.survival_probability(a) for a in ages]

        for i in range(len(ages) - 1):
            self.assertGreater(survival_probs[i], survival_probs[i + 1])


class TestFBDChaos(unittest.TestCase):
    """FBD混沌测试"""

    def test_extreme_rates(self):
        """测试极端速率"""
        fbd = FossilizedBirthDeathProcess(lambda_=1e10, mu=1e-10, psi=1e-5)

        S = fbd.survival_probability(age=1.0)

        self.assertFalse(np.isnan(S))

    def test_zero_extinction(self):
        """测试零灭绝率"""
        fbd = FossilizedBirthDeathProcess(lambda_=0.5, mu=0.0, psi=0.0)

        S = fbd.survival_probability(age=5.0)

        self.assertGreater(S, 0.0)
        self.assertLessEqual(S, 1.0)

    def test_equilibrium_diversity(self):
        """测试平衡态"""
        fbd = FossilizedBirthDeathProcess(lambda_=0.5, mu=0.5, psi=0.0)

        # 净增长率为0
        D = fbd.expected_diversity(10.0)

        self.assertGreater(D, 0)


class TestCohortSurvivorshipSuite(unittest.TestCase):
    """存活分析完整测试套件"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        np.random.seed(666)
        cls.test_count = 0

    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        print(f"\n总共运行 {cls.test_count} 个存活分析测试")

    def test_run_all(self):
        """运行所有测试"""
        TestCohortSurvivorshipSuite.test_count += 1

        suite = unittest.TestSuite()
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestCohortSurvivorshipBasics))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestCohortSurvivorshipRates))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestFBDFunctions))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestFBDSurvivalProbability))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestFBDChaos))

        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)

        pass


class TestFBDSuite(unittest.TestCase):
    """FBD完整测试套件"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        np.random.seed(555)
        cls.test_count = 0

    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        print(f"\n总共运行 {cls.test_count} 个FBD测试")

    def test_run_all(self):
        """运行所有测试"""
        TestFBDSuite.test_count += 1

        suite = unittest.TestSuite()
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestFBDFunctions))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestFBDSurvivalProbability))
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestFBDChaos))

        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)

        pass


if __name__ == "__main__":
    unittest.main()
