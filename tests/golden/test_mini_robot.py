# =============================================================================
# FILE: tests/golden/test_mini_robot.py
# =============================================================================
"""
``stratigraphy.mini_robot`` 的测试: 录下一串分析, 再重放。

这个模块对应 Acycle 的 mini-robot (``code/guicode/RobotGUI.m``)。读完之后
的结论是: 它是一条**硬编码的分级批处理流水线** —— 准备数据 (去 NaN/去 inf/
排序/重复点取均值) → 插值重采样 → 去趋势 → 功率谱 + 稳健 AR(1) 红噪声 →
进化 FFT → 小波变换, 每级可单独开关, 中间结果可存盘。它的本质是**一份
"配方"**: 一串带参数的步骤, 录下来, 再对一批数据重放。

PaleoAST 里的对应形式: 一步 = 目录里的一个分析 + 一组参数, 数据 = 电子表格,
执行 = 插件注册表。于是本文件断言的是**执行模型**的性质:

1. 一份由**真实目录条目**组成的配方能跑通, 每一步都成功, 结果按顺序返回;
2. 配方里放一个**不存在的分析名**, 它被记为失败（``succeeded=False`` 且带
   错误信息）, 而**不是**让整批抛异常;
3. ``stop_on_error=True`` 时才会在第一个失败处停下;
4. 配方能存成 JSON 再读回来, 且读回来的内容与存进去的一致;
5. 同一份配方能对一批数据各跑一遍（Acycle 的"批量"用法）。

参考来源
--------
* Acycle: https://github.com/mingsongli/acycle （GPL; 本仓库是独立实现,
  没有翻译其 MATLAB 源码）。
* Li, M., Kump, L.R., Hinnov, L.A., Mann, M.E., 2018. Cyclostratigraphic
  correlation of astronomical forcing and Earth system processes. *Earth and
  Planetary Science Letters*, 501: 165-179. (COCO 的方法本身, 供对照。)
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"mini_robot"}


import json

import numpy as np
import pytest

from stratigraphy.mini_robot import (
    ROBOT_FORMAT,
    RobotResult,
    RobotResultEntry,
    RobotStep,
    robot_from_preset,
    run_robot,
    run_robot_batch,
    save_robot,
)
from utils.exceptions import ValidationError

# =============================================================================
# 常量
# =============================================================================

#: 合成记录: 长度 (kyr)、两个周期 (kyr) 与噪声水平。
RECORD_LENGTH_KYR = 100.0
RECORD_POINTS = 256
CYCLE_LONG_KYR = 20.0
CYCLE_SHORT_KYR = 5.0
NOISE_LEVEL = 0.05

#: 一个不存在的分析名。
MISSING_ANALYSIS = "no_such_analysis_anywhere"

#: 配方里用的真实目录条目 (名称, 参数)。
RECIPE: tuple[tuple[str, dict], ...] = (
    ("derivative", {}),
    ("spectral", {"n_frequencies": 200}),
    ("wavelet", {}),
)


# =============================================================================
# 辅助
# =============================================================================


@pytest.fixture(scope="module", autouse=True)
def catalog() -> None:
    """把内置目录装进插件注册表。

    注册表是全局单例, 而 ``tests/plugins/test_catalog.py`` 会把它清空;
    ``register_builtin_analyses`` 是幂等的, 所以这里调用总是安全的。
    """
    from plugins.catalog import register_builtin_analyses

    register_builtin_analyses()


def _record(seed: int = 5, points: int = RECORD_POINTS) -> np.ndarray:
    """合成的两列记录: 时间轴 (kyr) 与观测值。"""
    rng = np.random.default_rng(seed)
    time_axis = np.linspace(0.0, RECORD_LENGTH_KYR, points)
    values = np.sin(2.0 * np.pi * time_axis / CYCLE_LONG_KYR)
    values += 0.3 * np.sin(2.0 * np.pi * time_axis / CYCLE_SHORT_KYR)
    values += NOISE_LEVEL * rng.standard_normal(points)
    return np.column_stack([time_axis, values])


def _recipe() -> list[RobotStep]:
    """由真实目录条目组成的配方。"""
    return [
        RobotStep(analysis=name, parameters=dict(parameters), label=f"第{index}步:{name}")
        for index, (name, parameters) in enumerate(RECIPE, start=1)
    ]


# =============================================================================
# 1. 正常执行
# =============================================================================


class TestRobotRunsRealAnalyses:
    """一份由真实目录条目组成的配方必须能跑通。"""

    def test_every_step_succeeds(self):
        """三步全部成功。"""
        result = run_robot(_recipe(), _record())

        assert isinstance(result, RobotResult)
        assert len(result.steps) == len(RECIPE)
        assert all(entry.succeeded for entry in result.steps), [
            (entry.label, entry.error) for entry in result.steps if not entry.succeeded
        ]
        assert result.succeeded is True
        assert result.failures == []

    def test_the_results_come_back_in_order(self):
        """结果必须按配方顺序返回, 且名字与配方对得上。"""
        result = run_robot(_recipe(), _record())

        assert [entry.analysis for entry in result.steps] == [name for name, _ in RECIPE]
        assert [entry.label for entry in result.steps] == [step.label for step in _recipe()]

    def test_each_step_carries_the_analysis_payload(self):
        """每一步都必须带回分析自己的结果对象。"""
        result = run_robot(_recipe(), _record())

        for entry in result.steps:
            assert entry.payload is not None, f"{entry.analysis} returned nothing"
            assert entry.error == ""

    def test_the_parameters_reach_the_analysis(self):
        """步骤上的参数必须真的传给了分析。"""
        steps = [RobotStep("spectral", {"n_frequencies": 64}, label="功率谱")]

        result = run_robot(steps, _record())

        assert result.steps[0].succeeded
        frequencies = getattr(result.steps[0].payload, "frequencies", None)
        assert frequencies is not None
        assert int(np.asarray(frequencies).size) == 64


# =============================================================================
# 2. 失败处理
# =============================================================================


class TestRobotFailureHandling:
    """一步失败不许把整批带崩。"""

    def test_an_unknown_analysis_is_recorded_as_a_failure(self):
        """不存在的分析名被记为失败, 其余步骤照跑, 整批不抛异常。"""
        steps = [
            RobotStep("derivative", label="一阶导数"),
            RobotStep(MISSING_ANALYSIS, label="不存在的分析"),
            RobotStep("wavelet", label="小波"),
        ]

        result = run_robot(steps, _record())

        assert len(result.steps) == 3
        assert result.steps[0].succeeded is True
        assert result.steps[1].succeeded is False
        assert result.steps[1].analysis == MISSING_ANALYSIS
        assert result.steps[1].error, "a failed step must carry an error message"
        assert MISSING_ANALYSIS in result.steps[1].error
        assert result.steps[2].succeeded is True
        assert result.succeeded is False
        assert len(result.failures) == 1

    def test_a_failing_analysis_is_recorded_as_a_failure(self):
        """分析自己抛出的异常也被记成失败, 而不是让整批崩掉。

        这里用的是"配方里参数写错了"的真实情形: 插值方法名不在允许的取值
        里, 分析会抛 ``ValidationError``。
        """
        steps = [
            RobotStep("interpolate_series", {"new_time": [0.0, 1.0, 2.0], "method": "bogus"}, label="坏参数"),
        ]

        result = run_robot(steps, _record())

        assert result.steps[0].succeeded is False
        assert result.steps[0].error, "a failed step must carry the analysis' own message"
        assert result.steps[0].payload is None

    def test_stop_on_error_stops_at_the_first_failure(self):
        """``stop_on_error=True`` 时, 第一个失败就把异常抛出来。"""
        steps = [
            RobotStep("derivative", label="一阶导数"),
            RobotStep(MISSING_ANALYSIS, label="不存在的分析"),
            RobotStep("wavelet", label="小波"),
        ]

        with pytest.raises(ValidationError):
            run_robot(steps, _record(), stop_on_error=True)

    def test_the_summary_counts_the_failures(self):
        """摘要必须说清楚几步成功、几步失败。"""
        steps = [
            RobotStep("derivative", label="一阶导数"),
            RobotStep(MISSING_ANALYSIS, label="不存在的分析"),
        ]

        text = run_robot(steps, _record()).summary()

        assert "mini-robot" in text
        assert "1" in text and "2" in text


# =============================================================================
# 3. 批量
# =============================================================================


class TestRobotBatch:
    """同一份配方对一批数据各跑一遍。"""

    def test_every_dataset_gets_its_own_result(self):
        results = run_robot_batch(_recipe(), [_record(seed=1), _record(seed=2)])

        assert len(results) == 2
        for result in results:
            assert isinstance(result, RobotResult)
            assert len(result.steps) == len(RECIPE)
            assert all(entry.succeeded for entry in result.steps)

    def test_a_failing_step_does_not_stop_the_batch(self):
        steps = [
            RobotStep("derivative", label="一阶导数"),
            RobotStep(MISSING_ANALYSIS, label="不存在的分析"),
        ]

        results = run_robot_batch(steps, [_record(seed=1), _record(seed=2)])

        assert len(results) == 2
        for result in results:
            assert result.steps[0].succeeded is True
            assert result.steps[1].succeeded is False


# =============================================================================
# 4. 存盘与读回
# =============================================================================


class TestRobotPresetRoundTrip:
    """配方能存成 JSON, 再读回来。"""

    def test_a_saved_recipe_reads_back_unchanged(self, tmp_path):
        steps = _recipe()
        path = str(tmp_path / "recipe.json")

        written = save_robot(steps, path)
        loaded = robot_from_preset(written)

        assert written == path
        assert [step.analysis for step in loaded] == [step.analysis for step in steps]
        assert [step.parameters for step in loaded] == [step.parameters for step in steps]
        assert [step.label for step in loaded] == [step.label for step in steps]

    def test_the_file_is_json_with_the_expected_shape(self, tmp_path):
        path = str(tmp_path / "recipe.json")
        save_robot(_recipe(), path)

        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)

        assert payload["format"] == ROBOT_FORMAT
        assert isinstance(payload["steps"], list)
        assert len(payload["steps"]) == len(RECIPE)
        assert payload["steps"][0]["analysis"] == RECIPE[0][0]
        assert payload["steps"][0]["params"] == dict(RECIPE[0][1])
        assert "created_at" in payload
        assert "app_version" in payload

    def test_a_saved_recipe_runs_after_being_loaded(self, tmp_path):
        path = str(tmp_path / "recipe.json")
        save_robot(_recipe(), path)

        result = run_robot(robot_from_preset(path), _record())

        assert all(entry.succeeded for entry in result.steps)

    def test_a_missing_file_is_rejected(self, tmp_path):
        with pytest.raises(ValidationError):
            robot_from_preset(str(tmp_path / "nothing_here.json"))

    def test_a_file_that_is_not_json_is_rejected(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("this is not json", encoding="utf-8")

        with pytest.raises(ValidationError):
            robot_from_preset(str(path))

    def test_a_preset_file_is_not_a_robot_file(self, tmp_path):
        """预置文件的格式标记不同, 不能当机器人文件读。"""
        path = tmp_path / "preset.json"
        path.write_text(
            json.dumps({"format": "paleoast_preset_v1", "analysis": "coco", "params": {}}),
            encoding="utf-8",
        )

        with pytest.raises(ValidationError):
            robot_from_preset(str(path))

    def test_a_step_without_an_analysis_name_is_rejected(self, tmp_path):
        path = tmp_path / "recipe.json"
        path.write_text(
            json.dumps({"format": ROBOT_FORMAT, "steps": [{"params": {}}]}),
            encoding="utf-8",
        )

        with pytest.raises(ValidationError):
            robot_from_preset(str(path))


# =============================================================================
# 5. 边界
# =============================================================================


class TestBoundaries:
    """非法配方必须抛 ``ValidationError``。"""

    def test_an_empty_analysis_name_is_rejected(self):
        with pytest.raises(ValidationError):
            run_robot([RobotStep("   ")], _record())

    def test_a_step_that_is_not_a_robot_step_is_rejected(self):
        with pytest.raises(ValidationError):
            run_robot([{"analysis": "derivative"}], _record())

    def test_non_mapping_parameters_are_rejected(self):
        with pytest.raises(ValidationError):
            run_robot([RobotStep("derivative", parameters=["not", "a", "mapping"])], _record())

    def test_an_empty_recipe_runs_and_returns_nothing(self):
        result = run_robot([], _record())

        assert result.steps == []
        assert result.succeeded is True

    def test_a_result_entry_can_be_built_by_hand(self):
        entry = RobotResultEntry(label="一步", analysis="derivative", succeeded=True, payload=None)

        assert entry.error == ""
