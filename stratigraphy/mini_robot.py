# =============================================================================
# FILE: stratigraphy/mini_robot.py
# =============================================================================
"""
Mini-robot: 把一串分析"录下来", 再"重放"。

Acycle 的 Mini-Robot 是什么
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Acycle 的 mini-robot 是 ``code/guicode/RobotGUI.m``。读完之后的结论是: 它
**不是**一个通用的"录屏回放"工具, 而是一条**硬编码的分级批处理流水线**,
共六级, 每级可单独开关, 级与级之间可以暂停, 中间结果可以存盘:

1. **准备数据** —— 去 NaN、去 inf、按时间排序、重复时间点取均值;
2. **插值** —— 按选定的采样率 (平均/中位/最大/最小/指定值) 重采样成等间距;
3. **去趋势** —— mean / linear / lowess / rlowess / loess / rloess, 窗口按
   百分比给;
4. **功率谱 + 噪声** —— 多锥窗 (2πMTM) 或周期图, 配稳健 AR(1) 红噪声置信;
5. **进化 FFT** —— 滑动窗口的 evoFFT;
6. **小波变换** —— 给定周期范围。

它的本质因此是**一份"配方"**: 一串带参数的步骤, 录下来, 再对一批数据重放。
本模块把同一件事表达成 PaleoAST 的原生形式:

* **一步** = 目录 (``plugins/catalog.py``) 里的一个分析名 + 一组参数
  (:class:`RobotStep`);
* **数据** = 电子表格 (两列: 时间轴与观测值);
* **执行** = 插件注册表 (:func:`plugins.registry.get_plugin_registry`), 所以
  mini-robot 能跑的分析, 就是用户界面里能跑的分析;
* **重放** = :func:`run_robot` 按顺序跑完每一步; **批量** =
  :func:`run_robot_batch` 把同一份配方对一批数据各跑一遍。

与 Acycle 的差别（要说清楚）
----------------------------
Acycle 的流水线每一级都改写同一份数据, 上一级的输出直接喂给下一级; 这里的
步骤是彼此独立的分析, 各自返回自己的结果对象（不是一个可以串起来的数组）,
所以**默认每一步吃的是同一份数据**, 而不是上一级的输出。需要"上一级结果接着
算"的地方, 由调用方自己把结果取出来再传进去。

存盘
----
配方存成 JSON, 沿用仓库既有的预置机制: 写盘走
``presets.schema.atomic_write_json``（临时文件 + ``os.replace``, 崩溃也不会留下
半个文件）, 字段命名与 ``presets`` 的预置文件一致 (``format`` / ``name`` /
``params`` / ``created_at`` / ``app_version``)。预置文件本身只描述**一个**
分析, 装不下"一串步骤", 因此机器人文件用自己的格式标记
``paleoast_robot_v1``, 但读写方式与预置文件同源。

失败处理
--------
一步失败**不会让整批崩掉**: 失败被记成 ``succeeded=False`` 并带上错误信息,
后面的步骤照跑。只有显式传 ``stop_on_error=True`` 时才会在第一个失败处停下
并把异常抛出来。

作者: PaleoAST 开发组
版本: 1.0.0
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from config import APP_VERSION
from plugins.registry import get_plugin_registry
from presets.schema import atomic_write_json
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "ROBOT_FORMAT",
    "RobotResult",
    "RobotResultEntry",
    "RobotStep",
    "robot_from_preset",
    "run_robot",
    "run_robot_batch",
    "save_robot",
]

#: 机器人配方文件的格式标记。
ROBOT_FORMAT = "paleoast_robot_v1"


# =============================================================================
# 数据容器
# =============================================================================


@dataclass(frozen=True)
class RobotStep:
    """
    一步操作。

    Attributes:
        analysis: 目录里的分析名（如 ``"spectral"``）。
        parameters: 传给该分析的参数（关键字参数）。
        label: 这一步的人类可读名字; 留空时用分析名。
    """

    analysis: str
    parameters: dict[str, Any] = field(default_factory=dict)
    label: str = ""


@dataclass(frozen=True)
class RobotResultEntry:
    """
    一步操作的执行结果。

    Attributes:
        label: 这一步的名字（来自 :attr:`RobotStep.label`, 为空时是分析名）。
        analysis: 分析名。
        succeeded: 成功与否。
        payload: 成功时是分析自己的返回值; 失败时是 ``None``。
        error: 失败时的错误信息。
    """

    label: str
    analysis: str
    succeeded: bool
    payload: Any
    error: str = ""


@dataclass(frozen=True)
class RobotResult:
    """
    一次重放的完整结果。

    Attributes:
        steps: 每一步的执行结果, 顺序与配方一致。
    """

    steps: list[RobotResultEntry]

    @property
    def succeeded(self) -> bool:
        """全部步骤都成功时为 True。"""
        return all(entry.succeeded for entry in self.steps)

    @property
    def failures(self) -> list[RobotResultEntry]:
        """失败的步骤。"""
        return [entry for entry in self.steps if not entry.succeeded]

    def summary(self) -> str:
        """
        返回一行人类可读的摘要。

        Returns:
            str: 摘要文本, 含成功/失败步数与失败步骤的名字。
        """
        ok = sum(1 for entry in self.steps if entry.succeeded)
        failed = len(self.steps) - ok
        text = f"mini-robot: {len(self.steps)} 步, {ok} 成功, {failed} 失败"
        if failed:
            names = ", ".join(f"{entry.label or entry.analysis}" for entry in self.failures)
            text += f" (失败的步骤: {names})"
        return text


# =============================================================================
# 校验
# =============================================================================


def _validate_steps(steps: Sequence[RobotStep]) -> list[RobotStep]:
    """检查每一步的形状; 非法配方抛 ``ValidationError``。"""
    if steps is None:
        raise ValidationError("steps must be a sequence of RobotStep, got None")
    recipe: list[RobotStep] = []
    for index, step in enumerate(steps):
        if not isinstance(step, RobotStep):
            raise ValidationError(
                f"step {index} is not a RobotStep",
                details={"index": index, "type": type(step).__name__},
            )
        name = step.analysis.strip() if isinstance(step.analysis, str) else ""
        if not name:
            raise ValidationError(
                f"step {index} has an empty analysis name",
                details={"index": index},
            )
        if not isinstance(step.parameters, dict):
            raise ValidationError(
                f"step {index} has non-mapping parameters",
                details={"index": index, "type": type(step.parameters).__name__},
            )
        if not isinstance(step.label, str):
            raise ValidationError(
                f"step {index} has a non-string label",
                details={"index": index, "type": type(step.label).__name__},
            )
        recipe.append(RobotStep(analysis=name, parameters=dict(step.parameters), label=step.label))
    return recipe


def _resolve_registry(registry: Any) -> Any:
    """拿到插件注册表; 注册表是空的就先把内置目录装进去。"""
    active = registry if registry is not None else get_plugin_registry()
    try:
        empty = not active.list_plugins()
    except Exception:  # 一个坏掉的注册表不该让机器人整个失效
        logger.warning("the plugin registry could not be listed", exc_info=True)
        return active
    if empty:
        try:
            from plugins.catalog import register_builtin_analyses

            register_builtin_analyses(registry=active)
        except Exception:  # 目录坏掉时只能让每一步都失败, 而不是崩掉
            logger.warning("could not register the built-in analyses; the robot will find none", exc_info=True)
    return active


# =============================================================================
# 执行
# =============================================================================


def _run_step(step: RobotStep, data: Any, registry: Any, stop_on_error: bool) -> RobotResultEntry:
    """跑一步; 失败被记成一条失败记录, 除非 ``stop_on_error``。"""
    label = step.label or step.analysis
    plugin = registry.get(step.analysis)
    if plugin is None:
        message = f"no analysis named {step.analysis!r} is registered"
        if stop_on_error:
            raise ValidationError(message, details={"analysis": step.analysis})
        return RobotResultEntry(
            label=label,
            analysis=step.analysis,
            succeeded=False,
            payload=None,
            error=message,
        )

    try:
        outcome = plugin.analyze(data, **dict(step.parameters))
    except Exception as exc:
        if stop_on_error:
            raise
        logger.warning("mini-robot step %r failed: %s", step.analysis, exc)
        return RobotResultEntry(
            label=label,
            analysis=step.analysis,
            succeeded=False,
            payload=None,
            error=str(exc) or exc.__class__.__name__,
        )

    if getattr(outcome, "success", True) is False:
        message = str(getattr(outcome, "error", "") or "the analysis reported a failure")
        if stop_on_error:
            raise ValidationError(message, details={"analysis": step.analysis})
        return RobotResultEntry(
            label=label,
            analysis=step.analysis,
            succeeded=False,
            payload=getattr(outcome, "data", None),
            error=message,
        )

    return RobotResultEntry(
        label=label,
        analysis=step.analysis,
        succeeded=True,
        payload=getattr(outcome, "data", outcome),
        error="",
    )


def run_robot(
    steps: Sequence[RobotStep],
    data: Any = None,
    registry: Any = None,
    stop_on_error: bool = False,
) -> RobotResult:
    """
    按顺序重放一串步骤。

    每一步都从插件注册表里取出分析, 把 ``data`` 与该步的参数交给它。一步
    失败不会让整批崩掉: 失败被记成 ``succeeded=False`` 并带上错误信息, 后面的
    步骤照跑。

    Parameters:
        steps: 步骤配方, 顺序即执行顺序。
        data: 交给每一步的数据。通常是电子表格 (两列: 时间轴与观测值);
            也可以是"槽位名 -> 数组"的字典, 或不需要数据的分析所配的
            ``None``。
        registry: 插件注册表; ``None`` 时用全局单例。
        stop_on_error: 为 True 时, 第一个失败的步骤就停下并抛出异常
            (分析自己抛的异常原样上抛; 未注册的分析名抛
            ``ValidationError``)。

    Returns:
        :class:`RobotResult`: 每一步的执行结果, 顺序与配方一致。

    Raises:
        ValidationError: 配方本身不合法（步不是 ``RobotStep``、分析名为空、
            参数不是映射等）, 或 ``stop_on_error`` 为 True 且某一步失败。

    Example:
        >>> steps = [RobotStep("derivative", label="一阶导数")]
        >>> result = run_robot(steps, data)  # doctest: +SKIP
        >>> result.succeeded  # doctest: +SKIP
        True
    """
    recipe = _validate_steps(steps)
    active = _resolve_registry(registry)

    entries: list[RobotResultEntry] = []
    for step in recipe:
        entries.append(_run_step(step, data, active, stop_on_error))
    return RobotResult(steps=entries)


def run_robot_batch(
    steps: Sequence[RobotStep],
    datasets: Sequence[Any],
    registry: Any = None,
    stop_on_error: bool = False,
) -> list[RobotResult]:
    """
    把同一份配方对一批数据各跑一遍（Acycle mini-robot 的"批量"用法）。

    Parameters:
        steps: 步骤配方。
        datasets: 一批数据, 每个元素就是一份给 ``run_robot`` 的数据。
        registry: 插件注册表; ``None`` 时用全局单例。
        stop_on_error: 传给 :func:`run_robot`。

    Returns:
        list[RobotResult]: 每个数据集一份结果, 顺序与 ``datasets`` 一致。

    Raises:
        ValidationError: 配方不合法, 或 ``stop_on_error`` 为 True 且某一步失败。
    """
    recipe = _validate_steps(steps)
    if datasets is None:
        raise ValidationError("datasets must be a sequence of datasets, got None")
    return [run_robot(recipe, data=dataset, registry=registry, stop_on_error=stop_on_error) for dataset in datasets]


# =============================================================================
# 存盘
# =============================================================================


def _step_payload(step: RobotStep) -> dict[str, Any]:
    """把一步变成 JSON 友好的字典; 参数不可序列化时抛 ``ValidationError``。"""
    payload = {
        "analysis": step.analysis,
        "params": dict(step.parameters),
        "label": step.label,
    }
    try:
        json.dumps(payload)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"the parameters of step {step.analysis!r} are not JSON-serialisable",
            details={"analysis": step.analysis, "reason": str(exc)},
        ) from exc
    return payload


def save_robot(steps: Sequence[RobotStep], path: str) -> str:
    """
    把配方写成 JSON 文件。

    写盘走 ``presets.schema.atomic_write_json``（临时文件 + ``os.replace``）,
    与仓库里预置文件的写法一致。

    Parameters:
        steps: 步骤配方。
        path: 目标文件路径。

    Returns:
        str: 写入的路径。

    Raises:
        ValidationError: 配方不合法, 参数不可序列化, 或写盘失败。
    """
    recipe = _validate_steps(steps)
    payload = {
        "format": ROBOT_FORMAT,
        "name": os.path.splitext(os.path.basename(path))[0],
        "steps": [_step_payload(step) for step in recipe],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "app_version": APP_VERSION,
    }
    try:
        atomic_write_json(path, payload)
    except OSError as exc:
        raise ValidationError(
            f"the robot file could not be written: {path}",
            details={"path": path, "reason": str(exc)},
        ) from exc
    logger.info("mini-robot recipe written: %s", path)
    return path


def robot_from_preset(path: str) -> list[RobotStep]:
    """
    读回一份机器人配方。

    Parameters:
        path: 配方文件路径。

    Returns:
        list[RobotStep]: 步骤列表, 顺序与文件里一致。

    Raises:
        ValidationError: 文件不存在、不是合法 JSON、格式标记不对, 或某一步
            的内容不合法。
    """
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ValidationError(f"the robot file does not exist: {path}", details={"path": path}) from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"the robot file is not valid JSON: {path}",
            details={"path": path, "reason": str(exc)},
        ) from exc
    except OSError as exc:
        raise ValidationError(
            f"the robot file could not be read: {path}",
            details={"path": path, "reason": str(exc)},
        ) from exc

    if not isinstance(payload, dict):
        raise ValidationError(
            "a robot file must be a JSON object",
            details={"path": path, "type": type(payload).__name__},
        )
    marker = payload.get("format")
    if marker != ROBOT_FORMAT:
        raise ValidationError(
            f"the robot file has format {marker!r}, expected {ROBOT_FORMAT!r}",
            details={"path": path, "format": marker},
        )

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        raise ValidationError(
            "a robot file must contain a 'steps' list",
            details={"path": path, "type": type(raw_steps).__name__},
        )

    steps: list[RobotStep] = []
    for index, item in enumerate(raw_steps):
        if not isinstance(item, dict):
            raise ValidationError(
                f"step {index} of the robot file is not an object",
                details={"path": path, "index": index, "type": type(item).__name__},
            )
        analysis = item.get("analysis")
        if not isinstance(analysis, str) or not analysis.strip():
            raise ValidationError(
                f"step {index} of the robot file has no usable 'analysis' name",
                details={"path": path, "index": index},
            )
        parameters = item.get("params", item.get("parameters", {}))
        if not isinstance(parameters, dict):
            raise ValidationError(
                f"step {index} of the robot file has a non-object 'params'",
                details={"path": path, "index": index, "type": type(parameters).__name__},
            )
        label = item.get("label", "")
        if not isinstance(label, str):
            raise ValidationError(
                f"step {index} of the robot file has a non-string 'label'",
                details={"path": path, "index": index, "type": type(label).__name__},
            )
        steps.append(RobotStep(analysis=analysis.strip(), parameters=dict(parameters), label=label))

    return steps
