"""数值后端配置与只读资产解析；角色模型配置由 runtime/configuration 管理。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from .schemas import JSON, INT, NULL, NUMBER, POS, STR, arr, obj, require, validate
from .quantization.config import OPTIONAL_SPEC

NULL_STR = {"anyOf": [STR, NULL]}
NULL_NUMBER = {"anyOf": [NUMBER, NULL]}
CONSTRAINTS_SCHEMA = obj({
    "precision_mode": {"enum": ["mixed", "fixed"]},
    "allowed_formats": arr(STR, minItems=1, uniqueItems=True),
    "fixed_weight_format": NULL_STR, "fixed_activation_format": NULL_STR,
    "protected_modules": arr(STR, uniqueItems=True), "excluded_modules": arr(STR, uniqueItems=True),
    "fixed_layers": obj({}, [], additional=obj({"weight": OPTIONAL_SPEC, "activation": OPTIONAL_SPEC}, [])),
})
CONFIG_SCHEMA = obj({
    "mode": {"enum": ["real"]},
    "task": obj({"name": STR, "metric": {"enum": ["task_accuracy", "feature_relative_mse"]},
                 "max_error": NULL_NUMBER, "goal_error": NULL_NUMBER}),
    "constraints": CONSTRAINTS_SCHEMA,
    "backend": obj({"kind": {"enum": ["server"]}, "seed": INT,
                    "calibration_samples": {"type": "integer", "minimum": 1, "maximum": 256},
                    "search_samples": {"type": "integer", "minimum": 1, "maximum": 50000},
                    "final_samples": {"type": "integer", "minimum": 1, "maximum": 50000},
                    "data_seeds": obj({"calibration": INT, "search": INT, "final": INT}),
                    "model_path": NULL_STR, "data_path": NULL_STR}),
    "budget": obj({"trials": INT, "tools": INT, "agents": INT, "measurements": INT,
                   "seconds": NUMBER, "consecutive_failures": POS}),
    "output_dir": STR, "memory_dir": STR,
})

# 服务器资产字段；文件相对配置所在目录解析。
CONFIG_SCHEMA["properties"]["backend"]["properties"].update({
    "asset_manifest": NULL_STR, "model_id": NULL_STR, "device": STR,
    "data_protocol": {"enum": ["imagefolder_manifest_v1", "imagefolder_manifest_v2"]},
    "data_sha256": STR,
    "task_head": obj({"kind": {"enum": ["normalized_cls_linear"]}, "path": STR,
                      "sha256": STR, "backbone_sha256": STR}),
})
from .execution.contracts import FORMATS, LOOP_SCHEMA, SPEC, validate_loop
CONFIG_SCHEMA["properties"]["loop"] = LOOP_SCHEMA
# 固定层 v2 使用同一格式契约（bits 由 format 推导），不能误用仅 minmax
# 的旧 toy schema 拒绝 FP16 或动态 per_token；v1 的定义保持不变。
LOOP_CONFIG_SCHEMA = deepcopy(CONFIG_SCHEMA)
loop_fixed_spec = obj({k: v for k, v in SPEC['properties'].items() if k != 'bits'})
loop_optional_spec = {'anyOf': [NULL, loop_fixed_spec]}
LOOP_CONFIG_SCHEMA['properties']['constraints']['properties']['fixed_layers'] = obj(
    {}, [], additional=obj({'weight': loop_optional_spec, 'activation': loop_optional_spec}, []))


def validate_config(config: JSON) -> None:
    validate(config, LOOP_CONFIG_SCHEMA if 'loop' in config else CONFIG_SCHEMA)
    if "loop" in config:
        validate_loop(config["loop"])
        require(config["backend"]["kind"] == "server", "可执行视觉闭环需要真实 PyTorch 后端")
        require(set(config['constraints']['allowed_formats']) <= FORMATS.keys(), "执行格式未注册")
        if config["backend"].get("task_head"):
            require(not config["loop"].get("quantize_classifier", False), "外接冻结分类头不得量化")
    metric = ("feature_relative_mse" if config.get('loop', {}).get('evaluation') == 'feature_fidelity'
              else "task_accuracy")
    require(config["mode"] == "real" and config["task"]["metric"] == metric,
            "server 后端需要 real 模式及真实任务指标", "$.task.metric")
    require(len(set(config["backend"]["data_seeds"].values())) == 3,
            "校准、搜索、终评必须使用不同种子", "$.backend.data_seeds")
    goal, limit = config["task"]["goal_error"], config["task"]["max_error"]
    require(goal is None or limit is None or goal <= limit, "目标误差不能大于可行性上限", "$.task.goal_error")
    c = config["constraints"]
    if c["precision_mode"] == "fixed":
        require(c["fixed_weight_format"] is not None or c["fixed_activation_format"] is not None,
                "固定规格至少量化 W/A 之一", "$.constraints")
    else:
        require(c["fixed_weight_format"] is None and c["fixed_activation_format"] is None,
                "mixed 模式使用 fixed_layers 表达局部固定约束", "$.constraints")


def load_config(path: Path) -> JSON:
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_config(config)
    for key in ("output_dir", "memory_dir"):
        config[key] = str((path.resolve().parent / config[key]).resolve())
    for key in ("model_path", "data_path", "asset_manifest"):
        if config["backend"].get(key):
            config["backend"][key] = str((path.resolve().parent / config["backend"][key]).resolve())
    if config["backend"].get("task_head"):
        head = config["backend"]["task_head"]
        head["path"] = str((path.resolve().parent / head["path"]).resolve())
    return config
