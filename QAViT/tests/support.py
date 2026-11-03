"""显式测试配置与协议夹具；这些科研陈述只用于控制流测试。"""
from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.paths import ROOT
from core.journal import read_json


def validation_config():
    config = read_json(ROOT / "QAViT/configs/research.json")
    config.update(name="qavit_v2_tiny", purpose="validation", enable_l3=False)
    config["budget"] = {"max_rounds": 2, "wall_seconds": 7200, "min_free_bytes": 1024 ** 3, "max_artifact_bytes": 2 * 1024 ** 3}
    config["runtime"]["allowed_gpus"] = []
    config["research_brief"] = {
        "scientific_question": "在随机小型ViT中，低比特激活输入MSE能否预测局部输出与整网输出偏差？请提出已有方法、工程改进和新机制三个可证伪假设，通过真实局部实验审查，不能外推为ImageNet/ADE精度。",
        "goals": {"validation": "两轮小规模真实Agent职责验证；主动取证、H4改进、Memory反馈。至少针对真实争议取得一次L1/L2测量，必要时请求Coding扩展量化器。"},
        "model": {"name": "seeded tiny ViT", "scope": {"model": "tiny_vit"}, "adapter": {
            "kind": "fixture", "factory": "QAViT/tests/fixtures.py:tiny_backend",
            "config_path": "QAViT/configs/deitb_backend.json", "backend_overrides": {
                "device": "cpu", "calibration_samples": 2, "search_samples": 4, "final_samples": 2,
                "data_seeds": {"calibration": 17, "search": 18, "final": 19}}}},
        "hard_constraints": {"training": False, "weight_bits": 4, "activation_bits": 4},
        "search_space": {"methods": "选择已有工具或明确定义新量化器；本次静态W4A4 training-free仅为验证输入。",
                         "sites": ["blocks.0.mlp.fc2", "blocks.0.attn.proj"],
                         "calibration_samples": [1, 2], "granularity": ["per_tensor", "per_channel"]},
        "evaluation_protocol": {"kind": "synthetic_feature_fidelity_only", "classification_accuracy": False,
                                 "segmentation": False, "l3_enabled": False},
        "experiment_example": experiment_payload(),
        "source_documents": [str(ROOT / "QAViT/docs/METHODS.md"), str(ROOT / "QAViT/docs/EXECUTION.md")]
    }
    return config


def experiment_payload(level="L1"):
    return {"level": level, "question": "局部输入量化误差与下游算子输出误差是否一致变化？",
            "prediction": "均匀量化会引入非零误差；输入误差与输出误差可能排序不同，需要测量而非预设。",
            "measurements": ["input_relative_mse", "operator_output_relative_mse"],
            "controls": {"baseline": "same original FP32 model", "intervention": "one activation site only"},
            "decision_impact": "确定输入MSE是否足以支持当前排序，识别是否需要输出敏感度机制。",
            "execution": {"service": "tensor_probe", "requires_code": False, "seed": 42,
                "site": "blocks.0.mlp.fc2", "method": "rtn",
                "parameters": {"bits": 4, "signed": True, "granularity": "per_tensor", "axis": None, "group_size": None},
                "calibration": {"split": "calibration", "count": 2}, "evaluation": {"split": "search", "count": 4},
                "resolved_choices": {"training": False, "weight_bits": 4, "activation_bits": 4,
                                     "granularity": {"activation": "per_tensor"}, "calibration_samples": 2,
                                     "activation_calibration": "static", "quantization_scope": {"sites": ["blocks.0.mlp.fc2"]}}},
            "resources": {"estimated_seconds": 120, "cpu_threads": 4},
            "replication": {"requested": False, "purpose": "", "configuration_difference": {}},
            "scope": {"model": "tiny_vit", "activation_bits": 4, "site": "blocks.0.mlp.fc2", "method": "rtn"},
            "validation": {"baseline": "FP32 original", "scope": "controlled local QDQ, not full-model accuracy"}}


def hypothesis(number, slot):
    return {"round": number, "slot": slot, "hypothesis_type": {"H1": "existing_method", "H2": "engineering", "H3": "new_algorithm", "H4": "evolved"}[slot],
            "title": "CONTROL FLOW TEST " + slot, "research_question": "fixture only", "mechanism": "scripted fixture, not real reasoning",
            "intervention": {"fixture": slot}, "predictions": [{"id": "p1", "statement": "test", "measurement": "test", "falsification": "test"}],
            "implementation": "fixture", "validation_design": "fixture", "sources": [{"kind": "test_fixture", "document": "scripted control flow only; no scientific citation claim"}], "mechanism_relations": [],
            "feedback_used": [], "scope": {"model": "tiny_vit"}}


def review(phase, **fields):
    if phase == 'verdict' and fields.get('outcome') == 'undetermined':
        fields.setdefault('continuation_assessment', {'experiment_available': False, 'next_dispute_available': False,
            'reason': 'Scripted CPU control-flow fixture has no scientific evidence path.'})
    return {"phase": phase, "rationale": "scripted infrastructure fixture, not scientific evidence",
            "evidence_refs": [], "uncertainties": ["not real LLM reasoning"], **fields}


def meta(number, judgment_refs=()):
    return {"round": number, "scientific_analysis": "control-flow fixture; no empirical conclusion",
            "judgment_audit": [{"judgment_ref": ref, "assessment": "untested", "evidence_refs": [],
                                "reasoning": "control-flow fixture; no empirical conclusion"} for ref in judgment_refs],
            "predictive_metrics": [], "knowledge_updates": [], "recurring_failures": [],
            "blind_spots": ["not a real agent test"], "next_questions": ["verify real numeric evidence"],
            "generation_guidance": "use previous feedback explicitly", "evidence_refs": []}


if __name__ == "__main__":
    import argparse
    import time
    from core.journal import atomic_json
    parser = argparse.ArgumentParser(description="生成显式小规模验证输入；不自动启动模型或实验")
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-budget", help="已有验证预算JSON，继承其绝对deadline；不重置计时")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if ROOT / "EXP" not in output.parents or output.exists():
        parser.error("输出必须是EXP中的新文件，不覆盖已有配置")
    config = validation_config()
    if args.parent_budget:
        budget = read_json(args.parent_budget)
        if budget["deadline_unix"] <= time.time():
            parser.error("原验证预算已耗尽；需用户明确延长，不能重置起点")
        config["budget"]["parent_deadline_unix"] = budget["deadline_unix"]
    atomic_json(output, config)
    print(str(output))
