"""V2 Profiling 服务目录、工具会话与已验证候选绑定。"""
from pathlib import Path
from core.cards import ContractError, fingerprint, validate_card
from core.journal import atomic_json, read_json, file_sha256
from tools.experiments import asset, verify_assets
from ptq_agent.profiling.host import ProfileHost

PROFILE_NAMES = ('query_structure', 'query_execution_capabilities', 'query_ptq_plugins', 'query_ptq_sites', 'profile_structure', 'profile_distribution', 'profile_hardware', 'profile_sensitivity', 'profile_plugin_sites', 'profile_ptq_candidates', 'profile_trial_diagnostics', 'profile_trial_ablation', 'read_profile_evidence', 'query_method_library', 'read_method_document')


def profile_host(backend, cfg, directory):
    """建立完整纯工具会话；计数只计量，时间行为保留。"""
    return ProfileHost(backend, cfg, directory)


def bind_profile_evidence(journal, reference):
    """从本次运行已提交的成功测量授权读取；不扫描孤立作业或相信调用方绑定。"""
    path = Path(reference)
    if path.is_absolute() or len(path.parts) != 2 or path.parts[0] != "evidence" or ".." in path.parts:
        raise ContractError("Profiling 证据引用必须是已授权的 evidence/ 文件")
    for card in journal.find_cards("EvidenceCard"):
        facts = card["payload"]
        if (card["run_id"] != journal.run_id or facts["status"] != "completed"
                or facts["measurement_source"] == "test_double"):
            continue
        source = journal.card(facts["experiment_ref"])
        if (source["payload"]["execution"]["service"] != "profile"
                or facts["observations"].get("result", {}).get("evidence") != reference):
            continue
        assets = {Path(a["path"]).resolve(): a for a in facts["artifacts"]}
        for result_path, result_asset in assets.items():
            if result_path.name != "profile_result.json" or not result_path.is_relative_to(journal.root / "jobs"):
                continue
            measurement_path = (result_path.parent / "profile" / path).resolve()
            if measurement_path not in assets or not measurement_path.is_relative_to(result_path.parent / "profile/evidence"):
                continue
            binding = {"reference": reference, "evidence_ref": card["id"],
                       "evidence_asset": asset(journal.card_path(card["id"])),
                       "result_asset": result_asset, "measurement_asset": assets[measurement_path]}
            verify_assets(binding)
            result = read_json(result_path)
            if (result["status"] == "success" and reference in result["artifacts"]
                    and result["data"] == facts["observations"]["result"]):
                return [binding]
    raise ContractError("只能读取本次运行已提交并绑定原始资产的成功 Profiling 测量")


def restore_profile_evidence(host, bindings, run_id):
    """把冻结授权的测量复制入新作业，并恢复成功返回账目；保留底层权限检查。"""
    from ptq_agent.profiling.evidence import read_measurement
    for binding in bindings:
        verify_assets(binding)
        source = read_json(binding["evidence_asset"]["path"])
        validate_card(source)
        result = read_json(binding["result_asset"]["path"])
        reference = binding["reference"]
        path = Path(reference)
        if (source["kind"] != "EvidenceCard" or source["run_id"] != run_id or source["id"] != binding["evidence_ref"]
                or source["payload"]["status"] != "completed" or source["payload"]["measurement_source"] == "test_double"
                or any(binding[key] not in source["payload"]["artifacts"] for key in ("result_asset", "measurement_asset"))
                or result["status"] != "success" or reference not in result["artifacts"]
                or result["data"] != source["payload"]["observations"]["result"]
                or path.is_absolute() or len(path.parts) != 2 or path.parts[0] != "evidence" or ".." in path.parts):
            raise ContractError("Profiling 恢复的证据归属或成功登记不匹配")
        measurement = read_measurement(Path(binding["measurement_asset"]["path"]), host.state["identity"])
        atomic_json(host.run_dir / path, measurement)
        host.state["tool_results"].append({"restored_from": source["id"], "result": result})
    host.save()


def register_trials(host, bindings):
    """把明确引用的 v2 实测候选投影为旧诊断接口的只读索引；不导入旧 Memory。"""
    from ptq_agent.profiling.evidence import seal
    for binding in bindings:
        verify_assets(binding)
        model = Path(binding["model_dir"])
        manifest, evaluation = read_json(model / "manifest.json"), read_json(binding["evaluation_asset"]["path"])
        if manifest["identity"] != host.state["identity"] or manifest["weights_sha256"] != file_sha256(model / "weights.pt"):
            raise ContractError("候选模型/数据/源码身份与当前诊断基准不一致")
        if evaluation.get("status") != "completed" or evaluation.get("split") != "search":
            raise ContractError("不完整或非search候选不能进入旧诊断接口")
        tid = binding["trial_id"]
        directory = host.run_dir.parent / "trials" / tid
        directory.mkdir(parents=True, exist_ok=True)
        link = directory / "artifact"
        if not link.exists():
            link.symlink_to(model, target_is_directory=True)
        elif link.resolve() != model.resolve():
            raise ContractError("候选只读索引发生冲突")
        trial = seal({"trial_id": tid, "status": "completed", "candidate": manifest["plan"], "artifact": manifest,
                      "evaluation": evaluation, "metrics": evaluation["metrics"], "error": None,
                      "source_evidence_ref": binding["evidence_ref"], "source_evidence_asset": binding["evidence_asset"],
                      "scope": evaluation["scope"]})
        atomic_json(directory / "result.json", trial)
        feedback = {"id": "execution-" + tid, "trial_id": tid, "split": "search", "metrics": [], "difficulties": [],
                    "summary": "本次v2运行已完成测量，实际样本数与subset标记保留；不是正式完整评测。",
                    "evidence": [str((directory / "result.json").resolve())]}
        reference = "feedback/" + fingerprint(feedback) + ".json"
        host.state.setdefault("execution_feedback", []).append({"source": "program_verified", "trial": tid,
            "evidence": [reference], "configuration": manifest["plan"], "input_shapes": manifest["input_shapes"],
            "metrics": evaluation["metrics"], "gate": None, "error": None})
        if host.feedback_source(feedback) != "program_verified":
            raise ContractError("v2 候选桥接未通过旧诊断来源检查")
        record = {"source": "program_verified", "data": feedback, "reference": reference}
        atomic_json(host.run_dir / reference, record)
        host.state["profile_feedback"].append(record)
    host.save()

