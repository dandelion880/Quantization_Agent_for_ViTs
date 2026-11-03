"""V2 纯数值工具会话；没有 Agent、LLM、Memory、选优策略或研究循环。

每个实验尝试独立建会话；任务级中断恢复在 runtime/task_runtime.py。
这里保存测量账目、证据和精确缓存。计数只记录，不设旧 measurement 上限；
活动时间检查保留原行为，外层任务同时受全局硬截止时间控制。
"""
from copy import deepcopy
from pathlib import Path
import time
from .adapters import ProfileAdapter
from ..registry import ToolRegistry
from ..schemas import require, fingerprint, PTQError
from ..storage import atomic_json, read_json
from ..execution.capabilities import detect, validate_plan, check_resources, plan_key


class ProfileHost:
    def __init__(self, backend, config, directory):
        self.config = deepcopy(config)
        self.run_dir = Path(directory).resolve()
        require(not (self.run_dir / "state.json").exists(), "运行目录已有状态，请使用新的任务尝试", code="run_exists")
        self.backend = backend
        self.profile_adapter = ProfileAdapter()
        self._last_tick = time.perf_counter()
        self.state = {
            "config": deepcopy(config), "identity": backend.identity(),
            "source": {"data": backend.identity()["source"], "llm": "v2_tools_only", "backend": config["backend"]["kind"]},
            "profile_session": True, "profile_adapter": self.profile_adapter.version,
            "profile_input": {}, "profile_feedback": [], "analysis_epoch": 0,
            "history": [], "tool_results": [], "tool_calls": [], "failures": [],
            "consecutive_failures": 0, "phase": "profiling", "stop_reason": None,
            "budget": {"trials": 0, "tools": 0, "agents": 0, "measurements": 0, "elapsed_seconds": 0.0},
            "v2_adapter": {"legacy_selection_policy_used": False, "measurement_count_limit": None,
                           "llm_calls_from_numerical_host": False},
        }
        self.registry = ToolRegistry(self)
        from ..execution.profile_tools import register_execution_tools
        register_execution_tools(self)
        self.save()

    def _tick(self):
        now = time.perf_counter()
        self.state["budget"]["elapsed_seconds"] += max(0, now - self._last_tick)
        self._last_tick = now

    def save(self):
        self._tick()
        atomic_json(self.run_dir / "state.json", self.state)

    def check_time(self):
        self._tick()
        if self.state["budget"]["elapsed_seconds"] >= self.config["budget"]["seconds"]:
            raise PTQError("budget_exhausted", "累计活动运行时间预算耗尽", path="$.budget.seconds")

    def consume(self, kind, count=1):
        self.check_time()
        self.state["budget"][kind] += count
        self.save()

    def _failure(self, error, source):
        self.state["failures"].append({"location": source, "error": error,
            "epoch": self.state["analysis_epoch"], "sequence": self.state["budget"]["agents"]})
        self.state["consecutive_failures"] += 1
        if error.get("code") == "disk_budget":
            self.state.update(phase="finished", stop_reason={"code": "disk_budget", "detail": error["message"]})
            self.save()

    def profile_module_permissions(self):
        """结构工具使用同一执行许可；显式保护/排除及scope仍优先。"""
        from ..execution.sites import site_catalog
        sites = site_catalog(self.backend, self.config)
        constraints = self.config["constraints"]
        protected = constraints["protected_modules"] + constraints["excluded_modules"]
        scope = self.config["loop"].get("scope_modules", [])
        return {row["name"]: bool(
                    sites.get(row["name"], row)["protected"]
                    or any(row["name"] == p or row["name"].startswith(p + ".") for p in protected)
                    or (scope and row["name"] not in scope))
                for row in self.backend.modules()}


    def _feedback_trial(self, item):
        """只信任本轮外层执行器提交的 search 证据，外部反馈不成为形状缓存。"""
        # diagnostic 可以没有量化 trial；保持 external_unverified，不能拼成路径。
        if item["trial_id"] is None:
            return None
        path = self.run_dir.parent / "trials" / item["trial_id"] / "result.json"
        if (not path.resolve().is_relative_to((self.run_dir.parent / "trials").resolve())
                or item["evidence"] != [str(path.resolve())] or not path.is_file()):
            return None
        trial = read_json(path)
        require(
            trial.get("content_sha256")
            == fingerprint({k: v for k, v in trial.items() if k != "content_sha256"}),
            "执行反馈证据损坏",
            code="invalid_evidence",
        )
        require(
            item["split"] == "search" and trial["trial_id"] == item["trial_id"]
            and any(
                r["trial"] == trial["trial_id"] and r["configuration"] == trial["candidate"]
                for r in self.state.get("execution_feedback", [])
            ),
            "执行反馈与已提交 trial 不同",
            code="invalid_evidence",
        )
        # 仅指向合法trial不够；反馈全文也必须对应外层执行器公布的确切引用。
        reference = "feedback/" + fingerprint(item) + ".json"
        if not any(r["trial"] == trial["trial_id"] and reference in r.get("evidence", [])
                   for r in self.state.get("execution_feedback", [])):
            return None
        return trial


    def feedback_source(self, item):
        return "program_verified" if self._feedback_trial(item) is not None else "external_unverified"


    def shapes(self):
        result = {}
        current_trials = {row["trial"] for row in self.state.get("execution_feedback", [])}
        for row in self.state.get("profile_feedback", []):
            item = row["data"]
            if (row["source"] != "program_verified" or item["split"] != "search"
                    or item["trial_id"] not in current_trials):
                continue
            trial = self._feedback_trial(item)
            if trial is None or trial["status"] != "completed":
                continue
            artifact, evaluation = trial.get("artifact", {}), trial.get("evaluation", {})
            require(
                evaluation.get("status") == "completed" and evaluation.get("split") == "search"
                and artifact.get("identity") == self.state["identity"]
                and artifact.get("content_sha256") == fingerprint(
                    {k: v for k, v in artifact.items() if k != "content_sha256"})
                and artifact.get("plan") == trial["candidate"]
                and artifact.get("configuration_hash") == fingerprint(trial["candidate"]),
                "执行产物形状证据的身份/完整性/配置不匹配", code="invalid_evidence")
            result.update(deepcopy(artifact.get("input_shapes", {})))
        # 新增 profile 仍可覆盖旧形状；不从自然语言或未验证的反馈推测维度。
        for row in self.state["tool_results"]:
            if row["result"]["status"] == "success":
                result.update(row["result"].get("data", {}).get("shapes", {}))
        return result


    def validate_plan_or_draft(self, plan=None, draft=None):
        require((plan is None) != (draft is None), "plan 与 draft 必须且只能提供一个")
        if draft is not None:
            from ..execution.planning import resolve_draft
            plan = resolve_draft(self, draft)
        return self.validate_experiment(plan)


    def validate_experiment(self, plan):
        from ..execution.plugin_contracts import site_id
        shapes = self.shapes()
        rows = validate_plan(self.backend, plan, self.config, detect(self.backend), shapes)
        # 和外层执行器使用同一数值 hash；只改理由/引用也仍是重复实验。
        key = plan_key(plan)
        repeated = [row["trial"] for row in self.state.get("tested_plan_keys", []) if row["key"] == key]
        require(not repeated, f"数值配置重复于已提交候选：{repeated}；请修改数值方案或明确 stop。",
                code="duplicate_candidate")
        # 通道/分组轴需要实测形状，不能把仅通过 schema 的草案称作可执行。
        # per_tensor 不依赖轴，仍允许按需 profile，而不强制全模型统计。
        missing = [site_id(layer) for layer in plan["layers"]
                   if layer["activation"] and layer["activation"]["granularity"] != "per_tensor"
                   and not shapes.get(site_id(layer))]
        require(not missing, f"缺少依赖量化轴的实测激活形状：{missing}。"
                "普通模块用 profile_distribution；语义位点用 profile_plugin_sites，"
                "读取 calibration 样本后再提交。此校验不消耗候选次数。", code="missing_profile")
        resources = check_resources(self.backend, plan, self.config, self.run_dir.parent)
        return {
            "valid": True,
            "configuration_hash": fingerprint(plan),
            "site_checks": rows,
            "resources": resources,
        }

