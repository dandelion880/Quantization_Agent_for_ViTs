"""v2 分角色订阅调用。复用已验证 CLI 隔离/解码，不继承旧512次限制。

一次调用对应一个持久任务尝试；重试与总预算由 GlobalTaskQueue 管理。
实际 model/effort 未回显时保持未知，不用请求配置冒充服务端回显。
"""
from dataclasses import asdict
from contextlib import contextmanager, ExitStack
import hashlib
from pathlib import Path
import time

from runtime.paths import ROOT  # 初始化已核验本地依赖与 EXP 缓存路径。
from runtime.subscription import SubscriptionTransport
from runtime.configuration import GENERATION_ROUTES, model_defaults
from runtime import codex_cli as cli
from runtime.subscription import output_schema, TRANSPORT, strict_json
from ptq_agent.schemas import LLMRequest, Message, PTQError

from core.cards import ContractError, canonical, fingerprint
from core.journal import atomic_json, file_lock


def client_for(role, settings):
    if settings.get("provider", "codex_subscription") != "codex_subscription":
        raise ContractError("本版仅核验过订阅通道；变更 provider 需要显式适配与验证")
    model, effort = settings.get("agent_models", model_defaults(critic=role == "critic"))[role]
    if role == "generation" and (model, effort) not in GENERATION_ROUTES:
        raise ContractError("G-only Generation 需使用已核验的 Astra / high 路由；禁止静默降级")
    return SubscriptionTransport(model, effort, settings)


def preflight(role, settings, *, timeout=60):
    return client_for(role, settings).preflight(timeout)


@contextmanager
def subscription_slot(role, deadline):
    """Project-wide cap of four model calls, including at most one Coding."""
    root = ROOT / "EXP/cache/codex_lock"
    slots = [root / f"subscription-{i}.lock" for i in range(4)]
    while True:
        if time.time() >= deadline:
            raise cli.fail("timeout", "等待订阅并发槽时到达硬截止")
        for path in slots:
            with ExitStack() as stack:
                try:
                    if role == 'coding':
                        stack.enter_context(file_lock(root / '.lock'))
                    if role == 'judger' and path == slots[-1]:
                        continue
                    stack.enter_context(file_lock(path))
                except ContractError:
                    continue
                yield path.name
                return
        time.sleep(min(.05, max(0, deadline - time.time())))


def transport_input(transport):
    if transport.get('role') not in ('coding', 'safeguard'):
        return canonical(transport)
    ordered = ['role', 'tools', 'response_schema', 'context', 'messages']
    keys = [k for k in ordered if k in transport] + sorted(set(transport) - set(ordered))
    return '{' + ','.join(canonical(k) + ':' + canonical(transport[k]) for k in keys) + '}'


def call_once(spec, directory, deadline):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    request = LLMRequest(role=spec["role"], system_prompt=spec["system_prompt"],
                         messages=[Message(**m) for m in spec.get("messages", [])],
                         tools=spec.get("tools", []), context=spec["context"],
                         response_schema=spec["response_schema"])
    client = client_for(spec["role"], spec["settings"])
    started = time.monotonic()
    audit = {"role": spec["role"], "requested": {"model": client.model, "reasoning_effort": client.effort},
             "actual": {"model": None, "reasoning_effort": None, "source": "not_reported_by_cli"},
             "request": asdict(request), "request_sha256": fingerprint(asdict(request)),
             "model_call_started": False, "usage": None, "error": None,
             "provider": "codex_subscription", "automatic_model_fallback": False}
    working = directory / "cli"
    working.mkdir(exist_ok=True)
    atomic_json(directory / "llm_audit.json", audit)
    def remaining():
        seconds = deadline - time.time()
        if seconds <= 0:
            raise cli.fail("timeout", "已到运行/任务硬截止时间")
        return seconds
    try:
        # 每次调用仍独立预检、账户校验和CLI隔离，等待并发槽计入硬预算。
        with subscription_slot(spec["role"], deadline) as slot:
            audit["subscription_slot"] = slot
            audit["preflight"] = client.preflight(min(60, remaining()))
            cap = spec['settings'].get('subscription_max_used_percent')
            limits = audit['preflight']['account']['rate_limits'] if cap is not None else {}
            if cap is not None and any(limits[k] and limits[k]['usedPercent'] >= cap for k in ('primary', 'secondary')):
                raise cli.fail('quota_reserve', '已到达冻结的订阅额度预留边界；未开始新的模型调用')
            atomic_json(working / "schema.json", output_schema(request))
            native, transport_prompt = None, TRANSPORT
            if spec['role'] == 'coding' and spec['settings'].get('coding_native_tools') is True:
                from runtime.coding_native import prepare
                native, audit['coding_channel'] = prepare(client,spec,directory,deadline)
                if native:
                    transport_prompt = ("在本任务工作目录用原生工具检索、编辑及 CPU 调试 implementation.py。只读绑定源码在 "
                        + audit['coding_channel']['sources'] + "。最终按外层 schema 返回 __agent_decision__，"
                        "arguments_json 内为 action=submit 和 content={source_file:implementation.py} 及必要执行选择。"
                        "科研工具仍以原工具名和 arguments_json 交还宿主；GPU 实验由 worker 执行。")
            (working / "instructions.txt").write_text(request.system_prompt + "\n\n" + transport_prompt, encoding="utf-8")
            audit["model_call_started"] = True
            audit["started_unix"] = time.time()
            atomic_json(directory / "llm_audit.json", audit)
            transport = asdict(request)
            # system_prompt 已作为 CLI instructions 完整传输，stdin 不再重复。
            transport.pop('system_prompt')
            event_path = working / 'events.jsonl' if native or spec['settings'].get('record_live_events') else None
            code, stdout, stderr = cli.run(client.command(working, native=native) if native else client.command(working),
                working, remaining(), transport_input(transport), **({'event_path': event_path} if event_path else {}))
            audit["exit_code"] = code
            audit["stderr"] = {"bytes": len(stderr.encode()), "sha256": hashlib.sha256(stderr.encode()).hexdigest(),
                               "category": cli.failure_category(stderr)}
            if code:
                # Keep a bounded redacted diagnostic, never account RPC bodies or credentials.
                import re
                diagnostic = re.sub(r'''(?i)(bearer\s+|(?:api[_-]?key|access_token|refresh_token|authorization)["']?\s*[:=]\s*["']?)[^"'\s,;}]+''', r'\1[REDACTED]', stderr)
                diagnostic = re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', '[EMAIL]', diagnostic)
                diagnostic = re.sub(r'\bsk-[A-Za-z0-9_-]+', '[KEY]', diagnostic)
                diagnostic = re.sub(r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b', '[TOKEN]', diagnostic)
                path = directory / 'cli_failure.txt'
                path.write_text(diagnostic[:16000], encoding='utf-8')
                path.chmod(0o600)
                audit['stderr']['diagnostic_path'] = str(path)
            reports = []
            for line in stdout.splitlines():
                try:
                    event = strict_json(line)
                    if isinstance(event, dict) and event.get("type") == "turn.completed":
                        reports.append(event.get("usage"))
                except (ValueError, TypeError):
                    pass
            audit["reported_usage"] = reports
            from runtime.coding_native import usage_total
            audit["usage"] = usage_total(reports)
            if code:
                raise cli.fail(cli.failure_category(stderr), "订阅 CLI 失败；未接受科学动作")
            if audit["stderr"]["category"] in ("authentication", "quota_exhausted"):
                raise cli.fail(audit["stderr"]["category"], "账户/额度错误；停止且不切换后端")
            if any(x in stderr.lower() for x in ("falling back", "fallback model", "switching model", "downgrad")):
                raise cli.fail("model_mismatch", "CLI 报告模型降级，拒绝结果")
            answer = working / "answer.json"
            if not answer.exists() or answer.stat().st_size > 16 * 1024 ** 2:
                raise cli.fail("format_error", "缺少完整最终响应")
            response, details = client.decode(stdout, answer.read_text(), request, **({"native": True} if native else {}))
            if native and response.decision and response.decision.get("action") == "submit":
                from runtime.coding_native import delivered
                content = response.decision["content"]
                if content.pop("source_file", None) != "implementation.py" or "source" in content or "revision" in content:
                    raise ContractError("native submission must deliver implementation.py")
                content["source"] = delivered(native[0])
            audit.update(response=details, actual=details["actual"], usage=response.usage)
            return {"response": asdict(response), "audit": str(directory / "llm_audit.json"),
                    "usage": response.usage, "actual": details["actual"],
                    "requested": audit["requested"], "source": client.source}
    except BaseException as exc:
        audit["error"] = exc.to_dict() if isinstance(exc, PTQError) else {"code": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        events = working / 'events.jsonl'
        if events.is_file() and audit.get('error') and audit.get('usage') is None:
            from runtime.coding_native import interrupted_usage
            audit['usage_status'] = interrupted_usage(events)
            audit['usage'] = audit['usage_status']['completed_turn_usage']
        audit["elapsed_seconds"] = time.monotonic() - started
        audit["finished_unix"] = time.time()
        atomic_json(directory / "llm_audit.json", audit)
