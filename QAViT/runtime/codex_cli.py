"""官方 Codex CLI 的受限进程与只读账户检查，不接触 PTQ 执行器。

复用现有 ChatGPT 登录；不复制凭据、不写全局配置、不接受 API endpoint/key。
CLI 升级须重新验证隔离开关，不能猜测新版本的默认权限。
"""

import json
import ctypes
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import tempfile
import time

from ptq_agent.schemas import PTQError, require

# 官方 CLI 把 warning 也编码为 item.type=error；仅此精确的隔离通知可接受。
# 来源：core/src/tools/code_mode/mod.rs::take_unavailable_warning。
ISOLATION_NOTICE = ("Code Mode is unavailable because code-mode host is disabled. "
                    "Code mode will fail closed; enable `features.code_mode_host` and install `codex-code-mode-host`.")
# 此版本的 gpt-6-luna 为 code_mode_only。关闭 host 后工具分派 fail closed；
# 同时关闭原生执行、网络/应用、子 agent，read-only 是额外的写保护。
DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "shell_snapshot", "code_mode", "code_mode_host",
    "apps", "plugins", "remote_plugin", "hooks", "workspace_dependencies",
    "skill_mcp_dependency_install", "skill_search", "memories", "multi_agent",
    "multi_agent_v2", "computer_use", "browser_use", "browser_use_external",
    "in_app_browser", "image_generation", "view_image", "sleep_tool", "goals",
    "tool_suggest", "send_message_to_user_async", "default_mode_request_user_input",
    "request_permissions_tool", "unbounded_connection_retries", "daemon_auto_start",
    "fast_mode", "ultrafast_mode", "step_model_switching",
)


def fail(code, message):
    # 后端已管理有限重试；外层 Agent 不能再嵌套重试，或切换其它后端。
    return PTQError("llm_" + code, message, retryable=False)


def environment():
    """保留登录位置和必要网络设置，剥离 API key、父 Codex 会话和 GPU 环境。"""
    names = {"PATH", "HOME", "CODEX_HOME", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
             "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy",
             "https_proxy", "all_proxy", "no_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR"}
    env = {k: v for k, v in os.environ.items() if k in names}
    env.update(CUDA_VISIBLE_DEVICES="", NO_COLOR="1")
    return env


def skill_discovery_notice():
    # 该开关用于隔离宿主 skills。只接受它自身的已核对启动提示，不忽略其它 warning。
    config = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "config.toml"
    return ("Under-development features enabled: skip_host_skill_discovery. "
            "Under-development features are incomplete and may behave unpredictably. "
            "To suppress this warning, set `suppress_unstable_features_warning = true` in " + str(config) + ".")


def executable(config):
    value = config.get("cli_path")
    candidates = ([Path(value)] if value else
                  [Path(shutil.which("codex") or "/nonexistent"),
                   Path.home() / ".codex/packages/app-server-daemon/current/bin/codex"])
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    raise fail("not_configured", "未找到官方 Codex CLI；请设置 llm.codex.cli_path")


def policy(directory):
    values = {
        "model_provider": '"openai"', "forced_login_method": '"chatgpt"',
        "approval_policy": '"never"', "sandbox_mode": '"read-only"',
        "web_search": '"disabled"', "mcp_servers": "{}", "plugins": "{}",
        "skills.bundled.enabled": "false", "skills.include_instructions": "false",
        "project_doc_max_bytes": "0", "project_doc_fallback_filenames": "[]",
        "history.persistence": '"none"', "analytics.enabled": "false",
        "tools.update_plan.enabled": "false",
        "tools.experimental_request_user_input.enabled": "false",
        "include_apps_instructions": "false", "include_collaboration_mode_instructions": "false",
        "model_auto_compact_token_limit": "1000000000",  # 不让 CLI 摘要完整任务；超窗则报错。
        "model_post_turn_compact_threshold_percent": "0",
        "log_dir": json.dumps(str(directory / "cli_logs")),
    }
    args = [part for k, v in values.items() for part in ("-c", f"{k}={v}")]
    for name in DISABLED_FEATURES:
        args += ["--disable", name]
    return args + ["--enable", "skip_host_skill_discovery"]


def terminate(process):
    """取消整个本次 CLI 进程组，避免超时后残留子进程继续生成。"""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)


def parent_guard(parent):
    """Linux 父进程被强制结束时终止 CLI；正常超时仍由进程组清理负责。"""
    def guard():
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            os._exit(126)
        if os.getppid() != parent:  # 处理设置 signal 前父进程退出的竞争窗口。
            os._exit(126)
    return guard


def run(command, directory, timeout, prompt=None):
    """stdout/stderr 落临时文件，不把 CLI 日志或潜在凭据直接输出到用户终端。"""
    if timeout <= 0:
        raise fail("timeout", "Codex 调用已超过截止时间")
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        process = subprocess.Popen(command, cwd=directory, env=environment(),
                                   stdin=subprocess.PIPE, stdout=out, stderr=err,
                                   start_new_session=True, preexec_fn=parent_guard(os.getpid()))
        try:
            process.communicate(input=prompt.encode("utf-8") if prompt is not None else None,
                                timeout=timeout)
        except subprocess.TimeoutExpired:
            raise fail("timeout", "Codex 调用超时；本次进程组已终止") from None
        finally:
            terminate(process)
        require(out.tell() <= 16 * 1024**2 and err.tell() <= 16 * 1024**2,
                "Codex 输出超过 16 MiB；拒绝不完整响应", code="llm_output_limit")
        out.seek(0)
        err.seek(0)
        return process.returncode, out.read().decode("utf-8"), err.read().decode("utf-8", errors="replace")


def failure_category(text):
    """只返回类别；原始 stderr 可能含账户信息，不存入实验日志。"""
    text = text.lower()
    # 不能在时间戳、UUID 或字节数中搜裸数字：例如 .614038Z 并非 HTTP 403。
    status = re.search(r'\b(?:http(?:/\d+(?:\.\d+)?)?(?: error)?\s+|'
                       r'(?:status(?:\s+code)?|status_code|http_status)["\s:=]+)(401|403|429|502|503)\b', text)
    code = status.group(1) if status else None
    if code == "429" or any(s in text for s in ("usage limit", "quota", "insufficient", "rate limit", "too many requests", "credits")):
        return "quota_exhausted"
    if code in ("401", "403") or any(s in text for s in
            ("unauthorized", "authentication failed", "authentication error", "authentication failure",
             "authentication required", "not logged", "token expired", "refresh_token")):
        return "authentication"
    if any(s in text for s in ("context length", "context window", "too many tokens", "context_length")):
        return "context_limit"
    if any(s in text for s in ("not supported", "unsupported", "model_not_found", "does not exist")):
        return "unsupported_model"
    if code in ("502", "503") or any(s in text for s in ("timed out", "timeout", "connection", "stream disconnect", "temporarily unavailable")):
        return "transport_error"
    return "cli_failed"


def check_account(account, limits):
    """只校验元数据，不根据套餐名称猜权限，也不消费购买额度。"""
    require(account.get("type") == "chatgpt", "必须使用 ChatGPT 订阅登录；禁止 API key 登录",
            code="llm_authentication")
    require(any(isinstance(limits.get(k), dict) for k in ("primary", "secondary")),
            "无法核实订阅额度，停止调用", code="llm_quota_unknown")
    for key in ("primary", "secondary"):
        window = limits.get(key)
        if window:
            require(isinstance(window.get("usedPercent"), (int, float)) and window["usedPercent"] < 100,
                    "ChatGPT 订阅额度已用尽；停止，不购买/使用额外额度", code="llm_quota_exhausted")
    credits = limits.get("credits")
    require(isinstance(credits, dict) and credits.get("hasCredits") is False and credits.get("unlimited") is False,
            "无法确认没有额外 credits；CLI 未提供本地禁用其自动消费的已验证开关，停止调用",
            code="llm_credit_policy")
    return {"auth_type": account["type"], "plan_type": account.get("planType"),
            "rate_limits": {k: limits.get(k) for k in ("primary", "secondary", "credits")}}


def account_rpc_error(method, error):
    """网络/服务错误不能冒充登录失效；不把可能含账户信息的原消息写进日志。"""
    category = failure_category(json.dumps(error, ensure_ascii=False))
    if category == "cli_failed":
        category = "preflight_failed"
    code = error.get("code") if isinstance(error, dict) else None
    code = code if type(code) is int else None
    return fail(category, f"官方 CLI {method} 只读检查失败（{category}，RPC code={code}）；未开始新的模型调用")


def account_status(cli, directory, timeout):
    """只发 account RPC，不创建 thread/turn，不调用模型，不记录 email/token。"""
    deadline = time.monotonic() + timeout
    command = [cli, "app-server", "--stdio", *policy(directory)]
    with tempfile.TemporaryFile() as err:
        process = subprocess.Popen(command, cwd=directory, env=environment(), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=err, start_new_session=True,
                                   preexec_fn=parent_guard(os.getpid()))
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        buffer = b""

        def rpc(number, method, params):
            nonlocal buffer
            process.stdin.write((json.dumps({"id": number, "method": method, "params": params}) + "\n").encode())
            process.stdin.flush()
            while time.monotonic() < deadline:
                if b"\n" not in buffer:
                    if not selector.select(max(0, deadline - time.monotonic())):
                        break
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    buffer += chunk
                    require(len(buffer) <= 4 * 1024**2, "账户 RPC 输出异常", code="llm_preflight_failed")
                    continue
                line, buffer = buffer.split(b"\n", 1)
                response = json.loads(line)
                if response.get("id") == number:
                    if "error" in response:
                        raise account_rpc_error(method, response["error"])
                    return response["result"]
            raise fail("preflight_failed", "账户只读检查超时或异常退出；未开始模型调用")

        try:
            rpc(1, "initialize", {"clientInfo": {"name": "ptq_subscription", "version": "1.0"},
                                  "capabilities": {"experimentalApi": True}})
            process.stdin.write(b'{"method":"initialized"}\n')
            process.stdin.flush()
            account = rpc(2, "account/read", {"refreshToken": False}).get("account") or {}
            limits = rpc(3, "account/rateLimits/read", {}).get("rateLimits") or {}
            return check_account(account, limits)
        finally:
            selector.close()
            terminate(process)
            process.stdin.close()
            process.stdout.close()

