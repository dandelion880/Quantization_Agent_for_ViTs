"""Bounded live role breakpoint on the manual catalog and current numerical receipts."""

from pathlib import Path
import sys, time, json
from runtime.paths import ROOT
from agents.roles import role_spec
from agents.source_workspace import asset_text_page
from core.cards import fingerprint
from core.journal import atomic_json, read_json
from runtime.model_router import call_once
from ptq_agent.execution.method_library import MethodLibrary

OUT = Path(__file__).resolve().parent
role = sys.argv[1]
settings = {"provider": "codex_subscription"}
library = MethodLibrary()
context = {
    "task": {
        "round": 1,
        "scientific_question": "在同一量化前缀上检验输出补偿，是可泛化的误差修正还是拟合位置均值。",
    },
    "research_brief": {
        "protocol": "ptq_extension/1",
        "extension_dimensions": ["additional_computation"],
        "goal": "有界实现验证，固定父包、输入顺序和保留集，只作机制判断，不作 L3。",
    },
    "method_catalog": [
        {
            "id": r["id"],
            "state": r.get("implementation_status"),
            "read_asset": {"asset": "methods/" + r["document"]},
        }
        for r in library.entries.values()
        if r["id"] in ("qwt", "qwt-v2", "biascomp")
    ],
    "memory": {"cards": []},
    "tool_results": [],
}
if role == "generation":
    purpose = "focus"
    context["research_brief"]["goal"] += (
        " 实际阅读至少一项方法原文，确定本次小实验的参考、轴和可区分预测。该输入是单轮 Focus 断点。"
    )
else:
    purpose = "record"
    from research_record import bind

    journal, evidence_ref = bind()
    context["task"]["observations"] = journal.card(evidence_ref)
    context["task"]["recording_batch"] = {"scientific_refs": [evidence_ref], "round": 1}
    context["task"]["prior_focus"] = read_json(OUT / "generation.json")
    context["task"]["goal"] = "据实际测量解释参考与轴的适用条件、反例及下一步。"
trace = []
for i in range(3):
    spec = role_spec(role, purpose, context, settings)
    try:
        response = call_once(
            spec, OUT / "role_calls" / role / str(i), time.time() + 165
        )["response"]
    except Exception as exc:
        trace.append({"type": "transport_error", "message": str(exc)})
        break
    decision = response.get("decision")
    if decision:
        trace.append({"decision": decision, "usage": response.get("usage")})
        break
    for call in response.get("tool_calls", []):
        args = call["arguments"]
        asset = args.get("asset", "")
        selected = next(
            (
                r
                for r in library.entries.values()
                if "methods/" + r["document"] == asset
            ),
            None,
        )
        if (
            role == "recorder"
            and call["name"] == "read_card"
            and args.get("card_id", args.get("ref")) in journal.cards
        ):
            result = journal.card(args.get("card_id", args.get("ref")))
            receipt = {"read_card": result["id"]}
        elif call["name"] == "read_asset" and selected:
            body = library.read(selected["id"])
            result = asset_text_page(body["content"], args, 12000)
            receipt = {
                "asset": asset,
                "sha256": body["sha256"],
                "characters": len(result["text"]),
            }
        else:
            result = {"status": "unavailable_in_bounded_breakpoint"}
            receipt = result
        context["tool_results"].append(
            {"name": call["name"], "arguments": args, "result": result}
        )
        trace.append(receipt)
atomic_json(OUT / (role + ".json"), trace)
print(json.dumps(trace, ensure_ascii=False)[:600])
