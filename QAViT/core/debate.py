"""三场初始比较与一场父子比较分开选择；不使用 Elo 或跨阶段全局排名。"""
from collections import Counter

from .cards import ContractError


INITIAL_PAIRS = (("H1", "H2"), ("H1", "H3"), ("H2", "H3"))
# 合法配对目录；每轮只从后三种中执行已持久化 H* 对应的一种。
PAIRS = INITIAL_PAIRS + (("H4", "H1"), ("H4", "H2"), ("H4", "H3"))
MAX_EXCHANGES = 5


def selection_pairs(stage, parent_slot=None):
    if stage == "parent":
        return INITIAL_PAIRS
    if stage == "final" and parent_slot in ("H1", "H2", "H3", None):
        return (("H4", parent_slot),) if parent_slot else ()
    raise ContractError("选择阶段或 H* 非法；只能先选 parent，再从父子中选 final")


def match_key(round_number, pair):
    if tuple(pair) not in PAIRS:
        raise ContractError("不是合法的初始或父子配对")
    return f"r{round_number:03d}/match/{pair[0]}-{pair[1]}"


def validate_verdict(payload, exchange):
    if not 1 <= exchange <= MAX_EXCHANGES:
        raise ContractError("辩论内部回合超限")
    if payload.get("outcome") not in ("A", "B", "tie", "undetermined", "continue", "ineligible"):
        raise ContractError("辩论裁决类型非法")
    if payload["outcome"] == "continue" and exchange >= MAX_EXCHANGES:
        raise ContractError("第5回合必须结束或诚实记录无法判定，不能继续第6回合")
    if payload["outcome"] in ("A", "B") and not payload.get("decisive_reason"):
        raise ContractError("胜负必须明确决定性依据")
    if payload['outcome'] == 'undetermined':
        assessment = payload.get('continuation_assessment')
        if (not isinstance(assessment, dict)
                or any(type(assessment.get(k)) is not bool for k in ('experiment_available', 'next_dispute_available'))
                or not isinstance(assessment.get('reason'), str) or not assessment['reason'].strip()):
            raise ContractError('undetermined 需要 continuation_assessment：experiment_available、next_dispute_available 布尔值及非空 reason')
        if assessment['experiment_available'] or (exchange < MAX_EXCHANGES and assessment['next_dispute_available']):
            raise ContractError('仍声明有可执行且有区分力的路径：请先取证，或在未到第5回合时 continue；不能同时提前未决结案')


def ranking_table(matches, eligible):
    stages = (INITIAL_PAIRS, *((pair,) for pair in PAIRS[3:]))
    pairs = next((group for group in stages if set(matches) == {"-".join(p) for p in group}), None)
    if not matches and not eligible:
        pairs = ()  # 初始全部不合格，最终选择为空；不生成虚构 H4 配对。
    if pairs is None:
        raise ContractError("选择只能使用初始三场或唯一父子场；禁止合并为四候选全局排名")
    eligible = set(eligible)
    if not eligible <= {slot for pair in pairs for slot in pair}:
        raise ContractError("eligible 包含本选择阶段之外的候选")
    wins, losses, undecided = Counter({s: 0 for s in eligible}), Counter(), []
    edges, head_to_head = [], {}
    for pair in pairs:
        key = pair[0] + "-" + pair[1]
        row = matches[key]
        outcome = row["outcome"]
        if not set(pair) <= eligible:
            if outcome != "ineligible":
                raise ContractError("不合格假设不得产生比赛胜负")
            continue
        if outcome in ("A", "B"):
            winner, loser = pair if outcome == "A" else pair[::-1]
            wins[winner] += 1
            losses[loser] += 1
            edges.append((winner, loser))
            head_to_head[key] = winner
        elif outcome in ("tie", "undetermined"):
            undecided.append(key)
            head_to_head[key] = outcome
        else:
            raise ContractError("合格比赛尚未完成")
    # 小图直接检测所有环，保留不确定性，禁止伪造全序。
    adjacency = {s: {b for a, b in edges if a == s} for s in eligible}
    cycles = set()
    def walk(path):
        for target in adjacency[path[-1]]:
            if target == path[0] and len(path) >= 3:
                rotations = [tuple(path[i:] + path[:i]) for i in range(len(path))]
                cycles.add(min(rotations))
            elif target not in path:
                walk(path + [target])
    for slot in sorted(eligible):
        walk([slot])
    groups = []
    for count in sorted(set(wins.values()), reverse=True):
        group = sorted(s for s in eligible if wins[s] == count)
        if len(group) == 2:
            edge = next(((a, b) for a, b in edges if {a, b} == set(group)), None)
            if edge:
                groups.extend([[edge[0]], [edge[1]]])
                continue
        groups.append(group)
    return {"wins": dict(wins), "losses": {s: losses[s] for s in sorted(eligible)},
            "groups_by_direct_results": groups, "head_to_head": head_to_head,
            "cycles": [list(c) for c in sorted(cycles)], "undetermined_matches": undecided,
            "requires_provisional_basis": bool(cycles or undecided or any(len(g) > 1 for g in groups)),
            "score_system": "direct_wins_only"}
