# G-only 当前伪代码：独立灵感与跨轮接续

```text
initialize_or_resume(run):
    verify frozen Brief, source identity, assets and original budget
    replay Journal -> queue, Memory, complete SolutionLibrary, InspirationLibrary
    # Idea library contains short notes and references, no duplicated full solution.

for round in original_budget:
    retain independent verified best B
    for slot in [H1, H2, H3]:
        Generation receives Brief, feedback, complete parents, ideas and current evidence
        H1/H2: improve own complete parent; record changes and rollback
        H3: default to one active idea; restore exact source/config/results/unfinished work
        Generation may update_inspiration without requesting any experiment
        Generation decides continue / pause / resume / switch using progress and next step
        switching pauses prior idea without dropping its code or results
        while a justified experiment fits existing budget:
            bind request to idea (H3 automatic; synthesis explicit inspiration_id)
            Coding receives exact prior implementations and returns one incremental implementation
            host preserves code hash + original request + source identity + parent code context
            Safeguard -> original execution -> immutable Evidence or failure
            return result to Generation; no fabricated success or metric guarantee
        submit final H; H3 retains idea identity, H1/H2 retain complete parent identity
    Generation synthesizes and may revise exact candidate versions
    for each final candidate in [H1, H2, H3]:
        plan and execute original full dual-task L3 with exact candidate binding
        valid outcomes update formal Top-3 by ADE20K primary metric and existing rules
        preserve independent B; partial/failure does not block siblings or count as full success
        all outcomes remain linked to idea/request even without promotion
    Recorder updates scientific Memory and idea progress/references/absorbed parts
    # Recorder does not activate/pause/switch; lower scores do not auto-terminate an idea.
    commit round; next H3 resumes same active ID and exact assets

on interruption:
    replay committed facts and original tool request; do not duplicate update or experiment
    frozen read sequence and hashes prevent future-state leaks and silent code replacement
```
