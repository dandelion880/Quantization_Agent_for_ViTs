# 当前 QAViT 信息流与执行流程

```text
Supervisor(program): restore Journal, frozen inputs, versions and original budget
for each permitted round:
  Generation(H1/H2/H3):
    repeat read information / request experiment / receive evidence or error
    interpret scientifically; optionally save research_note; submit hypothesis
  screen all three hypotheses
  compare H1-H2, then H1-H3, then H2-H3; persist each closed verdict
  select and persist H*
  Evolution(H*): same autonomous research loop; submit H4
  screen H4; compare only H4-H*; select final candidate
  if final candidate and L3 enabled:
    Judger.plan_l3 sees final candidate, rationale, explicit support evidence and protocol
    execute the submitted plan through the shared experiment service
    if scientific clarification, exhausted implementation repair or failed request:
      preserve original plan/request/result
      create plan_l3/revisionN owned by original Judger, with concrete feedback and sources
      either submit revised request or explain defer_reason; ask user only for user-only decisions
    valid numerical evidence returns immediately; no mandatory interpret task
  Recorder receives newly unsynthesized science and relevant old basis
    submit scientific_analysis, optional question changes and knowledge updates
    archive facts independently; isolate bad optional knowledge items
    if Recorder fails: preserve pending batch and continue under existing retry/defer policy
  Journal persists round completion; next round reads L1 and relevant prior feedback

Shared experiment service:
  preserve original scientific intent and precise sources
  normalize only uniquely determined built-in representation fields
  validate full numeric contract, resource and source constraints
  if new capability: single Coding -> host binds conditions -> Safeguard
  execute real worker; classify recoverable engineering failures with finite retries
  v2/v3 built-in plans -> RTNExecutor adapter -> proper numerical implementation
  capture real shapes during ordinary preparation, then fully validate
  audit coverage/bitwidth/baseline; export; independently reload; compare actual outputs
  validate actual data protocol and execution identity; save immutable Evidence
  reuse valid visible cached measurements with same execution identity; never count as repeat
  return to waiting scientific task, or next immutable version of already-submitted plan

Memory handoff:
  Journal -> L0 immutable facts, L1 actual progress/questions, L2 scoped knowledge, L3 reviews
  current H/Review/Request's explicit support -> bounded Evidence observations + own conditions
  only current snapshot/role-visible evidence; deduplicate; do not recursively expand history
  large statistics -> descriptive projection + coverage + exact read_card selectors
  methods -> method_catalog -> read_asset(asset="methods/<id>.md") without experiment task
  retrieve_memory(empty or query) -> useful short body/results/scope/status/limits/source
  comparison knowledge -> each Evidence retains its own scope; pending -> revision/publish/abandon

Report independently:
  workflow_completed
  numerical_validation_status / formal_validation_success (only protocol-valid evidence)
  interpretation_state (only historical separate tasks, if present)
  scientific_review_complete (Recorder synthesis may still be pending)
```

不新增摘要 Agent、异步 Recorder、跨 run 记忆继承或科研提交门禁。脚本验证不能代替真实 LLM 两轮科研结论。
