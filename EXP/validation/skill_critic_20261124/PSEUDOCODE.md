# QAViT 主线流程

```text
start(Brief, budget, configuration)
    freeze source, Brief, execution bindings and visible Skill release heads
    recover open outbox entries and committed Journal state
    for each round allowed by the run
        Generation.restore_or_choose_topic(Brief, history, evidence)
        launch existing H1, H2, H3 generation tasks under their existing concurrency
        H1/H2 derive reversible increments from complete parent solutions
        H3 may continue an inspiration or explore a new testable mechanism
        join terminal generator tasks; retain exactly the versions submitted
        if enabled and at least one candidate exists
            resume_or_create one batch Critic task for this round
            read bounded candidate facts, selected parent information and measurements
            optional request_experiment(exact H version, draft)
                bind topic, method and inspiration from that H
                use existing implementation, Safeguard, frozen execution and Evidence path
                return measured result or recorded failure to the same task
            commit suggestions with source versions, or record actual failure
        Generation.select(current candidates, measurements, optional review)
            optional revise_candidate(parent, full H, rationale)
            on revision: resume synthesis with new versions and original review
        for each final candidate, independently
            Generation.plan_l3(version, frozen Brief)
            run or defer under the existing protocol; preserve package identity
        update solutions and independent B using eligible numerical evidence
        Recorder.review(unarchived batch)
            commit scientific analysis, knowledge amendments, cursor and optional Skill drafts
            for each well-formed draft: check base, publish immutable package and release
            preserve invalid drafts, conflicts and retryable pending writes
        subsequent tasks see this run's newly published procedures
        frozen reads keep their exact versions; other ongoing runs keep their original heads
```

各科研角色均可选择按需恢复原件。判断研究范围、实验必要性和机制优先级由角色负责。正式评价的资格来自冻结协议与有效测量。详见[实现报告](REPORT.md)。
