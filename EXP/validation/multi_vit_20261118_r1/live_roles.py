"""Bounded component run using the existing role queue, workers, feedback and recorder."""
import os
from pathlib import Path
import argparse
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'QAViT'))
from runtime.paths import ROOT
from core.workflow import Workflow
from runtime.task_queue import process_identity
from core.journal import atomic_json, read_json


def finish(wf, tid):
    while True:
        wf.budget.check(allow_numeric_drain=wf.numeric_running())
        wf.poll()
        wf.update_numeric_wait()
        task = wf.queue.get(tid)
        if task['state'] == 'completed':
            return task
        if task['state'] in ('failed','cancelled','needs_clarification') or wf.transport_blocked:
            raise RuntimeError(str(task.get('error') or task['state']) + ' ' + str(wf.transport_blocked))
        wf.dispatch()
        wf.update_numeric_wait()
        time.sleep(.4)


def run(name, resume=False):
    cfg = read_json(ROOT/'EXP/inputs/dinov3b_static_w4a4/input.json')
    cfg.update(name=name,purpose='validation',enable_l3=False)
    cfg['budget'].update(max_rounds=1,wall_seconds=1100,llm_call_limit=22)
    cfg['runtime'].update(allowed_gpus=[2],cpu_threads=3,llm_concurrency=1,experiment_concurrency=1)
    cfg['llm'].update(cli_path=str((Path.home()/'.local/bin/codex').resolve()),max_transport_retries=1)
    brief = cfg['research_brief']
    brief['goals'] = {'validation':'真实研究角色小样本接入验收，形成一个局部可执行改动，用实测决定如何接续。'}
    brief['evaluation_protocol']={'kind':'paired_classification_subset','final_training':False,
        'formal_incumbent_allowed':False,'evaluations':[{'kind':'classification','split':'search','count':2}]}
    instruction = ('本次限定为一个局部科研组件。focus 只决定一个合适的共同问题。generate 自选研究机制和确切位点，'
        '通过 request_experiment 申请一次有真实导出/重载的 L2 局部候选。为验收单份 Coding 与 Safeguard，'
        '请用 quantize_evaluate、requires_code=true，经 executor_v1.build_plan 把你提出的改动落实为原生 v3 局部计划。'
        '此次只验收原生 build_plan 的一个已实现选择：从实际目录自选一个算子，用一个确定的 W4/A4 '
        '位点配置和已实现的 weight/activation_method、granularity 或 calibration 选项做一次测量，对照同图 FP。'
        '不新增拟合目标、码器、make_executor 或自定义统计。量化表示、位点、选择理由与后续判断由你决定。'
        'Coding 可直接复用 ptq_agent.execution.baselines.build_baseline(backend,name) 构造原生 schema v3，'
        '从返回的字典按精确 module 过滤 layers，移除该局部计划不用的 preprocess 和 operators，再施加所选变更。'
        '可从绑定快照 read_asset 的 snapshot_field=source_snapshot、member=tools/ptq_agent/execution/baselines.py '
        '取得该函数完整原件。库已提供评估、校准、导出和重载。只用1张 calibration 和2张 classification/search；其他主体与任务头暂留 FP。'
        '不做完整算法校准或 L3。可使用 context.model_metadata 及原件，自选干预、指标解释与后续决定。'
        '从返回的 Evidence 与实际实现形成候选，注明已实施位点、量化选择、重建来源，以及尚未测试的外推部分。')
    brief['execution_guidance']={'seed':42,'acceptance_scope':instruction}
    path=HERE/name
    wf=Workflow(path) if resume else Workflow(path,cfg)
    if resume:
        old = wf.queue.get_by_key('component/record')
        if old and old['state'] == 'running' and 'selection' not in old['inputs']['context']:
            wf.queue.fail(old['task_id'], 'component harness omitted recorder context; resubmitted via enqueue_recorder', retryable=False)
    wf.budget.begin_segment(owner=process_identity(os.getpid()))
    try:
        focus=finish(wf,wf.enqueue_role('component/focus','focus',refs=[wf.brief['id']],context={'round':1}))
        topic=wf.topics.active()
        task=finish(wf,wf.enqueue_role('component/generate','generate',refs=[wf.brief['id']],context={
            'round':1,'slot':'H3','hypothesis_type':'new_algorithm','topic_ref':topic['ref'],'component_scope':instruction}))
        record=finish(wf,wf.enqueue_recorder(1,incomplete=True))
        result={'focus':focus['outputs'],'candidate':task['outputs'],'recorder':record['outputs'],
                'evidence':[c['id'] for c in wf.journal.find_cards('EvidenceCard')],
                'B':wf.solutions.fallback(), 'budget':wf.budget.input_receipt()}
        atomic_json(path/'component_result.json',result)
    finally:
        wf.runtime.stop()
        wf.budget.end_segment(processes_stopped=True, reason="component finished or stopped")


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    run(args.name,args.resume)
