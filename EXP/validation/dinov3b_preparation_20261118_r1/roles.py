"""Two normal QAViT component tasks across a restart. Never advance a formal round."""
import os
from pathlib import Path
import sys
import time
LIVE = Path(__file__).resolve().parent.parents[2]

ROOT = LIVE
sys.path.insert(0, str(LIVE / 'QAViT'))
from core.workflow import Workflow
from core.journal import read_json, atomic_json
from runtime.task_queue import process_identity

SCOPE = ('本次是正式运行前的真实科研片段，保留原冻结协议。自选能改变研究判断的小实验，'
         '至少有少量为该问题专门定义并经 Coding、Safeguard 执行的数值逻辑。具体方向、假说、算子、'
         '分组、对照、目标、实验数和判据均由 Generation 决定，可继续修订。可以用便宜的局部范围。'
         '按真实结果更新候选或解释其局限，留存值得接续的具体问题及确切实验、代码、配置。')
def finish(wf, tid):
    while True:
        wf.budget.check(allow_numeric_drain=wf.numeric_running())
        wf.poll()
        wf.update_numeric_wait()
        t = wf.queue.get(tid)
        if t['state'] == 'completed':
            return t
        if t['state'] in ('failed', 'cancelled', 'needs_clarification') or wf.transport_blocked:
            raise RuntimeError(str(t.get('error') or t['state']) + ' ' + str(wf.transport_blocked))
        wf.dispatch()
        wf.update_numeric_wait()
        time.sleep(.6)

def run():
    path = Path(__file__).resolve().parent / sys.argv[1]
    cfg = read_json(LIVE/'EXP/inputs/dinov3b_static_w4a4/input.json')
    cfg.update(name='dinov3b_local_continuation', purpose='validation', enable_l3=False)
    cfg['budget'].update(max_rounds=2, wall_seconds=10800)
    cfg['runtime'].update(allowed_gpus=[2,3], cpu_threads=4)
    cfg['llm']['cli_path'] = str(Path('/root/.local/bin/codex').resolve())
    cfg['research_brief']['execution_guidance']['component_scope'] = SCOPE
    for index in (1, 2):
        if (path / f'component-{index}.json').is_file():
            continue
        wf = (Workflow(path, intervention=os.environ.get('QAVIT_COMPONENT_INTERVENTION'))
              if path.exists() else Workflow(path, cfg))
        wf.budget.begin_segment(owner=process_identity(os.getpid()))
        if os.environ.get('QAVIT_COMPONENT_RETRY'):
            wf.retry_failed(os.environ['QAVIT_COMPONENT_RETRY'])
        try:
            round_context = {'round':index}
            focus = finish(wf, wf.enqueue_role(f'component/{index}/focus', 'focus', refs=[wf.brief['id']], context=round_context))
            topic = wf.topics.active()
            scope = SCOPE if index == 1 else ('恢复前一片段的实际主题、灵感、未完动作和确切代码/配置。'
                    '围绕记录的同一灵感完成一个真实后续动作，可直接复用或修订已有实现，'
                    '由 Generation 选择下一项最值得做的区分、修订或验证，再根据反馈决定。')
            active = wf.inspirations.active()
            task = finish(wf, wf.enqueue_role(f'component/{index}/generate', 'generate', refs=[wf.brief['id']],
                context={**round_context, 'slot':'H3','hypothesis_type':'new_algorithm','topic_ref':topic['ref'],'component_scope':scope}))
            record = finish(wf, wf.enqueue_recorder(index, incomplete=True, key=f'component/{index}/record'))
            atomic_json(path / f'component-{index}.json', {'focus':focus['outputs'], 'topic':topic['ref'],
                       'restored_idea':active and active['id'], 'candidate':task['outputs'], 'recorder':record['outputs'],
                       'evidence':[c['id'] for c in wf.journal.find_cards('EvidenceCard')], 'budget':wf.budget.input_receipt()})
        finally:
            wf.runtime.stop()
            wf.budget.end_segment(processes_stopped=True, reason='local component boundary')
if __name__ == '__main__':
    run()
