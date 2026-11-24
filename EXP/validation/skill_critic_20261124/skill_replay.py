"""Bounded offline fork of a real checkpoint; only role calls and their read/submit tools run here."""
from copy import deepcopy
from pathlib import Path
import shutil, time
from core import skills
from core.journal import atomic_json, read_json, file_sha256
from core.cards import canonical
from core.workflow import Workflow
from runtime.model_router import call_once
from runtime.paths import ROOT

OUT=Path(__file__).resolve().parent
skills.SHARED_ROOT=OUT/'shared-skills'
SOURCE=ROOT/'EXP/evaluations/frozen_trials_20261124_focus_host_r3/branch'


def turn(wf, tid, name, extra, limit=4):
    folder=OUT/name
    trace=read_json(folder/'trace.json') if (folder/'trace.json').is_file() else []
    for i in range(len(trace),limit):
        wf.queue.ready()
        task=wf.queue.get(tid)
        request=wf.request_body(task)
        request['spec']['context']['validation_question']=extra + (' 本断点只剩本次调用，用已有事实给出可执行决定，或申请缺失的关键观测。' if i==4 else '')
        where=folder/str(i+1)
        path=where/'request.json'
        atomic_json(path,request)
        task=wf.queue.claim(tid,request_path=str(path))
        started=time.monotonic()
        response=call_once(request['spec'],where,time.time()+240)['response']
        atomic_json(where/'response.json',response)
        row={'seconds':time.monotonic()-started,'usage':response.get('usage'),
             'read_chars':0,'calls':[(c['name'],c['arguments']) for c in response['tool_calls']],
             'decision':response.get('decision'),'request_sha256':file_sha256(path)}
        trace.append(row);atomic_json(folder/'trace.json',trace)
        if not response['tool_calls']:
            if response['decision']['action']=='submit':
                wf.submit(task,response['decision']['content'],str(where/'response.json'))
            return
        wf.receive_tools(task,response,str(where/'response.json'))
        wf.queue.ready()
        for child_id in wf.queue.get(tid)['waiting_for']:
            child=wf.queue.get(child_id)
            if child['kind']!='host_tool':
                row['pending_request']=child['inputs']
                atomic_json(folder/'trace.json',trace)
                return
            child=wf.queue.claim(child_id)
            value=wf.host_tool(child)
            row['read_chars']+=len(value.get('text',''))
            wf.queue.finish(child_id,result=value)
        atomic_json(folder/'trace.json',trace)


def recorder():
    dest=OUT/'run-A'
    if not dest.exists():
        dest.mkdir()
        for name in ('commits','cards','solution_library'):
            shutil.copytree(SOURCE/name,dest/name)
        shutil.copy2(SOURCE/'research_run.json',dest/'research_run.json')
    wf=Workflow(dest,intervention='Isolated validation fork of the measured focus checkpoint; add the versioned procedure library.')
    tid=wf.enqueue_recorder(1,incomplete=True,key='validation/record')
    turn(wf,tid,'recorder','复盘本批真实的冻结干预及下游反馈。若某个可复用的判别步骤值得在下次实验使用，可沿现有 Skill 修订或新建。只选择能从本批精确字段恢复的依据。')
    meta=wf.result_card(wf.queue.get(tid))
    atomic_json(OUT/'recorder-result.json',{'metareview_ref':meta and meta['id'],'items':meta and meta['payload'].get('skill_updates'),
        'publications':[e for e in wf.journal.events if e['type']=='skill_update_result']})


def consumer(control=False):
    if control:
        skills.SHARED_ROOT=OUT/'control-skills'
    dest=OUT/('run-B-control' if control else 'run-B')
    config=deepcopy(read_json(SOURCE/'research_run.json')['config'])
    config['isolated_solution_library']=True
    config['research_brief']['scientific_question'] += ' 已有同父的冻结单块 FP 替换结果，局部、分组与端点响应不同。先确定应投入哪一个合法 PTQ 干预、哪个竞争对照及何时回退。'
    wf=Workflow(dest,config) if not dest.exists() else Workflow(dest,intervention='Resume isolated read-only breakpoint validation after package recovery fixes.')
    tid=wf.enqueue_role('validation/focus','focus',refs=[wf.brief['id']],context={'round':1})
    if wf.queue.get(tid)['state']=='completed': return
    old=OUT/('consumer-control' if control else 'consumer')/'trace.json'
    if old.exists() and read_json(old)[-1].get('pending_request'): return
    turn(wf,tid,'consumer-control' if control else 'consumer','依据本次 Brief 为首个合法低比特动作选主题。已有研究出现局部恢复与不同末端目标不一致；可按需从流程目录或来源附件取得已有配对事实。说明下一项实际干预或提出实验。',limit=5)
    atomic_json(OUT/('consumer-control-result.json' if control else 'consumer-result.json'),{'topic':wf.topics.active(),'state':wf.queue.get(tid)['state']})

if __name__=='__main__':
    import sys
    if sys.argv[1]=='recorder': recorder()
    else: consumer(sys.argv[1]=='control')
