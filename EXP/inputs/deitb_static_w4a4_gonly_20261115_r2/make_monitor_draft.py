#!/usr/bin/env python3
"""Build provisional report from read-only monitoring receipts; pending stays pending."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'QAViT/tools'))
from evaluation.metrics import union_seconds


def read(path):
    return json.loads(path.read_text())


def short(value, limit=180):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text[:limit].replace('|', '\\|').replace('\n', ' ')


def metric(value):
    return 'pending' if value is None else f'{value:.6f}'.rstrip('0').rstrip('.')


def main():
    audit = read(HERE / 'round_audit_latest.json')
    observed = datetime.fromisoformat(audit['observed_at']).timestamp()
    lib = audit['solution_library_history']
    initial = [x for x in lib[0]['entries'] if x.get('status') == 'evaluated'] if lib else []
    best = max(initial, key=lambda x:(x['metrics']['mIoU'],x['metrics']['top1'])) if initial else None
    initial_B, previous_B = best, best
    finished = {x['round']:x for x in audit['round_finishes']}
    rounds = []
    for number in range(1,4):
        hs = [h for h in audit['hypotheses'] if h['payload'].get('round')==number]
        sel = next((s for s in audit['selections'] if s['payload'].get('round')==number),None)
        ns = [n for n in audit['numeric'] if n.get('round')==number]
        formal = [n for n in ns if n['level']=='L3' and n.get('formal',{}).get('full_dual_task_observed')]
        rec = [r for r in audit['recorder'] if r['payload'].get('round')==number]
        sequence = finished.get(number,{}).get('commit_sequence')
        snapshots = [s for s in lib if sequence is not None and s['sequence']<=sequence]
        after = snapshots[-1]['entries'] if snapshots else None
        row = {'round':number,'status':'completed' if number in finished else 'pending_or_running',
               'hypotheses':[{'ref':h['id'],**{k:h['payload'].get(k) for k in
                   ('slot','title','feedback_used','evidence_refs')}} for h in hs],
               'selection':{'ref':sel['id'],'payload':sel['payload']} if sel else None,
               'recorder_refs':[r['id'] for r in rec],
               'B_before':previous_B if (hs or sel or ns or number in finished) else None,
               'executions':[{'execution_id':n['execution_id'],'level':n['level'],'status':n['status'],
                   'evidence_ref':n['evidence_ref'],'error':n['error'],'binding_issues':n['binding_issues'],
                   'worker_wall_seconds':(n.get('resources')or{}).get('wall_seconds'),
                   'result':n['result']} for n in ns],
               'formal':[], 'solution_considered':[s for s in audit['solution_considered'] if s['round']==number]}
        for n in formal:
            f=n['formal'];q=f['segmentation'].get('metrics',{}).get('quantized',{})
            m={'top1':f['classification']['metrics']['top1'],'mIoU':q.get('mIoU')}
            def delta(base):
                return {k:m[k]-base['metrics'][k] for k in ('top1','mIoU')} if base else None
            row['formal'].append({'execution_id':n['execution_id'],'evidence_ref':n['evidence_ref'],
                'experiment_ref':n['experiment_ref'],'source_sha256':n['source_sha256'],
                'core_sha256':n['core_sha256'],'model_identity':f['model_identity'],
                'classification':f['classification'],'segmentation':f['segmentation'],'reload':f['reload'],
                'actual_calibration':f.get('scope_observations',{}).get('data',{}).get('calibration'),
                'static_w4a4':f['static_w4a4'],
                'artifacts_hashes_match':all(x['sha256']==x['current_sha256'] for x in f['artifacts']),
                'delta_initial_B_pp':delta(initial_B),'delta_previous_B_pp':delta(previous_B)})
        if after:
            evaluated=[x for x in after if x.get('status')=='evaluated']
            previous_B=max(evaluated,key=lambda x:(x['metrics']['mIoU'],x['metrics']['top1'])) if evaluated else previous_B
        row['B_after']=previous_B if number in finished else None
        rounds.append(row)
    budget=audit['budget']
    segments=budget.get('segments',[])
    intervals=[(s['started_unix'],s.get('stopped_unix') or observed) for s in segments]
    timing={'as_of_unix':observed,'active_seconds':budget['elapsed_seconds'],
        'excluded_numeric_wait_seconds':budget['excluded_numeric_wait_seconds'],
        'remaining_seconds':budget['remaining_seconds'],'configured_budget':budget['config'],
        'extensions':budget.get('extensions'),'segments':segments,
        'segment_wall_union_seconds_as_of':union_seconds(intervals),
        'all_segments_closed':all(s.get('stopped_unix') is not None for s in segments),
        'active_segment':budget.get('active_segment')}
    tools=Counter()
    for t in audit['tool_counts']:
        tools[(t['tool'],t['state'])]+=t['count']
    reads=[x for x in audit['read_ledger'] if x['state']=='completed']
    def asset_path(x):
        a=x.get('arguments') or {}
        return a.get('asset') or a.get('path') or a.get('ref') or ''
    assets=Counter(asset_path(x) for x in reads if x['tool']=='read_asset')
    components={'skill_catalog_deliveries':audit['skill_catalog_deliveries'],
        'skill_reads':[{'parent':x['parent'],'arguments':x['arguments'],'key':x['key']}
            for x in reads if x['tool']=='read_asset' and 'skills/' in str(x['arguments'])],
        'asset_read_counts':[{'path':k,'count':v} for k,v in sorted(assets.items())],
        'successful_read_card_count':tools[('read_card','completed')],
        'memory_retrievals':[{'key':x['key'],'state':x['state'],'arguments':x['arguments'],'parent':x['parent']}
            for x in audit['memory_retrievals']],
        'recorder_refs':[r['id'] for r in audit['recorder']],
        'implementation_revision_count':len(audit['implementation_revisions']),
        'safeguard_review_count':len(audit['safeguard_reviews'])}
    result={'schema':'qavit.gonly.monitor_draft/1','draft':True,'as_of':audit['observed_at'],
        'source_audit':'round_audit_latest.json','run':audit['run'],'status':audit['status'],
        'journal':{'sequence':audit['sequence'],'head':audit['head']},'initial_B':initial_B,
        'rounds':rounds,'timing':timing,'role_costs':audit['role_costs'],
        'all_agent_costs':audit['all_agent_costs'],'numeric_costs':audit['numeric_costs'],
        'model_routes':audit['model_routes'],'components':components,'failed_tasks':audit['failed_tasks'],
        'interventions':audit['interventions'],
        'experiment_request_outcomes':audit.get('experiment_request_outcomes',[]),
        'managed_terminal':read(HERE/'monitor_terminal_review.json') if
            (HERE/'monitor_terminal_review.json').exists() else None,
        'scientific_memory_diagnosis':'scientific_memory_diagnosis.json' if
            (HERE/'scientific_memory_diagnosis.json').exists() else None,
        'cache_reused_request_count':sum(x['result'].get('cache_reused') is True
            for x in audit.get('experiment_request_outcomes',[])),
        'independent_payload_comparisons':[n for n in
            ('r2_frozen_payload_comparison.json','r3_frozen_payload_comparison.json') if (HERE/n).exists()],
        'limitations':audit['limitations']+[
            'Provisional report; root will add preparation verification, external-session tokens and Git/archive receipts.',
            'No new L0 evaluation was started by this monitor. Pending results remain pending.',
            'GPU allocation wall is not measured GPU compute time or energy. No subscription charge is inferred.']}
    (HERE/'RESULTS_DRAFT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 三轮 G-only 正式实验监控报告（草稿）','',
        f"截至 `{audit['observed_at']}`，状态 `{audit['status']}`；Journal `{audit['sequence']}`。这是待三轮和托管清理完成后定稿的报告，未得结果保留 pending。",'',
        '## 已冻结的协议与证据边界','',
        'DeiT-B 官方非蒸馏模型、training-free、FP32 reference QDQ、static_w4a4/1；分类校准32张，正式 ImageNet 50,000张及 ADE20K 2,000张。ADE直接迁移分类冻结状态，既有neck/head不训练或重新校准。三轮合计14,400秒Agent活跃预算，排除纯数值/资源/结果等待；重叠推理、Coding与审核计时。L1/L2/L3时限3,600/10,800/86,400秒，恢复不重置。',
        '', 'Generation请求astra/high；Coding/Safeguard sol/medium，Recorder luna/high。实际服务端身份未回显。G-only没有Judger/Evolution/H4；监控者未提出候选或代替预选。',
        '', '初始历史B与当前执行区分；方案库/B按既有mIoU、Top-1次序保留，历史指标不算本轮新增测量。', '',
        '## 逐轮轨迹与正式结果','',
        '| 轮次 | 状态 / 预选 | 有效数值执行 | Recorder |',
        '|---|---|---|---|']
    for row in rounds:
        counts=Counter(n['level'] for n in row['executions'] if n['status']=='completed' and n['evidence_ref'])
        lines.append(f"| R{row['round']} | {row['status']} / {row['selection']['payload']['selected_slot'] if row['selection'] else 'pending'} | {dict(counts)} | {', '.join(row['recorder_refs']) or 'pending'} |")
    lines += ['', '| 方案 | ImageNet Top-1 / Top-5 | ADE mIoU / mAcc / aAcc | 相对初始B Top-1 / mIoU pp | 相对上一轮B Top-1 / mIoU pp |',
              '|---|---|---|---|---|']
    if initial_B:
        lines.append(f"| 历史初始B（本次未重测） | {metric(initial_B['metrics']['top1'])} / 90.294 | {metric(initial_B['metrics']['mIoU'])} / 37.788110 / 71.391383 | — | — |")
    for row in rounds:
        if not row['formal']:
            lines.append(f"| R{row['round']} | pending | pending | pending | pending |")
            continue
        for f in row['formal']:
            c=f['classification']['metrics'];s=f['segmentation']['metrics']['quantized']
            di=f['delta_initial_B_pp'];dp=f['delta_previous_B_pp']
            lines.append(f"| R{row['round']} | {metric(c['top1'])} / {metric(c['top5'])} | {metric(s['mIoU'])} / {metric(s['mAcc'])} / {metric(s['aAcc'])} | {metric(di['top1'])} / {metric(di['mIoU'])} | {metric(dp['top1'])} / {metric(dp['mIoU'])} |")
    lines += ['', '每份有效L3另记录配对FP、样本数、重载和代码binding，详见 `RESULTS_DRAFT.json#/rounds`；重载parity只覆盖回执实际记录的search样本，不扩大为全图逐样本等价。任务方向不一致时分别报告；局部诊断、理论推导、历史结果和完整任务证据不互相替代。','']
    for row in rounds:
        lines += [f"### R{row['round']}", '']
        for h in row['hypotheses']:
            lines.append(f"- {h['slot']}：{h['title']}；`{h['ref']}`。")
        if not row['hypotheses']:
            lines.append('pending。')
        if row['selection']:
            lines += ['', row['selection']['payload']['rationale']]
        for f in row['formal']:
            c=f['classification']['metrics'];s=f['segmentation']['metrics']['fp']
            lines += ['',f"正式Evidence `{f['evidence_ref']}`；execution `{f['execution_id']}`；代码 `{f['source_sha256']}`，core `{f['core_sha256']}`。配对FP ImageNet={metric(c['baseline_top1'])}/{metric(c['baseline_top5'])}，ADE={metric(s['mIoU'])}/{metric(s['mAcc'])}/{metric(s['aAcc'])}。全部原件hash匹配={f['artifacts_hashes_match']}。"]
        lines.append('')
    lines += ['## 真实失败、缓存与干预','', '| 执行 | 轮次 / 级别 | 状态 | worker秒 | 错误 |', '|---|---|---|---:|---|']
    for row in rounds:
        for n in row['executions']:
            lines.append(f"| `{n['execution_id']}` | R{row['round']} {n['level']} | {n['status']} | {metric(n['worker_wall_seconds'])} | {short(n['error']) if n['error'] else '—'} |")
    lines += ['', '成功执行须有真实Evidence；多臂不算独立重复。失败不填精度、缓存不冒充新实测。外部干预以Journal记录为准；内部修订、原始失败、Safeguard和代码版本保留。H3来源绑定两次失败的原始诊断见 `h3_binding_diagnosis.json`；关键人工核验见 `monitor_reviews.jsonl`。', '',
        '## 预算、LLM与GPU成本','',
        f"当前活跃 {metric(timing['active_seconds'])} 秒；纯数值等待排除 {metric(timing['excluded_numeric_wait_seconds'])} 秒；余额 {metric(timing['remaining_seconds'])} 秒；extensions={timing['extensions']}。Segment墙钟并集（截至采样，未闭合时会继续增长）{metric(timing['segment_wall_union_seconds_as_of'])}秒。worker墙钟和预算排除边界不同，不能相加当成同一成本。",'',
        '| role | 调用执行次数 | input | output | cached input（input子字段） | reasoning output（output子字段） | Router累计秒 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for c in audit['role_costs']:
        t=c['tokens'];known=lambda k:t[k]['known_sum']
        lines.append(f"| {c['role']} | {c['execution_attempts']} | {known('input_tokens')} | {known('output_tokens')} | {known('cached_input_tokens')} | {known('reasoning_output_tokens')} | {metric(c['duration']['router_elapsed_sum_seconds'])} |")
    nc=audit['numeric_costs']
    lines += ['',f"数值worker墙钟累计 {metric(nc['worker_wall_seconds_sum'])} 秒；失败 {metric(nc['failed_worker_wall_seconds_sum'])} 秒；已闭合执行区间并集 {metric(nc['closed_worker_interval_union_seconds'])} 秒；worker峰值并发 {nc['peak_numeric_workers']}。GPU分配墙钟 {metric(nc.get('allocated_gpu_wall_seconds_sum'))} GPU秒；该值不是纯GPU计算/能耗。实际货币账单、纯推理时延、服务端actual模型均未知。",'',
        '## Skill、知识、Memory与B的实际利用','',
        f"Skill目录交付 {audit['skill_catalog_deliveries']}；正文/参考实际成功读取 {len(components['skill_reads'])} 次；Memory retrieve {len(components['memory_retrievals'])} 次；read_card成功 {components['successful_read_card_count']} 次。逐次路径、角色和来源保留于机器账本，不以目录交付冒充Skill使用。",'',
        'Recorder及后轮HypothesisCard的feedback_used、实际retrieve_memory和read_card分别核验；后轮若尚未发生则保留pending。方案库和独立B保留来源/确切包/历史与本轮标识，不因某一新候选入选丢失已验证结果。', '',
        '## 待根代理合并与终态核验','',
        '- 启动前准备、验证及冻结身份。', '- R2/R3尚未闭合部分，以及三轮后托管退出、进程/GPU释放、state_saved。',
        '- 外部root与子agent会话Token账、Git/文件清单归档。',
        '- 无新增L0评分；不作Full-vs-G-only因果比较、首创性或原生INT4加速声明。','']
    (HERE/'REPORT_DRAFT.md').write_text('\n'.join(lines))
    print(json.dumps({'as_of':audit['observed_at'],'draft_rounds':[(r['round'],r['status']) for r in rounds],
                      'formal_results':sum(len(r['formal']) for r in rounds)},ensure_ascii=False))


if __name__=='__main__':
    main()
