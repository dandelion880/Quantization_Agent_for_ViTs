"""Real Generation host-tool loop on a fresh synthetic tiny ViT validation state.

Initial hypothesis cards are explicitly labeled validation fixtures. Decisions,
scientific revision text and experiment requests come from the real Generation.
No formal L3, real-dataset result or production-science decision is produced.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'QAViT'))
from core.cards import make_card, fingerprint
from core.journal import atomic_json, read_json
from core.workflow import Workflow
from runtime.identity import code_identity
from runtime.task_queue import TERMINAL
from tests.support import validation_config, hypothesis, experiment_payload

BASE = Path(__file__).parent


def create(name, cli):
    config = validation_config()
    config.update(name=name, purpose='validation', enable_l3=False)
    config.pop('scope_target', None)
    config['budget'].update(max_rounds=1, wall_seconds=1200, closeout_reserve_seconds=0,
                            shutdown_reserve_seconds=15, max_artifact_bytes=1024 ** 3)
    config['runtime'].update(cpu_threads=2, allowed_gpus=[], llm_concurrency=1,
                             experiment_concurrency=1, task_timeout_seconds=300,
                             numeric_task_timeout_seconds={'L1': 90, 'L2': 120, 'L3': 120})
    config['llm'].update(max_transport_retries=0)
    if cli:
        config['llm']['cli_path'] = cli
    brief = config['research_brief']
    brief['scientific_question'] = (
        '这是隔离的真实Generation工具闭环能力验证，不是正式科研轮次。初始H1/H2/H3卡均为宿主明确标注的未测验证fixture。'
        '在seed42随机tiny ViT blocks.0.mlp.fc2仅做局部static A4，权重和其它位点FP，合成RGB32，'
        '检验H2的主张：按calibration算子输出误差选per_channel，能否保证4张search整网输出误差也优于per_tensor？'
        '请实际调用request_experiment取得两种粒度的L2局部/整网误差，先用1张calibration、同4张search；'
        '如有必要再用2张calibration区分竞争解释。结论和科学修订内容由你根据真实反馈决定，不预填结果。'
        '本验证要实际覆盖revise_candidate及重新预选能力；若实测反证改变原科学定义，请单独调用该工具提交完整修订，'
        '然后在宿主新预选中比较当前确切版本。若不存在修订依据，诚实说明该能力未被覆盖，不捏造修订或结果。'
        '可用原生tensor_probe，method=rtn，per_channel axis=-1，per_tensor axis=null；固定bits=4,signed=true，'
        'calibration split=calibration、count=1或2，evaluation split=search、count=4；不训练、不读final、不联网。'
        '请聚焦这个小场景，避免开发大型代码或无关重复实验。不能外推ImageNet/ADE20K准确率。')
    brief['goals'] = {'validation': 'real tool evidence return, scientific revision, reselection and exact-version L2 execution; no L3'}
    brief['execution_guidance'] = {'notes': 'This validation uses only tensor_probe L1/L2. No complete W4A4 accuracy claim. All identities and dataset scope checks remain enabled.'}
    atomic_json(BASE / (name + '_input.json'), config)
    wf = Workflow(BASE / name, config)
    proposals = {
        'H1': ('Unmeasured validation fixture: per_tensor reference',
               'A4 per_tensor provides a fixed comparison; no claim of superiority.',
               {'site': 'blocks.0.mlp.fc2', 'method': 'rtn', 'granularity': 'per_tensor', 'axis': None,
                'bits': 4, 'signed': True, 'calibration_count': 1}),
        'H2': ('Unmeasured validation fixture: local-error-based channel selection',
               'Per_channel with lower calibration operator-output error is predicted also to reduce held-out network-output error compared with per_tensor, for calibration count 1 or 2. This untested transfer claim may be false.',
               {'site': 'blocks.0.mlp.fc2', 'method': 'rtn', 'granularity': 'per_channel', 'axis': -1,
                'bits': 4, 'signed': True, 'calibration_count': 1,
                'selection_rule': 'lower calibration operator-output NMSE predicts lower search network-output NMSE'}),
        'H3': ('Unmeasured validation fixture: downstream sensitivity explanation',
               'Local output error direction may interact with downstream Jacobians; scalar local NMSE alone need not order whole-network error. This fixture is a theoretical alternative, without implemented mechanism or measured gain.',
               {'status': 'theoretical alternative, unimplemented', 'measurement': 'paired local and whole-network errors only'}),
    }
    for slot, (title, mechanism, intervention) in proposals.items():
        body = hypothesis(1, slot)
        body.update(title=title, research_question=brief['scientific_question'], mechanism=mechanism,
                    intervention=intervention,
                    predictions=[{'id': 'p1', 'statement': mechanism,
                                  'measurement': 'paired calibration operator and search network relative MSE',
                                  'falsification': 'A reversed ordering refutes the claimed sufficient ranking rule.'}],
                    sources=[{'kind': 'explicit_validation_fixture', 'document': str(Path(__file__).resolve()),
                              'meaning': 'Host-supplied unmeasured setup, not generated research or empirical evidence.'}],
                    scope={'model': 'tiny_vit', 'validation_fixture': True, 'formal_l3': False})
        card = make_card('HypothesisCard', wf.journal.run_id, body, author='program')
        wf.journal.commit('validation/fixture/' + slot, cards=[card], events=[
            wf.memory.live_update(round_number=1, hypothesis_ref=card['id'])])
    return wf


def summary(wf, status, error):
    tasks = list(wf.queue.snapshot().values())
    evidence = wf.journal.find_cards('EvidenceCard')
    revisions = [e for e in wf.journal.events if e['type'] == 'candidate_revised']
    selected = wf.selection_card(1)
    followup = wf.queue.get_by_key('validation/exact_selected_version')
    bound = []
    if followup:
        for t in tasks:
            if t['kind'] != 'experiment_request' or t['inputs'].get('parent_task') != followup['task_id']:
                continue
            if not t.get('experiment_ref'):
                continue
            spec = wf.journal.card(t['experiment_ref'])
            outputs = [c for c in evidence if c['payload']['experiment_ref'] == spec['id']]
            bound.append({'task_id': t['task_id'], 'state': t['state'],
                          'experiment_ref': spec['id'], 'candidate_binding': spec['payload'].get('candidate_binding'),
                          'refs': spec['refs'], 'execution': {k: spec['payload']['execution'].get(k) for k in ('site', 'method', 'parameters', 'calibration', 'evaluation', 'seed')},
                          'original_request': t['inputs'].get('original_request'),
                          'evidence_refs': [c['id'] for c in outputs],
                          'measurement_source': [c['payload']['measurement_source'] for c in outputs],
                          'execution_id': [c['payload']['execution_id'] for c in outputs]})
    calls = []
    for p in sorted(wf.run_dir.glob('jobs/*/llm_audit.json')):
        a = read_json(p)
        calls.append({'path': str(p), **{k: a.get(k) for k in ('role', 'requested', 'actual', 'usage', 'error', 'elapsed_seconds')}})
    binding = {'selection_ref': selected['id'], 'hypothesis_ref': selected['payload']['selected_hypothesis_ref']} if selected else None
    exact = [row for row in bound if row['candidate_binding'] == binding
             and binding['hypothesis_ref'] in row['refs'] and binding['selection_ref'] in row['refs']
             and row['state'] == 'completed' and row['evidence_refs']
             and all(x in ('real', 'synthetic_computation') for x in row['measurement_source'])]
    chosen = wf.journal.card(binding['hypothesis_ref']) if binding else None
    intervention = chosen['payload']['intervention'] if chosen else {}
    def matches_intervention(row):
        e = row['execution']
        actual = {**e.get('parameters', {}), 'site': e.get('site'), 'method': e.get('method'),
                  'seed': e.get('seed'), 'calibration_count': e.get('calibration', {}).get('count')}
        fields = ('site', 'method', 'seed', 'calibration_count', 'granularity', 'axis', 'bits', 'signed')
        return isinstance(intervention, dict) and all(k in intervention and actual.get(k) == intervention[k] for k in fields)
    checks = {'actual_tool_numeric_evidence_returned': bool(evidence) and all(c['payload']['measurement_source'] in ('real', 'synthetic_computation') for c in evidence),
              'scientific_revision_published': bool(revisions),
              'reselection_uses_current_versions': bool(selected) and selected['payload']['candidate_refs'] == wf.candidate_refs(1)
                    and any(t['inputs'].get('purpose') == 'select' and t['key'].startswith('r001/select/') for t in tasks),
              'selected_exact_version_actually_executed': bool(exact),
              'executed_parameters_match_selected_intervention': bool(exact) and all(matches_intervention(row) for row in exact),
              'no_l3_executed': not any(c['payload']['level'] == 'L3' for c in wf.journal.find_cards('ExperimentCard')),
              'no_retired_roles': not any(t['role'] in ('judger', 'evolution') for t in tasks)}
    result = {'status': status, 'error': error, 'actual_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Real Generation and real torch on synthetic tiny ViT; initial H cards are explicit unmeasured host fixtures. Not a formal L3 or a complete research round.',
              'core_identity': code_identity(), 'checks': checks,
              'validation_passed': status == 'validation_completed' and all(checks.values()),
              'revision_events': revisions, 'selection_ref': selected['id'] if selected else None,
              'candidate_refs': wf.candidate_refs(1), 'selected_candidate_payload': chosen['payload'] if chosen else None, 'exact_execution': bound,
              'evidence': [{'ref': c['id'], 'experiment_ref': c['payload']['experiment_ref'],
                            'execution_id': c['payload']['execution_id'], 'observations': c['payload']['observations']} for c in evidence],
              'tasks': [{k: t.get(k) for k in ('task_id', 'key', 'kind', 'state', 'role', 'error')} for t in tasks],
              'calls': calls, 'budget': wf.budget.snapshot()}
    atomic_json(BASE / (wf.run_dir.name + '_result.json'), result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='live_generation_r1')
    p.add_argument('--cli')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    wf = Workflow(BASE / args.name) if args.resume else create(args.name, args.cli)
    status, error = 'validation_incomplete', None
    last = None
    try:
        while True:
            wf.budget.check()
            wf.poll()
            wf.progress_preselection(1, wf.round_hypotheses(1), [])
            selected = wf.selection_card(1)
            if selected:
                if selected['payload']['selected_slot'] is None:
                    raise RuntimeError('Generation selected no candidate; exact execution remains uncovered')
                if not any(e['type'] == 'candidate_revised' for e in wf.journal.events):
                    raise RuntimeError('Real Generation completed selection without scientific revision; capability remains uncovered')
                identifier = wf.enqueue_role('validation/exact_selected_version', 'select',
                    refs=[wf.brief['id'], selected['id'], *wf.candidate_refs(1).values(),
                          *[c['id'] for c in wf.journal.find_cards('EvidenceCard')]], context={
                        'round': 1, 'selection_stage': 'gonly_preselection', 'candidate_refs': wf.candidate_refs(1),
                        'validation_followup': {
                            'selection_ref': selected['id'],
                            'hypothesis_ref': selected['payload']['selected_hypothesis_ref'],
                            'instruction': '当前是预选后的工程能力验证。请基于你已选择的确切候选版本，用request_experiment实际提交一个L2 tensor_probe小样本请求以检查该版本可执行。'
                                '保留同一科学干预和参数，不改预选、不造L3。experiment顶层candidate_binding明确写selection_ref/hypothesis_ref（见本对象），'
                                '并以hypothesis_ref明确绑定来源。不要复用过期父版本或偷偷换干预；测量结束后解释实际返回并提交与已选相同的selected_slot/rationale。'
                                '即使数值参数与先前测量相同，也须通过replication.requested=true和purpose明确这是同条件工程重复检查，不能把它称为独立科研重复或新正式成绩。'}})
                followup = wf.queue.get(identifier)
                if followup['state'] == 'completed':
                    status = 'validation_completed'
                    break
                if followup['state'] in TERMINAL | {'needs_clarification'}:
                    raise RuntimeError('Exact-version followup did not complete: ' + str(followup.get('error') or followup['state']))
            active = [(t['key'], t['state'], t['attempt']) for t in wf.queue.snapshot().values() if t['state'] not in TERMINAL]
            if active != last:
                print(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'active': active,
                                  'evidence_count': len(wf.journal.find_cards('EvidenceCard'))}, ensure_ascii=False), flush=True)
                last = active
            wf.dispatch()
            time.sleep(.2)
    except BaseException as exc:
        error = {'code': type(exc).__name__, 'message': str(exc)}
    finally:
        wf.runtime.stop()
        wf.poll()
        wf.queue.cancel_unfinished('local_validation_stopped')
    result = summary(wf, status, error)
    print(json.dumps({k: result[k] for k in ('status', 'error', 'checks', 'validation_passed')}, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result['validation_passed'] else 2)


if __name__ == '__main__':
    main()
