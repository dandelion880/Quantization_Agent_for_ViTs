"""Explicit authorized post-run maintenance; no resume, model call or evaluation."""
from pathlib import Path
import argparse
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'QAViT'))
from core.journal import Journal, read_json, atomic_json, file_sha256
from core.memory import ResearchMemory, MemoryRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    run = ROOT / 'EXP/runs/deitb_static_w4a4_gonly_20261115_r2'
    manifest = read_json(run / 'research_run.json')
    journal = Journal(run, manifest['run_id'])
    memory = ResearchMemory(journal)
    diagnosis = read_json(ROOT / 'EXP/inputs/deitb_static_w4a4_gonly_20261115_r2/scientific_memory_diagnosis.json')
    key = 'maintenance/20261116/recorder-B-timing'
    if key in journal.keys:
        plan = read_json(HERE / 'memory_correction_plan.json')
        commit = journal.keys[key]
        retrieved = MemoryRetriever(journal).retrieve('generation', query={'layer': 'L2'}, limit=50)
        selected = []
        for update in plan['updates']:
            row = next(r for r in retrieved['entries'] if r.get('claim_id') == update['claim_id'])
            assert row['body']['text'] == update['text'] and row['version'] == 2
            historical = MemoryRetriever(journal).retrieve('generation', explicit_refs=[update['supersedes']])
            assert any(r['claim_id'] == update['claim_id'] for r in historical['entries'][0]['knowledge_revisions'])
            selected.append(row)
        atomic_json(HERE / 'memory_correction_results.json', {'status': 'passed',
            'original_sequence': plan['original_sequence'], 'commit_sequence': commit['sequence'],
            'commit': commit, 'comparisons': plan['comparisons'], 'current_knowledge': selected,
            'chain_audit': journal.audit(), 'formal_run_resumed': False, 'model_called': False,
            'validation_note': 'Initial original cards/commits hash comparison passed. Initial final assertion incorrectly included derived checkpoint.json, which Journal.commit must update. Recheck verifies chain and current retrieval; no second correction was appended.',
            'unchanged_terminal_status': read_json(run / 'status.json'),
            'checkpoint_is_derived': read_json(run / 'checkpoint.json')})
        print('Existing append verified; no duplicate correction.')
        return
    original_sequence = journal.sequence
    originals = {str(p.relative_to(run)): file_sha256(p) for folder in ('cards', 'commits') for p in (run / folder).glob('*.json')}
    preserved = {name: file_sha256(run / name) for name in ('research_run.json', 'summary.json', 'status.json') if (run / name).exists()}
    prior_ev = 'EvidenceCard-34132bb00ead4c6f8f6b8ecbc4a26107@1'
    updates, comparisons = [], []
    state = memory.snapshot()
    for finding in diagnosis['findings']:
        old = state['L2'][finding['persisted_L2_claims'][0]['claim_id']]
        ev = old['evidence_refs'][0]
        before, after = journal.card(prior_ev), journal.card(ev)
        assert before['payload']['provenance']['model_ref'] == finding['pre_round_B_model_ref']
        assert after['payload']['provenance']['model_ref'] == finding['new_candidate_model_ref']
        assert before['payload']['provenance']['model_ref'] != after['payload']['provenance']['model_ref']
        metrics = lambda card: {metric: card['payload']['observations']['evaluations'][task]['metrics'][metric]
                     for task, metric in [('classification', 'top1'), ('segmentation', 'mIoU')]}
        b, a = metrics(before), metrics(after)
        diff = {k: a[k] - b[k] for k in a}
        text = (f"R{finding['round']}候选与预选时B属于不同冻结包；候选后来晋升为当前B，晋升后的身份相等不能否定本轮新执行或方案改动。"
                f"相对上一轮已完成方案，Top-1变化{diff['top1']:+.6f}、ADE mIoU变化{diff['mIoU']:+.6f}个百分点。"
                "这是既定协议下相邻完整方案的单次实测差异；不构成AdaLog-only同期配对消融、独立重复或机制因果证明。")
        updates.append({'claim_id': old['claim_id'], 'supersedes': old['revision_ref'],
            'text': text, 'status': 'conditional', 'evidence_refs': [prior_ev, ev],
            'rationale': '更正原知识将晋升后B当作晋升前对照的时间混淆。两份原Evidence记录不同model_ref及独立execution_id；保留原始数值与限制。'})
        comparisons.append({'round': finding['round'], 'before_ref': prior_ev, 'after_ref': ev,
            'before_model_ref': finding['pre_round_B_model_ref'], 'after_model_ref': finding['new_candidate_model_ref'], 'delta_percentage_points': diff})
        prior_ev = ev
    reason = '用户授权的事后三轮Recorder知识更正；仅追加修订，不改正式结果、原卡片或原执行身份。'
    guidance = ('区分预选时B与晋升后B；检索L2当前版本。R2/R3完整方案均发生实际改变和独立执行。'
                '保留代理指标冲突和H3尚未获得数值证据的限制；下一研究位点及校准策略由Generation依据证据选择。')
    atomic_json(HERE / 'memory_correction_plan.json', {'original_sequence': original_sequence, 'updates': updates, 'comparisons': comparisons, 'original_immutable_hashes': originals, 'preserved_files': preserved})
    if not args.apply:
        return
    commit = memory.append_correction(updates, reason=reason, operator='Codex engineering maintenance, explicitly requested by user; no Recorder model call', guidance=guidance, key=key)
    retrieved = MemoryRetriever(journal).retrieve('generation', query={'layer': 'L2'}, limit=50)
    for update in updates:
        row = next(r for r in retrieved['entries'] if r.get('claim_id') == update['claim_id'])
        assert row['body']['text'] == update['text'] and row['version'] == 2
        history = MemoryRetriever(journal).retrieve('generation', explicit_refs=[update['supersedes']])
        assert any(r['claim_id'] == update['claim_id'] for r in history['entries'][0]['knowledge_revisions'])
    assert all(file_sha256(run / p) == digest for p, digest in originals.items())
    assert all(file_sha256(run / p) == digest for p, digest in preserved.items())
    atomic_json(HERE / 'memory_correction_results.json', {'status': 'passed', 'commit_sequence': journal.sequence,
        'original_sequence': original_sequence, 'original_immutable_files_unchanged': len(originals),
        'preserved_files': preserved, 'commit': commit, 'comparisons': comparisons,
        'current_knowledge': [r for r in retrieved['entries'] if r.get('claim_id') in {u['claim_id'] for u in updates}],
        'formal_run_resumed': False, 'model_called': False})
    print('Appended and verified correction at sequence', journal.sequence)

if __name__ == '__main__':
    main()
